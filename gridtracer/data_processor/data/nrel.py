"""
NREL data handler for gridtracer.

This module provides functionality to process NREL residential building
typology datasets.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, OrderedDict

import pandas as pd
from tqdm import tqdm

from gridtracer.data_processor.data.base import DataHandler

if TYPE_CHECKING:
    from gridtracer.data_processor.workflow import WorkflowOrchestrator

# Expected NREL vintage categories (based on actual NREL data)
EXPECTED_VINTAGE_BINS = [
    "<1940", "1940s", "1950s", "1960s", "1970s",
    "1980s", "1990s", "2000s", "2010s"
]

# Set up logging
logger = logging.getLogger(__name__)


class NRELDataHandler(DataHandler):
    """
    Handler for NREL data.

    This class handles processing NREL residential building typology data,
    which provides information for building classification and energy demand estimation.
    It uses the WorkflowOrchestrator for context and configuration.
    """

    def __init__(self, orchestrator: 'WorkflowOrchestrator'):
        """
        Initialize the NREL data handler.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance,
                providing access to configuration, FIPS, output paths, and input data paths.
        """
        super().__init__(orchestrator)  # Call base class init with orchestrator

        # Get the NREL input file path from the orchestrator (which gets it from config)
        input_data_paths = self.orchestrator.get_input_data_paths()
        self.input_file_path: Optional[Path] = None
        nrel_path_str = input_data_paths.get('nrel_data')

        if nrel_path_str:
            self.input_file_path = Path(nrel_path_str)
            self.logger.info(f"NREL input file path set to: {self.input_file_path}")
        else:
            self.logger.warning(
                "NREL input data path ('nrel_data') not found in configuration."
            )
            # Depending on requirements, you might raise an error here if essential
            # raise ValueError("NREL input file path ('nrel_data') required.")

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name.
        """
        return "NREL"

    def download(self) -> Dict[str, Optional[Path]]:
        """
        Process local NREL file and extract data for the target region.
        Returns paths to processed files.
        """
        if not self._validate_inputs():
            return {"parquet_path": None, "csv_path": None}

        fips = self.orchestrator.get_fips_dict()
        state_fips = fips['state_fips']
        county_fips = fips['county_fips']

        # Define output file paths
        filename_base = f"NREL_residential_typology_{state_fips}_{county_fips}"
        parquet_path = self.dataset_output_dir / f"{filename_base}.parquet"
        csv_path = self.dataset_output_dir / f"{filename_base}.csv"

        # Check if files already exist
        if parquet_path.exists() and csv_path.exists():
            region_name = f"{fips['state']}, {fips['county']}"
            self.logger.info(f"NREL files already exist for {region_name}")
            return {"parquet_path": parquet_path, "csv_path": csv_path}

        # Extract data if files don't exist
        return self._extract_and_save_nrel_data(parquet_path, csv_path)

    def process(self) -> Dict[str, any]:
        """
        Process NREL data for the region with consistent output structure.

        Returns:
            Dict containing parquet_path, csv_path, data, and vintage_distribution
        """
        # Standard result structure
        result: Dict[str, any] = {
            'parquet_path': None,
            'csv_path': None,
            'data': None,
            'vintage_distribution': OrderedDict()
        }

        if not self._validate_inputs():
            return result

        # Get file paths (download if needed)
        paths = self.download()
        result.update(paths)

        parquet_path = result['parquet_path']
        if not parquet_path or not parquet_path.exists():
            return result

        # Compute vintage distribution
        result['vintage_distribution'] = self.compute_vintage_distribution(parquet_path)

        # Load data
        try:
            result['data'] = pd.read_parquet(parquet_path)
            self.logger.info(f"Loaded {len(result['data'])} NREL records")
        except Exception as e:
            self.logger.error(f"Error loading NREL data: {e}")

        return result

    def _validate_inputs(self) -> bool:
        """Validate required inputs for NREL processing."""
        if not self.input_file_path or not self.input_file_path.exists():
            self.logger.error(f"NREL input file not found: {self.input_file_path}")
            return False

        fips = self.orchestrator.get_fips_dict()
        if not fips:
            self.logger.error("FIPS dictionary not available")
            return False

        return True

    def _extract_and_save_nrel_data(self, parquet_path: Path,
                                    csv_path: Path) -> Dict[str, Optional[Path]]:
        """Extract NREL data for the region and save to files."""
        fips = self.orchestrator.get_fips_dict()
        state_fips = fips['state_fips']
        county_fips = fips['county_fips']
        region_name = f"{fips['state']}, {fips['county']}"

        self.logger.info(f"Extracting NREL data for {region_name}")

        str_state_fips = str(state_fips).zfill(2)
        str_county_fips = str(county_fips).zfill(3)

        county_data_frames = []
        chunk_size = 100_000

        try:
            # Get total lines for progress bar
            try:
                with open(self.input_file_path, 'r') as f:
                    total_lines = sum(1 for _ in f) - 1
                total_chunks = max(
                    1, (total_lines // chunk_size) + (1 if total_lines % chunk_size > 0 else 0)
                )
            except Exception:
                total_chunks = None
                self.logger.warning("Could not determine file size for progress tracking")

            # Process file in chunks
            with tqdm(
                total=total_chunks, desc=f"Processing NREL for {region_name}", unit="chunk"
            ) as pbar:
                for chunk in pd.read_csv(
                    self.input_file_path, sep="\t", chunksize=chunk_size, low_memory=False
                ):

                    if 'in.county' not in chunk.columns:
                        pbar.update(1)
                        continue

                    # Filter for target county
                    county_ids_no_g = chunk['in.county'].astype(str).str.removeprefix('G')

                    state_match = pd.Series(False, index=county_ids_no_g.index)
                    valid_state = county_ids_no_g.str.len() >= 2
                    state_match[valid_state] = (
                        county_ids_no_g[valid_state].str[:2] == str_state_fips
                    )

                    county_match = pd.Series(False, index=county_ids_no_g.index)
                    valid_county = county_ids_no_g.str.len() >= 6
                    county_match[valid_county] = (
                        county_ids_no_g[valid_county].str[3:6] == str_county_fips
                    )

                    county_chunk = chunk[state_match & county_match]

                    if not county_chunk.empty:
                        county_data_frames.append(county_chunk)
                        pbar.set_postfix_str(
                            f"Found {sum(len(df) for df in county_data_frames)} total rows"
                        )

                    pbar.update(1)

            # Save results if data found
            if county_data_frames:
                county_data = pd.concat(county_data_frames, ignore_index=True)
                county_data.to_parquet(parquet_path, index=False)
                county_data.to_csv(csv_path, index=False)

                self.logger.info(f"Saved {len(county_data)} NREL records to {parquet_path}")
                return {"parquet_path": parquet_path, "csv_path": csv_path}
            else:
                self.logger.warning(f"No NREL data found for {region_name}")
                return {"parquet_path": None, "csv_path": None}

        except Exception as e:
            self.logger.error(f"Error extracting NREL data: {e}", exc_info=True)
            return {"parquet_path": None, "csv_path": None}

    def compute_vintage_distribution(
        self,
        parquet_path: Path,
        vintage_col: str = "in.vintage",
    ) -> OrderedDict[str, float]:
        """
        Weighted percentage distribution of NREL building‐stock 'vintage' bins.

        Parameters
        ----------
        parquet_path : Path
            Path to parquet file with NREL data for the region
        vintage_col : str, default ``"in.vintage"``
            Column holding the construction-period label
        Returns
        -------
        OrderedDict[str, float]
            Keys are the nine bins defined in ``EXPECTED_VINTAGE_BINS``. Values are percentages
        """
        df = pd.read_parquet(parquet_path)

        if vintage_col not in df.columns:
            msg = f"Column '{vintage_col}' not found – cannot build vintage distribution."
            logger.warning(msg)
            return OrderedDict((k, 0.0) for k in EXPECTED_VINTAGE_BINS)

        # Map NREL vintage labels directly to our bins
        nrel_to_bins_mapping = {
            '<1940': '<1940',
            '1940s': '1940s',
            '1950s': '1950s',
            '1960s': '1960s',
            '1970s': '1970s',
            '1980s': '1980s',
            '1990s': '1990s',
            '2000s': '2000s',
            '2010s': '2010s'
        }

        # Map each record to a bin using direct label mapping
        bin_labels = df[vintage_col].map(nrel_to_bins_mapping)
        bin_labels = bin_labels.fillna("Unknown")

        # Count records in each bin
        counts = bin_labels.value_counts(dropna=False)

        # Ensure all expected bins are present (fill missing with 0)
        counts = counts.reindex(EXPECTED_VINTAGE_BINS, fill_value=0)

        # Convert to percentages
        total = counts.sum()
        if total > 0:
            perc = (counts / total).round(3)
        else:
            perc = pd.Series(0.0, index=EXPECTED_VINTAGE_BINS)

        return OrderedDict(perc.to_dict())

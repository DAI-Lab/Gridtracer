"""
NREL data handler

This module provides functionality to process NREL residential building
typology datasets.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, OrderedDict

import pandas as pd

from gridtracer.config.config_loader import INPUT_DATA
from gridtracer.data.imports.base import DataHandler

if TYPE_CHECKING:
    from gridtracer.data.workflow import WorkflowOrchestrator

# Expected NREL vintage categories (based on actual NREL data)
EXPECTED_VINTAGE_BINS = [
    "<1940",
    "1940s",
    "1950s",
    "1960s",
    "1970s",
    "1980s",
    "1990s",
    "2000s",
    "2010s",
]

CHUNK_SIZE = 100_000
NREL_COUNTY_COLUMN = "in.county"
NREL_VINTAGE_COLUMN = "in.vintage"


class NRELDataHandler(DataHandler):
    """
    Handler for NREL data.

    This class handles processing NREL residential building typology data,
    which provides information for building classification and energy demand estimation.
    It uses the WorkflowOrchestrator for context and configuration.
    """

    def __init__(self, orchestrator: "WorkflowOrchestrator"):
        """
        Initialize the NREL data handler.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance,
                providing access to configuration, FIPS, output paths, and input data paths.
        """
        super().__init__(orchestrator)  # Call base class init with orchestrator

        self.input_file_path: Optional[Path] = None
        nrel_path_str = INPUT_DATA.get("NREL_FILE")

        if nrel_path_str:
            self.input_file_path = Path(nrel_path_str)
        else:
            self.logger.warning("NREL input data path ('nrel_data') not found in configuration.")

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name.
        """
        return "NREL"

    def download(self) -> Dict[str, Optional[Path]]:
        """Get or create NREL data files for the target region."""
        if not self._validate_inputs():
            return {"parquet_path": None, "csv_path": None}

        file_paths = self._get_output_file_paths()

        # Check if files already exist
        if self._files_exist(file_paths):
            self._log_files_exist()
            return file_paths

        # Extract data if files don't exist
        return self._extract_and_save_data(file_paths)

    def process(self) -> Dict[str, Any]:
        """
        Process NREL data for the region with consistent output structure.

        Returns:
            Dict containing parquet_path, csv_path, data, and vintage_distribution
        """
        # Standard result structure
        result: Dict[str, Any] = {
            "parquet_path": None,
            "csv_path": None,
            "data": None,
            "vintage_distribution": OrderedDict(),
        }

        if not self._validate_inputs():
            return result

        # Get file paths (download if needed)
        paths = self.download()
        result.update(paths)

        parquet_path = result["parquet_path"]
        if not parquet_path or not parquet_path.exists():
            return result

        # Compute vintage distribution
        result["vintage_distribution"] = self._compute_vintage_distribution(parquet_path)

        # Load data
        try:
            result["data"] = pd.read_parquet(parquet_path)
            self.logger.info(f"Loaded {len(result['data'])} NREL records")
        except Exception as e:
            self.logger.error(f"Error loading NREL data: {e}")

        return result

    def _validate_inputs(self) -> bool:
        """Validate required inputs for NREL processing."""
        if not self.input_file_path or not self.input_file_path.exists():
            self.logger.error(f"NREL input file not found: {self.input_file_path}")
            return False

        return True

    def _get_output_file_paths(self) -> Dict[str, Path]:
        """Generate output file paths based on FIPS codes."""
        fips = self.orchestrator.fips_dict
        filename_base = f"NREL_residential_typology_{fips['state_fips']}_{fips['county_fips']}"

        return {
            "parquet_path": self.dataset_output_dir / f"{filename_base}.parquet",
            "csv_path": self.dataset_output_dir / f"{filename_base}.csv",
        }

    def _files_exist(self, file_paths: Dict[str, Path]) -> bool:
        """Check if both output files already exist."""
        return file_paths["parquet_path"].exists() and file_paths["csv_path"].exists()

    def _log_files_exist(self) -> None:
        """Log that files already exist."""
        fips = self.orchestrator.fips_dict
        region_name = f"{fips['state']}, {fips['county']}"
        self.logger.info(f"NREL files already exist for {region_name}")

    def _extract_and_save_data(self, file_paths: Dict[str, Path]) -> Dict[str, Optional[Path]]:
        """Extract NREL data and save to files."""
        fips = self.orchestrator.fips_dict
        region_name = f"{fips['state']}, {fips['county']}"

        self.logger.info(f"Extracting NREL data for {region_name}")

        try:
            county_data = self._process_file_chunks(fips["state_fips"], fips["county_fips"])

            if county_data.empty:
                self.logger.warning(f"No NREL data found for {region_name}")
                return {"parquet_path": None, "csv_path": None}

            # Save files
            self._save_data_files(county_data, file_paths)
            return file_paths

        except Exception as e:
            self.logger.error(f"Error extracting NREL data: {e}", exc_info=True)
            return {"parquet_path": None, "csv_path": None}

    def _process_file_chunks(self, state_fips: str, county_fips: str) -> pd.DataFrame:
        """Process CSV file in chunks and extract county data."""
        county_data_frames = []

        for chunk in pd.read_csv(self.input_file_path, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
            county_chunk = self._extract_county_chunk(chunk, state_fips, county_fips)
            if not county_chunk.empty:
                county_data_frames.append(county_chunk)

        return pd.concat(county_data_frames, ignore_index=True) if county_data_frames else pd.DataFrame()

    def _save_data_files(self, data: pd.DataFrame, file_paths: Dict[str, Path]) -> None:
        """Save data to both parquet and CSV files."""
        data.to_parquet(file_paths["parquet_path"], index=False)
        data.to_csv(file_paths["csv_path"], index=False)

    def _extract_county_chunk(self, chunk: pd.DataFrame, state_fips: str, county_fips: str) -> pd.DataFrame:
        """Extract records matching the target county from a data chunk."""
        if NREL_COUNTY_COLUMN not in chunk.columns:
            return pd.DataFrame()

        # Clean county IDs (remove 'G' prefix)
        county_ids = chunk[NREL_COUNTY_COLUMN].astype(str).str.removeprefix("G")

        # Create state and county filters
        state_filter = self._create_state_filter(county_ids, state_fips)
        county_filter = self._create_county_filter(county_ids, county_fips)

        return chunk[state_filter & county_filter]

    def _create_state_filter(self, county_ids: pd.Series, state_fips: str) -> pd.Series:
        """Create boolean filter for state FIPS matching."""
        state_match = pd.Series(False, index=county_ids.index)
        valid_length = county_ids.str.len() >= 2
        state_match[valid_length] = county_ids[valid_length].str[:2] == state_fips.zfill(2)
        return state_match

    def _create_county_filter(self, county_ids: pd.Series, county_fips: str) -> pd.Series:
        """Create boolean filter for county FIPS matching."""
        county_match = pd.Series(False, index=county_ids.index)
        valid_length = county_ids.str.len() >= 6
        county_match[valid_length] = county_ids[valid_length].str[3:6] == county_fips.zfill(3)
        return county_match

    def _compute_vintage_distribution(
        self,
        parquet_path: Path,
        vintage_col: str = NREL_VINTAGE_COLUMN,
    ) -> OrderedDict[str, float]:
        """
        Weighted percentage distribution of NREL building‐stock 'vintage' bins.

        Parameters
        ----------
        parquet_path : Path
            Path to parquet file with NREL data for the region
        vintage_col : str, default ``NREL_VINTAGE_COLUMN``
            Column holding the construction-period label
        Returns
        -------
        OrderedDict[str, float]
            Keys are the nine bins defined in ``EXPECTED_VINTAGE_BINS``. Values are percentages
        """
        df = pd.read_parquet(parquet_path)

        if vintage_col not in df.columns:
            msg = f"Column '{vintage_col}' not found – cannot build vintage distribution."
            self.logger.warning(msg)
            return OrderedDict((k, 0.0) for k in EXPECTED_VINTAGE_BINS)

        # Map NREL vintage labels directly to our bins
        nrel_to_bins_mapping = {
            "<1940": "<1940",
            "1940s": "1940s",
            "1950s": "1950s",
            "1960s": "1960s",
            "1970s": "1970s",
            "1980s": "1980s",
            "1990s": "1990s",
            "2000s": "2000s",
            "2010s": "2010s",
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

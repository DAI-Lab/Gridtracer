"""
NREL data handler for SynGrid.

This module provides functionality to process NREL residential building
typology datasets.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from syngrid.data_processor.data.base import DataHandler

if TYPE_CHECKING:
    from syngrid.data_processor.workflow import WorkflowOrchestrator

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
                "NREL input data path ('nrel_data') not found in configuration via orchestrator."
            )
            # Depending on requirements, you might raise an error here if it's essential
            # raise ValueError("NREL input file path ('nrel_data') is required but not configured.")

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name.
        """
        return "NREL"

    def _extract_nrel_data_for_region(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Processes an existing NREL dataset file to extract data for the target region.

        Returns:
            tuple: (parquet_path, csv_path) if successful, (None, None) otherwise.
        """
        if self.input_file_path is None:
            self.logger.error("No input file path configured for NREL data processing.")
            return None, None

        if not self.input_file_path.exists():
            self.logger.error(f"NREL input file not found: {self.input_file_path}")
            return None, None

        fips = self.orchestrator.get_fips_dict()
        if not fips:
            self.logger.error(
                "FIPS dictionary not available from orchestrator for NREL processing.")
            return None, None  # Or raise error

        state_fips = fips['state_fips']
        county_fips = fips['county_fips']
        region_name_for_log = f"{fips['state']}, {fips['county']}"

        self.logger.info(f"Processing NREL data for {region_name_for_log}")

        # Output filenames are now constructed using self.dataset_output_dir (from
        # base DataHandler)
        filename_base = f"NREL_Residential_typology_{state_fips}_{county_fips}"
        parquet_file = self.dataset_output_dir / f"{filename_base}.parquet"
        csv_file = self.dataset_output_dir / f"{filename_base}.csv"

        if parquet_file.exists() and csv_file.exists():
            self.logger.info(
                f"NREL files already exist for {region_name_for_log} "
                f"(FIPS: {state_fips}_{county_fips}) at {self.dataset_output_dir}"
            )
            return parquet_file, csv_file

        county_data_frames = []
        try:
            chunk_size = 100_000  # Underscore for readability

            try:
                # More efficient way to count lines for tqdm progress bar
                with open(self.input_file_path, 'r') as f_count:
                    total_lines = sum(1 for _ in f_count) - 1  # -1 for header
                if total_lines < 0: total_lines = 0 # Handle empty or header-only file
                total_chunks = (total_lines // chunk_size) + \
                    (1 if total_lines % chunk_size > 0 and total_lines > 0 else 0)
            except Exception:  # Fallback if line counting fails
                self.logger.warning("Could not determine total lines for NREL progress bar.", exc_info=True)
                total_chunks = None  # tqdm will run without a total

            with tqdm(total=total_chunks, desc=f"Processing NREL for {region_name_for_log}", unit="chunk") as pbar:
                for chunk in pd.read_csv(self.input_file_path, sep="\t",
                                         chunksize=chunk_size, low_memory=False):
                    if 'in.county' not in chunk.columns:
                        self.logger.error("'in.county' column not found in NREL data chunk.")
                        pbar.update(1)
                        continue

                    # Ensure FIPS codes from orchestrator are strings for comparison
                    # (They should already be, but this is a safe check)
                    str_state_fips = str(state_fips).zfill(2)
                    str_county_fips = str(county_fips).zfill(3)

                    # KISS filtering logic based on your previous working snippet
                    county_ids_no_g = chunk['in.county'].astype(str).str.removeprefix('G')
                    
                    # Pandas Series boolean conditions for filtering
                    # Ensure string is long enough before slicing
                    state_match = pd.Series(False, index=county_ids_no_g.index)
                    valid_for_state_slice = county_ids_no_g.str.len() >= 2
                    state_match[valid_for_state_slice] = county_ids_no_g[valid_for_state_slice].str[:2] == str_state_fips
                    
                    county_match = pd.Series(False, index=county_ids_no_g.index)
                    # Need at least 6 characters for slice [3:6] (e.g., SSXCCC)
                    valid_for_county_slice = county_ids_no_g.str.len() >= 6 
                    county_match[valid_for_county_slice] = county_ids_no_g[valid_for_county_slice].str[3:6] == str_county_fips
                    
                    county_chunk = chunk[state_match & county_match]

                    if not county_chunk.empty:
                        county_data_frames.append(county_chunk)
                        pbar.set_postfix_str(
                            f"Found {len(county_chunk)} rows, total {sum(len(df) for df in county_data_frames)}",
                            refresh=True)
                    pbar.update(1)

            if county_data_frames:
                county_data = pd.concat(county_data_frames, ignore_index=True)
                rows_count = len(county_data)
                self.logger.info(f"Saving {rows_count} NREL rows for {region_name_for_log}")
                county_data.to_parquet(parquet_file, index=False)
                county_data.to_csv(csv_file, index=False)
                self.logger.info(f"Saved NREL data to: {parquet_file} and {csv_file}")
                return parquet_file, csv_file
            else:
                self.logger.warning(
                    f"No NREL data found for {region_name_for_log} (FIPS: {state_fips}_{county_fips})")
                return None, None
        except Exception as e:
            self.logger.error(
                f"Error processing NREL file {self.input_file_path}: {e}",
                exc_info=True)
            return None, None

    def download(self) -> Dict[str, Optional[Path]]:  # Adjusted return type to match likely usage
        """
        Abstract download method implementation for NRELDataHandler.
        For NREL, "download" means processing the local file.
        This method calls the main data extraction logic.
        """
        self.logger.debug(
            "NRELDataHandler.download() called, deferring to _extract_nrel_data_for_region.")
        parquet_path, csv_path = self._extract_nrel_data_for_region()
        return {
            "parquet_path": parquet_path,
            "csv_path": csv_path
        }

    def process(self, boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> Dict[str, any]:
        """
        Process the NREL data for the region.

        Args:
            boundary_gdf (Optional[gpd.GeoDataFrame]): Not used by this method as NREL data
                                                     is tabular and filtered by FIPS codes.
        Returns:
            dict: Dictionary containing:
                - 'parquet_path': Path to the Parquet file for the target region.
                - 'csv_path': Path to the CSV file for the target region.
                - 'data': Pandas DataFrame with the NREL data if successfully loaded.
        """


        fips = self.orchestrator.get_fips_dict()
        if not fips:  # Redundant if base class validated, but safe
            self.logger.error(
                "FIPS dictionary not available from orchestrator for NREL processing.")
            raise ValueError("FIPS dictionary missing for NREL processing.")

        region_name_for_log = f"{fips['state']}, {fips['county']}"
        self.logger.info(f"Processing NREL data for {region_name_for_log}")

        paths = self.download()  # Calls the refined download which calls _extract_nrel_data
        parquet_path = paths.get("parquet_path")
        csv_path = paths.get("csv_path")

        result: Dict[str, any] = {
            'parquet_path': parquet_path,
            'csv_path': csv_path,
            'data': None
        }

        if parquet_path and parquet_path.exists():
            try:
                data = pd.read_parquet(parquet_path)
                result['data'] = data
                self.logger.info(
                    f"Successfully loaded {len(data)} rows of processed NREL data from {parquet_path}")
            except Exception as e:
                self.logger.error(
                    f"Error loading processed NREL data from Parquet {parquet_path}: {e}",
                    exc_info=True)
        elif parquet_path:  # Path exists but file doesn't - indicates processing might have failed to write
            self.logger.warning(
                f"NREL Parquet file path exists ({parquet_path}) but file not found. Data not loaded.")

        return result

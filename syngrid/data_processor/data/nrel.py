"""
NREL data handler for SynGrid.

This module provides functionality to process NREL residential building
typology datasets.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from syngrid.data_processor.data.base import DataHandler

# Set up logging
logger = logging.getLogger(__name__)


class NRELDataHandler(DataHandler):
    """
    Handler for NREL data.

    This class handles processing NREL residential building typology data,
    which provides information for building classification and energy demand estimation.
    """

    def __init__(self, fips_dict, input_file_path=None, output_dir=None):
        """
        Initialize the NREL data handler.

        Args:
            fips_dict (dict): Dictionary containing region information
            input_file_path (str or Path, optional): Path to the NREL data file
            output_dir (str or Path, optional): Base output directory
        """
        super().__init__(fips_dict, output_dir)
        self.input_file_path = input_file_path

    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name
        """
        return "NREL"

    def download(self):
        """
        Process NREL data for the specified region.

        Note: Unlike other data handlers, this doesn't download data from the web
        but rather processes an existing NREL dataset file provided at initialization.

        Returns:
            tuple: (parquet_path, csv_path) if successful, (None, None) otherwise
        """
        if self.input_file_path is None:
            logger.error("No input file path provided for NREL data processing")
            return None, None

        input_file = Path(self.input_file_path)
        if not input_file.exists():
            logger.error(f"NREL input file not found: {input_file}")
            return None, None

        logger.info(
            f"Processing NREL data for {self.fips_dict['state']}, {self.fips_dict['county']}")

        # Create output filenames
        state_fips = self.fips_dict['state_fips']
        county_fips = self.fips_dict['county_fips']
        filename_base = f"NREL_Residential_typology_{state_fips}_{county_fips}"
        parquet_file = self.dataset_output_dir / f"{filename_base}.parquet"
        csv_file = self.dataset_output_dir / f"{filename_base}.csv"

        # Check if files already exist
        if parquet_file.exists() and csv_file.exists():
            logger.info(
                f"NREL files already exist for {self.fips_dict['state']}, "
                f"{self.fips_dict['county']} (FIPS: {state_fips}_{county_fips})"
            )
            return parquet_file, csv_file

        # Track if we found any matching data
        county_data_frames = []

        try:
            # Process the file in chunks
            chunk_size = 100000

            # Count total number of chunks for tqdm
            total_chunks = sum(1 for _ in pd.read_csv(input_file, sep="\t", chunksize=chunk_size))

            # Process chunks with progress bar
            with tqdm(total=total_chunks, desc=f"Processing {self.fips_dict['state']}, {self.fips_dict['county']} data") as pbar:
                for chunk in pd.read_csv(input_file, sep="\t", chunksize=chunk_size):
                    # Filter chunk for the target county
                    # Remove G prefix and extract state/county codes
                    county_ids = chunk['in.county'].astype(str).str.replace('G', '', regex=False)
                    state_match = county_ids.str[:2] == state_fips
                    county_match = county_ids.str[3:6] == county_fips
                    county_chunk = chunk[state_match & county_match]

                    # If found rows for this county, add to our list
                    if not county_chunk.empty:
                        county_data_frames.append(county_chunk)
                        total_rows = sum(len(df) for df in county_data_frames)
                        pbar.set_postfix(found_rows=len(county_chunk), total=total_rows)

                    pbar.update(1)

            # Combine all chunks with county data
            if county_data_frames:
                county_data = pd.concat(county_data_frames, ignore_index=True)
                rows_count = len(county_data)

                logger.info(
                    f"Saving {rows_count} rows for {self.fips_dict['state']}, {self.fips_dict['county']}")

                # Save to parquet
                county_data.to_parquet(parquet_file, index=False)

                # Save to CSV
                county_data.to_csv(csv_file, index=False)

                logger.info(f"Saved to: {parquet_file} and {csv_file}")

                return parquet_file, csv_file
            else:
                logger.warning(
                    f"No data found for {self.fips_dict['state']}, {self.fips_dict['county']} "
                    f"(FIPS: {state_fips}_{county_fips})"
                )
                return None, None

        except Exception as e:
            logger.error(f"Error processing NREL file: {e}")
            return None, None

    def process(self, boundary_gdf=None):
        """
        Process the NREL data for the region.

        Args:
            boundary_gdf (GeoDataFrame, optional): Not used by this method

        Returns:
            dict: Dictionary containing:
                - parquet_path: Path to the Parquet file
                - csv_path: Path to the CSV file
                - data: DataFrame with the NREL data if successfully loaded
        """
        logger.info(
            f"Processing NREL data for {self.fips_dict['state']} - {self.fips_dict['county']}")

        # Extract data
        parquet_path, csv_path = self.download()

        # Return the results
        result = {
            'parquet_path': parquet_path,
            'csv_path': csv_path,
            'data': None
        }

        # Load the data into a DataFrame if available
        if parquet_path and parquet_path.exists():
            try:
                data = pd.read_parquet(parquet_path)
                result['data'] = data
                logger.info(f"Loaded {len(data)} rows of NREL data")
            except Exception as e:
                logger.error(f"Error loading NREL data: {e}")

        return result

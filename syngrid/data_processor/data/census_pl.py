import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
import requests
import us

from syngrid.data_processor.data.base import DataHandler

logger = logging.getLogger(__name__)


class CensusPLDataHandler(DataHandler):
    """Handles downloading and processing Census PL 94-171 Redistricting Data."""

    BASE_URL = "https://www2.census.gov/programs-surveys/decennial/2020/data/" + \
               "01-Redistricting_File--PL_94-171/"

    # Census PL 94-171 file schema definitions
    GEO_SCHEMA = {
        'FILEID': (0, 6),
        'STUSAB': (6, 8),
        'SUMLEV': (8, 11),
        'GEOCOMP': (11, 13),
        'CHARITER': (13, 16),
        'CIFSN': (16, 18),
        'LOGRECNO': (18, 25),
        'REGION': (25, 26),
        'DIVISION': (26, 27),
        'STATE': (27, 29),
        'COUNTY': (29, 32),
        'COUNTYCC': (32, 34),
        'COUNTYSC': (34, 36),
        'COUSUB': (36, 41),
        'COUSUBCC': (41, 43),
        'COUSUBSC': (43, 45),
        'PLACE': (45, 50),
        'PLACECC': (50, 52),
        'PLACESC': (52, 54),
        'TRACT': (54, 60),
        'BLKGRP': (60, 61),
        'BLOCK': (61, 65),
        'IUC': (65, 67),
        'CONCIT': (67, 72),
        'CONCITCC': (72, 74),
        'CONCITSC': (74, 76),
        'AIANHH': (76, 80),
        'AIANHHFP': (80, 85),
        'AIANHHCC': (85, 87),
        'AIHHTLI': (87, 88),
        'AITSCE': (88, 91),
        'AITS': (91, 96),
        'AITSCC': (96, 98),
        'TTRACT': (98, 104),
        'TBLKGRP': (104, 105),
        'ANRC': (105, 110),
        'ANRCCC': (110, 112),
        'CBSA': (112, 117),
        'CBSASC': (117, 119),
        'METDIV': (119, 124),
        'CSA': (124, 127),
        'NECTA': (127, 132),
        'NECTASC': (132, 134),
        'NECTADIV': (134, 139),
        'CNECTA': (139, 142),
        'CBSAPCI': (142, 143),
        'NECTAPCI': (143, 144),
        'UA': (144, 149),
        'UASC': (149, 151),
        'UATYPE': (151, 152),
        'UR': (152, 153),
        'CD116': (153, 155),
        'CD118': (155, 157),
        'CD119': (157, 159),
        'CD120': (159, 161),
        'CD121': (161, 163),
        'SLDU18': (163, 166),
        'SLDU22': (166, 169),
        'SLDU24': (169, 172),
        'SLDU26': (172, 175),
        'SLDU28': (175, 178),
        'SLDL18': (178, 181),
        'SLDL22': (181, 184),
        'SLDL24': (184, 187),
        'SLDL26': (187, 190),
        'SLDL28': (190, 193),
        'VTD': (193, 199),
        'VTDI': (199, 200),
        'ZCTA': (200, 205),
        'SDELM': (205, 210),
        'SDSEC': (210, 215),
        'SDUNI': (215, 220),
        'PUMA': (220, 225),
        'AREALAND': (225, 239),
        'AREAWATR': (239, 253),
        'BASENAME': (253, 353),
        'NAME': (353, 453),
        'FUNCSTAT': (453, 455),
        'GCUNI': (455, 456),
        'POP100': (456, 466),
        'HU100': (466, 476),
        'INTPTLAT': (476, 487),
        'INTPTLON': (487, 500),
        'LSADC': (500, 502),
        'PARTFLAG': (502, 503),
        'UGA': (503, 508)
    }

    def __init__(
        self,
        fips_dict: Dict[str, str],
        output_dir: Optional[str | Path] = None,
    ):
        """Initialize handler with region information and output directory."""
        super().__init__(fips_dict, output_dir)
        self.state_abbr = self.fips_dict['state'].upper()
        self.state_name = us.states.lookup(self.state_abbr).name
        self.download_url = f"{self.BASE_URL}{self.state_name}/"

        # Set up raw data directory within the dataset output directory
        self.raw_data_dir = self.dataset_output_dir / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_data_dir / self.state_abbr).mkdir(parents=True, exist_ok=True)

    def _get_dataset_name(self) -> str:
        """Return the unique name for this dataset."""
        return "PL94171"

    def download(self) -> Dict[str, Any]:
        """
        Download and extract PL 94-171 data files for the region.

        Returns:
            Dict[str, Any]: Dictionary with status, file paths, and other information
        """
        # Check if files already exist
        if self._check_raw_files_exist():
            logger.info(f"Raw data already exists for {self.state_abbr}")
            files_dict = {
                "geo_file": str(self._get_raw_file_path("geo")),
                "data_file01": str(self._get_raw_file_path("file01")),
                "data_file02": str(self._get_raw_file_path("file02")),
            }
            return {
                "status": "success",
                "message": f"Raw data available at {self.raw_data_dir / self.state_abbr}",
                "download_path": str(self.raw_data_dir / self.state_abbr),
                "files": files_dict,
            }

        # Files don't exist, need to download
        logger.info(f"Downloading raw data for {self.state_abbr}...")
        zip_file_name = f"{self.state_abbr.lower()}2020.pl.zip"
        file_url = f"{self.download_url}{zip_file_name}"

        local_zip_path = self.raw_data_dir / self.state_abbr / zip_file_name
        extract_path = self.raw_data_dir / self.state_abbr

        try:
            logger.info(f"Downloading: {file_url}")
            response = requests.get(
                file_url,
                stream=True,
                verify=False
            )
            response.raise_for_status()

            with open(local_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Download complete: {local_zip_path}")

            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                logger.info(f"Extracted files to {extract_path}")

            # Clean up zip file
            local_zip_path.unlink(missing_ok=True)

            # Return success with file information
            files_dict = {
                "geo_file": str(self._get_raw_file_path("geo")),
                "data_file01": str(self._get_raw_file_path("file01")),
                "data_file02": str(self._get_raw_file_path("file02")),
            }
            return {
                "status": "success",
                "message": f"Raw data downloaded to {self.raw_data_dir / self.state_abbr}",
                "download_path": str(self.raw_data_dir / self.state_abbr),
                "files": files_dict,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Download error: {e}")
        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file: {local_zip_path}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        return {
            "status": "failure",
            "message": f"Failed to get raw data for {self.state_abbr}",
            "download_path": str(self.raw_data_dir / self.state_abbr),
            "files": {},
        }

    def _get_raw_file_path(self, file_type: str) -> Path:
        """Get path to a specific raw data file."""
        base_path = self.raw_data_dir / self.state_abbr
        if file_type == "geo":
            return base_path / f"{self.state_abbr.lower()}geo2020.pl"
        if file_type == "file01":  # Contains P1, P2 tables
            return base_path / f"{self.state_abbr.lower()}000012020.pl"
        if file_type == "file02":  # Contains P3, P4, H1 tables
            return base_path / f"{self.state_abbr.lower()}000022020.pl"
        raise ValueError(f"Unknown file type: {file_type}")

    def _check_raw_files_exist(self) -> bool:
        """Check if all required raw data files exist."""
        try:
            return all(
                self._get_raw_file_path(ft).exists()
                for ft in ["geo", "file01", "file02"]
            )
        except ValueError:
            return False

    def _parse_geo_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse the geographic header file (geo2020.pl) using pipe delimiter.

        Returns:
            DataFrame with geographic identifiers for each logical record
        """
        # Define the correct column names for the geo file
        geo_columns = [
            "FILEID", "STUSAB", "SUMLEV", "GEOVAR", "GEOCOMP", "CHARITER", "CIFSN",
            "LOGRECNO", "GEOID", "GEOCODE", "REGION", "DIVISION", "STATE", "STATENS",
            "COUNTY", "COUNTYCC", "COUNTYNS", "COUSUB", "COUSUBCC", "COUSUBNS",
            "SUBMCD", "SUBMCDCC", "SUBMCDNS", "ESTATE", "ESTATECC", "ESTATENS",
            "CONCIT", "CONCITCC", "CONCITNS", "PLACE", "PLACECC", "PLACENS",
            "TRACT", "BLKGRP", "BLOCK", "AIANHH", "AIHHTLI", "AIANHHFP", "AIANHHCC",
            "AIANHHNS", "AITS", "AITSFP", "AITSCC", "AITSNS", "TTRACT", "TBLKGRP",
            "ANRC", "ANRCCC", "ANRCNS", "CBSA", "MEMI", "CSA", "METDIV", "NECTA",
            "NMEMI", "CNECTA", "NECTADIV", "CBSAPCI", "NECTAPCI", "UA", "UATYPE",
            "UR", "CD116", "CD118", "CD119", "CD120", "CD121", "SLDU18", "SLDU22",
            "SLDU24", "SLDU26", "SLDU28", "SLDL18", "SLDL22", "SLDL24", "SLDL26",
            "SLDL28", "VTD", "VTDI", "ZCTA", "SDELM", "SDSEC", "SDUNI", "PUMA",
            "AREALAND", "AREAWATR", "BASENAME", "NAME", "FUNCSTAT", "GCUNI",
            "POP100", "HU100", "INTPTLAT", "INTPTLON", "LSADC", "PARTFLAG", "UGA"
        ]

        try:
            # Read file as pipe-delimited with correct column names
            df_geo = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                names=geo_columns,
                dtype=str,
                na_values=[''],
                keep_default_na=False
            )
            # Filter to block level records (SUMLEV=750 for blocks)
            block_df = df_geo[df_geo['SUMLEV'] == '750']
            logger.info(f"Records with SUMLEV=750 (blocks): {len(block_df)}")

            # Extract FIPS codes from our input
            county_fips = self.fips_dict.get('county_fips')
            subdivision_fips = self.fips_dict.get('subdivision_fips')

            # Filter to our county of interest
            if county_fips:
                # Make sure FIPS codes are properly formatted (may need leading zeros)
                county_fips_padded = county_fips.zfill(3)

                county_df = block_df[block_df['COUNTY'] == county_fips_padded]
            else:
                raise ValueError("County FIPS code is required")

            # Check if we have subdivision info
            subdivision_present = subdivision_fips is not None
            final_df = county_df

            # Filter to our subdivision if specified
            if subdivision_present and not county_df.empty:

                # Pad the FIPS code to 5 digits
                padded_fips = subdivision_fips.zfill(5)

                # Filter using padded FIPS
                subdiv_df = county_df[county_df['COUSUB'] == padded_fips]

                # Apply the filter and update final_df if we found matches
                if len(subdiv_df) > 0:
                    final_df = subdiv_df
                    logger.info(
                        f"Found {len(subdiv_df)} blocks in subdivision "
                        f"{self.fips_dict.get('subdivision')}"
                    )
                else:
                    logger.warning(
                        f"No blocks found for subdivision. "
                        "Using county-level data."
                    )
            else:
                if subdivision_fips:
                    logger.warning("Subdivision filtering skipped")
                else:
                    logger.info("No subdivision filter applied")

            # Save final filtered data
            final_csv_path = self.dataset_output_dir / "final_filtered_geo_data.csv"
            final_df.to_csv(final_csv_path, index=False)
            logger.info(f"Saved final filtered geo data to {final_csv_path}")

            return final_df

        except Exception as e:
            logger.error(f"Error parsing geo file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_data_file(self, file_path: Path, segment_num: str) -> pd.DataFrame:
        """
        Parse data segment files (segment 01=Population, segment 02=Housing).

        Args:
            file_path: Path to the data file
            segment_num: "01" or "02" indicating which segment to parse

        Returns:
            DataFrame with data values keyed by LOGRECNO
        """
        try:
            # Define column headers based on segment type
            if segment_num == "01":
                # Headers for segment 01 (Population data)
                headers = [
                    "FILEID", "STUSAB", "CHARITER", "CIFSN", "LOGRECNO",
                    "P0010001", "P0010002", "P0010003", "P0010004", "P0010005",
                    "P0010006", "P0010007", "P0010008", "P0010009", "P0010010",
                    # Additional columns omitted for brevity
                ]

                # For segment 01, we only need P0010001 (Total Population)
                columns_to_keep = ["LOGRECNO", "P0010001"]

                # Read only the specific columns we need to save memory
                df = pd.read_csv(
                    file_path,
                    sep='|',
                    header=None,
                    names=headers,
                    usecols=range(len(headers)),
                    dtype=str
                )

                # Extract just the columns we need
                result_df = df[columns_to_keep].copy()

            elif segment_num == "02":
                # Headers for segment 02 (Housing data)
                # First fields are the same, but then H001xxxx fields for housing
                # Since we don't have the exact headers, we'll read the file dynamically
                df = pd.read_csv(
                    file_path,
                    sep='|',
                    header=None,
                    dtype=str
                )

                # In segment 02, the 5th column (index 4) is LOGRECNO, and then housing data
                # H0010001 (Total Housing Units) - index 5
                # H0010002 (Occupied Housing Units) - index 6
                # H0010003 (Vacant Housing Units) - index 7
                result_df = pd.DataFrame()
                result_df['LOGRECNO'] = df.iloc[:, 4]
                result_df['H0010001'] = df.iloc[:, 5]
                result_df['H0010002'] = df.iloc[:, 6]
                result_df['H0010003'] = df.iloc[:, 7]

            else:
                logger.error(f"Unknown segment number: {segment_num}")
                return pd.DataFrame()

            # Save extracted data
            extracted_csv_path = self.dataset_output_dir / f"extracted_data_file{segment_num}.csv"
            result_df.to_csv(extracted_csv_path, index=False)
            return result_df

        except Exception as e:
            logger.error(f"Error parsing data file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _generate_processed_dataframe(self, download_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Process raw data files into a structured DataFrame.

        Args:
            download_result: Dictionary with download status and file paths

        Returns:
            pd.DataFrame: Processed data frame with population and housing data
        """
        if download_result.get('status') != 'success':
            logger.error("Cannot process data without successful download")
            return pd.DataFrame()

        # Get file paths from download result
        files = download_result.get('files', {})
        geo_file_path = files.get('geo_file')
        file01_path = files.get('data_file01')
        file02_path = files.get('data_file02')

        if not all([geo_file_path, file01_path, file02_path]):
            logger.error("Missing required data files")
            return pd.DataFrame()

        # Parse the files
        geo_df = self._parse_geo_file(Path(geo_file_path))
        pop_df = self._parse_data_file(Path(file01_path), "01")
        housing_df = self._parse_data_file(Path(file02_path), "02")

        if geo_df.empty or pop_df.empty or housing_df.empty:
            logger.error("One or more required datasets is empty")
            return pd.DataFrame()

        # Check for LOGRECNO in all dataframes
        if 'LOGRECNO' not in geo_df.columns:
            logger.error("LOGRECNO column missing in geo data")
            return pd.DataFrame()
        if 'LOGRECNO' not in pop_df.columns:
            logger.error("LOGRECNO column missing in population data")
            return pd.DataFrame()
        if 'LOGRECNO' not in housing_df.columns:
            logger.error("LOGRECNO column missing in housing data")
            return pd.DataFrame()

        # Merge the data using LOGRECNO as the key
        logger.info("Merging geo data with population data")
        merged_data = pd.merge(geo_df, pop_df, on='LOGRECNO', how='inner')
        logger.info(f"Records after first merge: {len(merged_data)}")

        merged_data = pd.merge(merged_data, housing_df, on='LOGRECNO', how='inner')
        logger.info(f"Records after second merge: {len(merged_data)}")

        # Save merged data for inspection
        merged_csv_path = self.dataset_output_dir / "merged_data.csv"
        merged_data.to_csv(merged_csv_path, index=False)
        logger.info(f"Saved merged data to {merged_csv_path}")

        if merged_data.empty:
            logger.error("No data after merging - likely no matching LOGRECNO keys")
            return pd.DataFrame()

        # Check if required columns are present
        required_columns = ['GEOID', 'P0010001', 'H0010001', 'H0010002', 'H0010003']
        for col in required_columns:
            if col not in merged_data.columns:
                logger.error(f"Required column {col} missing in merged data")

        # Create the final output format
        try:
            processed_df = pd.DataFrame({
                'GEOID': merged_data['GEOID'],
                'STATE_ID': merged_data['STATE'],
                'COUNTY_ID': merged_data['COUNTY'],
                'COUNTY_SUBDIVISION_ID': merged_data['COUSUB'],
                'TRACT_ID': merged_data['TRACT'],
                'Block_ID': merged_data['BLOCK'],
                'Total_Population': pd.to_numeric(merged_data['P0010001'], errors='coerce'),
                'Housing_Units': pd.to_numeric(merged_data['H0010001'], errors='coerce'),
                'Occupied_Units': pd.to_numeric(merged_data['H0010002'], errors='coerce'),
                'Vacant_Units': pd.to_numeric(merged_data['H0010003'], errors='coerce')
            })

            return processed_df

        except Exception as e:
            logger.error(f"Error creating final DataFrame: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def process(self, boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
        """
        Download and process Census PL data for the specified region.

        This is the main governing function that manages the entire workflow:
        1. Download raw PL 94-171 data
        2. Process the raw data into a structured format
        3. Join with census blocks geometry (if available)
        4. Save the processed data to output files

        Args:
            boundary_gdf: Optional boundary for spatial filtering (not used directly)

        Returns:
            Dictionary with processing status, message, and output information
        """
        logger.info(f"Processing {self.dataset_name} for {self.fips_dict['state']}")

        # Step 1: Download raw data
        download_result = self.download()
        if download_result.get("status") != "success":
            return {
                "status": "failure",
                "message": f"Raw data download failed for {self.state_abbr}",
                "processed_dataframe": pd.DataFrame(),
                "output_filepath": None
            }

        # Step 2: Process the raw data
        df_processed = self._generate_processed_dataframe(download_result)

        if df_processed.empty:
            logger.warning(f"Empty DataFrame for {self.state_abbr}")
            return {
                "status": "partial_success",
                "message": f"No data extracted for {self.state_abbr}",
                "processed_dataframe": pd.DataFrame(),
                "output_filepath": None
            }

        # Step 3: Join with census block geometries if available

        subdivision_fips = self.fips_dict.get('subdivision_fips')

        # Construct possible paths to census block geometries
        census_output_dir = self.get_base_output_dir() / "Census"

        # Try subdivision-specific blocks first, then county-wide blocks
        if subdivision_fips:
            blocks_filepath = census_output_dir / \
                f"{self.fips_dict['state_fips']}_{self.fips_dict['county_fips']}_{self.fips_dict['subdivision_fips']}_blocks.geojson"
        else:
            blocks_filepath = census_output_dir / \
                f"{self.fips_dict['state_fips']}_{self.fips_dict['county_fips']}_blocks.geojson"

        # If we can find block geometries, join with them
        gdf = None
        if blocks_filepath.exists():
            try:
                logger.info(f"Joining with census block geometries from: {blocks_filepath}")
                blocks_gdf = gpd.read_file(blocks_filepath)

                # Ensure GEOID is a string in both dataframes for proper joining
                blocks_gdf['GEOID'] = blocks_gdf['GEOID'].astype(str).str.removeprefix("7500000US")
                df_processed['GEOID'] = df_processed['GEOID'].astype(str)

                # Join the data with geometries
                gdf = blocks_gdf.merge(df_processed, on='GEOID', how='inner')
                logger.info(f"Successfully joined data with geometries: {len(gdf)} blocks")
            except Exception as e:
                logger.error(f"Failed to join with geometries: {e}")
                # Continue with just tabular data

        # Step 4: Save output files
        state_part = self.state_abbr.lower()
        county_part = self.fips_dict.get('county', 'all').replace(' ', '_').lower()
        output_base_name = f"{state_part}_{county_part}_pl_data"

        # Always save the CSV
        csv_filepath = self.dataset_output_dir / f"{output_base_name}.csv"
        df_processed.to_csv(csv_filepath, index=False)
        logger.info(f"Saved tabular data to: {csv_filepath}")

        return {
            "status": "success",
            "message": f"Processing successful for {self.state_abbr}",
            "processed_dataframe": df_processed,
            "geospatial_dataframe": gdf,
            "csv_filepath": str(csv_filepath),
        }


if __name__ == '__main__':
    # Simple test harness
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    test_fips = {
        "state": "MA",
        "state_fips": "25",
        "county": "Middlesex County",
        "county_fips": "017",
        "subdivision": "Cambridge city",
        "subdivision_fips": "11000"
    }

    # Use the default output directory from the base class
    handler = CensusPLDataHandler(fips_dict=test_fips)

    result = handler.process()
    print(f"Processing result: {result['status']}")
    if result['output_filepath']:
        print(f"Output saved to: {result['output_filepath']}")

import csv
import os
import tempfile
import urllib.request
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm

from gridtracer.config.config_loader import EPSG, OUTPUT_DIR
from gridtracer.data.imports.base import DataHandler


class CountySubdivisionHandler(DataHandler):
    """
    Handler for US Census County Subdivision Segmentation data.

    This class processes US Census county subdivision data to augment it with
    population, area, and geometry information. It can work within the
    orchestrator framework or as a standalone processor.
    """

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name.
        """
        return "CENSUS"

    def _download_file(self, url: str, output_dir: Path) -> Path:
        """
        Downloads a file from a URL to a specified directory.

        Args:
            url (str): The URL of the file to download.
            output_dir (Path): The directory to save the file in.

        Returns:
            Path: The local path to the downloaded file.
        """
        filename = os.path.basename(url)
        local_path = output_dir / filename
        if not local_path.exists():
            urllib.request.urlretrieve(url, local_path)
        else:
            self.logger.debug(f"File already exists: {local_path}")
        return local_path

    def _read_fips_lookup_file(self, file_path: Path) -> pd.DataFrame:
        """
        Reads the national FIPS lookup file and handles formatting issues.

        Args:
            file_path (Path): Path to the national_cousub.txt file.

        Returns:
            pd.DataFrame: A DataFrame containing the processed FIPS data.
        """
        with open(file_path, encoding="latin-1") as infile:
            reader = csv.reader(infile)
            processed_rows = []
            for i, row in enumerate(reader):
                if i == 0 or (row and row[0] == "STATE"):  # Skip header
                    continue
                if len(row) == 7:
                    processed_rows.append(row)
                elif len(row) == 8:  # Handle known inconsistency
                    merged_row = row[:5] + [row[5] + " " + row[6]] + [row[7]]
                    processed_rows.append(merged_row)

        column_names = [
            "state_abbr",
            "state_fips",
            "county_fips",
            "county_name",
            "subdivision_fips",
            "subdivision_name",
            "funcstat",
        ]
        return pd.DataFrame(processed_rows, columns=column_names)

    def _download_and_read_shapefile(self, url: str) -> Optional[gpd.GeoDataFrame]:
        """
        Downloads and reads a zipped shapefile from a URL.

        Args:
            url (str): The URL to the .zip shapefile.

        Returns:
            Optional[gpd.GeoDataFrame]: A GeoDataFrame if successful, else None.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                urllib.request.urlretrieve(url, tmp_file.name)

                # Read the downloaded file
                gdf = gpd.read_file(tmp_file.name)

                # Clean up temporary file
                Path(tmp_file.name).unlink()
            return gdf
        except Exception as e:
            self.logger.error(f"Failed to download or read {url}: {e}")
            return None

    def _process_state(self, state_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Processes all county subdivisions for a single state to add population,
        area, and geometry.

        Args:
            state_df (pd.DataFrame): DataFrame with subdivisions for a single
                                   state.

        Returns:
            Tuple[pd.DataFrame, List[Dict]]: A tuple containing:
                - The augmented DataFrame for the state.
                - A list of subdivisions that were not found.
        """
        state_fips = state_df["state_fips"].iloc[0]
        state_abbr = state_df["state_abbr"].iloc[0]
        self.logger.info(f"--- Processing State: {state_abbr} (FIPS: {state_fips}) ---")

        # 1. Download state-wide data ONCE
        base_url = "https://www2.census.gov/geo/tiger/TIGER2020"
        cousub_url = f"{base_url}/COUSUB/tl_2020_{state_fips}_cousub.zip"
        all_cousubs_gdf = self._download_and_read_shapefile(cousub_url)
        if all_cousubs_gdf is None:
            self.logger.warning(f"Could not load subdivisions for state {state_fips}, " "skipping.")
            return pd.DataFrame(), []

        blocks_url = f"{base_url}/TABBLOCK20/tl_2020_{state_fips}_tabblock20.zip"
        all_blocks_gdf = self._download_and_read_shapefile(blocks_url)
        if all_blocks_gdf is None:
            self.logger.warning(f"Could not load blocks for state {state_fips}, skipping.")
            return pd.DataFrame(), []

        results = []
        not_found_subdivisions = []
        # Group by county to process one county at a time
        for county_fips, group in state_df.groupby("county_fips"):
            # Filter state-wide data for the current county
            county_cousubs_gdf = all_cousubs_gdf[all_cousubs_gdf["COUNTYFP"] == county_fips]

            fips_col = "COUNTYFP20" if "COUNTYFP20" in all_blocks_gdf.columns else "COUNTYFP"
            county_blocks_gdf = all_blocks_gdf[all_blocks_gdf[fips_col] == county_fips]

            if county_blocks_gdf.empty:
                self.logger.warning(f"  No blocks found for county {county_fips}, skipping.")
                continue

            # Process each subdivision in the county
            for _, row in group.iterrows():
                subdiv_fips = row["subdivision_fips"]
                subdivision_geom_df = county_cousubs_gdf[county_cousubs_gdf["COUSUBFP"] == subdiv_fips]

                if subdivision_geom_df.empty:
                    self.logger.warning(
                        f"    Subdivision {subdiv_fips} " f"({row['subdivision_name']}) not found in shapefile."
                    )
                    not_found_subdivisions.append(
                        {
                            "state": state_abbr,
                            "subdivision_fips": subdiv_fips,
                            "subdivision_name": row["subdivision_name"],
                        }
                    )
                    continue

                # Clip blocks to the subdivision geometry.
                # Align CRS if necessary.
                if county_blocks_gdf.crs != subdivision_geom_df.crs:
                    county_blocks_gdf = county_blocks_gdf.to_crs(subdivision_geom_df.crs)

                clipped_blocks = gpd.clip(county_blocks_gdf, subdivision_geom_df)

                # --- Augment Data ---
                regional_identifier = f"{row['state_fips']}{row['county_fips']}" f"{row['subdivision_fips']}"
                population = clipped_blocks["POP20"].astype(int).sum()

                # Unify geometry and project to EPSG for area calculation
                # and output
                unified_geom = unary_union(subdivision_geom_df.geometry)
                projected_gds = gpd.GeoSeries([unified_geom], crs=subdivision_geom_df.crs).to_crs(f"EPSG:{EPSG}")
                projected_geom = projected_gds.iloc[0]

                # Calculate area and get WKB from the projected geometry
                area_sq_km = projected_geom.area / 1_000_000
                geom_wkb_hex = projected_geom.wkb_hex

                results.append(
                    {
                        **row.to_dict(),
                        "regional_identifier": regional_identifier,
                        "population": population,
                        "qkm": area_sq_km,
                        "geom": geom_wkb_hex,
                    }
                )

        return pd.DataFrame(results), not_found_subdivisions

    def process_state(self, args: Tuple[pd.DataFrame, Path, str]) -> List[Dict]:
        """
        A wrapper function for a single worker process.
        Processes a state and saves the resulting chunk file.

        Args:
            args (Tuple): A tuple containing state_df, chunk_path, and state_abbr.

        Returns:
            List[Dict]: A list of subdivisions that were not found in this state.
        """
        state_df, chunk_path, state_abbr = args
        try:
            augmented_df, not_found_list = self._process_state(state_df)
            if not augmented_df.empty:
                augmented_df.to_csv(chunk_path, index=False)
            return not_found_list
        except Exception as e:
            self.logger.error(f"Error processing state {state_abbr}: {e}", exc_info=True)
            return [
                {
                    "state": state_abbr,
                    "subdivision_fips": "ERROR",
                    "subdivision_name": str(e),
                }
            ]

    def _write_not_found_log(self, all_not_found: List[Dict]) -> None:
        """
        Write the log of not-found subdivisions to a file.

        Args:
            all_not_found (List[Dict]): List of subdivisions that were not found.
        """
        if all_not_found:
            total_not_found = len(all_not_found)
            self.logger.info(f"Found {total_not_found} subdivisions that were not " "in the shapefiles.")
            not_found_df = pd.DataFrame(all_not_found)
            # Reorder columns for clarity
            not_found_df = not_found_df[["state", "subdivision_fips", "subdivision_name"]]
            log_path = self.dataset_output_dir / "Subdivisions_not_found_log.txt"

            with open(log_path, "w") as f:
                f.write(f"Total Subdivisions Not Found: {total_not_found}\n\n")
                f.write(not_found_df.to_string(index=False))

    def download(self) -> Dict[str, Any]:
        """
        Download and process subcounty segmentation data.

        Returns:
            Dict[str, Any]: Dictionary containing processed data and file paths.
        """
        # URL for the national county subdivision lookup file
        lookup_url = "https://www2.census.gov/geo/docs/reference/codes/files/" "national_cousub.txt"

        # Download to the base output directory (output/) level
        base_output_dir = Path(OUTPUT_DIR)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        local_fips_file = self._download_file(lookup_url, base_output_dir)
        fips_df = self._read_fips_lookup_file(local_fips_file)

        return {"fips_data": fips_df, "fips_file": local_fips_file}

    def process(self, state_filter: Optional[str] = None, num_processes: int = 4) -> Dict[str, Any]:
        """
        Process subcounty segmentation data for all states or a specific state.

        Args:
            state_filter (Optional[str]): State abbreviation to process.
                                        If None, processes all states.
            num_processes (int): Number of parallel processes to use.

        Returns:
            Dict[str, Any]: Dictionary containing results and file paths.
        """
        # Check if final output files already exist
        if state_filter:
            state_abbr_upper = state_filter.upper()
            final_output_path = self.dataset_output_dir / f"{state_abbr_upper}_census_subdivisions.csv"
            if final_output_path.exists():
                self.logger.info(f"Segmentation file already exists for {state_abbr_upper}")
                return {
                    "output_file": final_output_path,
                    "not_found": [],
                }
        else:
            # Check for national file
            national_output_path = self.dataset_output_dir / "US_census_subdivisions.csv"
            if national_output_path.exists():
                self.logger.info(f"National segmentation file already exists: {national_output_path}")
                return {
                    "output_file": national_output_path,
                    "not_found": [],
                }

        # Download the lookup data
        download_results = self.download()
        fips_df = download_results["fips_data"]

        # Filter for a single state if provided
        if state_filter:
            state_abbr_upper = state_filter.upper()
            if state_abbr_upper not in fips_df["state_abbr"].unique():
                self.logger.error(
                    f"Invalid state abbreviation: '{state_filter}'. "
                    "Please use a valid 2-letter US state abbreviation."
                )
                return {"error": f"Invalid state: {state_filter}"}
            fips_df = fips_df[fips_df["state_abbr"] == state_abbr_upper].copy()
            self.logger.info(f"Processing state: {state_abbr_upper}")

        # Only create chunks directory for all-states processing
        if state_filter is None:
            chunks_directory = self.dataset_output_dir / "fips_code_chunks"
            chunks_directory.mkdir(exist_ok=True)
        else:
            # For single state, we'll write directly to the output file
            chunks_directory = self.dataset_output_dir

        all_state_fips = sorted(fips_df["state_fips"].unique())

        # Prepare arguments for parallel processing
        tasks = []
        for state_fips in all_state_fips:
            state_abbr = fips_df[fips_df["state_fips"] == state_fips]["state_abbr"].iloc[0]

            # For single state, write directly to final output
            if state_filter:
                chunk_path = self.dataset_output_dir / f"{state_abbr}_census_subdivisions.csv"
            else:
                # For all states, use chunk naming
                chunk_path = chunks_directory / f"{state_abbr}_census_chunk.csv"

            if chunk_path.exists():
                self.logger.info(f"Chunk for state {state_abbr} already exists. Skipping.")
                continue

            state_df = fips_df[fips_df["state_fips"] == state_fips].copy()
            tasks.append((state_df, chunk_path, state_abbr))

        if len(tasks) <= 1:
            results = [self.process_state(task) for task in tasks]
        else:
            # Parallel processing for multiple tasks
            with Pool(min(num_processes, len(tasks))) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(self.process_state, tasks),
                        total=len(tasks),
                        desc="Processing All States",
                    )
                )
        # Flatten the list of lists of not-found subdivisions
        all_not_found = [item for sublist in results if sublist for item in sublist]

        # Write the log of not-found subdivisions (do this before returns)
        self._write_not_found_log(all_not_found)

        # Handle output based on processing scope
        if state_filter:
            # For single state, the file is already at the final location
            state_abbr_upper = state_filter.upper()
            final_output_path = self.dataset_output_dir / f"{state_abbr_upper}_census_subdivisions.csv"
            if final_output_path.exists():
                self.logger.info(f"Processing complete. Final output saved to: " f"{final_output_path}")
                return {
                    "output_file": final_output_path,
                    "not_found": all_not_found,
                }
            else:
                self.logger.warning(f"No output file was generated for state {state_abbr_upper}.")
                return {"error": f"No output for state {state_abbr_upper}"}

        else:
            # Combine all chunks into one file
            self.logger.info("Combining all state chunks into a single file...")
            all_chunks = []
            for chunk_file in tqdm(
                sorted(chunks_directory.glob("*_census_chunk.csv")),
                desc="Combining Chunks",
            ):
                all_chunks.append(pd.read_csv(chunk_file))

            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                output_csv_path = self.dataset_output_dir / "US_census_subdivisions.csv"
                final_df.to_csv(output_csv_path, index=False)
                self.logger.info(f"Processing complete. Final output saved to: " f"{output_csv_path}")
                return {
                    "output_file": output_csv_path,
                    "not_found": all_not_found,
                    "data": final_df,
                }
            else:
                self.logger.warning("No chunks were generated. Final file not created.")
                return {"error": "No chunks generated"}

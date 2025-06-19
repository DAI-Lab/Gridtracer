import csv
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm
from multiprocessing import Pool

# --- Configuration ---
NUM_PROCESSES = 5  # Number of states to process in parallel

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_file(url: str, output_dir: Path) -> Path:
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
        logging.info(f"Downloading {url} to {local_path}")
        urllib.request.urlretrieve(url, local_path)
    else:
        logging.info(f"File already exists: {local_path}")
    return local_path


def read_fips_lookup_file(file_path: Path) -> pd.DataFrame:
    """
    Reads the national FIPS lookup file and handles formatting issues.

    Args:
        file_path (Path): Path to the national_cousub.txt file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed FIPS data.
    """
    with open(file_path, "r", encoding="latin-1") as infile:
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


def download_and_read_shapefile(
    url: str,
) -> Optional[gpd.GeoDataFrame]:
    """
    Downloads and reads a zipped shapefile from a URL.

    Args:
        url (str): The URL to the .zip shapefile.

    Returns:
        Optional[gpd.GeoDataFrame]: A GeoDataFrame if successful, else None.
    """
    try:
        gdf = gpd.read_file(url)
        return gdf
    except Exception as e:
        logging.error(f"Failed to download or read {url}: {e}")
        return None


def process_state(
    state_df: pd.DataFrame, data_cache_dir: Path
) -> Tuple[pd.DataFrame, list]:
    """
    Processes all county subdivisions for a single state to add population, area, and geometry.
    Args:
        state_df (pd.DataFrame): DataFrame with subdivisions for a single state.
        data_cache_dir (Path): Directory to cache downloaded census files.
    Returns:
        Tuple[pd.DataFrame, list]: A tuple containing:
            - The augmented DataFrame for the state.
            - A list of subdivisions that were not found.
    """
    state_fips = state_df["state_fips"].iloc[0]
    state_abbr = state_df["state_abbr"].iloc[0]
    logging.info(f"--- Processing State: {state_abbr} (FIPS: {state_fips}) ---")

    # 1. Download state-wide data ONCE
    cousub_url = f"https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/tl_2020_{state_fips}_cousub.zip"
    all_cousubs_gdf = download_and_read_shapefile(cousub_url)
    if all_cousubs_gdf is None:
        logging.warning(f"Could not load subdivisions for state {state_fips}, skipping.")
        return pd.DataFrame(), []

    blocks_url = f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{state_fips}_tabblock20.zip"
    all_blocks_gdf = download_and_read_shapefile(blocks_url)
    if all_blocks_gdf is None:
        logging.warning(f"Could not load blocks for state {state_fips}, skipping.")
        return pd.DataFrame(), []

    results = []
    not_found_subdivisions = []
    # Group by county to process one county at a time
    for county_fips, group in tqdm(
        state_df.groupby("county_fips"), desc=f"Counties in {state_abbr}"
    ):
        county_name = group['county_name'].iloc[0]
        # This logging can be noisy in parallel, but useful for debugging.
        # logging.info(f"  Processing County: {county_name} (FIPS: {county_fips})")

        # Filter state-wide data for the current county
        county_cousubs_gdf = all_cousubs_gdf[all_cousubs_gdf["COUNTYFP"] == county_fips]

        fips_col = "COUNTYFP20" if "COUNTYFP20" in all_blocks_gdf.columns else "COUNTYFP"
        county_blocks_gdf = all_blocks_gdf[all_blocks_gdf[fips_col] == county_fips]

        if county_blocks_gdf.empty:
            logging.warning(f"  No blocks found for county {county_fips}, skipping.")
            continue

        # Process each subdivision in the county
        for _, row in group.iterrows():
            subdiv_fips = row["subdivision_fips"]
            subdivision_geom_df = county_cousubs_gdf[
                county_cousubs_gdf["COUSUBFP"] == subdiv_fips
            ]

            if subdivision_geom_df.empty:
                logging.warning(
                    f"    Subdivision {subdiv_fips} ({row['subdivision_name']}) not found in shapefile."
                )
                not_found_subdivisions.append(
                    {
                        "state": state_abbr,
                        "subdivision_fips": subdiv_fips,
                        "subdivision_name": row["subdivision_name"],
                    }
                )
                continue

            # Clip blocks to the subdivision geometry. Align CRS if necessary.
            if county_blocks_gdf.crs != subdivision_geom_df.crs:
                county_blocks_gdf = county_blocks_gdf.to_crs(subdivision_geom_df.crs)

            clipped_blocks = gpd.clip(county_blocks_gdf, subdivision_geom_df)

            # --- Augment Data ---
            fipscode = f"{row['state_fips']}{row['county_fips']}{row['subdivision_fips']}"
            population = clipped_blocks["POP20"].astype(int).sum()

            # Unify geometry, project, and calculate area
            unified_geom = unary_union(subdivision_geom_df.geometry)
            geom_wkb_hex = unified_geom.wkb_hex

            # Project to an equal-area projection for area calculation (EPSG:5070)
            projected_geom = gpd.GeoSeries([unified_geom], crs=subdivision_geom_df.crs).to_crs("EPSG:5070")
            area_sq_meters = projected_geom.area.iloc[0]
            area_sq_km = area_sq_meters / 1_000_000

            results.append(
                {
                    **row.to_dict(),
                    "FIPSCODE": fipscode,
                    "POPULATION": population,
                    "QKM": area_sq_km,
                    "geom": geom_wkb_hex,
                }
            )

    logging.info(f"--- Finished processing State: {state_abbr} ---")
    return pd.DataFrame(results), not_found_subdivisions


def worker(args: Tuple[pd.DataFrame, Path, Path, str]):
    """
    A wrapper function for a single worker process.
    Processes a state and saves the resulting chunk file.

    Args:
        args (Tuple): A tuple containing state_df, data_cache_dir, chunk_path, and state_abbr.
    Returns:
        list: A list of subdivisions that were not found in this state.
    """
    state_df, data_cache_dir, chunk_path, state_abbr = args
    try:
        augmented_df, not_found_list = process_state(state_df, data_cache_dir)
        if not augmented_df.empty:
            augmented_df.to_csv(chunk_path, index=False)
            logging.info(f"Saved chunk for state {state_abbr} to {chunk_path}")
        return not_found_list
    except Exception as e:
        logging.error(f"Error processing state {state_abbr}: {e}", exc_info=True)
        return [{"state": state_abbr, "subdivision_fips": "ERROR", "subdivision_name": str(e)}]


def main():
    """
    Main function to run the data processing pipeline.
    """
    output_directory = Path("./gridtracer/data_processor/output/")
    output_directory.mkdir(exist_ok=True)

    chunks_directory = output_directory / "fips_code_chunks"
    chunks_directory.mkdir(exist_ok=True)

    data_cache_dir = output_directory / "census_cache"
    data_cache_dir.mkdir(exist_ok=True)

    # URL for the national county subdivision lookup file
    lookup_url = "https://www2.census.gov/geo/docs/reference/codes/files/national_cousub.txt"

    # Download and process the data
    local_fips_file = download_file(lookup_url, output_directory)
    fips_df = read_fips_lookup_file(local_fips_file)

    all_state_fips = sorted(fips_df["state_fips"].unique())

    # Prepare arguments for parallel processing
    tasks = []
    for state_fips in all_state_fips:
        state_abbr = fips_df[fips_df['state_fips'] == state_fips]['state_abbr'].iloc[0]
        chunk_path = chunks_directory / f"extended_{state_abbr}.csv"

        if chunk_path.exists():
            logging.info(f"Chunk for state {state_abbr} already exists. Skipping.")
            continue

        state_df = fips_df[fips_df["state_fips"] == state_fips].copy()
        tasks.append((state_df, data_cache_dir, chunk_path, state_abbr))

    # Run tasks in parallel
    logging.info(f"Processing {len(tasks)} states using {NUM_PROCESSES} parallel workers.")
    with Pool(NUM_PROCESSES) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="Processing All States")
        )
    logging.info("Parallel processing finished.")

    # Flatten the list of lists of not-found subdivisions
    all_not_found = [item for sublist in results if sublist for item in sublist]

    # Combine all chunks into one file
    logging.info("Combining all state chunks into a single file...")
    all_chunks = []
    for chunk_file in tqdm(
        sorted(chunks_directory.glob("extended_*.csv")), desc="Combining Chunks"
    ):
        all_chunks.append(pd.read_csv(chunk_file))

    if all_chunks:
        final_df = pd.concat(all_chunks, ignore_index=True)
        output_csv_path = output_directory / "national_cousub_extended.csv"
        final_df.to_csv(output_csv_path, index=False)
        logging.info(f"Processing complete. Final output saved to: {output_csv_path}")
    else:
        logging.warning("No chunks were generated. Final file not created.")

    # Write the log of not-found subdivisions
    if all_not_found:
        total_not_found = len(all_not_found)
        logging.info(f"Found {total_not_found} subdivisions that were not in the shapefiles.")
        not_found_df = pd.DataFrame(all_not_found)
        # Reorder columns for clarity
        not_found_df = not_found_df[['state', 'subdivision_fips', 'subdivision_name']]
        log_path = output_directory / "cousub_not_found_log.txt"

        with open(log_path, 'w') as f:
            f.write(f"Total Subdivisions Not Found: {total_not_found}\n\n")
            f.write(not_found_df.to_string(index=False))

        logging.info(f"Log of not-found subdivisions saved to: {log_path}")
        print(f"\nTotal number of subdivisions not found: {total_not_found}")
    else:
        logging.info("All subdivisions were found successfully.")


if __name__ == "__main__":
    main() 
import csv
import logging
import os
import urllib.request
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_region_path(fips_dict, dataset_name, base_output_dir=None, create_dir=True):
    """
    Create a standardized directory path structure for data output based on FIPS region
    information.

    The path follows the structure:
    base_output_dir/state/county/subdivision/dataset_name/

    Args:
        fips_dict (dict): Dictionary containing FIPS codes (result from lookup_fips_codes)
            Must include at least 'state' and 'county', optionally 'subdivision'
        dataset_name (str): Name of the dataset directory (e.g., 'NREL', 'Census', 'OSM')
        base_output_dir (str or Path, optional): Base output directory
            If None, defaults to 'syngrid/data_processor/output'
        create_dir (bool): Whether to create the directory if it doesn't exist

    Returns:
        Path: Path object to the output directory
    """
    if not isinstance(fips_dict, dict):
        logger.error("fips_dict must be a dictionary")
        return None

    # Extract necessary fields, with fallbacks
    state = fips_dict.get('state', 'unknown')
    county = fips_dict.get('county', 'unknown')
    subdivision = fips_dict.get('subdivision')

    # Sanitize names for directory paths
    state_dir = state.replace(' ', '_')
    county_dir = county.replace(' ', '_')

    # Set default base output directory if not provided
    if base_output_dir is None:
        base_output_dir = Path("syngrid/data_processor/output")
    elif isinstance(base_output_dir, str):
        base_output_dir = Path(base_output_dir)

    # Create hierarchical path
    output_path = base_output_dir / state_dir / county_dir

    # Add subdivision level if available
    if subdivision:
        subdivision_dir = subdivision.replace(' ', '_')
        output_path = output_path / subdivision_dir

    # Add dataset directory
    output_path = output_path / dataset_name

    # Create directory if requested
    if create_dir:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {output_path}")

    return output_path


def lookup_fips_codes(region):
    state = region.get('state')
    county = region.get('county')
    subdivision = region.get('county_subdivision')
    lookup_url = region.get('lookup_url')

    if not state or not county or not lookup_url:
        logger.error("Missing required parameters: state, county, and lookup_url are required")
        return None

    # Create output directory
    output_dir = Path("syngrid/data_processor/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Local file path
    filename = os.path.basename(lookup_url)
    local_file_path = output_dir / filename

    # Download file if it doesn't exist
    if not local_file_path.exists():
        logger.info(f"Downloading Census data file from {lookup_url}")
        urllib.request.urlretrieve(lookup_url, local_file_path)
        logger.debug(f"Census data file saved to {local_file_path}")
    else:
        logger.debug(f"Using existing Census data file: {local_file_path}")

    try:
        # Read and clean the data
        with open(local_file_path, 'r', encoding='latin-1') as infile:
            reader = csv.reader(infile)
            processed_rows = []
            merged_count = 0

            for i, row in enumerate(reader):
                # Skip header
                if i == 0 or (row and row[0] == 'STATE'):
                    continue
                # Process rows based on column count
                if len(row) == 7:
                    processed_rows.append(row)
                elif len(row) == 8:
                    # Merge third-last and second-last columns (5 and 6 in zero-based indexing)
                    merged_row = row[:5]  # First 5 columns
                    merged_row.append(row[5] + ' ' + row[6])  # Merge columns 5 and 6
                    merged_row.append(row[7])  # Last column
                    processed_rows.append(merged_row)
                    merged_count += 1

        logger.debug(f"Merged {merged_count} rows with 8 columns into 7 columns")

        # Create DataFrame from clean data
        column_names = [
            'state',
            'state_fips',
            'county_fips',
            'county',
            'subdivision_fips',
            'subdivision',
            'funcstat']
        df = pd.DataFrame(processed_rows, columns=column_names)

        # Now look up the specific data we need
        state_df = df[df['state'] == state]
        if state_df.empty:
            raise ValueError(f"State '{state}' not found in Census data")

        # Filter by exact county match
        county_matches = state_df[state_df['county'] == county]
        if county_matches.empty:
            raise ValueError(f"County '{county}' not found in state '{state}'")

        # Get the county data
        county_data = county_matches.iloc[0]
        state_fips = county_data['state_fips']
        county_fips = county_data['county_fips']

        # Initialize result with consistent structure
        result = {
            'state': state,
            'state_fips': state_fips,
            'county': county,
            'county_fips': county_fips,
            'subdivision': None,
            'subdivision_fips': None,
            'funcstat': None
        }

        # If subdivision provided, get exact match
        if subdivision:
            subdiv_match = county_matches[county_matches['subdivision'] == subdivision]

            if subdiv_match.empty:
                raise ValueError(
                    f"Subdivision '{subdivision}' not found in county '{county}', state '{state}'")
            subdiv_data = subdiv_match.iloc[0]
            result['subdivision'] = subdivision
            result['subdivision_fips'] = subdiv_data['subdivision_fips']
            result['funcstat'] = subdiv_data['funcstat']

        return result

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        logger.error(f"Error processing Census data: {str(e)}")
        raise ValueError(f"Failed to lookup FIPS codes: {str(e)}")


def get_region_data(fips_dict):
    """
    Download and process census data based on FIPS codes.

    Args:
        fips_dict (dict): Dictionary containing FIPS codes from lookup_fips_codes

    Returns:
        dict: Dictionary containing census data for the specified region, including:
            - blocks: GeoDataFrame of census blocks
            - block_filepath: Path to the blocks GeoJSON file
            - subdivision: GeoDataFrame of the subdivision (if applicable)
            - subdivision_filepath: Path to the subdivision GeoJSON file
            - boundary: GeoDataFrame of the region's outer boundary
            - boundary_filepath: Path to the boundary GeoJSON file
    """
    state_fips = fips_dict.get('state_fips')
    county_fips = fips_dict.get('county_fips')
    subdivision_fips = fips_dict.get('subdivision_fips')

    if not state_fips or not county_fips:
        logger.error("Missing required FIPS codes: state_fips and county_fips are required")
        return None

    # Create output directory using the new utility function
    output_dir = create_region_path(fips_dict, "Census")

    # Use shorter variables for readability
    state = fips_dict['state']
    county = fips_dict['county']

    logger.info(f"Downloading census data for {state} - {county}")

    # Results dictionary
    results = {
        'blocks': None,
        'block_filepath': None,
        'subdivision': None,
        'subdivision_filepath': None,
        'boundary': None,
        'boundary_filepath': None
    }

    try:
        # 1. Download Subdivision boundaries first (if we have a subdivision)
        subdivision_present = subdivision_fips is not None
        subdivision_filepath = output_dir / f"{state_fips}_{county_fips}_subdivisions.geojson"
        subdivision_gdf = None
        target_subdivision = None

        if not subdivision_filepath.exists():
            # Construct URL for county subdivisions
            subdivision_url = (
                f"https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/"
                f"tl_2020_{state_fips}_cousub.zip"
            )
            logger.info(f"Downloading county subdivisions from: {subdivision_url}")

            # Download and read the subdivision data
            subdivisions = gpd.read_file(subdivision_url)
            # Show columns
            logger.debug(f"Subdivisions columns: {subdivisions.columns}")
            # Filter for specific county
            county_subdivisions = subdivisions[subdivisions['COUNTYFP'] == county_fips]
            logger.info(f"Subdivisions in county {county}: {len(county_subdivisions)}")

            # Save to file
            county_subdivisions.to_file(subdivision_filepath, driver='GeoJSON')
            logger.info(f"Saved subdivisions to: {subdivision_filepath}")
            subdivision_gdf = county_subdivisions
        else:
            logger.info(f"Loading subdivisions from existing file: {subdivision_filepath}")
            subdivision_gdf = gpd.read_file(subdivision_filepath)

        # If we have a specific subdivision, find it
        if subdivision_present and subdivision_fips:
            target_subdivision = subdivision_gdf[
                subdivision_gdf['COUSUBFP'] == subdivision_fips
            ]

            if len(target_subdivision) == 0:
                logger.warning(
                    f"Subdivision with FIPS {subdivision_fips} not found in county {county}"
                )
            else:
                logger.info(
                    f"Found subdivision: {target_subdivision.iloc[0]['NAME']} "
                    f"(FIPS: {subdivision_fips})"
                )

                # Save specific subdivision to a separate file
                target_subdiv_filepath = (
                    output_dir
                    / f"{state_fips}_{county_fips}_{subdivision_fips}_subdivision.geojson"
                )

                target_subdivision.to_file(target_subdiv_filepath, driver='GeoJSON')
                logger.info(f"Saved target subdivision to: {target_subdiv_filepath}")

                results['subdivision'] = target_subdivision
                results['subdivision_filepath'] = target_subdiv_filepath

        # 2. Download Census Blocks
        blocks_filepath = output_dir / f"{state_fips}_{county_fips}_blocks.geojson"

        if not blocks_filepath.exists():
            # Construct URL for census blocks
            blocks_url = (
                f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
                f"tl_2020_{state_fips}_tabblock20.zip"
            )
            logger.info(f"Downloading Census blocks from: {blocks_url}")

            # Download and read the blocks data
            blocks = gpd.read_file(blocks_url)
            logger.info(f"Total blocks downloaded: {len(blocks)}")

            # Filter for specific county
            blocks = blocks[blocks['COUNTYFP20'] == county_fips]
            logger.info(f"Blocks in county {county}: {len(blocks)}")

            # Show blocks columns
            logger.debug(f"Blocks columns: {blocks.columns}")

            # Save all county blocks first
            blocks.to_file(blocks_filepath, driver='GeoJSON')
            logger.info(f"Saved all county blocks to: {blocks_filepath}")

            # If we have a target subdivision, filter blocks by spatial intersection
            if target_subdivision is not None and not target_subdivision.empty:
                logger.info("Filtering blocks by intersection with subdivision")

                # Ensure both GeoDataFrames have the same CRS
                if target_subdivision.crs != blocks.crs:
                    target_subdivision = target_subdivision.to_crs(blocks.crs)

                # Perform spatial join to find blocks that intersect with the subdivision
                blocks_in_subdiv = gpd.sjoin(
                    blocks,
                    target_subdivision,
                    how='inner',
                    predicate='intersects'
                )

                # Format for log message to avoid long line
                subdiv_name = fips_dict.get('subdivision')
                block_count = len(blocks_in_subdiv)
                logger.info(f"Found {block_count} blocks in subdivision {subdiv_name}")

                # Save the filtered blocks
                subdiv_blocks_filepath = (
                    output_dir
                    / f"{state_fips}_{county_fips}_{subdivision_fips}_blocks.geojson"
                )
                blocks_in_subdiv.to_file(subdiv_blocks_filepath, driver='GeoJSON')
                logger.info(f"Saved subdivision blocks to: {subdiv_blocks_filepath}")

                # Update results
                results['blocks'] = blocks_in_subdiv
                results['block_filepath'] = subdiv_blocks_filepath
            else:
                # Use all county blocks if no subdivision specified
                results['blocks'] = blocks
                results['block_filepath'] = blocks_filepath
        else:
            logger.info(f"Loading blocks from existing file: {blocks_filepath}")

            # If we have a target subdivision and corresponding blocks file
            if target_subdivision is not None and not target_subdivision.empty:
                subdiv_blocks_filepath = (
                    output_dir
                    / f"{state_fips}_{county_fips}_{subdivision_fips}_blocks.geojson"
                )

                if subdiv_blocks_filepath.exists():
                    logger.info(f"Loading subdivision blocks from: {subdiv_blocks_filepath}")
                    blocks = gpd.read_file(subdiv_blocks_filepath)
                    results['blocks'] = blocks
                    results['block_filepath'] = subdiv_blocks_filepath
                else:
                    # We need to load county blocks and filter them
                    blocks = gpd.read_file(blocks_filepath)

                    # Ensure both GeoDataFrames have the same CRS
                    if target_subdivision.crs != blocks.crs:
                        target_subdivision = target_subdivision.to_crs(blocks.crs)

                    # Perform spatial join
                    blocks_in_subdiv = gpd.sjoin(
                        blocks,
                        target_subdivision,
                        how='inner',
                        predicate='intersects'
                    )

                    # Format for log message to avoid long line
                    subdiv_name = fips_dict.get('subdivision')
                    block_count = len(blocks_in_subdiv)
                    logger.info(f"Found {block_count} blocks in subdivision {subdiv_name}")

                    # Save the filtered blocks
                    blocks_in_subdiv.to_file(subdiv_blocks_filepath, driver='GeoJSON')
                    logger.info(f"Saved subdivision blocks to: {subdiv_blocks_filepath}")

                    # Update results
                    results['blocks'] = blocks_in_subdiv
                    results['block_filepath'] = subdiv_blocks_filepath
            else:
                # Just load county blocks if no subdivision
                blocks = gpd.read_file(blocks_filepath)
                results['blocks'] = blocks
                results['block_filepath'] = blocks_filepath

        # After processing blocks and updating 'blocks' and 'block_filepath' in results
        # Add boundary creation
        if results['blocks'] is not None and not results['blocks'].empty:
            try:
                # Generate the boundary filepath
                if results['block_filepath']:
                    blocks_path = Path(results['block_filepath'])
                    boundary_path = blocks_path.parent / f"{blocks_path.stem}_boundary.geojson"

                    # Create a unified outer boundary by unary_union of all block geometries
                    unified_geometry = results['blocks'].geometry.unary_union

                    # Create a new GeoDataFrame with the boundary
                    blocks_crs = results['blocks'].crs
                    boundary_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=blocks_crs)

                    # Save the boundary to file
                    boundary_gdf.to_file(boundary_path, driver='GeoJSON')
                    logger.info(f"Saved region boundary to: {boundary_path}")

                    # Update results
                    results['boundary'] = boundary_gdf
                    results['boundary_filepath'] = boundary_path
            except Exception as e:
                logger.error(f"Error creating region boundary: {str(e)}")
                # Continue even if boundary creation fails

        # We still return partial results if available
        if any(value is not None for value in results.values()):
            return results

        logger.error("Could not download or load Census data")
        raise ValueError("Failed to get census data")

    except Exception as e:
        logger.error(f"Error downloading census data: {str(e)}")
        # Try to load from local file if download fails
        if 'blocks_filepath' in locals() and blocks_filepath.exists():
            logger.info(f"Loading blocks from local file: {blocks_filepath}")
            blocks = gpd.read_file(blocks_filepath)
            results['blocks'] = blocks
            results['block_filepath'] = blocks_filepath

            # Try to create boundary even in exception case
            try:
                boundary_path = blocks_filepath.parent / f"{blocks_filepath.stem}_boundary.geojson"
                unified_geometry = blocks.geometry.unary_union
                boundary_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=blocks.crs)
                boundary_gdf.to_file(boundary_path, driver='GeoJSON')
                logger.info(f"Saved region boundary to: {boundary_path}")
                results['boundary'] = boundary_gdf
                results['boundary_filepath'] = boundary_path
            except Exception as boundary_err:
                logger.error(f"Error creating region boundary: {str(boundary_err)}")

        # We still return partial results if available
        if any(value is not None for value in results.values()):
            return results

        logger.error("Could not download or load Census data")
        raise ValueError("Failed to get census data")


def visualize_blocks(blocks_gdf, subdivision_gdf=None, output_dir=None, fips_dict=None,
                     title="Census Blocks", show_block_ids=True):
    """
    Visualize census blocks on a map with their individual boundaries.

    Args:
        blocks_gdf (GeoDataFrame): GeoDataFrame containing census blocks
        subdivision_gdf (GeoDataFrame, optional): GeoDataFrame containing subdivision boundary
        output_dir (Path, optional): Directory to save the output plot
        fips_dict (dict, optional): Dictionary containing FIPS codes for standardized path creation
        title (str, optional): Title for the plot
        show_block_ids (bool, optional): Whether to show block IDs on the map

    Returns:
        str: Path to the saved plot file
    """
    if blocks_gdf is None or blocks_gdf.empty:
        logger.error("No blocks to plot")
        return None

    logger.info(f"Plotting {len(blocks_gdf)} blocks")

    # Set up output directory
    if output_dir is None:
        if fips_dict is not None:
            # Use the standardized directory structure if FIPS dict is provided
            output_dir = create_region_path(fips_dict, "Plots")
        else:
            # Fall back to generic plots directory if no FIPS info
            output_dir = Path("syngrid/data_processor/output/plots")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    _, ax = plt.subplots(figsize=(15, 15))

    # Convert to Web Mercator for basemap compatibility
    blocks_mercator = blocks_gdf.to_crs(epsg=3857)

    # Define initial bounds based on blocks
    bounds = list(blocks_mercator.total_bounds)

    # Plot blocks with styling
    blocks_mercator.plot(
        ax=ax,
        alpha=0.2,  # Transparent fill
        edgecolor='red',  # Red boundaries
        facecolor='skyblue',  # Light fill color
        linewidth=1.0  # Line thickness
    )

    # If subdivision provided, plot it too
    if subdivision_gdf is not None and not subdivision_gdf.empty:
        # Convert to same CRS
        subdivision_mercator = subdivision_gdf.to_crs(epsg=3857)

        # Plot subdivision boundary
        subdivision_mercator.plot(
            ax=ax,
            facecolor='none',
            edgecolor='green',
            linewidth=2.0,
            linestyle='--'
        )

        # Update bounds to include subdivision
        subdivision_bounds = subdivision_mercator.total_bounds
        bounds[0] = min(bounds[0], subdivision_bounds[0])
        bounds[1] = min(bounds[1], subdivision_bounds[1])
        bounds[2] = max(bounds[2], subdivision_bounds[2])
        bounds[3] = max(bounds[3], subdivision_bounds[3])

    # Add basemap
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom='auto',  # Auto-determine zoom level
            crs=blocks_mercator.crs,
            attribution_size=8
        )
    except Exception as e:
        logger.warning(f"Could not add basemap: {e}")

    # Set the axis limits to match the bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # Set title and remove axes
    plt.title(title, pad=20, fontsize=16)
    ax.set_axis_off()

    # Generate filename based on title and timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.replace(" ", "_").lower()
    output_file = output_dir / f"{safe_title}_{timestamp}.png"
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {output_file}")

    # Close plot to free memory
    plt.close()

    return str(output_file)


def nrel_data_preprocessing(fips_dict, input_file_path=None, output_dir=None):
    """
    Extract data for a specific county from NREL Residential dataset and save as parquet and csv.

    Parameters:
    -----------
    fips_dict : dict
        Dictionary containing region information with the following keys:
        - state: State name (e.g., 'California')
        - state_fips: State FIPS code (2 digits)
        - county: County name (e.g., 'Los Angeles')
        - county_fips: County FIPS code (3 digits)
        - subdivision: County subdivision name (optional)
        - subdivision_fips: County subdivision FIPS code (optional)
        - funcstat: Functional status code (optional)
    input_file_path : str or Path, optional
        Path to the NREL Residential typology TSV file
    output_dir : str or Path, optional
        Directory where to save the output files
        If None, uses a default output directory

    Returns:
    --------
    tuple : (parquet_path, csv_path) if successful, (None, None) otherwise
    """
    # Extract FIPS codes from dictionary
    if not isinstance(fips_dict, dict):
        logger.error("fips_dict must be a dictionary")
        return None, None

    state = fips_dict.get('state')
    county = fips_dict.get('county')
    state_fips = fips_dict.get('state_fips')
    county_fips = fips_dict.get('county_fips')

    # Convert input file to Path
    if input_file_path is None:
        logger.error("No input file path provided for NREL data preprocessing")
        return None, None

    input_file = Path(input_file_path)

    # Use the new utility function to create directory path
    if output_dir is None:
        output_dir = create_region_path(fips_dict, "NREL")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Create output filenames
    filename_base = f"NREL_Residential_typology_{state_fips}_{county_fips}"
    parquet_file = output_dir / f"{filename_base}.parquet"
    csv_file = output_dir / f"{filename_base}.csv"

    # Check if files already exist
    if parquet_file.exists() and csv_file.exists():
        logger.info(
            f"Files already exist for {state}, {county} "
            f"(FIPS: {state_fips}_{county_fips})"
        )
        return parquet_file, csv_file

    logger.info(f"Extracting data for {state}, {county} (FIPS: {state_fips}_{county_fips})")

    # Track if we found any matching data
    county_data_frames = []

    try:
        # Process the file in chunks
        chunk_size = 100000

        # Count total number of chunks for tqdm
        total_chunks = sum(1 for _ in pd.read_csv(input_file, sep="\t", chunksize=chunk_size))

        # Process chunks with progress bar
        with tqdm(total=total_chunks, desc=f"Processing {state}, {county} data") as pbar:
            for i, chunk in enumerate(pd.read_csv(input_file, sep="\t", chunksize=chunk_size)):
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

            logger.info(f"Saving {rows_count} rows for {state}, {county}")

            # Save to parquet
            county_data.to_parquet(parquet_file, index=False)

            # Save to CSV
            county_data.to_csv(csv_file, index=False)

            logger.info(f"Saved to: {parquet_file} and {csv_file}")

            return parquet_file, csv_file
        else:
            logger.warning(
                f"No data found for {state}, {county} "
                f"(FIPS: {state_fips}_{county_fips})"
            )
            return None, None

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return None, None

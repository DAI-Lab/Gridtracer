import csv
import logging
import os
import urllib.request
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


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

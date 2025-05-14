import csv
import logging
import os
import urllib.request
from pathlib import Path

import contextily as ctx
import geopandas as gpd
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
                     title="Census Blocks"):
    """
    Visualize census blocks on a map with their individual boundaries.

    Args:
        blocks_gdf (GeoDataFrame): GeoDataFrame containing census blocks
        subdivision_gdf (GeoDataFrame, optional): GeoDataFrame containing subdivision boundary
        output_dir (Path, optional): Directory to save the output plot
        fips_dict (dict, optional): Dictionary containing FIPS codes for standardized path creation
        title (str, optional): Title for the plot

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
            # Extract fields for directory structure
            state = fips_dict.get('state', 'unknown')
            county = fips_dict.get('county', 'unknown')
            subdivision = fips_dict.get('subdivision')

            # Sanitize names for directory paths
            state_dir = state.replace(' ', '_')
            county_dir = county.replace(' ', '_')

            # Create base path
            base_output_dir = Path("syngrid/data_processor/output")
            output_dir = base_output_dir / state_dir / county_dir

            # Add subdivision level if available
            if subdivision:
                subdivision_dir = subdivision.replace(' ', '_')
                output_dir = output_dir / subdivision_dir

            # Add plots directory
            output_dir = output_dir / "Plots"
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
            zoom='auto',
            crs="EPSG:3857",
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


def visualize_osm_data(fips_dict, boundary_gdf=None, output_dir=None,
                       plot_buildings=True, plot_pois=True, plot_power=True, plot_landuse=False):
    """
    Visualize OpenStreetMap data (buildings, POIs, power infrastructure, and land use).

    This function finds the GeoJSON files for buildings, POIs, power
    infrastructure, and land use in the OSM output directory based on the provided FIPS codes,
    then plots them together on a map.

    Args:
        fips_dict (dict): Dictionary containing region information with keys:
                         'state', 'county', and optionally 'subdivision'
        boundary_gdf (GeoDataFrame, optional): Region boundary to overlay
        output_dir (Path, optional): Custom output directory for the plot
        plot_buildings (bool, optional): Whether to plot buildings. Default: True
        plot_pois (bool, optional): Whether to plot POIs. Default: True
        plot_power (bool, optional): Whether to plot power infrastructure. Default: True
        plot_landuse (bool, optional): Whether to plot land use data. Default: False

    Returns:
        str: Path to the saved plot file
    """
    if fips_dict is None:
        logger.error("FIPS dictionary required to locate OSM data files")
        return None

    # Determine OSM data directory based on FIPS info
    state = fips_dict.get('state', 'unknown')
    county = fips_dict.get('county', 'unknown')
    subdivision = fips_dict.get('subdivision')

    # Sanitize names for directory paths
    state_dir = state.replace(' ', '_')
    county_dir = county.replace(' ', '_')

    # Create base path
    base_output_dir = Path("syngrid/data_processor/output")
    osm_data_dir = base_output_dir / state_dir / county_dir

    # Add subdivision level if available
    if subdivision:
        subdivision_dir = subdivision.replace(' ', '_')
        osm_data_dir = osm_data_dir / subdivision_dir

    # Complete path to OSM folder
    osm_data_dir = osm_data_dir / "OSM"

    logger.info(f"Looking for OSM data in: {osm_data_dir}")

    # Check for existence of required files
    buildings_file = osm_data_dir / "buildings.geojson"
    pois_file = osm_data_dir / "pois.geojson"
    power_file = osm_data_dir / "power.geojson"

    # Load available data
    buildings_gdf = None
    pois_gdf = None
    power_gdf = None
    landuse_gdf = None

    if buildings_file.exists():
        logger.info(f"Loading buildings from {buildings_file}")
        buildings_gdf = gpd.read_file(buildings_file)
    else:
        logger.warning(f"Buildings file not found: {buildings_file}")

    if pois_file.exists():
        logger.info(f"Loading POIs from {pois_file}")
        pois_gdf = gpd.read_file(pois_file)
    else:
        logger.warning(f"POIs file not found: {pois_file}")

    if power_file.exists():
        logger.info(f"Loading power infrastructure from {power_file}")
        power_gdf = gpd.read_file(power_file)
    else:
        logger.warning(f"Power file not found: {power_file}")
        
    # Add land use data loading
    landuse_file = osm_data_dir / "landuse.geojson"
    if plot_landuse and landuse_file.exists():
        logger.info(f"Loading land use data from {landuse_file}")
        landuse_gdf = gpd.read_file(landuse_file)
    elif plot_landuse:
        logger.warning(f"Land use file not found: {landuse_file}")

    # Check if any data was loaded
    if buildings_gdf is None and pois_gdf is None and power_gdf is None and landuse_gdf is None:
        logger.error("No OSM data found to visualize")
        return None

    # Set up output directory for the plot
    if output_dir is None:
        # Use the standard directory structure based on FIPS info
        output_dir = osm_data_dir.parent / "Plots"
    else:
        # If output_dir is provided, use it as the base and add region-specific folders
        base_output_dir = Path(output_dir)

        # Extract fields for directory structure
        state = fips_dict.get('state', 'unknown')
        county = fips_dict.get('county', 'unknown')
        subdivision = fips_dict.get('subdivision')

        # Sanitize names for directory paths
        state_dir = state.replace(' ', '_')
        county_dir = county.replace(' ', '_')

        # Create path structure
        output_dir = base_output_dir / state_dir / county_dir

        # Add subdivision level if available
        if subdivision:
            subdivision_dir = subdivision.replace(' ', '_')
            output_dir = output_dir / subdivision_dir

        # Add Plots directory
        output_dir = output_dir / "Plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"OSM plot will be saved to: {output_dir}")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))

    # Initialize bounds
    bounds = None

    # Convert data to Web Mercator for basemap compatibility and plot
    
    # Plot land use data first (as a background layer)
    if plot_landuse and landuse_gdf is not None and not landuse_gdf.empty:
        try:
            landuse_mercator = landuse_gdf.to_crs(epsg=3857)
            
            # Define a color map for land use categories
            category_colors = {
                'Agriculture': 'yellowgreen',
                'Natural': 'darkgreen',
                'Developed': 'lightgray',
                'Leisure': 'lightgreen',
                'Amenity': 'khaki',
                'Transportation': 'silver',
                'Other': 'white'
            }
            
            # If the category field exists, use it for coloring
            if 'category' in landuse_mercator.columns:
                # Plot each category with its own color
                for category, color in category_colors.items():
                    category_data = landuse_mercator[landuse_mercator['category'] == category]
                    if not category_data.empty:
                        category_data.plot(
                            ax=ax,
                            color=color,
                            alpha=0.5,
                            label=f"Land Use: {category}"
                        )
            else:
                # Just plot all land use polygons with a single color
                landuse_mercator.plot(
                    ax=ax,
                    color='lightblue',
                    alpha=0.3,
                    label='Land Use'
                )
            
            if bounds is None:
                bounds = list(landuse_mercator.total_bounds)
            else:
                lu_bounds = landuse_mercator.total_bounds
                bounds[0] = min(bounds[0], lu_bounds[0])
                bounds[1] = min(bounds[1], lu_bounds[1])
                bounds[2] = max(bounds[2], lu_bounds[2])
                bounds[3] = max(bounds[3], lu_bounds[3])
                
            logger.info(f"Added {len(landuse_gdf)} land use polygons to plot")
        except Exception as e:
            logger.error(f"Error plotting land use data: {e}")
    
    # Plot buildings (now on top of land use)
    if plot_buildings and buildings_gdf is not None and not buildings_gdf.empty:
        try:
            buildings_mercator = buildings_gdf.to_crs(epsg=3857)
            buildings_mercator.plot(
                ax=ax,
                color='skyblue',
                alpha=0.6,
                edgecolor='navy',
                linewidth=0.2,
                label='Buildings'
            )

            if bounds is None:
                bounds = list(buildings_mercator.total_bounds)
            else:
                bldg_bounds = buildings_mercator.total_bounds
                bounds[0] = min(bounds[0], bldg_bounds[0])
                bounds[1] = min(bounds[1], bldg_bounds[1])
                bounds[2] = max(bounds[2], bldg_bounds[2])
                bounds[3] = max(bounds[3], bldg_bounds[3])

            logger.info(f"Added {len(buildings_gdf)} buildings to plot")
        except Exception as e:
            logger.error(f"Error plotting buildings: {e}")

    if plot_pois and pois_gdf is not None and not pois_gdf.empty:
        try:
            pois_mercator = pois_gdf.to_crs(epsg=3857)
            pois_mercator.plot(
                ax=ax,
                marker='o',
                color='red',
                markersize=5,
                alpha=0.7,
                label='POIs'
            )

            if bounds is None:
                bounds = list(pois_mercator.total_bounds)
            else:
                poi_bounds = pois_mercator.total_bounds
                bounds[0] = min(bounds[0], poi_bounds[0])
                bounds[1] = min(bounds[1], poi_bounds[1])
                bounds[2] = max(bounds[2], poi_bounds[2])
                bounds[3] = max(bounds[3], poi_bounds[3])

            logger.info(f"Added {len(pois_gdf)} POIs to plot")
        except Exception as e:
            logger.error(f"Error plotting POIs: {e}")

    if plot_power and power_gdf is not None and not power_gdf.empty:
        try:
            power_mercator = power_gdf.to_crs(epsg=3857)

            # Plot power features with different styles based on type
            # First, check if 'power' column exists
            if 'power' in power_gdf.columns:
                # Plot transformers
                transformers = power_mercator[power_gdf['power'] == 'transformer']
                if not transformers.empty:
                    transformers.plot(
                        ax=ax,
                        marker='s',  # Square
                        color='yellow',
                        edgecolor='black',
                        markersize=30,
                        alpha=0.8,
                        label='Transformers'
                    )

                # Plot substations
                substations = power_mercator[power_gdf['power'] == 'substation']
                if not substations.empty:
                    substations.plot(
                        ax=ax,
                        marker='*',  # Star
                        color='orange',
                        edgecolor='black',
                        markersize=100,
                        alpha=0.8,
                        label='Substations'
                    )

                # Plot poles
                poles = power_mercator[power_gdf['power'] == 'pole']
                if not poles.empty:
                    poles.plot(
                        ax=ax,
                        marker='x',  # X
                        color='green',
                        markersize=15,
                        alpha=0.6,
                        label='Poles'
                    )
            else:
                # If no type information, plot all power features the same way
                power_mercator.plot(
                    ax=ax,
                    marker='D',  # Diamond
                    color='purple',
                    markersize=25,
                    alpha=0.7,
                    label='Power Infrastructure'
                )

            if bounds is None:
                bounds = list(power_mercator.total_bounds)
            else:
                power_bounds = power_mercator.total_bounds
                bounds[0] = min(bounds[0], power_bounds[0])
                bounds[1] = min(bounds[1], power_bounds[1])
                bounds[2] = max(bounds[2], power_bounds[2])
                bounds[3] = max(bounds[3], power_bounds[3])

            logger.info(f"Added {len(power_gdf)} power infrastructure features to plot")
        except Exception as e:
            logger.error(f"Error plotting power infrastructure: {e}")

    # If region boundary provided, plot it too
    if boundary_gdf is not None and not boundary_gdf.empty:
        try:
            boundary_mercator = boundary_gdf.to_crs(epsg=3857)
            boundary_mercator.plot(
                ax=ax,
                facecolor='none',
                edgecolor='green',
                linewidth=2.0,
                linestyle='--',
                label='Region Boundary'
            )

            # Update bounds to include boundary
            boundary_bounds = boundary_mercator.total_bounds
            if bounds is not None:
                bounds[0] = min(bounds[0], boundary_bounds[0])
                bounds[1] = min(bounds[1], boundary_bounds[1])
                bounds[2] = max(bounds[2], boundary_bounds[2])
                bounds[3] = max(bounds[3], boundary_bounds[3])
            else:
                bounds = list(boundary_bounds)

            logger.info("Added region boundary to plot")
        except Exception as e:
            logger.error(f"Error plotting region boundary: {e}")

    # Add basemap
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom='auto',
            crs="EPSG:3857",
            attribution_size=8
        )
    except Exception as e:
        logger.warning(f"Could not add basemap: {e}")

    # If we have bounds, set the axis limits
    if bounds:
        # Add some padding around the bounds
        pad_x = (bounds[2] - bounds[0]) * 0.05
        pad_y = (bounds[3] - bounds[1]) * 0.05
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    # Add legend
    plt.legend(loc='upper right', fontsize=10)

    # Set title with region information
    title_parts = []
    if subdivision:
        title_parts.append(subdivision)
    title_parts.extend([county, state])

    # Create base title with region info
    region_title = ", ".join(title_parts)

    # Add data type indicators to title
    title_elements = []
    if plot_buildings:
        title_elements.append("Buildings")
    if plot_pois:
        title_elements.append("POIs")
    if plot_power:
        title_elements.append("Power Infrastructure")

    if len(title_elements) > 0:
        data_title = " & ".join(title_elements)
        title = f"{data_title} for {region_title}"
    else:
        title = f"OpenStreetMap Data for {region_title}"

    plt.title(title, pad=20, fontsize=16)

    # Remove axes
    ax.set_axis_off()

    # Generate filename based on region and timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    region_name = "_".join([x.replace(" ", "_").lower() for x in title_parts if x])

    # Add data type indicators to filename
    filename_parts = ["osm"]
    if plot_buildings:
        filename_parts.append("bldg")
    if plot_pois:
        filename_parts.append("poi")
    if plot_power:
        filename_parts.append("power")

    filename_prefix = "_".join(filename_parts)
    output_file = output_dir / f"{filename_prefix}_{region_name}_{timestamp}.png"

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved OSM data visualization to: {output_file}")

    # Close plot to free memory
    plt.close()

    return str(output_file)


def visualize_road_network(network_data, boundary_gdf=None, output_dir=None, title="Road Network"):
    """
    Simple visualization of a road network.

    Args:
        network_data: Either a GeoDataFrame of roads or a path to a GeoJSON file
        boundary_gdf: Optional boundary GeoDataFrame for overlay
        output_dir: Directory to save the output plot, defaults to current directory
        title: Title for the plot

    Returns:
        str: Path to the saved plot file
    """
    import matplotlib.pyplot as plt
    import contextily as ctx
    from pathlib import Path
    import pandas as pd
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path("syngrid/data_processor/output/plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load network data if it's a file path
    if isinstance(network_data, (str, Path)):
        logger.info(f"Loading network from: {network_data}")
        network_gdf = gpd.read_file(network_data)
    else:
        network_gdf = network_data
    
    if network_gdf is None or network_gdf.empty:
        logger.error("No network data to visualize")
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Convert to Web Mercator for basemap compatibility
    network_mercator = network_gdf.to_crs(epsg=3857)
    
    # Plot network
    network_mercator.plot(ax=ax, color='blue', linewidth=0.8)
    
    # Add boundary if provided
    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_mercator = boundary_gdf.to_crs(epsg=3857)
        boundary_mercator.plot(
            ax=ax,
            facecolor='none',
            edgecolor='green',
            linewidth=2.0,
            linestyle='--'
        )
    
    # Get bounds for the map
    bounds = list(network_mercator.total_bounds)
    
    # Add basemap
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom='auto',
            crs="EPSG:3857"
        )
    except Exception as e:
        logger.warning(f"Could not add basemap: {e}")
    
    # Set title and remove axes
    plt.title(title, fontsize=16)
    ax.set_axis_off()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"road_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Road network visualization saved to: {output_file}")
    return str(output_file)

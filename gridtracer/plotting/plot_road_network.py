from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt

from gridtracer.data_processor.utils.log_config import logger


def visualize_road_network(network_data, boundary_gdf=None,
                           output_dir=None, title="Road Network"):
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

    # Set up output directory
    if output_dir is None:
        output_dir = Path("gridtracer/output/plots")
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
    _, ax = plt.subplots(figsize=(12, 12))

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
    list(network_mercator.total_bounds)

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
    output_file = output_dir / "road_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Road network visualization saved to: {output_file}")
    return str(output_file)

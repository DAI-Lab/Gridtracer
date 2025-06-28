import argparse
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkb

from gridtracer.config import config
from gridtracer.utils import create_logger

logger = create_logger(
    name="PlotStateSubdivisions",
    log_level=config.log_level,
    log_file=config.log_file,
)


def plot_subdivisions(csv_path: Path, show_labels: bool = True):
    """
    Reads a state's processed CSV file, converts hex geometries, and plots
    the subdivisions on a map with annotations.

    Args:
        csv_path (Path): The path to the input CSV file (e.g., .../extended_AK.csv).
        show_labels (bool): If True, adds name and population labels to subdivisions.
    """
    if not csv_path.exists():
        logger.error(f"Input file not found at: {csv_path}")
        return

    logger.info(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)

    if 'geom' not in df.columns:
        logger.error("CSV file must contain a 'geom' column with WKB hex strings.")
        return

    # 1. Convert WKB hex strings to Shapely geometry objects
    try:
        df["geometry"] = df["geom"].apply(lambda x: wkb.loads(x, hex=True))
    except Exception as e:
        logger.error(f"Failed to parse geometries: {e}")
        return

    # 2. Create a GeoDataFrame
    # The original TIGER data is typically NAD83 (EPSG:4269)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4269")
    logger.info(f"Successfully created GeoDataFrame with {len(gdf)} subdivisions.")

    # Calculate total population for the title
    total_population = int(df["POPULATION"].sum())

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))

    # Project to Web Mercator (EPSG:3857) for plotting with contextily
    gdf_web_mercator = gdf.to_crs(epsg=3857)

    # Plot the subdivision boundaries
    gdf_web_mercator.plot(
        ax=ax, edgecolor="blue", facecolor="yellow", alpha=0.5, linewidth=1.5
    )

    # 4. Add annotations for each subdivision if enabled
    if show_labels:
        logger.info("Adding annotations to subdivisions...")
        for idx, row in gdf_web_mercator.iterrows():
            # Use representative_point to ensure the label is within the polygon
            centroid = row.geometry.representative_point()
            subdivision_name = row["subdivision_name"]
            population = int(row["POPULATION"])
            annotation_text = f"{subdivision_name}\nPop: {population:,}"

            ax.text(
                centroid.x,
                centroid.y,
                annotation_text,
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
            )

    # 5. Add a basemap
    logger.info("Adding basemap...")
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom="auto")
    except Exception as e:
        logger.warning(f"Could not add basemap: {e}")

    # Final plot styling
    state_name = gdf.iloc[0]['state_abbr']
    ax.set_title(
        f"Subdivisions of {state_name}\n(Total Population: {total_population:,})",
        fontsize=20
    )
    ax.set_axis_off()
    plt.tight_layout()

    # 6. Save the plot
    output_filename = csv_path.parent.parent / f"{state_name}_subdivisions_map.png"
    plt.savefig(output_filename, dpi=300)
    logger.info(f"Map saved to: {output_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot state subdivisions from a processed CSV file."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV file for a single state (e.g., 'output/fips_code_chunks/extended_AK.csv')",
    )
    parser.add_argument(
        "--no-labels",
        action="store_false",
        dest="show_labels",
        help="Run the script without showing labels on the plot.",
    )
    args = parser.parse_args()

    plot_subdivisions(Path(args.csv_path), show_labels=args.show_labels)

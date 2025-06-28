"""
Standalone script to extract and build a routable road network.

This script initializes the WorkflowOrchestrator to access project configurations
and then uses the RoadNetworkBuilder to generate the road network for the
configured region.

Usage:
    python -m gridtracer.scripts.extract_road_network --plot
"""
import argparse
import time

import geopandas as gpd

from gridtracer.config import config
from gridtracer.data_processor.data_imports.osm.road_network_builder import RoadNetworkBuilder
from gridtracer.data_processor.workflow import WorkflowOrchestrator
from gridtracer.plotting.plot_road_network import visualize_road_network
from gridtracer.utils import create_logger

logger = create_logger(
    name="ExtractRoadNetwork",
    log_level=config.log_level,
    log_file=config.log_file,
)


def set_boundary_gdf(file_path: str):
    """
    Set the boundary GeoDataFrame for the region.
    """
    boundary_gdf = gpd.read_file(file_path)
    return boundary_gdf


def extract_road_network(boundary_file_path: str = None, plot: bool = False):
    """
    Initializes and runs the road network extraction process.

    Args:
        plot (bool): Whether to generate and save plots of the road network.
    """
    start_time = time.time()
    logger.info("Starting Road Network Extraction")

    try:
        # Initialize the orchestrator to load configuration and set up paths
        orchestrator = WorkflowOrchestrator()
        if boundary_file_path:
            boundary_gdf = set_boundary_gdf(boundary_file_path)
            orchestrator.set_region_boundary(boundary_gdf)

        # --- Routable Road Network Generation ---
        road_network_builder = RoadNetworkBuilder(orchestrator=orchestrator)
        road_network_results = road_network_builder.process()
        if plot:
            visualize_road_network(road_network_results['geojson_file'], boundary_gdf)

        logger.info("Road Network Extraction completed successfully.")

    except ValueError as ve:
        logger.error("Configuration or validation error: %s", ve, exc_info=True)
    except RuntimeError as re:
        logger.error("Runtime error during execution: %s", re, exc_info=True)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
    finally:
        # Calculate and log total execution time
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(
            "Road Network Extraction finished in %.2f seconds", total_time
        )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract and build a routable road network for a "
        "configured region."
    )
    parser.add_argument(
        "--boundary_file_path",
        type=str,
        default=None,
        help="Path to the boundary GeoJSON file. Optional.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generate and save plots of the road network.",
    )
    args = parser.parse_args()

    extract_road_network(boundary_file_path=args.boundary_file_path, plot=args.plot)


if __name__ == "__main__":
    main()

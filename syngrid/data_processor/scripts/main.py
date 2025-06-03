"""
Entrypoint for the SynGrid data processing pipeline (Version 2.0).

This script orchestrates the entire data processing workflow using the
WorkflowOrchestrator.
"""
import time

import geopandas as gpd

from syngrid.data_processor.data.census import CensusDataHandler
from syngrid.data_processor.data.microsoft_buildings import MicrosoftBuildingsDataHandler
from syngrid.data_processor.data.nrel import NRELDataHandler
from syngrid.data_processor.data.osm.osm_data_handler import OSMDataHandler
from syngrid.data_processor.data.osm.road_network_builder import RoadNetworkBuilder
from syngrid.data_processor.processing.building_processor import BuildingHeuristicsProcessor
from syngrid.data_processor.utils import logger
from syngrid.data_processor.workflow import WorkflowOrchestrator


def dev_fill_census_results():
    """
    Just for testing purposes os i dotn have tpo always rerun this  fills this from precreated links :
          - 'target_region_blocks': GeoDataFrame of census blocks for the target region.
                - 'target_region_blocks_filepath': Path to the saved blocks GeoJSON.
                - 'target_region_boundary': GeoDataFrame representing the final authoritative
                                            boundary for the processing run.
                - 'target_region_boundary_filepath': Path to the final region boundary GeoJSON.
    """
    region_blocks_filepath = "/Users/magic-rabbit/Documents/00_Tech-Repositories/05_MASTER_THESIS/SynGrid/syngrid/data_processor/output/MA/Middlesex_County/Cambridge_city_old/Census/25_017_11000_blocks.geojson"
    region_boundary_filepath = "/Users/magic-rabbit/Documents/00_Tech-Repositories/05_MASTER_THESIS/SynGrid/syngrid/data_processor/output/MA/Middlesex_County/Cambridge_city_old/Census/25_017_11000_blocks_boundary.geojson"
    census_data = {}
    census_data['target_region_blocks'] = gpd.read_file(region_blocks_filepath)
    census_data['target_region_blocks_filepath'] = region_blocks_filepath
    census_data['target_region_boundary'] = gpd.read_file(region_boundary_filepath)
    census_data['target_region_boundary_filepath'] = region_boundary_filepath
    return census_data


def run_pipeline_v2():
    """
    Runs the main SynGrid data processing pipeline using the WorkflowOrchestrator.
    """
    start_time = time.time()
    logger.info("Starting SynGrid Data Processing Pipeline v2.0")

    try:
        # # Initialize the orchestrator, loading config, setting up FIPS, and creating all output directories
        orchestrator = WorkflowOrchestrator()

        # TODO: Uncomment this when you want to use the precreated census data
        # census_data = dev_fill_census_results()
        # orchestrator.set_region_boundary(census_data['target_region_boundary'])

        # # --- STEP 1: REGIONAL DATA EXTRACTION & PREPARATION ---
        logger.info("STEP 1: Regional Data Extraction & Preparation")

        census_handler = CensusDataHandler(orchestrator)
        census_data = census_handler.process(plot=False)
        if not census_data or 'target_region_boundary' not in census_data:  # Check for primary output
            logger.error(
                "Census data processing failed or did not yield a target_region_boundary. Halting.")
            return

        # --- STEP 2: Process NREL Data ---

        logger.info("STEP 2: Processing NREL data")
        nrel_handler = NRELDataHandler(orchestrator)
        nrel_data = nrel_handler.process()
        if nrel_data.get('parquet_path'):
            logger.info(
                f"NREL data processing complete. Parquet at: {nrel_data['parquet_path']}")
        else:
            logger.warning("NREL data processing did not yield a parquet path.")

        # --- STEP 3: Extract OSM Data ---
        osm_handler = OSMDataHandler(orchestrator)
        osm_data = osm_handler.process(plot=False)
        if osm_data is not None:
            logger.info("OSM data processing complete.")
        else:
            logger.warning("OSM data processing did not yield a result.")

        # --- STEP 3.5: Process Microsoft Buildings Data ---
        logger.info("STEP 3.5: Processing Microsoft Buildings data")
        microsoft_buildings_handler = MicrosoftBuildingsDataHandler(orchestrator)
        microsoft_buildings_data = microsoft_buildings_handler.process()
        if microsoft_buildings_data:
            logger.info(
                f"Microsoft Buildings data processing complete. "
                f"Found {len(microsoft_buildings_data['ms_buildings'])} buildings."
            )
        else:
            logger.warning("Microsoft Buildings data processing did not yield buildings data.")

        # # --- STEP 4: Building Classification Heuristic ---
        building_classification_heuristic = BuildingHeuristicsProcessor(
            orchestrator.base_output_dir)
        #
        building_classification_heuristic.process(
            census_data, osm_data, microsoft_buildings_data, nrel_data["vintage_distribution"])

        # --- STEP 5: ROUTABLE ROAD NETWORK GENERATION ---
        road_network_builder = RoadNetworkBuilder(orchestrator=orchestrator)
        road_network_results = road_network_builder.process(
            boundary_gdf=census_data['target_region_boundary'], plot=True)
        if road_network_results.get('geojson_file'):
            logger.info(
                f"Road network generation complete. Network GPKG at: {road_network_results['geojson_file']}"
            )
        else:
            logger.warning("Road network generation did not yield a GPKG path.")

        logger.info("SynGrid Data Processing Pipeline v2.0 completed successfully.")

    except ValueError as ve:
        logger.error(f"Configuration or validation error during pipeline: {ve}", exc_info=True)
    except RuntimeError as re:
        logger.error(f"Runtime error during pipeline execution: {re}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)
    finally:
        # Calculate and log total execution time
        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"SynGrid Data Processing Pipeline completed in {total_time} seconds")


if __name__ == '__main__':
    run_pipeline_v2()

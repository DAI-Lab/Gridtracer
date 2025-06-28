"""GridTracer Data Processing Pipeline

This script serves as the main entrypoint for the GridTracer data processing
workflow. It orchestrates a series of modules to download, process, and
integrate various geospatial datasets to model electrical grid infrastructure.

The pipeline executes the following stages in order:
1.  **Census Data Processing:** Fetches census block and geometry data to
    define the target region's boundary.
2.  **NREL Data Processing:** Processes NREL RESstock/Comstock data to
    determine building vintage distributions.
3.  **OpenStreetMap (OSM) Data Extraction:** Downloads power infrastructure,
    buildings, and road networks from OSM for the target region.
4.  **Microsoft Buildings Integration:** Downloads and enriches the buildings with height data.
5.  **Building Classification:** Combines all building data sources and applies
    heuristics to classify buildings and estimate electrical loads.
6.  **Routable Road Network Generation:** Builds a clean, routable road
    network for use with pgRouting.

Prerequisites:
  - Define your region of interest in the `config.yaml` file.

Usage:
  # Run the entire pipeline for the configured region
  $ python -m gridtracer.scripts.main
"""
import time
from typing import Any, Dict, Optional

from gridtracer.config import config
from gridtracer.data_processor.data_imports.census import CensusDataHandler
from gridtracer.data_processor.data_imports.microsoft_buildings import (
    MicrosoftBuildingsDataHandler,)
from gridtracer.data_processor.data_imports.nrel import NRELDataHandler
from gridtracer.data_processor.data_imports.osm.osm_data_handler import OSMDataHandler
from gridtracer.data_processor.data_imports.osm.road_network_builder import RoadNetworkBuilder
from gridtracer.data_processor.processing.building_processor import BuildingProcessor
from gridtracer.data_processor.workflow import WorkflowOrchestrator
from gridtracer.utils import create_logger

logger = create_logger(
    name="Main",
    log_level=config.log_level,
    log_file=config.log_file,
)


def run_full_pipeline(
    census_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Run the main gridtracer data processing pipeline.

    Uses the WorkflowOrchestrator to manage the pipeline steps.

    Args:
        census_data: A dictionary containing pre-loaded census data,
            including 'target_region_boundary'. If provided, the census
            data processing step is skipped. Defaults to None.
    """
    start_time = time.time()
    logger.info("Starting gridtracer Data Processing Pipeline v2.0")

    try:
        # # Initialize the orchestrator, loading config, setting up FIPS, and creating all output directories
        orchestrator = WorkflowOrchestrator()

        # --- STEP 1: REGIONAL DATA EXTRACTION & PREPARATION ---
        logger.info("STEP 1: Regional Data Extraction & Preparation")

        census_handler = CensusDataHandler(orchestrator)
        census_data = census_handler.process(plot=False)

        # --- STEP 2: Process NREL Data ---

        logger.info("STEP 2: Processing NREL data")
        nrel_handler = NRELDataHandler(orchestrator)
        nrel_data = nrel_handler.process()

        # --- STEP 3: Extract OSM Data ---
        logger.info("STEP 3: Extracting OSM data")
        osm_handler = OSMDataHandler(orchestrator)
        osm_data = osm_handler.process(plot=False)

        # --- STEP 3.5: Process Microsoft Buildings Data ---
        logger.info("STEP 3.5: Processing Microsoft Buildings data")
        microsoft_buildings_handler = MicrosoftBuildingsDataHandler(orchestrator)
        microsoft_buildings_data = microsoft_buildings_handler.process()

        # # --- STEP 4: Building Classification ---
        logger.info("STEP 4: Building Classification")
        building_processor = BuildingProcessor(
            orchestrator.get_dataset_specific_output_directory("BUILDINGS_OUTPUT"))

        building_processor.process(
            census_data, osm_data, microsoft_buildings_data, nrel_data["vintage_distribution"])

        # --- STEP 5: ROUTABLE ROAD NETWORK GENERATION ---
        logger.info("STEP 5: Routable Road Network Generation")
        road_network_builder = RoadNetworkBuilder(orchestrator=orchestrator)
        _ = road_network_builder.process()

        logger.info("gridtracer Data Processing Pipeline v2.0 completed successfully.")

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

        logger.info(
            f"gridtracer Data Processing Pipeline completed in {total_time} seconds"
        )


if __name__ == "__main__":
    run_full_pipeline()

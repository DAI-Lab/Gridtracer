"""GridTracer: Geospatial Data Processing Pipeline for Electrical Grid Modeling

This module provides the main execution pipeline for GridTracer, a comprehensive
geospatial data processing framework designed for synthetic electrical grid
infrastructure modeling in the United States.

The pipeline integrates multiple authoritative data sources to create detailed,
georeferenced datasets suitable for energy system modeling, urban planning, and
infrastructure analysis.

Pipeline Stages:
    1. **Census Boundary Definition**: Establishes precise geographic scope using
       US Census TIGER/Line data for states, counties, and subdivisions.

    2. **Census Subdivision Segmentation**: Generates comprehensive subdivision
       datasets with population and area metrics for the target region.

    3. **NREL Data Integration**: Processes residential/commercial building stock
       characteristics and vintage distributions from NREL datasets.

    4. **OpenStreetMap Extraction**: Retrieves building footprints, road networks,
       points of interest, and power infrastructure from OSM.

    5. **Microsoft Buildings Enhancement**: Augments building data with high-resolution
       footprints and height information from Microsoft's ML-derived datasets.

    6. **Building Classification**: Applies energy-focused heuristics to classify
       buildings by type and estimate electrical load characteristics.

    7. **Road Network Generation**: Creates pgRouting-compatible road networks for
       transportation and utility routing analysis.

Configuration:
    Region specification and processing parameters are defined in config.yaml:
    - Region: state, county, optional subdivision (using FIPS codes)
    - Data paths: OSM PBF files, NREL datasets
    - Processing thresholds: building classification parameters

Usage:
    As a module:
        python -m gridtracer.scripts.main

    As a script:
        python gridtracer/scripts/main.py

Output:
    Hierarchical directory structure organized by administrative region:
    output/{STATE}/{COUNTY}/{SUBDIVISION}/
        ├── CENSUS/          # Boundaries and census blocks
        ├── NREL/            # Building typology data
        ├── OSM/             # OpenStreetMap extracts
        ├── MICROSOFT_BUILDINGS/  # Building footprints
        ├── BUILDINGS_OUTPUT/     # Classified buildings
        └── STREET_NETWORK/       # Routable networks

Author: MIT Data To AI Lab
License: MIT
"""
import time

from gridtracer.config.config_loader import LOG_FILE, LOG_LEVEL
from gridtracer.data.census_subdivision import CountySubdivisionHandler
from gridtracer.data.imports.census import CensusDataHandler
from gridtracer.data.imports.msft_building_footprints import MicrosoftBuildingsDataHandler
from gridtracer.data.imports.nrel import NRELDataHandler
from gridtracer.data.imports.osm.osm_data_handler import OSMDataHandler
from gridtracer.data.imports.osm.road_network_builder import RoadNetworkBuilder
from gridtracer.data.processing.building_processor import BuildingProcessor
from gridtracer.data.workflow import WorkflowOrchestrator
from gridtracer.utils import create_logger

logger = create_logger(
    name="Main",
    log_level=LOG_LEVEL,
    log_file=LOG_FILE,
)


def run_full_pipeline(
) -> None:
    """
    Run the full data import pipeline for the target region.

    This function initializes the WorkflowOrchestrator, processes census data,
    census subdivision data, NREL data, OSM data, Microsoft buildings data,
    and building classification. It also generates a routable road network.

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting Data Import Pipeline for Target Region")

    try:
        # # Initialize the orchestrator, loading config, setting up FIPS,
        # and creating all output directories
        logger.info("--------------------------------")
        logger.info("STEP 1: Initializing WorkflowOrchestrator")
        logger.info("--------------------------------")
        orchestrator = WorkflowOrchestrator()

        # --- STEP 2: Census Data Extraction & Preparation ---
        logger.info("--------------------------------")
        logger.info("STEP 2: Census Data Extraction & Preparation")
        logger.info("--------------------------------")
        census_handler = CensusDataHandler(orchestrator)
        census_data = census_handler.process()

        subcounty_segmentation_handler = CountySubdivisionHandler(
            orchestrator=orchestrator)
        subcounty_segmentation_handler.process(
            state_filter=orchestrator.fips_dict['state'])

        # --- STEP 3: Census Subdivision Segmentation ---
        logger.info("--------------------------------")
        logger.info("STEP 3: Census Subdivision Segmentation")
        logger.info("--------------------------------")
        nrel_handler = NRELDataHandler(orchestrator)
        nrel_data = nrel_handler.process()

        # --- STEP 4: Extract OSM Data ---
        logger.info("--------------------------------")
        logger.info("STEP 4: Extracting OSM data")
        logger.info("--------------------------------")
        osm_handler = OSMDataHandler(orchestrator)
        osm_data = osm_handler.process(plot=False)

        # --- STEP 5: Process Microsoft Buildings Data ---
        logger.info("STEP 5: Processing Microsoft Buildings data")
        logger.info("--------------------------------")
        microsoft_buildings_handler = MicrosoftBuildingsDataHandler(
            orchestrator)
        microsoft_buildings_data = microsoft_buildings_handler.process()

        # # --- STEP 6: Building Classification ---
        logger.info("--------------------------------")
        logger.info("STEP 6: Building Classification")
        logger.info("--------------------------------")
        building_processor = BuildingProcessor(
            orchestrator.get_dataset_specific_output_directory(
                "BUILDINGS_OUTPUT"))

        building_processor.process(
            census_data,
            osm_data,
            microsoft_buildings_data,
            nrel_data["vintage_distribution"]
        )

        # --- STEP 7: ROUTABLE ROAD NETWORK GENERATION ---
        logger.info("--------------------------------")
        logger.info("STEP 7: Routable Road Network Generation")
        logger.info("--------------------------------")
        road_network_builder = RoadNetworkBuilder(orchestrator=orchestrator)
        _ = road_network_builder.process()

        logger.info("--------------------------------")
        logger.info(
            " ✓ Data Import Pipeline for Target Region completed successfully.")

    except ValueError as ve:
        logger.error(
            f"Configuration or validation error during pipeline: {ve}",
            exc_info=True)
    except RuntimeError as re:
        logger.error(
            f"Runtime error during pipeline execution: {re}",
            exc_info=True)
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the pipeline: {e}",
            exc_info=True)
    finally:
        # Calculate and log total execution time
        end_time = time.time()
        total_time = end_time - start_time

        logger.info(
            "Import Pipeline completed in "
            f"{total_time} seconds"
        )


if __name__ == "__main__":
    run_full_pipeline()

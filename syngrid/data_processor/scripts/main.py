# Entrypoint for the SynGrid data processing pipeline
#
# This script orchestrates the entire data processing workflow:
# 1. Regional Data Extraction & Preparation
# 2. Building Classification Pipeline
# 3. Routable Road Network Generation
# 4. Transformer Network Extraction

from syngrid.data_processor.config import ConfigLoader
from syngrid.data_processor.utils import (
    get_region_data, 
    logger, 
    lookup_fips_codes, 
    nrel_data_preprocessing, 
    visualize_blocks,
)

# Import OSM data handling functions (these need to be implemented)
# from syngrid.data_processor.data.osm import osm_data_extraction


def main():
    #####################################################################
    # STEP 1: REGIONAL DATA EXTRACTION & PREPARATION
    #####################################################################
    logger.info("Starting Step 1: Regional Data Extraction & Preparation")

    # 1.1: Parse the YAML config, validate inputs
    logger.info("1.1: Loading and validating configuration")
    config = ConfigLoader()
    region = config.get_region()
    output_dir = config.get_output_dir()
    input_file_paths = config.get_input_data_paths()

    # 1.2: Lookup FIPS codes for the specified region
    logger.info("1.2: Looking up FIPS codes for region")
    fips_dict = lookup_fips_codes(region)
    logger.info(f"Extracting data for region: {region}")

    # 1.3: Download/load administrative boundaries and census blocks
    logger.info("1.3: Downloading Census boundaries and blocks")
    region_data = get_region_data(fips_dict)
    
    # 1.4: Visualize the region's census blocks
    logger.info("1.4: Visualizing census blocks")
    # Create visualization title
    title = None
    if region_data['subdivision'] is not None and not region_data['subdivision'].empty:
        # For subdivision
        subdiv_name = region_data['subdivision'].iloc[0]['NAME']
        title = f"Census Blocks in {subdiv_name}"
    else:
        # For county
        title = f"Census Blocks in {fips_dict['county']}, {fips_dict['state']}"

    # Generate and log visualization
    if region_data['blocks'] is not None and not region_data['blocks'].empty:
        plot_file = visualize_blocks(
            blocks_gdf=region_data['blocks'],
            subdivision_gdf=region_data['subdivision'],
            title=title
        )
        logger.info(f"Generated visualization: {plot_file}")

    logger.info(f"Region boundary: {region_data['boundary']}")

    # 1.5: Download and process NREL data for the region
    logger.info("1.5: Processing NREL data")
    nrel_data_files = nrel_data_preprocessing(
        fips_dict,
        input_file_path=input_file_paths['nrel_data'],
        output_dir=output_dir)
    logger.info(f"NREL data processing complete: {nrel_data_files}")
    
    # 1.6: Download and process NLCD land cover data (to be implemented)
    logger.info("1.6: Downloading NLCD data (not implemented yet)")
    # TODO: Implement NLCD land cover data processing
    
    # 1.7: Extract OSM data for the region 
    logger.info("1.7: Extracting OpenStreetMap data")
    # TODO: Implement OSM data extraction for:
    # - Buildings 
    # - Roads
    # - POIs
    # - Power infrastructure (transformers, substations)
    # osm_data_extracted = osm_data_extraction(region_data)
    
    # 1.8: Clip all datasets to the region boundary
    logger.info("1.8: Clipping all datasets to region boundary")
    # TODO: Implement clipping for all datasets that haven't been clipped yet
    
    # 1.9: Project all datasets to a consistent CRS if needed
    logger.info("1.9: Ensuring consistent coordinate reference system")
    # TODO: Implement CRS transformation if needed
    
    logger.info("Step 1 completed: Regional Data Extraction & Preparation")

    #####################################################################
    # STEP 2: BUILDING CLASSIFICATION PIPELINE (placeholder)
    #####################################################################
    logger.info("Step 2 not implemented yet: Building Classification Pipeline")
    
    #####################################################################
    # STEP 3: ROUTABLE ROAD NETWORK GENERATION (placeholder)
    #####################################################################
    logger.info("Step 3 not implemented yet: Routable Road Network Generation")
    
    #####################################################################
    # STEP 4: TRANSFORMER NETWORK EXTRACTION (placeholder)
    #####################################################################
    logger.info("Step 4 not implemented yet: Transformer Network Extraction")
    
    logger.info("SynGrid data processing pipeline completed.")


if __name__ == '__main__':
    main()

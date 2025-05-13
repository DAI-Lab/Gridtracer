# Entrypoint for the SynGrid data processing pipeline
#
# This script orchestrates the entire data processing workflow:
# 1. Regional Data Extraction & Preparation
# 2. Building Classification Pipeline
# 3. Routable Road Network Generation
# 4. Transformer Network Extraction

from syngrid.data_processor.config import ConfigLoader
from syngrid.data_processor.data import CensusDataHandler, NRELDataHandler, OSMDataHandler
from syngrid.data_processor.utils import logger, lookup_fips_codes, visualize_blocks, visualize_osm_data


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

    # 1.3: Download/load administrative boundaries and census blocks using the Census handler
    logger.info("1.3: Downloading Census boundaries and blocks")
    census_handler = CensusDataHandler(fips_dict, output_dir=output_dir)
    region_data = census_handler.process()

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
            fips_dict=fips_dict,
            title=title
        )
        logger.info(f"Generated visualization: {plot_file}")

    logger.info(f"Region boundary: {region_data['boundary']}")

    # 1.5: Download and process NREL data for the region
    logger.info("1.5: Processing NREL data")
    nrel_handler = NRELDataHandler(
        fips_dict,
        input_file_path=input_file_paths['nrel_data'],
        output_dir=output_dir
    )
    nrel_data = nrel_handler.process()
    logger.info(f"NREL data processing complete: {nrel_data['parquet_path']}")

    # 1.6: Download and process NLCD land cover data (to be implemented)
    logger.info("1.6: Downloading NLCD data (not implemented yet)")
    # TODO: Implement NLCD data handler class and use it here
    # nlcd_handler = NLCDDataHandler(fips_dict, output_dir=output_dir)
    # nlcd_data = nlcd_handler.process(boundary_gdf=region_data['boundary'])

    # 1.7: Extract OSM data for the region
    logger.info("1.7: Extracting OpenStreetMap data")
    
    osm_handler = OSMDataHandler(
        fips_dict,
        output_dir=output_dir
    )
    
    # Process OSM data with the region boundary for efficient extraction
    if region_data['boundary'] is not None and not region_data['boundary'].empty:
        logger.info("Extracting OSM data using exact boundary polygon via OSMnx")
        osm_data = osm_handler.process(region_data['boundary'])
    else:
        # We cannot process without a boundary when using OSMnx with Overpass API
        logger.error("No region boundary available. OSMnx extraction requires a boundary polygon.")
        logger.error("Please ensure a valid region boundary is available.")
        osm_data = None
    
    if osm_data:
        if osm_data['buildings'] is not None:
            logger.info(f"Extracted {len(osm_data['buildings'])} OSM buildings")
        if osm_data['pois'] is not None:
            logger.info(f"Extracted {len(osm_data['pois'])} OSM POIs")
        if osm_data['power'] is not None:
            logger.info(f"Extracted {len(osm_data['power'])} power infrastructure features")
            logger.info(f"Power features saved to: {osm_data['power_filepath']}")
            
        # 1.7.1: Visualize the OSM data
        logger.info("1.7.1: Visualizing OpenStreetMap data")
        
        # First, create a visualization with all data elements
        osm_plot_file = visualize_osm_data(
            fips_dict=fips_dict,
            boundary_gdf=region_data['boundary'],
            output_dir=output_dir,
            plot_buildings=True,
            plot_pois=True,
            plot_power=True
        )
        if osm_plot_file:
            logger.info(f"Generated complete OSM data visualization: {osm_plot_file}")
        
        # Then create a visualization without POIs for clarity
        osm_plot_no_pois = visualize_osm_data(
            fips_dict=fips_dict,
            boundary_gdf=region_data['boundary'],
            output_dir=output_dir,
            plot_buildings=True,
            plot_pois=False,
            plot_power=True
        )
        if osm_plot_no_pois:
            logger.info(f"Generated OSM visualization without POIs: {osm_plot_no_pois}")
   
    # 1.8: Clip all datasets to the region boundary
    logger.info("1.8: Clipping all datasets to region boundary")
    # Note: Now handled within each data handler's process method

    # 1.9: Project all datasets to a consistent CRS if needed
    logger.info("1.9: Ensuring consistent coordinate reference system")
    # Note: Now handled within each data handler's process method

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

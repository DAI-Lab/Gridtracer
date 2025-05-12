# Entrypoint for the user

from syngrid.data_processor.config import ConfigLoader
from syngrid.data_processor.utils import (
    get_region_data, logger, lookup_fips_codes, nrel_data_preprocessing, visualize_blocks,)

if __name__ == '__main__':

    # Read the config file
    config = ConfigLoader()
    region = config.get_region()
    output_dir = config.get_output_dir()
    input_file_paths = config.get_input_data_paths()

    # Get FIPS codes
    fips_dict = lookup_fips_codes(region)
    logger.info("Extracting data for region: {}".format(region))

    # Get boundaries for the region of interest
    region_data = get_region_data(fips_dict)
    # Create visualization
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

    # # NREL Data Extraction
    # # Download the NREL data for the region of interest.
    nrel_data = nrel_data_preprocessing(
        fips_dict,
        input_file_path=input_file_paths['nrel_data'],
        output_dir=output_dir)

    # logger.info(f"NREL data: {nrel_data}")

    # # Landuse Data Extraction - TBD

    # Now OSM data reader: Starting from a geofabrik .pbf file
    # Extract OSM data for the region of interest:
    osm_data_extracted = osm_data_extraction(region_data)

    # logger.info(f"NLCD Landuse data: {nlcd_landuse}")

    # # Now that we have the boundaries we can  download the buildings data from OSM. and then store to three different shapefiels for POI's, buildings and landuse.
    # # Download the buildings data from OSM which fall into all of the blocks of the region of interest.
    # buildings_data = osm_data_extraction(region_data)

    # # Store the buildings data to three different shapefiles for POI's, buildings and landuse.

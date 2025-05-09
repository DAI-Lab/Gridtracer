# Entrypoint for the user

from syngrid.data_processor.config import ConfigLoader
from syngrid.data_processor.utils import logger, lookup_fips_codes

if __name__ == '__main__':

    # Read the config file
    config = ConfigLoader()
    region = config.get_region()
    # Get FIPS codes
    fips_dict = lookup_fips_codes(region)
    logger.info(f"FIPS codes: {fips_dict}")

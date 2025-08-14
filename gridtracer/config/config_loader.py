import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """
    Load and manage configuration settings from YAML files for the gridtracer data processor.

    This class is designed to be used as a singleton. A single instance is
    created at the module level, which should be imported by other parts of
    the application.
    """

    def __init__(self, config_path=None):
        """
        Initialize the ConfigLoader, load the YAML file, and set key config
        properties as attributes.
        """
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            # Default to config.yaml in the same directory as this script
            self.config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        else:
            self.config_path = config_path

        self.config: Dict[str, Any] = self._load_config()

        # Expose logging configuration as direct attributes for simple access
        self.log_level: int = self._parse_log_level()
        self.log_file: str = self._parse_log_file()

        # Expose constants from config as direct attributes
        self.EPSG: int = self.config.get('EPSG', 5070)
        self.BUILDING_TYPE_THRESHOLDS: Dict[str, Any] = self.config.get(
            'BUILDING_TYPE_THRESHOLDS', {})
        self.CENSUS_URLS: Dict[str, str] = self.config.get('CENSUS_URLS', {})

        self._validate_region()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            dict: Configuration as a dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config_data
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {str(e)}")
            raise

    def _parse_log_level(self) -> int:
        """Get the logging level from the configuration.
        Returns:
            int: The logging level (e.g., logging.INFO, logging.DEBUG).
                 Defaults to logging.INFO if not specified or invalid.
        """
        log_level_str = self.config.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_str, None)

        if not isinstance(level, int):
            # Use root logger for this warning, as the class logger may not
            # be fully configured yet.
            logging.warning(
                "Invalid log level '%s' in config. Defaulting to INFO.",
                log_level_str,
            )
            return logging.INFO

        return level

    def _parse_log_file(self) -> str:
        """Get the log file path from the configuration.
        Returns:
            str: The path to the log file.
        """
        return self.config.get('LOG_FILE', 'log.txt')

    def _validate_region(self):
        """
        Validate that the region configuration contains necessary information.
        """
        region = self.get_region()
        if not region.get('STATE'):
            self.logger.warning("No state specified in configuration")
        if not region.get('COUNTY'):
            self.logger.warning("No county specified in configuration")

    def get_region(self):
        """
        Get the region configuration.

        Returns:
            dict: Region configuration with state, county, and count_subdivision
        """
        return self.config.get('REGION', {})

    def get_input_data_paths(self):
        """
        Get all configured input data paths.

        Returns:
            dict: Dictionary of input data paths
        """
        return self.config.get('INPUT_DATA', {})

    def get_output_dir(self):
        """
        Get the output directory path.

        Returns:
            str: Output directory path
        """
        return Path(self.config.get('OUTPUT_DIR', 'gridtracer/output/'))

    def get_epsg(self) -> int:
        """
        Get the EPSG code for spatial data.

        Returns:
            int: EPSG code (default: 5070 for NAD83 / Conus Albers)
        """
        return self.EPSG

    def get_building_type_thresholds(self) -> Dict[str, Any]:
        """
        Get building type classification thresholds.

        Returns:
            dict: Building type thresholds for classification
        """
        return self.BUILDING_TYPE_THRESHOLDS

    def get_census_urls(self) -> Dict[str, str]:
        """
        Get Census TIGER data URLs.

        Returns:
            dict: Census URLs configuration
        """
        return self.CENSUS_URLS


# --- Singleton Instance ---
# This single, pre-initialized instance should be imported by other modules
# to ensure consistent configuration access across the application.
config = ConfigLoader()

# --- Module-level constants for convenient access ---
# These constants are available as direct imports for backward compatibility
EPSG = config.EPSG
BUILDING_TYPE_THRESHOLDS = config.BUILDING_TYPE_THRESHOLDS
CENSUS_URLS = config.CENSUS_URLS
LOG_LEVEL = config.log_level
LOG_FILE = config.log_file
REGION = config.get_region()
INPUT_DATA = config.get_input_data_paths()
OUTPUT_DIR = config.get_output_dir()

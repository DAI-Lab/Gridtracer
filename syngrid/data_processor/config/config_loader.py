import logging
import os
from pathlib import Path

import yaml


class ConfigLoader:
    """
    Load and manage configuration settings from YAML files for the SynGrid data processor
    """

    def __init__(self, config_path=None):
        """
        Initialize the ConfigLoader with a path to a YAML configuration file.

        Args:
            config_path (str, optional): Path to the configuration file.
                If None, will look for config.yaml in the same directory as this script.
        """
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            # Default to config.yaml in the same directory as this script
            self.config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        else:
            self.config_path = config_path

        self.config = self._load_config()
        self._validate_region()

    def _load_config(self):
        """
        Load configuration from YAML file.

        Returns:
            dict: Configuration as a dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {str(e)}")
            raise

    def _validate_region(self):
        """
        Validate that the region configuration contains necessary information.
        """
        region = self.get_region()
        if not region.get('state'):
            self.logger.warning("No state specified in configuration")
        if not region.get('county'):
            self.logger.warning("No county specified in configuration")

    def get_region(self):
        """
        Get the region configuration.

        Returns:
            dict: Region configuration with state, county, and count_subdivision
        """
        return self.config.get('region', {})

    def get_input_data_paths(self):
        """
        Get all configured input data paths.

        Returns:
            dict: Dictionary of input data paths
        """
        return self.config.get('input_data', {})

    def get_output_dir(self):
        """
        Get the output directory path.

        Returns:
            str: Output directory path
        """
        return Path(self.config.get('output_dir', 'syngrid/data_processor/output/'))

    def get_output_path(self, filename=None):
        """
        Get the output path, optionally with a filename appended.

        Args:
            filename (str, optional): Filename to append to the output directory

        Returns:
            str: Output path
        """
        output_dir = self.get_output_dir()

        # Create region-based subdirectory
        region = self.get_region()
        state = region.get('state', '')
        county = region.get('county', '')
        subdivision = region.get('count_subdivision', '')

        # Create a path structure based on region information
        if state and county:
            subdirectory = f"{state}/{county.replace(' ', '_')}"
            if subdivision:
                subdirectory = f"{subdirectory}/{subdivision.replace(' ', '_')}"
            output_dir = os.path.join(output_dir, subdirectory)

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        if filename:
            return os.path.join(output_dir, filename)
        return output_dir

    def get_overpass_config(self):
        """
        Get Overpass API configuration.

        Returns:
            dict: Overpass API configuration
        """
        return self.config.get('overpass', {})

    def get_processing_params(self):
        """
        Get processing parameters.

        Returns:
            dict: Processing parameters
        """
        return self.config.get('processing', {})

    def update_region(self, state=None, county=None, count_subdivision=None):
        """
        Update the region configuration and save to file.

        Args:
            state (str, optional): State abbreviation
            county (str, optional): County name
            count_subdivision (str, optional): County subdivision name
        """
        if not any([state, county, count_subdivision]):
            return

        if 'region' not in self.config:
            self.config['region'] = {}

        if state:
            self.config['region']['state'] = state
        if county:
            self.config['region']['county'] = county
        if count_subdivision:
            self.config['region']['count_subdivision'] = count_subdivision

        self._save_config()
        self._validate_region()

    def _save_config(self):
        """
        Save the current configuration to the YAML file.
        """
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                self.logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise

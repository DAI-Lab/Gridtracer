"""
Base class for data handlers in the SynGrid data processing pipeline.

This module provides the base DataHandler class which defines common functionality
for downloading, processing, and saving different types of data sources.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd

# Set up logging
logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """
    Abstract base class for all data source handlers.

    This class defines the common interface and functionality that all data source
    handlers should implement.
    """

    def __init__(self, fips_dict, output_dir=None):
        """
        Initialize the DataHandler.

        Args:
            fips_dict (dict): Dictionary containing region information including:
                - state: State abbreviation
                - state_fips: State FIPS code
                - county: County name
                - county_fips: County FIPS code
                - subdivision: County subdivision name (optional)
                - subdivision_fips: County subdivision FIPS code (optional)
            output_dir (str or Path, optional): Base output directory
                If None, defaults to a directory within the data processor output
        """
        self.fips_dict = fips_dict
        self._validate_fips_dict()

        # Set up output directory
        if output_dir is None:
            self.output_dir = Path("syngrid/data_processor/output")
        else:
            self.output_dir = Path(output_dir)

        # Create dataset-specific directory structure
        self.dataset_name = self._get_dataset_name()
        self.dataset_output_dir = self._create_dataset_output_dir()

    def _validate_fips_dict(self):
        """
        Validate that the FIPS dictionary contains required fields.

        Raises:
            ValueError: If required FIPS information is missing
        """
        required_fields = ['state', 'county', 'state_fips', 'county_fips']
        missing_fields = [field for field in required_fields if field not in self.fips_dict]

        if missing_fields:
            logger.error(f"Missing required FIPS information: {', '.join(missing_fields)}")
            msg = f"FIPS dictionary missing required fields: {', '.join(missing_fields)}"
            raise ValueError(msg)

    def _create_dataset_output_dir(self):
        """
        Create standardized dataset-specific output directory.

        The path follows the structure:
        base_output_dir/state/county/subdivision/dataset_name/

        Returns:
            Path: Path object to the output directory
        """
        # Extract fields for directory structure
        state = self.fips_dict.get('state', 'unknown')
        county = self.fips_dict.get('county', 'unknown')
        subdivision = self.fips_dict.get('subdivision')

        # Sanitize names for directory paths
        state_dir = state.replace(' ', '_')
        county_dir = county.replace(' ', '_')

        # Create hierarchical path
        output_path = self.output_dir / state_dir / county_dir

        # Add subdivision level if available
        if subdivision:
            subdivision_dir = subdivision.replace(' ', '_')
            output_path = output_path / subdivision_dir

        # Add dataset-specific directory
        output_path = output_path / self.dataset_name

        # Create directory
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory for {self.dataset_name}: {output_path}")

        return output_path

    @abstractmethod
    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name
        """

    @abstractmethod
    def download(self):
        """
        Download the data for the region specified in fips_dict.

        Returns:
            dict: Dictionary containing the downloaded data
        """

    def clip_to_boundary(self, data_gdf, boundary_gdf, crs=None):
        """
        Clip GeoDataFrame to the specified boundary.

        Args:
            data_gdf (GeoDataFrame): Data to clip
            boundary_gdf (GeoDataFrame): Boundary to clip to
            crs (str, optional): Coordinate reference system to use
                If None, uses the CRS of the boundary_gdf

        Returns:
            GeoDataFrame: Clipped GeoDataFrame
        """
        logger.info(f"Clipping {len(data_gdf)} features to region boundary")

        if data_gdf is None or data_gdf.empty:
            logger.warning("No data to clip")
            return data_gdf

        if boundary_gdf is None or boundary_gdf.empty:
            logger.warning("No boundary to clip to")
            return data_gdf

        # Ensure both GeoDataFrames have the same CRS
        if crs is not None:
            data_gdf = data_gdf.to_crs(crs)
            boundary_gdf = boundary_gdf.to_crs(crs)
        elif data_gdf.crs != boundary_gdf.crs:
            data_gdf = data_gdf.to_crs(boundary_gdf.crs)

        # Perform the clip operation
        try:
            # Get the geometry from the boundary
            if len(boundary_gdf) == 1:
                boundary_geom = boundary_gdf.geometry.iloc[0]
            else:
                # If multiple geometries, dissolve them
                boundary_geom = boundary_gdf.geometry.unary_union

            # Clip the data
            clipped_gdf = gpd.clip(data_gdf, boundary_geom)
            logger.info(f"Clipped data to {len(clipped_gdf)} features")
            return clipped_gdf

        except Exception as e:
            logger.error(f"Error clipping data: {str(e)}")
            return data_gdf

    def save_to_file(self, data_gdf, filename, driver="GeoJSON"):
        """
        Save GeoDataFrame to file.

        Args:
            data_gdf (GeoDataFrame): Data to save
            filename (str): Filename (without path)
            driver (str, optional): Format driver to use. Default is "GeoJSON"

        Returns:
            Path: Path to the saved file
        """
        if data_gdf is None or data_gdf.empty:
            logger.warning(f"No data to save to {filename}")
            return None

        try:
            # Construct full output path
            output_path = self.dataset_output_dir / filename

            # Save the data
            data_gdf.to_file(output_path, driver=driver)
            logger.info(f"Saved data to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving data to {filename}: {str(e)}")
            return None

    @abstractmethod
    def process(self, boundary_gdf=None):
        """
        Process the data for the region.

        This method should implement the complete data processing workflow for
        the specific data source, including downloading, transforming, and saving.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to use for clipping
                If None, the method should determine or download the boundary

        Returns:
            dict: Dictionary containing processed data and file paths
        """

    def get_base_output_dir(self) -> Path:
        """
        Get the standardized output directory path up to (but not including) the dataset name.

        Returns:
            Path: Path object to the base output directory (state/county/subdivision)
        """
        state = self.fips_dict.get('state', 'unknown')
        county = self.fips_dict.get('county', 'unknown')
        subdivision = self.fips_dict.get('subdivision')

        state_dir = state.replace(' ', '_')
        county_dir = county.replace(' ', '_')

        output_path = self.output_dir / state_dir / county_dir

        if subdivision:
            subdivision_dir = subdivision.replace(' ', '_')
            output_path = output_path / subdivision_dir

        return output_path

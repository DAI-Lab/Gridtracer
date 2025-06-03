import csv
import logging
import os
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
from shapely.geometry import MultiPolygon, Polygon

from syngrid.data_processor.config import ConfigLoader

# Define all known dataset names for directory creation
ALL_DATASETS: List[str] = [
    "CENSUS",
    "NREL",
    "OSM",
    "MICROSOFT_BUILDINGS",
    "BUILDINGS_OUTPUT",
    "STREET_NETWORK",
    "PLOTS",
    "TMP"
]


class WorkflowOrchestrator:
    """
    Orchestrates the SynGrid data processing pipeline.

    This class manages configuration, regional context (FIPS codes, boundaries),
    output directory structures for all datasets, and the overall workflow execution.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the WorkflowOrchestrator.

        Args:
            config_path (Optional[str]): Path to the YAML configuration file.
                If None, ConfigLoader will use its default path.
        """
        self.logger = logging.getLogger(__name__)
        self.config_loader: ConfigLoader = ConfigLoader(config_path)
        self.base_output_dir: Path = self.config_loader.get_output_dir()

        self.fips_dict: Optional[Dict[str, str]] = None
        self.region_boundary_gdf: Optional[gpd.GeoDataFrame] = None
        self.is_county_subdivision: bool = False
        self._osm_parser: Optional[OSM] = None

        self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> None:
        """Initialize critical components of the orchestrator."""
        self.logger.info("Initializing Workflow Orchestrator...")
        self._resolve_fips_codes()
        self.is_county_subdivision = self.fips_dict.get('subdivision') is not None
        self._create_output_directories()

        self.logger.info(
            f"Orchestrator initialized. Subdivision scope: {self.is_county_subdivision}"
        )

    def _resolve_fips_codes(self) -> None:
        """
        Lookup FIPS codes for the configured region.

        The FIPS lookup file is downloaded to the root of the configured output directory.
        """
        region_config = self.config_loader.get_region()
        state = region_config.get('state')
        county = region_config.get('county')
        subdivision = region_config.get('county_subdivision')
        lookup_url = region_config.get('lookup_url')

        if not all([state, county, lookup_url]):
            self.logger.error(
                "Missing parameters for FIPS lookup: 'state', 'county', and 'lookup_url' are required in config."
            )
            raise ValueError(
                "State, county, and lookup_url must be provided in config for FIPS lookup."
            )

        filename = os.path.basename(lookup_url)
        local_file_path = self.base_output_dir / filename
        self.logger.info(f"Local file path: {local_file_path}")

        if not local_file_path.exists():
            self.logger.info(
                f"Downloading FIPS lookup file from {lookup_url} to {local_file_path}")
            try:
                urllib.request.urlretrieve(lookup_url, local_file_path)  # type: ignore
                self.logger.debug(f"FIPS lookup file saved to {local_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to download FIPS lookup file: {e}")
                raise
        else:
            self.logger.debug(f"Using existing FIPS lookup file: {local_file_path}")

        try:
            with open(local_file_path, 'r', encoding='latin-1') as infile:
                reader = csv.reader(infile)
                processed_rows = []
                for i, row in enumerate(reader):
                    if i == 0 or (row and row[0] == 'STATE'):  # Skip header
                        continue
                    if len(row) == 7:
                        processed_rows.append(row)
                    elif len(row) == 8:  # Handle known inconsistency in some files
                        merged_row = row[:5] + [row[5] + ' ' + row[6]] + [row[7]]
                        processed_rows.append(merged_row)

            column_names = ['state_abbr', 'state_fips', 'county_fips', 'county_name',
                            'subdivision_fips', 'subdivision_name', 'funcstat']
            df = pd.DataFrame(processed_rows, columns=column_names)

            state_df = df[df['state_abbr'] == state]
            if state_df.empty:
                raise ValueError(f"State abbreviation '{state}' not found in lookup file.")

            county_matches = state_df[state_df['county_name'] == county]
            if county_matches.empty:
                raise ValueError(f"County '{county}' not found in state '{state}'.")

            county_data = county_matches.iloc[0]
            self.fips_dict = {
                'state': state,
                'state_fips': county_data['state_fips'],
                'county': county,
                'county_fips': county_data['county_fips'],
                'subdivision': None,
                'subdivision_fips': None,
                'funcstat': None
            }

            if subdivision:
                subdiv_match = county_matches[county_matches['subdivision_name'] == subdivision]
                if subdiv_match.empty:
                    raise ValueError(
                        f"Subdivision '{subdivision}' not found in county '{county}', state '{state}'."
                    )
                subdiv_data = subdiv_match.iloc[0]
                self.fips_dict['subdivision'] = subdivision
                self.fips_dict['subdivision_fips'] = subdiv_data['subdivision_fips']
                self.fips_dict['funcstat'] = subdiv_data['funcstat']

            self.logger.info(f"FIPS codes resolved: {self.fips_dict}")

        except Exception as e:
            self.logger.error(f"Error processing FIPS lookup file: {e}")
            raise ValueError(f"Failed to lookup FIPS codes: {e}")

    def _create_output_directories(self) -> None:
        """
        Determines and creates the specific output directory for the current region,
        including subdirectories for all defined datasets.
        This uses FIPS codes and subdivision information.
        """
        regional_path = self.base_output_dir

        if self.fips_dict and self.fips_dict.get('state') and self.fips_dict.get('county'):
            state_dir_name = self.fips_dict['state'].replace(' ', '_')
            county_dir_name = self.fips_dict['county'].replace(' ', '_')
            regional_path = regional_path / state_dir_name / county_dir_name

            if self.is_county_subdivision and self.fips_dict.get('subdivision'):
                subdivision_name = self.fips_dict['subdivision']
                if subdivision_name:
                    subdivision_dir_name = subdivision_name.replace(' ', '_')
                    regional_path = regional_path / subdivision_dir_name
        else:
            self.logger.warning(
                "FIPS dictionary not fully available. Regional output directory structure might be generic."
            )
            # If FIPS isn't fully resolved, regional_path remains the base_output_dir_str

        self.regional_base_output_dir = regional_path
        self.regional_base_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set regional base output directory to: {self.regional_base_output_dir}")

        # Create subdirectories for all known datasets
        for dataset_name in ALL_DATASETS:
            dataset_path = self.regional_base_output_dir / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured dataset directory exists: {dataset_path}")

    def get_fips_dict(self) -> Optional[Dict[str, str]]:
        """Return the FIPS dictionary for the current region."""
        return self.fips_dict

    def get_base_output_directory(self) -> Path:  # Renamed getter
        """Return the Path object for the current regional base output directory (e.g., .../State/County/[Subdivision]/)."""
        if not self.base_output_dir:
            self.logger.error("Regional base output directory accessed before initialization.")
            raise RuntimeError("Regional base output directory has not been initialized.")
        return self.base_output_dir

    def get_dataset_specific_output_directory(self, dataset_name: str) -> Path:
        """
        Returns the pre-created output directory for a specific dataset.

        Args:
            dataset_name (str): The name of the dataset (must be in ALL_DATASETS).

        Returns:
            Path: The Path object to the dataset-specific output directory.

        Raises:
            ValueError: If the dataset_name is not recognized or base_output_dir is not set.
        """

        if dataset_name not in ALL_DATASETS:
            self.logger.error(
                f"Dataset '{dataset_name}' is not a recognized dataset in ALL_DATASETS.")
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Must be one of {ALL_DATASETS}")

        dataset_dir = self.regional_base_output_dir / dataset_name
        # Ensure it exists, though it should have been created during initialization
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def get_path_in_output_dir(self, *path_segments: str) -> Path:
        """
        Constructs a path relative to the current regional base output directory.
        DEPRECATED: Prefer get_dataset_specific_output_directory(dataset_name).joinpath(*path_segments)

        Args:
            *path_segments: Segments of the path to append to the output directory.

        Returns:
            Path: The fully constructed Path object.

        Raises:
            RuntimeError: If the output directory has not been initialized.
        """
        warnings.warn(
            "get_path_in_output_dir is deprecated. Use get_dataset_specific_output_directory().joinpath() instead.",
            DeprecationWarning)
        if not self.regional_base_output_dir:  # Check renamed attribute
            self.logger.error("Output directory accessed before initialization.")
            raise RuntimeError("Output directory has not been initialized.")
        return self.regional_base_output_dir.joinpath(*path_segments)  # Use renamed attribute

    def get_region_config(self) -> Dict[str, Any]:
        """Return the raw region configuration dictionary."""
        return self.config_loader.get_region()

    def get_input_data_paths(self) -> Dict[str, Any]:
        """Return the configured input data paths."""
        return self.config_loader.get_input_data_paths()

    def get_overpass_config(self) -> Dict[str, Any]:
        """Get Overpass API configuration."""
        return self.config_loader.get_overpass_config()

    def is_subdivision_processing(self) -> bool:
        """Return True if processing a county subdivision, False otherwise."""
        return self.is_county_subdivision

    def set_region_boundary(self, boundary_gdf: gpd.GeoDataFrame) -> None:
        """
        Set the authoritative GeoDataFrame for the region's boundary.

        Args:
            boundary_gdf (gpd.GeoDataFrame): The GeoDataFrame representing the region's boundary.
        """
        self.region_boundary_gdf = boundary_gdf
        self.logger.info("Region boundary has been set in the orchestrator.")

    def get_region_boundary(self) -> gpd.GeoDataFrame:
        """
        Return the region boundary GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: The region boundary.

        Raises:
            ValueError: If the region boundary has not been set yet.
        """
        if self.region_boundary_gdf is None:
            self.logger.error("Attempted to access region boundary before it was set.")
            raise ValueError("Region boundary has not been set yet. Process census data first.")
        return self.region_boundary_gdf

    def _initialize_osm_parser(self) -> Optional[OSM]:
        """
        Lazily initializes and returns the pyrosm.OSM parser object.
        This method is called by get_osm_parser() when the parser is first needed.

        Returns:
            Optional[pyrosm.OSM]: The initialized OSM parser, or None on failure.
        """
        self.logger.info("Attempting to lazily initialize OSM parser...")
        input_paths = self.get_input_data_paths()
        osm_pbf_path = Path(input_paths.get("osm_pbf_file"))

        if not osm_pbf_path.exists():
            self.logger.error(
                f"OSM PBF file path ('osm_pbf_file') not found at: {osm_pbf_path}. "
                "Cannot initialize OSM parser."
            )
            return None

        boundary_gdf = self.get_region_boundary()

        try:
            # Ensure boundary is in WGS84 (EPSG:4326) as pyrosm expects
            if boundary_gdf.crs is None or boundary_gdf.crs.to_epsg() != 4326:
                self.logger.info("Re-projecting boundary to EPSG:4326 for pyrosm.")
                boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

            # Get a single geometry for the bounding box.
            # If multiple, use unary_union to get the overall extent.
            if len(boundary_gdf.geometry) > 1:
                boundary_geometry = boundary_gdf.unary_union
                self.logger.info(
                    "Multiple geometries found in boundary_gdf, "
                    "using unary_union for OSM parser bounding box."
                )
            elif len(boundary_gdf.geometry) == 1:
                boundary_geometry = boundary_gdf.geometry.iloc[0]

            # Ensure the geometry is a Polygon or MultiPolygon as expected by pyrosm
            if not isinstance(boundary_geometry, (Polygon, MultiPolygon)):
                self.logger.error(
                    f"Boundary geometry is not a Polygon or MultiPolygon (type: {type(boundary_geometry)}). "
                    "OSM parser might not work as expected."
                )
            self.logger.info(
                f"Initializing pyrosm.OSM with PBF: {osm_pbf_path} and derived bounding box."
            )
            osm_parser = OSM(str(osm_pbf_path), bounding_box=boundary_geometry)
            self.logger.info("pyrosm.OSM parser initialized successfully.")
            return osm_parser
        except FileNotFoundError:
            self.logger.error(
                f"OSM PBF file not found by pyrosm at: {osm_pbf_path}",
                exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error initializing pyrosm.OSM parser: {e}", exc_info=True)
            return None

    def get_osm_parser(self) -> Optional[OSM]:
        """
        Provides access to the pyrosm.OSM parser object.
        Initializes it on the first call if not already initialized (lazy loading).

        Returns:
            Optional[pyrosm.OSM]: The initialized OSM parser, or None if initialization fails.
        """
        if self._osm_parser is None:
            self.logger.info(
                "OSM parser not yet initialized. Attempting lazy initialization."
            )
            self._osm_parser = self._initialize_osm_parser()

        if self._osm_parser is None:
            self.logger.warning("OSM parser could not be initialized or is not available.")

        return self._osm_parser

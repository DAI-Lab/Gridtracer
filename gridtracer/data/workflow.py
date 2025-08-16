import csv
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
from shapely.geometry import MultiPolygon, Polygon

from gridtracer.config.config_loader import (
    EPSG,
    INPUT_DATA,
    LOG_FILE,
    LOG_LEVEL,
    OUTPUT_DIR,
    REGION,
)
from gridtracer.utils import create_logger

# Define all known dataset names for directory creation
ALL_DATASETS: List[str] = [
    "CENSUS",
    "NREL",
    "OSM",
    "MICROSOFT_BUILDINGS",
    "BUILDINGS_OUTPUT",
    "ROAD_NETWORK",
    "PLOTS",
]


class WorkflowOrchestrator:
    """
    Orchestrates the data import pipeline.

    This class manages configuration, regional context (FIPS codes, boundaries),
    output directory structures for all datasets, and the overall workflow execution.
    """

    def __init__(self) -> None:
        """
        Initialize the WorkflowOrchestrator.
        """
        self.logger = create_logger(name="WorkflowOrchestrator", log_level=LOG_LEVEL, log_file=LOG_FILE)
        self.base_output_dir: Path = Path(OUTPUT_DIR)

        self.fips_dict: Optional[Dict[str, str]] = None
        self.region_boundary_gdf: Optional[gpd.GeoDataFrame] = None
        self.is_county_subdivision: bool = False
        self._osm_parser: Optional[OSM] = None

        self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> None:
        """Initialize critical components of the orchestrator."""
        self._resolve_fips_codes()
        self.is_county_subdivision = self.fips_dict.get("subdivision") is not None
        self._create_output_directories()

        self.logger.info(
            f"Orchestrator initialized. Operating on County Subdivision: {
                self.is_county_subdivision}"
        )

    def _resolve_fips_codes(self) -> None:
        """
        Lookup FIPS codes for the configured region.

        The FIPS lookup file is downloaded to the root of the configured output directory.
        """
        region_config = REGION
        state = region_config.get("STATE")
        county = region_config.get("COUNTY")
        subdivision = region_config.get("COUNTY_SUBDIVISION")
        lookup_url = region_config.get("LOOKUP_URL")

        if not all([state, county, lookup_url]):
            self.logger.error(
                "Missing parameters for FIPS lookup: 'state', 'county', and 'lookup_url' are required in config."
            )
            raise ValueError("State, county, and lookup_url must be provided in config for FIPS lookup.")

        filename = os.path.basename(lookup_url)
        local_file_path = self.base_output_dir / filename
        self.logger.debug(f"Local file path: {local_file_path}")

        # Ensure the output directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        if not local_file_path.exists():
            self.logger.info(f"Downloading FIPS lookup file from {lookup_url} to {local_file_path}")
            try:
                # Download file first to temporary location, then move it
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    urllib.request.urlretrieve(lookup_url, tmp_file.name)

                    # Move the temporary file to the final location
                    Path(tmp_file.name).rename(local_file_path)

                self.logger.debug(f"FIPS lookup file saved to {local_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to download FIPS lookup file: {e}")
                raise
        else:
            self.logger.debug(f"Using existing FIPS lookup file: {local_file_path}")

        try:
            with open(local_file_path, encoding="latin-1") as infile:
                reader = csv.reader(infile)
                processed_rows = []
                for i, row in enumerate(reader):
                    if i == 0 or (row and row[0] == "STATE"):  # Skip header
                        continue
                    if len(row) == 7:
                        processed_rows.append(row)
                    elif len(row) == 8:  # Handle known inconsistency in some files
                        merged_row = row[:5] + [row[5] + " " + row[6]] + [row[7]]
                        processed_rows.append(merged_row)

            column_names = [
                "state_abbr",
                "state_fips",
                "county_fips",
                "county_name",
                "subdivision_fips",
                "subdivision_name",
                "funcstat",
            ]
            df = pd.DataFrame(processed_rows, columns=column_names)

            state_df = df[df["state_abbr"] == state]
            if state_df.empty:
                raise ValueError(f"State abbreviation '{state}' not found in lookup file.")

            county_matches = state_df[state_df["county_name"] == county]
            if county_matches.empty:
                raise ValueError(f"County '{county}' not found in state '{state}'.")

            county_data = county_matches.iloc[0]
            self.fips_dict = {
                "state": state,
                "state_fips": county_data["state_fips"],
                "county": county,
                "county_fips": county_data["county_fips"],
                "subdivision": None,
                "subdivision_fips": None,
                "funcstat": None,
            }

            if subdivision:
                subdiv_match = county_matches[county_matches["subdivision_name"] == subdivision]
                if subdiv_match.empty:
                    raise ValueError(f"Subdivision '{subdivision}' not found in county '{county}', state '{state}'.")
                subdiv_data = subdiv_match.iloc[0]
                self.fips_dict["subdivision"] = subdivision
                self.fips_dict["subdivision_fips"] = subdiv_data["subdivision_fips"]
                self.fips_dict["funcstat"] = subdiv_data["funcstat"]

            # Simple formatted logging
            if self.fips_dict["subdivision"]:
                self.logger.info(
                    f"FIPS codes resolved: {
                        self.fips_dict['state']} - {
                        self.fips_dict['county']} - {
                        self.fips_dict['subdivision']} "
                    f"({self.fips_dict['state_fips']}-{self.fips_dict['county_fips']}-{self.fips_dict['subdivision_fips']})"
                )
            else:
                self.logger.info(
                    f"FIPS codes resolved: {self.fips_dict['state']} - {self.fips_dict['county']} "
                    f"({self.fips_dict['state_fips']}-{self.fips_dict['county_fips']})"
                )

        except Exception as e:
            self.logger.error(f"Error processing FIPS lookup file: {e}")
            raise ValueError("Failed to lookup FIPS codes") from e

    def _create_output_directories(self) -> None:
        """
        Determines and creates the specific output directory for the current region,
        including subdirectories for all defined datasets.
        This uses FIPS codes and subdivision information.
        """
        regional_path = self.base_output_dir

        if self.fips_dict and self.fips_dict.get("state") and self.fips_dict.get("county"):
            state_dir_name = self.fips_dict["state"].replace(" ", "_")
            county_dir_name = self.fips_dict["county"].replace(" ", "_")
            regional_path = regional_path / state_dir_name / county_dir_name

            if self.is_county_subdivision and self.fips_dict.get("subdivision"):
                subdivision_name = self.fips_dict["subdivision"]
                if subdivision_name:
                    subdivision_dir_name = subdivision_name.replace(" ", "_")
                    regional_path = regional_path / subdivision_dir_name
        else:
            self.logger.warning(
                "FIPS dictionary not fully available. Regional output directory structure might be generic."
            )
            # If FIPS isn't fully resolved, regional_path remains the
            # base_output_dir_str

        self.regional_base_output_dir = regional_path
        self.regional_base_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"Set output directory to: {
                self.regional_base_output_dir}"
        )

        # Create subdirectories for all known datasets
        for dataset_name in ALL_DATASETS:
            dataset_path = self.regional_base_output_dir / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured dataset directory exists: {dataset_path}")

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
            self.logger.error(f"Dataset '{dataset_name}' is not a recognized dataset in ALL_DATASETS.")
            raise ValueError(f"Unknown dataset name: {dataset_name}. Must be one of {ALL_DATASETS}")

        dataset_dir = self.regional_base_output_dir / dataset_name

        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

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

    def has_region_boundary(self) -> bool:
        """
        Check if a region boundary has been set.

        Returns:
            bool: True if region boundary is available, False otherwise.
        """
        return self.region_boundary_gdf is not None

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

        If a region boundary is set, it will be used as a bounding box for spatial filtering.
        If no boundary is set, the entire PBF file will be processed.

        Returns:
            Optional[pyrosm.OSM]: The initialized OSM parser, or None on failure.
        """
        self.logger.info("Initializing OSM parser")
        input_paths = INPUT_DATA
        osm_pbf_path = Path(input_paths.get("OSM_PBF_FILE"))

        if not osm_pbf_path.exists():
            self.logger.error(
                f"OSM PBF file path ('osm_pbf_file') not found at: {osm_pbf_path}. " "Cannot initialize OSM parser."
            )
            return None

        try:
            # Check if a region boundary is available
            if self.has_region_boundary():
                boundary_gdf = self.get_region_boundary()  # Original is in EPSG:4269

                # Project to a meter-based CRS (EPSG) for accurate
                # buffering
                boundary_gdf_EPSG = boundary_gdf.to_crs(f"EPSG:{EPSG}")

                # Assume a single geometry entry as per system design
                boundary_geometry_EPSG = boundary_gdf_EPSG.geometry.iloc[0]

                # Apply a 25-meter buffer in the projected CRS (EPSG)
                buffered_geometry_EPSG = boundary_geometry_EPSG.buffer(25.0)

                # Create a temporary GeoDataFrame to hold the buffered geometry
                buffered_gdf_EPSG = gpd.GeoDataFrame([buffered_geometry_EPSG], columns=["geometry"], crs=f"EPSG:{EPSG}")
                # Reproject to WGS84 (EPSG:4326) as expected by pyrosm
                boundary_gdf_4326 = buffered_gdf_EPSG.to_crs("EPSG:4326")
                final_boundary_geometry = boundary_gdf_4326.geometry.iloc[0]

                # Ensure the geometry is a Polygon or MultiPolygon as expected
                # by pyrosm
                if not isinstance(final_boundary_geometry, (Polygon, MultiPolygon)):
                    self.logger.error(
                        f"Boundary geometry is not a Polygon or MultiPolygon "
                        f"(type: {type(final_boundary_geometry)}). "
                        "OSM parser might not work as expected."
                    )

                osm_parser = OSM(str(osm_pbf_path), bounding_box=final_boundary_geometry)
                self.logger.info("pyrosm.OSM parser initialized successfully with bounding box.")
            else:
                # No boundary set - process entire PBF file
                osm_parser = OSM(str(osm_pbf_path))
                self.logger.info("pyrosm.OSM parser initialized successfully for entire PBF file.")

            return osm_parser
        except FileNotFoundError:
            self.logger.error(f"OSM PBF file not found by pyrosm at: {osm_pbf_path}", exc_info=True)
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
            self._osm_parser = self._initialize_osm_parser()

        if self._osm_parser is None:
            self.logger.warning("OSM parser could not be initialized or is not available.")

        return self._osm_parser

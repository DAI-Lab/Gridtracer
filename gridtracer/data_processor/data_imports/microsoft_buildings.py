"""
Microsoft Buildings data handler for gridtracer.

This module provides functionality to process Microsoft building footprints data,
including QuadKey mapping, state-level filtering, and region-specific clipping.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, shape
from tqdm import tqdm

from gridtracer.data_processor.data_imports.base import DataHandler

if TYPE_CHECKING:
    from gridtracer.data_processor.workflow import WorkflowOrchestrator


class MicrosoftBuildingsDataHandler(DataHandler):
    """
    Handler for Microsoft Building Footprints data.

    This class handles downloading and processing Microsoft building footprints data,
    including QuadKey to state mapping, state-level building downloads, and
    region-specific filtering using the WorkflowOrchestrator boundary.
    """

    def __init__(self, orchestrator: 'WorkflowOrchestrator'):
        """
        Initialize the Microsoft Buildings data handler.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance.
        """
        super().__init__(orchestrator)
        self.mapping_file = self.orchestrator.base_output_dir / "us_state_quadkey_mapping.json"
        self.state_mapping: Optional[Dict] = None

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name.
        """
        return "MICROSOFT_BUILDINGS"

    def _quadkey_to_tile_xy(self, quadkey: str) -> Tuple[int, int, int]:
        """
        Convert a QuadKey string to tile X, Y coordinates and zoom level.

        Parameters:
        -----------
        quadkey : str
            QuadKey string (e.g., "0313102310")

        Returns:
        --------
        Tuple[int, int, int] : (tile_x, tile_y, zoom_level)
        """
        quadkey = str(quadkey)
        tile_x = tile_y = 0
        zoom_level = len(quadkey)

        for i in range(zoom_level):
            mask = 1 << (zoom_level - i - 1)
            if quadkey[i] == '1':
                tile_x |= mask
            elif quadkey[i] == '2':
                tile_y |= mask
            elif quadkey[i] == '3':
                tile_x |= mask
                tile_y |= mask

        return tile_x, tile_y, zoom_level

    def _tile_xy_to_lat_lon(
        self, tile_x: int, tile_y: int, zoom_level: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert tile X, Y coordinates to latitude/longitude bounding box.
        """
        import math

        def pixel_to_lat_lon(
            pixel_x: float, pixel_y: float, zoom_level: int
        ) -> Tuple[float, float]:
            map_size = 256 << zoom_level
            lon = (pixel_x / map_size) * 360.0 - 180.0
            y = 0.5 - (pixel_y / map_size)
            lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
            return lat, lon

        min_lat, min_lon = pixel_to_lat_lon(tile_x * 256, tile_y * 256, zoom_level)
        max_lat, max_lon = pixel_to_lat_lon((tile_x + 1) * 256, (tile_y + 1) * 256, zoom_level)

        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat

        return min_lat, min_lon, max_lat, max_lon

    def _quadkey_to_lat_lon(self, quadkey: str) -> Tuple[float, float, float, float]:
        """
        Convert QuadKey directly to latitude/longitude bounding box.
        """
        tile_x, tile_y, zoom_level = self._quadkey_to_tile_xy(quadkey)
        return self._tile_xy_to_lat_lon(tile_x, tile_y, zoom_level)

    def _extract_quadkey_from_url(self, url: str) -> str:
        """Extract QuadKey with leading zeros from Microsoft building URL."""
        match = re.search(r'quadkey=(\d+)', url)
        if match:
            return match.group(1)
        return None

    def _create_state_quadkey_mapping(self) -> Dict:
        """
        Create a mapping of US states to QuadKey URLs for efficient state-based building downloads.

        Returns:
        --------
        Dict : Dictionary mapping state abbreviations to QuadKey data
        """
        self.logger.info("Creating comprehensive US state to QuadKey URL mapping...")

        # Load Microsoft building footprints index
        self.logger.info("Loading Microsoft building footprints index...")
        dataset_links = pd.read_csv(
            "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
        )
        us_links = dataset_links[dataset_links.Location == 'UnitedStates'].copy()
        self.logger.info(f"Found {len(us_links)} total QuadKey tiles for United States")

        # Extract properly formatted QuadKeys with leading zeros from URLs
        us_links['proper_quadkey'] = us_links['Url'].apply(self._extract_quadkey_from_url)
        us_links = us_links[us_links['proper_quadkey'].notna()].copy()

        self.logger.info(f"Successfully extracted {len(us_links)} properly formatted QuadKeys")

        # Create QuadKey bounding box polygons
        self.logger.info("Creating QuadKey bounding box polygons...")
        quadkey_polygons = []

        for _, row in tqdm(us_links.iterrows(), total=len(us_links), desc="Processing QuadKeys"):
            quadkey = str(row['proper_quadkey'])
            url = row['Url']

            try:
                min_lat, min_lon, max_lat, max_lon = self._quadkey_to_lat_lon(quadkey)

                # Validate coordinates
                if not (-180 <= min_lon <= 180 and -90 <= min_lat <= 90 and
                        -180 <= max_lon <= 180 and -90 <= max_lat <= 90):
                    continue

                bbox_coords = [
                    (min_lon, min_lat), (max_lon, min_lat),
                    (max_lon, max_lat), (min_lon, max_lat),
                    (min_lon, min_lat)
                ]
                bbox_polygon = Polygon(bbox_coords)

                if not bbox_polygon.is_valid:
                    continue

                quadkey_polygons.append({
                    'quadkey': quadkey,
                    'url': url,
                    'size': row['Size'],
                    'geometry': bbox_polygon
                })

            except Exception as e:
                self.logger.debug(f"Error processing QuadKey {quadkey}: {e}")
                continue

        quadkey_gdf = gpd.GeoDataFrame(quadkey_polygons, crs=4326)
        self.logger.info(f"Created {len(quadkey_gdf)} valid QuadKey bounding boxes")

        # Load US state boundaries
        self.logger.info("Loading US state boundaries...")
        states_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip"
        states = gpd.read_file(states_url)
        states = states[['NAME', 'STUSPS', 'geometry']].copy()
        states = states.to_crs(4326)

        # Perform spatial intersection
        self.logger.info("Performing spatial intersection of QuadKeys with states...")
        joined = gpd.sjoin(quadkey_gdf, states, how='inner', predicate='intersects')

        self.logger.info(f"Found {len(joined)} QuadKey-state intersections")

        # Create state mapping using abbreviations as keys
        state_mapping = {}
        for state_abbr in joined['STUSPS'].unique():
            state_quadkeys = joined[joined['STUSPS'] == state_abbr]

            # Create quadkey dictionary with geometry and URL for each quadkey
            quadkeys_dict = {}
            for _, qk_row in state_quadkeys.iterrows():
                quadkey = qk_row['quadkey']
                url = qk_row['url']
                geometry = qk_row['geometry']

                quadkeys_dict[quadkey] = {
                    'url': url,
                    'geometry': geometry.wkt  # Store as WKT string for JSON serialization
                }

            state_mapping[state_abbr] = {
                'state_name': state_quadkeys['NAME'].iloc[0],
                'num_quadkeys': len(quadkeys_dict),
                'quadkeys': quadkeys_dict
            }

        # Save mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(state_mapping, f, indent=2)

        self.logger.info(f"Saved state -> QuadKey mapping to {self.mapping_file}")
        self.logger.info(f"Total states with building data: {len(state_mapping)}")

        return state_mapping

    def _load_state_mapping(self) -> Dict:
        """Load or create the state -> QuadKey mapping."""
        if self.mapping_file.exists():
            self.logger.info(f"Loading existing state mapping from {self.mapping_file}")
            with open(self.mapping_file, 'r') as f:
                return json.load(f)
        else:
            self.logger.info("State mapping not found, creating new mapping...")
            return self._create_state_quadkey_mapping()

    def _filter_quadkeys_by_region(self, state_abbr: str) -> List[str]:
        """
        Spatially filter QuadKeys for a state based on intersection with region boundary.

        Parameters:
        -----------
        state_abbr : str
            State abbreviation

        Returns:
        --------
        List[str] : List of QuadKey IDs that intersect with the region
        """
        if not self.state_mapping:
            self.state_mapping = self._load_state_mapping()

        if state_abbr not in self.state_mapping:
            raise ValueError(f"State '{state_abbr}' not found in mapping.")

        # Get region boundary from orchestrator
        region_boundary = self.orchestrator.get_region_boundary()

        # Create GeoDataFrame of QuadKey bounding boxes for the state
        quadkeys_data = []
        state_quadkeys = self.state_mapping[state_abbr]['quadkeys']

        for quadkey_id, quadkey_info in state_quadkeys.items():
            geometry_wkt = quadkey_info['geometry']
            # Convert WKT back to geometry
            from shapely import wkt
            geometry = wkt.loads(geometry_wkt)

            quadkeys_data.append({
                'quadkey': quadkey_id,
                'url': quadkey_info['url'],
                'geometry': geometry
            })

        quadkeys_gdf = gpd.GeoDataFrame(quadkeys_data, crs=4326)

        # Ensure same CRS for spatial intersection
        if quadkeys_gdf.crs != region_boundary.crs:
            quadkeys_gdf = quadkeys_gdf.to_crs(region_boundary.crs)

        # Perform spatial intersection
        intersecting_quadkeys = gpd.sjoin(
            quadkeys_gdf, region_boundary, how='inner', predicate='intersects'
        )

        filtered_quadkey_ids = intersecting_quadkeys['quadkey'].unique().tolist()

        self.logger.info(
            f"Filtered {len(state_quadkeys)} QuadKeys to {len(filtered_quadkey_ids)} "
            f"that intersect with region boundary"
        )

        return filtered_quadkey_ids

    def _download_state_buildings(
        self, state_abbr: str, max_tiles: Optional[int] = None
    ) -> List[Path]:
        """
        Download building data for QuadKeys that intersect with the region.

        Parameters:
        -----------
        state_abbr : str
            Abbreviation of the US state (e.g., 'MA', 'CA')
        max_tiles : int, optional
            Maximum number of tiles to download (for testing)

        Returns:
        --------
        List[Path] : List of GeoJSON file paths for intersecting QuadKeys
        """
        self.logger.info(f"Processing building data for {state_abbr}...")

        if not self.state_mapping:
            self.state_mapping = self._load_state_mapping()

        if state_abbr not in self.state_mapping:
            available_states = sorted(self.state_mapping.keys())
            raise ValueError(
                f"State '{state_abbr}' not found in mapping. Available: {available_states}"
            )

        # Spatially filter QuadKeys to only those intersecting with region
        filtered_quadkey_ids = self._filter_quadkeys_by_region(state_abbr)

        if max_tiles:
            filtered_quadkey_ids = filtered_quadkey_ids[:max_tiles]
            self.logger.info(f"Limited to first {len(filtered_quadkey_ids)} tiles for testing")

        if not filtered_quadkey_ids:
            self.logger.warning("No QuadKeys intersect with the region boundary")
            return []

        # Get QuadKey data for filtered IDs
        state_quadkeys = self.state_mapping[state_abbr]['quadkeys']

        # Create state-specific subdirectory
        state_dir = self.dataset_output_dir / state_abbr
        state_dir.mkdir(exist_ok=True)

        geojson_files = []
        for quadkey_id in tqdm(filtered_quadkey_ids, desc=f"Downloading {state_abbr}"):
            quadkey_info = state_quadkeys[quadkey_id]
            url = quadkey_info['url']

            try:
                # Check if file already exists
                filename = state_dir / f"{quadkey_id}.geojson"
                if filename.exists():
                    geojson_files.append(filename)
                    continue

                # Download and process
                df = pd.read_json(url, lines=True)
                df['geometry'] = df['geometry'].apply(shape)
                gdf = gpd.GeoDataFrame(df, crs=4326)

                # Flatten nested properties if they exist
                if 'properties' in gdf.columns:
                    # Extract properties and flatten them into separate columns
                    properties_expanded = pd.json_normalize(gdf['properties'])

                    # Add the flattened properties as new columns
                    for col in properties_expanded.columns:
                        gdf[col] = properties_expanded[col].values

                    # Drop the original nested properties column
                    gdf = gdf.drop('properties', axis=1)

                # Add metadata
                gdf['quadkey'] = quadkey_id
                gdf['state_abbr'] = state_abbr

                # Save file
                gdf.to_file(filename, driver="GeoJSON")
                geojson_files.append(filename)

            except Exception as e:
                self.logger.warning(f"Error downloading QuadKey {quadkey_id}: {e}")
                continue

        self.logger.info(
            f"Successfully downloaded {len(geojson_files)} intersecting tiles for {state_abbr}"
        )
        return geojson_files

    def _filter_buildings_to_region(self, building_files: List[Path]) -> gpd.GeoDataFrame:
        """
        Load and filter buildings to the specific region boundary.

        Parameters:
        -----------
        building_files : List[Path]
            List of GeoJSON files containing buildings

        Returns:
        --------
        gpd.GeoDataFrame : Filtered buildings for the region
        """
        self.logger.info(
            f"Loading and filtering {len(building_files)} building files to region..."
        )

        # Get region boundary from orchestrator
        region_boundary = self.orchestrator.get_region_boundary()

        # Load all building files
        all_buildings = []
        for file_path in tqdm(building_files, desc="Loading building files"):
            try:
                gdf = gpd.read_file(file_path)
                all_buildings.append(gdf)
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")
                continue

        if not all_buildings:
            self.logger.warning("No building files could be loaded")
            return gpd.GeoDataFrame()

        # Combine all buildings
        combined_buildings = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True))
        self.logger.info(f"Loaded {len(combined_buildings)} total buildings")

        # Ensure same CRS
        if combined_buildings.crs != region_boundary.crs:
            self.logger.info("Aligning CRS for region filtering")
            combined_buildings = combined_buildings.to_crs(region_boundary.crs)

        # Filter buildings to region
        filtered_buildings = gpd.clip(combined_buildings, region_boundary)
        self.logger.info(f"Filtered to {len(filtered_buildings)} buildings within region")

        return filtered_buildings

    def download(self) -> Dict[str, any]:
        """
        Download Microsoft building data for the region's state.

        Returns:
        --------
        Dict[str, any] : Dictionary containing download results
        """
        # Check if the output file already exists
        output_path = self.dataset_output_dir / "ms_buildings_output.geojson"
        if output_path.exists():
            self.logger.info(f"Microsoft building data already downloaded to {output_path}")
            ms_buildings = gpd.read_file(output_path)
            return {
                'ms_buildings': ms_buildings,
                'ms_buildings_filepath': output_path,
            }

        try:
            # Get state abbreviation from FIPS data
            fips = self.orchestrator.get_fips_dict()
            if not fips:
                raise ValueError("FIPS dictionary not available from orchestrator.")

            state_abbr = fips['state']

            # Download state buildings
            building_files = self._download_state_buildings(state_abbr, max_tiles=None)

            if not building_files:
                self.logger.warning("No building files were downloaded")
                return {
                    'ms_buildings': gpd.GeoDataFrame(),
                    'ms_buildings_filepath': None,
                }

            # Filter buildings to region
            filtered_buildings = self._filter_buildings_to_region(building_files)

            # Save filtered buildings
            if len(filtered_buildings) > 0:
                output_path = self.dataset_output_dir / "ms_buildings_output.geojson"
                filtered_buildings.to_file(output_path, driver="GeoJSON")

                self.logger.info(
                    f"Saved {len(filtered_buildings)} filtered buildings to {output_path}"
                )

                return {
                    'ms_buildings': filtered_buildings,
                    'ms_buildings_filepath': output_path,
                }
            else:
                self.logger.warning("No buildings found in region after filtering")
                return {
                    'ms_buildings': filtered_buildings,
                    'ms_buildings_filepath': None,
                }

        except Exception as e:
            self.logger.error(f"Error downloading Microsoft building data: {e}", exc_info=True)
            return {'error': str(e)}

    def process(self) -> Dict[str, any]:
        """
        Process Microsoft building data for the region.

        Returns:
        --------
        Dict[str, any] : Dictionary containing processed data and file paths
        """
        fips = self.orchestrator.get_fips_dict()
        if not fips:
            raise ValueError("FIPS dictionary not available from orchestrator.")

        region_name = f"{fips['county']}, {fips['state']}"
        if fips.get('subdivision'):
            region_name = f"{fips['subdivision']}, {region_name}"

        self.logger.info(f"Processing Microsoft building data for {region_name}")

        try:
            # The download() method already handles everything:
            # 1. Check if file exists (return existing data)
            # 2. Download building files
            # 3. Filter to region
            # 4. Save results
            download_results = self.download()

            self.logger.info("Microsoft Buildings processing complete")
            return download_results

        except Exception as e:
            self.logger.error(f"Error processing Microsoft building data: {e}", exc_info=True)
            return {'error': str(e)}

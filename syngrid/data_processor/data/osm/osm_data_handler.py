"""
OpenStreetMap data handler for SynGrid.

This module provides functionality to extract building, POI, and power infrastructure data
from OpenStreetMap using pyrosm via PYROSM from the WorkflowOrchestrator.
"""

import logging
import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional

import geopandas as gpd
from pyrosm import OSM
from shapely.geometry import Point

from syngrid.data_processor.data.base import DataHandler

if TYPE_CHECKING:
    from syngrid.data_processor.workflow import WorkflowOrchestrator

# Set up logging
logger = logging.getLogger(__name__)


class OSMDataHandler(DataHandler):
    """
    Handler for OpenStreetMap data.

    This class handles the extraction of buildings, POIs, and power infrastructure
    from OpenStreetMap using pyrosm via a shared parser from the WorkflowOrchestrator.
    """

    def __init__(self, orchestrator: 'WorkflowOrchestrator'):
        """
        Initialize the OSM data handler.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance.
        """
        super().__init__(orchestrator)
        self.orchestrator = orchestrator

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name ("OSM").
        """
        return "OSM"

    def set_boundary(self, boundary_gdf: Optional[gpd.GeoDataFrame]) -> bool:
        """
        Set a specific boundary for potential post-filtering of extracted data.
        The main bounding box for pyrosm is handled by the orchestrator.

        Args:
            boundary_gdf (Optional[gpd.GeoDataFrame]): Boundary to use for data extraction.

        Returns:
            bool: True if boundary was set successfully, False otherwise.
        """
        if boundary_gdf is None or boundary_gdf.empty:
            logger.debug("No specific boundary_gdf provided to OSMDataHandler.set_boundary.")
            return True

        try:
            # Store the actual boundary polygon for precise filtering
            # Ensure the boundary is in WGS84
            if boundary_gdf.crs is None:
                logger.warning(
                    "Provided boundary_gdf to OSMDataHandler has no CRS. Assuming EPSG:4326.")
                boundary_gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
            elif boundary_gdf.crs.to_string() != "EPSG:4326":
                boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

            # If multiple geometries, unify them
            if len(boundary_gdf) > 1:
                self.boundary_polygon_for_filtering = boundary_gdf.unary_union
            else:
                self.boundary_polygon_for_filtering = boundary_gdf.geometry.iloc[0]

            logger.info(
                "Specific boundary polygon for post-filtering set successfully for "
                "OSMDataHandler."
            )
            return True
        except Exception as e:
            logger.error(f"Error setting specific boundary for OSMDataHandler: {e}", exc_info=True)
            return False

    def deduplicate_power_features(self, power_gdf, distance_threshold_meters=15):
        """
        Deduplicate power infrastructure features that are within a distance threshold.
    

        Args:
            power_gdf (GeoDataFrame): GeoDataFrame containing power infrastructure features
            distance_threshold_meters (float): Distance threshold in meters

        Returns:
            GeoDataFrame: Deduplicated GeoDataFrame
        """
        if power_gdf is None or power_gdf.empty:
            return power_gdf

        logger.info(f"Deduplicating power features (threshold: {distance_threshold_meters}m)")
        initial_count = len(power_gdf)

        # Project to US National Grid (EPSG:5070) for accurate distance calculations
        power_projected = power_gdf.to_crs('EPSG:5070')

        # Define priority: substation > transformer > pole
        power_priority = {'substation': 3, 'transformer': 2, 'pole': 1}
        power_projected['priority'] = power_projected['power'].map(power_priority).fillna(0)

        # Sort by priority (highest first) to process important features first
        power_sorted = power_projected.sort_values('priority', ascending=False).reset_index()

        # Track which features to keep
        selected_indices = []
        processed_geometries = []

        for _, row in power_sorted.iterrows():
            current_geom = row.geometry
            original_idx = row['index']  # This is the original index from power_gdf

            # Check if this feature is too close to any already selected feature
            is_duplicate = False
            for selected_geom in processed_geometries:
                if current_geom.distance(selected_geom) <= distance_threshold_meters:
                    is_duplicate = True
                    break

            # If not a duplicate, keep this feature
            if not is_duplicate:
                selected_indices.append(original_idx)
                processed_geometries.append(current_geom)

        # Create deduplicated GeoDataFrame
        deduplicated_gdf = power_gdf.loc[selected_indices].copy()

        removed_count = initial_count - len(deduplicated_gdf)
        logger.info(
            f"Removed {removed_count} duplicate features, {len(deduplicated_gdf)} remaining"
        )

        return deduplicated_gdf

    def filter_by_voltage(self, power_gdf: gpd.GeoDataFrame,
                          max_voltage: float = 130_000) -> gpd.GeoDataFrame:
        """
        Filter out high-voltage transmission infrastructure by checking the 'voltage' tag.

        Args:
            power_gdf: GeoDataFrame containing power infrastructure
            max_voltage: Maximum voltage in volts to keep (default 130,000V for distribution)

        Returns:
            GeoDataFrame: Filtered GeoDataFrame
        """
        logger.info(f"Filtering power features by voltage (max: {max_voltage} Volts)")

        # If 'tags' column doesn't exist, there's nothing to filter by voltage.
        if 'tags' not in power_gdf.columns:
            logger.warning("No 'tags' column found, skipping voltage filtering.")
            return power_gdf

        def parse_voltage_simple(voltage_str):
            if not voltage_str:
                return None
            try:
                # Handle "132000;33000" format - take first value
                if ';' in str(voltage_str):
                    voltage_str = str(voltage_str).split(';')[0]
                # Handle "115000" format - just convert to int
                return int(str(voltage_str).strip())
            except (ValueError, TypeError):
                return None

        def get_voltage_from_tags(tags):
            if not isinstance(tags, dict):
                return None
            return parse_voltage_simple(tags.get('voltage'))

        # Get voltage values from the 'tags' column
        voltage_values = power_gdf['tags'].apply(get_voltage_from_tags)
        
        # Keep features with no voltage info or voltage <= threshold
        # Using Series.isna() is a robust way to check for both None and NaN
        voltage_mask = voltage_values.isna() | (voltage_values <= max_voltage)

        # Log the voltage distribution
        voltage_distribution = voltage_values.value_counts(dropna=False)
        logger.info(f"Voltage distribution: {voltage_distribution}")

        filtered_count = len(power_gdf) - voltage_mask.sum()
        logger.info(f"Removed {filtered_count} high-voltage features")

        return power_gdf[voltage_mask]

    def filter_transmission_tags(self, power_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove features tagged as transmission infrastructure.
        Simple check: skip if substation or transformer tag equals 'transmission'.

        Args:
            power_gdf: GeoDataFrame containing power infrastructure

        Returns:
            GeoDataFrame: Filtered GeoDataFrame
        """
        def is_transmission_feature(row):
            """
            Check if feature is transmission-level infrastructure.
            Simple check: substation == 'transmission' OR transformer == 'transmission'
            """
            return (row.get('substation') == 'transmission'
                    or row.get('transformer') == 'transmission')

        # Apply transmission filter
        transmission_mask = ~power_gdf.apply(is_transmission_feature, axis=1)

        filtered_count = len(power_gdf) - transmission_mask.sum()
        logger.info(f"Removed {filtered_count} transmission features")

        return power_gdf[transmission_mask]

    def remove_contained_points(self, power_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove point features that fall within polygon substations.

        Args:
            power_gdf: GeoDataFrame containing power infrastructure

        Returns:
            GeoDataFrame: Filtered GeoDataFrame
        """
        # Separate points and polygons
        points = power_gdf[power_gdf.geometry.geom_type == 'Point'].copy()
        polygons = power_gdf[power_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()

        if polygons.empty or points.empty:
            logger.info("No polygon-point conflicts to resolve")
            return power_gdf

        # Create union of all polygons
        polygon_union = polygons.geometry.unary_union

        # Find points within polygons
        contained_mask = points.geometry.within(polygon_union)
        indices_to_remove = points[contained_mask].index

        filtered_count = len(indices_to_remove)
        logger.info(f"Removed {filtered_count} points contained within polygons")

        return power_gdf.drop(indices_to_remove)

    def convert_to_centroids(self, power_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert all geometries to centroids (points only output).

        Args:
            power_gdf: GeoDataFrame containing power infrastructure

        Returns:
            GeoDataFrame: GeoDataFrame with all geometries as points
        """
        logger.info("Converting all geometries to centroids")

        # Project to US National Grid (EPSG:5070) for accurate centroid calculation
        power_projected = power_gdf.to_crs('EPSG:5070')

        # Store original geometry type and area
        power_projected['geom_type'] = power_projected.geometry.geom_type
        power_projected['area'] = power_projected.geometry.apply(
            lambda geom: (geom.area if hasattr(geom, 'area')
                          and geom.geom_type in ['Polygon', 'MultiPolygon'] else 0)
        )

        # Convert all geometries to centroids
        power_projected['geometry'] = power_projected.geometry.centroid

        # Convert back to WGS84 for final output
        power_centroids = power_projected.to_crs('EPSG:4326')

        logger.info(f"Converted {len(power_centroids)} features to centroid points")

        return power_centroids

    def extract_power_infrastructure(self, osm_parser: OSM):
        """
        Extract power infrastructure using the shared pyrosm parser from the orchestrator.
        Enhanced with voltage, area, and transmission filtering.
        """
        if osm_parser is None:
            logger.error(
                "OSM parser not available from orchestrator. "
                "Cannot extract power infrastructure."
            )
            return None, None

        power_tags = ["transformer", "substation", "pole"]
        # Extract power features
        power_features = osm_parser.get_data_by_custom_criteria(
            custom_filter={
                "power": power_tags},
            keep_nodes=True,
            keep_ways=True,
            keep_relations=True)

        if power_features is None or power_features.empty:
            logger.warning("No power infrastructure found in OpenStreetMap")
            return None, None

        # Save RAW power features to file
        raw_power_filepath = self.dataset_output_dir / "raw" / "raw_power.geojson"
        raw_power_filepath.parent.mkdir(parents=True, exist_ok=True)
        power_features.to_file(raw_power_filepath, driver="GeoJSON")

        # Initial logging with detailed breakdown
        initial_count = len(power_features)

        # Add element_type column based on geometry
        power_features['element_type'] = power_features.geometry.apply(
            lambda g: 'node' if isinstance(g, Point) else 'way'
        )

        # Step 1: Filter by voltage (keep only distribution level ≤130kV)
        power_features = self.filter_by_voltage(power_features)

        # Step 2: Remove points contained within polygons
        power_features = self.remove_contained_points(power_features)

        # Step 3: Filter distribution poles
        def has_distribution_transformer(row):
            return row.get('transformer') == 'distribution'

        transformer_substation_mask = power_features['power'].isin(['transformer', 'substation'])
        poles_mask = ((power_features['power'] == 'pole')
                      & power_features.apply(has_distribution_transformer, axis=1))

        final_mask = transformer_substation_mask | poles_mask
        power_features = power_features[final_mask]

        # Step 4: Spatial deduplication
        deduplicated_power_features = self.deduplicate_power_features(power_features)

        # Step 5: Filter by transmission tags
        final_features = self.filter_transmission_tags(deduplicated_power_features)

        # Step 6: Convert all geometries to centroids (FINAL STEP)
        centroids_power_features = self.convert_to_centroids(final_features)

        # Add properties for output format
        centroids_power_features['osm_id'] = centroids_power_features.apply(
            lambda row: f"{row['element_type']}/{row.get('id', row.name)}", axis=1
        )

        # Save final power features (centroids only)
        power_filepath = self.dataset_output_dir / "power.geojson"
        centroids_power_features.to_file(power_filepath, driver="GeoJSON")

        # Final summary
        total_removed = initial_count - len(centroids_power_features)
        reduction_percent = (total_removed / initial_count) * 100 if initial_count > 0 else 0

        final_power_types = centroids_power_features['power'].value_counts()

        logger.info("=== PIPELINE SUMMARY ===")
        logger.info("Initial features: %s", initial_count)
        logger.info("Final features: %s", len(centroids_power_features))
        logger.info("Total removed: %s (%.1f%%)", total_removed, reduction_percent)
        logger.info("Final power type distribution: %s", dict(final_power_types))
        logger.info(
            "Successfully extracted %s power features as centroids",
            len(centroids_power_features)
        )

        return centroids_power_features, power_filepath

    def extract_buildings(self, osm_parser: OSM):
        """
        Extract buildings using the shared pyrosm parser from the orchestrator.

        The pyrosm parser is already initialized with a bounding box by the orchestrator.
        This method can optionally perform further precise clipping if a specific
        boundary_polygon_for_filtering is set on this handler.

        Returns:
            tuple: (GeoDataFrame of buildings, Path to saved GeoJSON file) or (None, None)
                on failure.
        """
        if osm_parser is None:
            logger.error("OSM parser not available from orchestrator. Cannot extract buildings.")
            return None, None

        logger.info("Extracting buildings using shared pyrosm parser.")

        try:
            buildings_gdf = osm_parser.get_buildings()

            if buildings_gdf is None or buildings_gdf.empty:
                logger.warning("No buildings found by pyrosm parser for the given extent.")
                return None, None

            logger.info(
                f"Initially extracted {len(buildings_gdf)} building features using pyrosm.")

            raw_buildings_filepath = self.dataset_output_dir / "raw" / "raw_buildings.geojson"
            raw_buildings_filepath.parent.mkdir(
                parents=True, exist_ok=True)
            buildings_gdf.to_file(raw_buildings_filepath, driver="GeoJSON")
            logger.debug(f"Saved raw extracted buildings to {raw_buildings_filepath}")

            relevant_tags = set([
                "element",
                "id",
                "addr:housenumber",
                "addr:postcode",
                "addr:street",
                "building",
                "leisure",
                "name",
                "operator",
                "amenity",
                "office",
                "building:levels",
                "building:material",
                "height",
                "location",
                "roof:shape",
                "year_of_construction",
                "building:colour",
                "roof:colour",
                "roof:material",
                "name:en",
                "roof:levels",
                "operator:type",
                "description",
                "air_conditioning",
                "check_date",
                "architect:renovation",
                "building:levels:underground",
                "renovation_date",
                "parking",
                "shop",
                "man_made",
                "construction_date",
                "heritage",
                "heritage:operator",
                "ref:nrhp",
                "university",
                "landuse",
                "material",
                "historic",
                "building:architecture",
                "building:levels:roof",
                "max_level",
                "building:flats",
                "building:use",
                "source:height",
                "social_facility:for",
                "rooms",
                "level",
                "government",
                "telecom",
                "military",
                "type",
                "building:units",
                "building:part",
                "min_height",
                "plant:method",
                "plant:output:electricity",
                "plant:output:hot_water",
                "plant:source",
                "building:min_level",
                "construction:building:levels",
                "construction",
                "renovation",
                "house",
                "maxheight",
                "building:floor"
            ])

            columns_to_keep = ['geometry', 'id']  # 'id' is usually the OSM id from pyrosm
            available_cols = buildings_gdf.columns
            for col_name in relevant_tags:
                if col_name in available_cols and col_name not in columns_to_keep:
                    columns_to_keep.append(col_name)

            processed_buildings_gdf = buildings_gdf.copy()  # Or apply column filtering

            if processed_buildings_gdf.empty:
                logger.warning("No buildings remained after processing/filtering.")
                return None, None

            buildings_filepath = self.dataset_output_dir / "buildings.geojson"
            processed_buildings_gdf.to_file(buildings_filepath, driver="GeoJSON")

            return processed_buildings_gdf, buildings_filepath

        except Exception as e:
            logger.error(f"Error extracting buildings with pyrosm: {e}", exc_info=True)
            return None, None

    def extract_pois(self, osm_parser: OSM):
        """
        Extract POIs using OSMnx with direct polygon boundary filtering.

        This method uses the Overpass API to directly query POIs within
        the exact polygon boundary.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract POIs for

        Returns:
            tuple: (GeoDataFrame of POIs, Path to saved file)
        """
        if osm_parser is None:
            logger.error("OSM parser not available from orchestrator. Cannot extract POIs.")
            return None, None

        logger.info("Extracting POIs")

        try:

            # Define POI tags
            poi_tags = {
                "amenity": True,
                "shop": True,
                "tourism": True,
                "leisure": True,
                "office": True
            }

            # Extract POIs using OSMnx's current API
            pois = osm_parser.get_pois(
                custom_filter=poi_tags,
            )

            raw_pois_filepath = self.dataset_output_dir / "raw" / "raw_pois.geojson"
            # create the raw directory if it doesn't exist
            raw_pois_filepath.parent.mkdir(parents=True, exist_ok=True)
            pois.to_file(raw_pois_filepath, driver="GeoJSON")

            if pois is None or pois.empty:
                logger.warning("No POIs found in OpenStreetMap")
                return None, None

            # Properties to keep
            poi_keep_tags = set([
                "id", "name", "amenity", "shop", "tourism", "leisure", "office",
                "building", "building:use", "landuse", "man_made", "industrial",
                "craft", "public_transport", "operator:type", "government", "military",
                "description", "addr:street", "addr:housenumber", "addr:city", "name:en"
            ])

            # Filter the GeoDataFrame to keep only relevant columns
            columns_to_keep = ['geometry']  # Always keep geometry
            if 'osmid' in pois.columns:  # Keep osmid if present
                columns_to_keep.append('osmid')

            for col in pois.columns:
                if col in poi_keep_tags and col not in columns_to_keep:
                    columns_to_keep.append(col)

            pois = pois[columns_to_keep]

            logger.info(f"Successfully extracted {len(pois)} POIs with OSMnx")

            # Save POIs
            pois_filepath = self.dataset_output_dir / "pois.geojson"
            pois.to_file(pois_filepath, driver="GeoJSON")
            logger.info(f"Saved POIs to {pois_filepath}")

            return pois, pois_filepath

        except Exception as e:
            logger.error(f"Error extracting POIs: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def extract_landuse(self, osm_parser: OSM):
        """
        Extract land use polygons and classify them as residential, industrial, or public.
        All other landuse types are ignored.

        Returns:
            tuple: (GeoDataFrame of filtered landuse, Path to saved file)
        """
        if osm_parser is None:
            logger.error("OSM parser not available from orchestrator. Cannot extract landuse.")
            return None, None

        try:
            relevant_tags = set([
                "landuse",
                "name"
            ])

            landuse_gdf = osm_parser.get_landuse()

            raw_landuse_filepath = self.dataset_output_dir / "raw" / "raw_landuse.geojson"
            landuse_gdf.to_file(raw_landuse_filepath, driver="GeoJSON")

            # Filter the GeoDataFrame to keep only relevant columns
            columns_to_keep = ['geometry']  # Always keep geometry
            if 'osmid' in landuse_gdf.columns:  # Keep osmid if present
                columns_to_keep.append('osmid')

            for col in landuse_gdf.columns:
                if col in relevant_tags and col not in columns_to_keep:
                    columns_to_keep.append(col)

            landuse_gdf = landuse_gdf[columns_to_keep]

            if landuse_gdf is None or landuse_gdf.empty:
                logger.warning("No land use data found in OpenStreetMap")
                return None, None

            logger.info(f"Extracted {len(landuse_gdf)} total landuse features")

            # Define your simplified classification mapping
            landuse_categories = {
                "residential": "residential",
                "retail": "industrial",
                "commercial": "industrial",
                "industrial": "industrial",
                "garages": "industrial",
                "construction": "industrial",
                "brownfield": "industrial",
                "railway": "industrial",
                "landfill": "industrial",
                "quarry": "industrial",
                "military": "public",
                "religious": "public",
                "cemetery": "public",
                "education": "public",
                "school": "public",
                "college": "public",
                "university": "public",
                "hospital": "public",
                "institutional": "public"
            }

            # Filter only relevant values
            landuse_gdf = landuse_gdf[
                landuse_gdf["landuse"].isin(landuse_categories.keys())
            ].copy()
            landuse_gdf["category"] = landuse_gdf["landuse"].map(landuse_categories)

            logger.info(f"Filtered down to {len(landuse_gdf)} categorized landuse polygons")

            # Save file
            landuse_filepath = self.dataset_output_dir / "landuse.geojson"
            landuse_gdf.to_file(landuse_filepath, driver="GeoJSON")
            logger.info(f"Saved land use data to {landuse_filepath}")

            return landuse_gdf, landuse_filepath

        except Exception as e:
            logger.error(f"Error extracting land use data: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def download(self) -> Dict[str, Any]:
        """
        Extract data from OpenStreetMap using the shared pyrosm parser.

        Args:
            boundary_gdf (Optional[gpd.GeoDataFrame]): A specific boundary for post-filtering.
                The main OSM data query uses the boundary set in the orchestrator's OSM parser.

        Returns:
            dict: Dictionary containing extracted data.
        """
        results = {
            'buildings': None,
            'buildings_filepath': None,
            'pois': None,
            'pois_filepath': None,
            'landuse': None,
            'landuse_filepath': None,
            'power': None,
            'power_filepath': None,
        }

        # Initialize the OSM parser
        osm_parser = self.orchestrator.get_osm_parser()

        # Check if buildings, pois, and landuse already exist
        buildings_filepath = self.dataset_output_dir / "buildings.geojson"
        pois_filepath = self.dataset_output_dir / "pois.geojson"
        landuse_filepath = self.dataset_output_dir / "landuse.geojson"
        power_filepath = self.dataset_output_dir / "power.geojson"

        # Extract buildings
        if not buildings_filepath.exists():
            buildings, buildings_filepath = self.extract_buildings(osm_parser)
            if buildings is not None:
                results['buildings'] = buildings
                results['buildings_filepath'] = buildings_filepath
        else:
            logger.info(f"Using existing buildings file: {buildings_filepath}")
            results['buildings'] = gpd.read_file(buildings_filepath)
            results['buildings_filepath'] = buildings_filepath

        # Extract POIs
        if not pois_filepath.exists():
            pois, pois_filepath = self.extract_pois(osm_parser)  # This still uses osmnx
            if pois is not None:
                results['pois'] = pois
            results['pois_filepath'] = pois_filepath
        else:
            logger.info(f"Using existing POIs file: {pois_filepath}")
            results['pois'] = gpd.read_file(pois_filepath)
            results['pois_filepath'] = pois_filepath

        # Extract landuse
        if not landuse_filepath.exists():
            landuse, landuse_filepath = self.extract_landuse(osm_parser)
            if landuse is not None:
                results['landuse'] = landuse
                results['landuse_filepath'] = landuse_filepath
        else:
            logger.info(f"Using existing landuse file: {landuse_filepath}")
            results['landuse'] = gpd.read_file(landuse_filepath)
            results['landuse_filepath'] = landuse_filepath

        # Extract power infrastructure
        if not power_filepath.exists():
            power, power_filepath = self.extract_power_infrastructure(osm_parser)
            if power is not None:
                results['power'] = power
                results['power_filepath'] = power_filepath
        else:
            logger.info(f"Using existing power file: {power_filepath}")
            results['power'] = gpd.read_file(power_filepath)
            results['power_filepath'] = power_filepath

        return results

    def plot_osm_data(self, osm_data: Dict[str, Any]):
        """
        Plot the OSM data.
        """
        # plot buildings
        ax_buildings = osm_data['buildings'].plot(
            column="building", figsize=(
                12, 12), legend=True, legend_kwds=dict(
                loc='upper left', ncol=3, bbox_to_anchor=(
                    1, 1)))
        fig_buildings = ax_buildings.get_figure()
        fig_buildings.savefig(self.dataset_output_dir / "buildings.png")
        # plot pois
        ax_pois = osm_data['pois'].plot(
            column='amenity', markersize=3, figsize=(
                12, 12), legend=True, legend_kwds=dict(
                loc='upper left', ncol=5, bbox_to_anchor=(
                    1, 1)))
        fig_pois = ax_pois.get_figure()
        fig_pois.savefig(self.dataset_output_dir / "pois.png")
        # plot
        ax_landuse = osm_data['landuse'].plot(column='landuse', legend=True, figsize=(10, 6))
        fig_landuse = ax_landuse.get_figure()
        fig_landuse.savefig(self.dataset_output_dir / "landuse.png")
        # plot power
        ax_power = osm_data['power'].plot(column='power', legend=True, figsize=(10, 6))
        fig_power = ax_power.get_figure()
        fig_power.savefig(self.dataset_output_dir / "power.png")

    def process(self, plot: bool = False) -> Dict[str, Any]:
        """
        Process OSM data for the region using the shared pyrosm parser.

        Args:
            plot (bool): Whether to plot the data.

        Returns:
            dict: Dictionary containing processed data and file paths.
        """

        osm_data = self.download()
        if plot:
            self.plot_osm_data(osm_data)

        return osm_data

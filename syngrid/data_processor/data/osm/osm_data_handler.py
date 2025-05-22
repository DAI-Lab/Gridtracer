"""
OpenStreetMap data handler for SynGrid.

This module provides functionality to extract building, POI, and power infrastructure data
from OpenStreetMap using pyrosm via PYROSM from the WorkflowOrchestrator.
"""

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import geopandas as gpd
import osmnx as ox
import pyproj
from shapely.geometry import Point
from shapely.ops import transform

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
                "Specific boundary polygon for post-filtering set successfully for OSMDataHandler.")
            return True
        except Exception as e:
            logger.error(f"Error setting specific boundary for OSMDataHandler: {e}", exc_info=True)
            return False

    def deduplicate_power_features(self, power_gdf, distance_threshold_meters=10):
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

        logger.info(
            f"Deduplicating power features (threshold: {distance_threshold_meters}m)")

        # Create a projection for accurate distance calculation
        wgs84 = pyproj.CRS('EPSG:4326')
        utm = pyproj.CRS('EPSG:32619')  # UTM zone 19N (Northeast US), adjust as needed
        project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

        # Create UTM-projected geometries for distance calculation
        power_gdf['geometry_utm'] = power_gdf.geometry.apply(
            lambda geom: transform(project, geom) if geom else None
        )

        # Create a copy to avoid modifying during iteration
        deduplicated_indices = []
        to_remove = set()

        # Get feature indices by type priority (Way > Relation > Node)
        way_indices = power_gdf[power_gdf['element_type'] == 'way'].index.tolist()
        relation_indices = power_gdf[power_gdf['element_type'] == 'relation'].index.tolist()
        node_indices = power_gdf[power_gdf['element_type'] == 'node'].index.tolist()

        # Process in priority order
        for priority_indices in [way_indices, relation_indices, node_indices]:
            for idx in priority_indices:
                if idx in to_remove:
                    continue

                # Get the UTM geometry for this feature
                geom_utm = power_gdf.loc[idx, 'geometry_utm']

                if geom_utm is None:
                    continue

                # Find all features within threshold distance
                for other_idx, other_row in power_gdf.iterrows():
                    if (other_idx == idx or other_idx in to_remove
                            or other_idx in deduplicated_indices):
                        continue

                    other_geom_utm = other_row['geometry_utm']
                    if other_geom_utm is None:
                        continue

                    # Calculate distance in meters
                    distance = geom_utm.distance(other_geom_utm)

                    if distance <= distance_threshold_meters:
                        # Mark for removal based on type priority
                        idx_type = power_gdf.loc[idx, 'element_type']
                        other_type = other_row['element_type']

                        if idx_type == 'way' and other_type != 'way':
                            to_remove.add(other_idx)
                        elif idx_type == 'relation' and other_type == 'node':
                            to_remove.add(other_idx)
                        elif idx_type == other_type:
                            # If same type, keep the one with more tags/information
                            idx_tags = (len(power_gdf.loc[idx, 'tags'])
                                        if isinstance(power_gdf.loc[idx, 'tags'], dict) else 0)
                            other_tags = (len(other_row['tags'])
                                          if isinstance(other_row['tags'], dict) else 0)

                            if idx_tags >= other_tags:
                                to_remove.add(other_idx)
                            else:
                                to_remove.add(idx)
                                break  # Stop checking this feature as it's being removed
                        else:
                            to_remove.add(idx)
                            break  # Stop checking this feature as it's being removed

                # If this feature wasn't removed, add it to deduplicated list
                if idx not in to_remove:
                    deduplicated_indices.append(idx)

        # Remove temporary UTM geometry column
        power_gdf = power_gdf.drop(columns=['geometry_utm'])

        # Create new GeoDataFrame with deduplicated features
        deduplicated_gdf = power_gdf.loc[deduplicated_indices].copy()

        logger.info(
            f"Removed {len(to_remove)} duplicate features, {len(deduplicated_gdf)} remaining")
        return deduplicated_gdf

    def extract_power_infrastructure(self, boundary_gdf=None):
        """
        Extract power infrastructure features (transformers, substations, poles) from OSM
        using OSMnx to query the Overpass API directly.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract data for

        Returns:
            GeoDataFrame: GeoDataFrame containing power infrastructure features
        """
        try:
            # Set boundary if provided
            if boundary_gdf is not None and not boundary_gdf.empty:
                if not self.set_boundary(boundary_gdf):
                    return None

            # Ensure we have a boundary polygon
            if self.boundary_polygon is None:
                logger.error("No boundary polygon available for extraction")
                return None

            logger.info("Extracting power infrastructure using OSMnx with Overpass API")

            # Define power infrastructure tags
            power_tags = {
                "power": ["transformer", "substation", "pole"]
            }

            # Extract power infrastructure using OSMnx's current API
            power_features = ox.features.features_from_polygon(
                polygon=self.boundary_polygon,
                tags=power_tags
            )

            if power_features is None or power_features.empty:
                logger.warning("No power infrastructure found in OpenStreetMap")
                return None

            logger.info(f"Successfully extracted {len(power_features)} power features with OSMnx")

            # Add element_type column based on geometry
            power_features['element_type'] = power_features.geometry.apply(
                lambda g: 'node' if isinstance(g, Point) else 'way'
            )

            # Filter abandoned infrastructure and keep only poles with transformers
            filtered_features = power_features.copy()

            # Check if feature is abandoned
            def is_abandoned(tags):
                if not isinstance(tags, dict):
                    return False

                tag_dict = tags if isinstance(tags, dict) else {}
                return (tag_dict.get('abandoned') == 'yes'
                        or tag_dict.get('abandoned:substation') == 'yes'
                        or tag_dict.get('abandoned:building') == 'transformer')

            # Check if pole has a distribution transformer
            def has_distribution_transformer(tags):
                if not isinstance(tags, dict):
                    return False

                tag_dict = tags if isinstance(tags, dict) else {}
                return tag_dict.get('transformer') == 'distribution'

            # In OSMnx, tags are in a series of columns, not a dictionary
            # We'll need to reconstruct the tags dictionary
            filtered_features['tags'] = filtered_features.apply(
                lambda row: {col: row[col] for col in row.index if col not in
                             ['geometry', 'element_type', 'osmid', 'tags']},
                axis=1
            )

            # Get power type from columns
            filtered_features['power'] = filtered_features['tags'].apply(
                lambda tag_dict: tag_dict.get('power') if isinstance(tag_dict, dict) else None
            )

            # Apply filters
            non_abandoned_mask = ~filtered_features['tags'].apply(is_abandoned)
            transformer_substation_mask = filtered_features['power'].isin(
                ['transformer', 'substation'])

            # For poles, keep only those with distribution transformers
            poles_mask = ((filtered_features['power'] == 'pole')
                          & filtered_features['tags'].apply(has_distribution_transformer))

            # Combine masks
            final_mask = (transformer_substation_mask & non_abandoned_mask) | poles_mask
            filtered_features = filtered_features[final_mask]

            logger.info(f"Power features after filtering: {len(filtered_features)}")

            # Save power features
            power_filepath = self.dataset_output_dir / "power.geojson"
            filtered_features.to_file(power_filepath, driver="GeoJSON")
            logger.info(f"Saved power features to {power_filepath}")

            # Deduplicate features
            deduplicated_features = self.deduplicate_power_features(filtered_features)

            return deduplicated_features

        except Exception as e:
            logger.error(f"Error extracting power infrastructure: {e}")
            logger.error(traceback.format_exc())
            return None

    def extract_buildings(self, osm_parser) -> Tuple[Optional[gpd.GeoDataFrame], Optional[Path]]:
        """
        Extract buildings using the shared pyrosm parser from the orchestrator.

        The pyrosm parser is already initialized with a bounding box by the orchestrator.
        This method can optionally perform further precise clipping if a specific
        boundary_polygon_for_filtering is set on this handler.

        Returns:
            tuple: (GeoDataFrame of buildings, Path to saved GeoJSON file) or (None, None) on failure.
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

    def extract_pois(self, osm_parser):
        """
        Extract POIs using OSMnx with direct polygon boundary filtering.

        This method uses the Overpass API to directly query POIs within
        the exact polygon boundary.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract POIs for

        Returns:
            tuple: (GeoDataFrame of POIs, Path to saved file)
        """

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

    def extract_landuse(self, osm_parser):
        """
        Extract land use polygons and classify them as residential, industrial, or public.
        All other landuse types are ignored.

        Returns:
            tuple: (GeoDataFrame of filtered landuse, Path to saved file)
        """
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
            'power': None,
            'power_filepath': None,
            'landuse': None,
            'landuse_filepath': None
        }

        # Initialize the OSM parser
        osm_parser = self.orchestrator.get_osm_parser()

        # Check if buildings, pois, and landuse already exist
        buildings_filepath = self.dataset_output_dir / "buildings.geojson"
        pois_filepath = self.dataset_output_dir / "pois.geojson"
        landuse_filepath = self.dataset_output_dir / "landuse.geojson"

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

        try:
            power = self.extract_power_infrastructure(boundary_gdf)  # This still uses osmnx
            if power is not None and not power.empty:
                power_filepath = self.dataset_output_dir / "power.geojson"
                power.to_file(power_filepath, driver="GeoJSON")
                results['power'] = power
                results['power_filepath'] = power_filepath
        except Exception as e:
            logger.error(f"Error during (old) power infrastructure extraction: {e}")
        logger.warning(
            "Power infrastructure extraction with pyrosm is not yet implemented in this refactoring step.")

        return results

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
            ax_buildings = osm_data['buildings'].plot(
                column="building", figsize=(
                    12, 12), legend=True, legend_kwds=dict(
                    loc='upper left', ncol=3, bbox_to_anchor=(
                        1, 1)))
            fig_buildings = ax_buildings.get_figure()
            fig_buildings.savefig(self.dataset_output_dir / "buildings.png")
            ax_pois = osm_data['pois'].plot(
                column='amenity', markersize=3, figsize=(
                    12, 12), legend=True, legend_kwds=dict(
                    loc='upper left', ncol=5, bbox_to_anchor=(
                        1, 1)))
            fig_pois = ax_pois.get_figure()
            fig_pois.savefig(self.dataset_output_dir / "pois.png")
            ax_landuse = osm_data['landuse'].plot(column='landuse', legend=True, figsize=(10, 6))
            fig_landuse = ax_landuse.get_figure()
            fig_landuse.savefig(self.dataset_output_dir / "landuse.png")

        return osm_data

"""
OpenStreetMap data handler for SynGrid.

This module provides functionality to extract building, POI, and power infrastructure data
from OpenStreetMap using OSMnx to query the Overpass API directly.
"""

import logging
import time
import traceback

import osmnx as ox
import pyproj
from shapely.geometry import Point
from shapely.ops import transform

from syngrid.data_processor.data.base import DataHandler

# Set up logging
logger = logging.getLogger(__name__)


class OSMDataHandler(DataHandler):
    """
    Handler for OpenStreetMap data.

    This class handles the extraction of buildings, POIs, and power infrastructure
    from OpenStreetMap using OSMnx to query the Overpass API directly.
    """

    def __init__(self, fips_dict, osm_pbf_file=None, output_dir=None):
        """
        Initialize the OSM data handler.

        Args:
            fips_dict (dict): Dictionary containing region information
            osm_pbf_file (str or Path, optional): Path to the OSM PBF file (not used with OSMnx)
            output_dir (str or Path, optional): Base output directory
        """
        super().__init__(fips_dict, output_dir)
        self.boundary_polygon = None

        # Configure OSMnx - using current API
        ox.settings.use_cache = True

    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name
        """
        return "OSM"

    def set_boundary(self, boundary_gdf):
        """
        Set the boundary for data extraction.

        This method stores the boundary polygon for data extraction from OSM.

        Args:
            boundary_gdf (GeoDataFrame): Boundary to use for data extraction

        Returns:
            bool: True if boundary was set successfully, False otherwise
        """
        if boundary_gdf is None or boundary_gdf.empty:
            logger.warning("No boundary provided")
            self.boundary_polygon = None
            return False

        try:
            # Store the actual boundary polygon for precise filtering
            # Ensure the boundary is in WGS84
            if boundary_gdf.crs != "EPSG:4326":
                boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

            self.boundary_polygon = boundary_gdf.geometry.iloc[0]

            # For safety with the Overpass API, simplify the polygon if it's very complex
            if len(self.boundary_polygon.exterior.coords) > 1000:
                logger.info(
                    f"Complex polygon with {len(self.boundary_polygon.exterior.coords)} points, "
                    "simplifying..."
                )
                self.boundary_polygon = self.boundary_polygon.simplify(
                    0.0001)  # ~10m simplification

            logger.info("Boundary polygon set successfully for OSM extraction")
            return True
        except Exception as e:
            logger.error(f"Error setting boundary: {e}")
            logger.error(traceback.format_exc())
            self.boundary_polygon = None
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

    def extract_buildings(self, boundary_gdf=None):
        """
        Extract buildings using OSMnx with direct polygon boundary filtering.

        This method uses the Overpass API to directly query buildings within
        the exact polygon boundary.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract buildings for

        Returns:
            tuple: (GeoDataFrame of buildings, Path to saved file)
        """
        try:
            start_time = time.time()

            # Set boundary if provided
            if boundary_gdf is not None and not boundary_gdf.empty:
                if not self.set_boundary(boundary_gdf):
                    return None, None

            # Ensure we have a boundary polygon
            if self.boundary_polygon is None:
                logger.error("No boundary polygon available for extraction")
                return None, None

            logger.info("Extracting buildings using OSMnx with Overpass API")

            relevant_tags = set([
                "element",
                "id",
                "access",
                "addr:city",
                "addr:housenumber",
                "addr:postcode",
                "addr:street",
                "building",
                "leisure",
                "name",
                "operator",
                "note",
                "amenity",
                "capacity",
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
                "massgis:school_id",
                "type",
                "isced:level",
                "social_centre:for",
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

            # Extract only buildings with the tags you care about
            buildings = ox.features.features_from_polygon(
                polygon=self.boundary_polygon,
                tags={"building": True},
            )
            raw_buildings_filepath = self.dataset_output_dir / "raw" / "raw_buildings.geojson"
            buildings.to_file(raw_buildings_filepath, driver="GeoJSON")

            # Filter the GeoDataFrame to keep only relevant columns
            columns_to_keep = ['geometry']  # Always keep geometry
            if 'osmid' in buildings.columns:  # Keep osmid if present
                columns_to_keep.append('osmid')

            for col in buildings.columns:
                if col in relevant_tags and col not in columns_to_keep:
                    columns_to_keep.append(col)

            buildings = buildings[columns_to_keep]

            logger.info(f"Extracted {len(buildings)} buildings within the boundary.")
            buildings_filepath = self.dataset_output_dir / "buildings.geojson"
            buildings.to_file(buildings_filepath, driver="GeoJSON")

            extraction_time = time.time() - start_time
            logger.info(
                f"Successfully extracted {len(buildings)} buildings in {extraction_time:.2f}s")

            # Save buildings
            save_start = time.time()
            buildings_filepath = self.dataset_output_dir / "buildings.geojson"
            buildings.to_file(buildings_filepath, driver="GeoJSON")
            save_time = time.time() - save_start
            logger.info(f"Saved buildings to {buildings_filepath} in {save_time:.2f}s")

            total_time = time.time() - start_time
            logger.info(f"Total building extraction completed in {total_time:.2f}s")

            return buildings, buildings_filepath

        except Exception as e:
            logger.error(f"Error extracting buildings: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def extract_pois(self, boundary_gdf=None):
        """
        Extract POIs using OSMnx with direct polygon boundary filtering.

        This method uses the Overpass API to directly query POIs within
        the exact polygon boundary.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract POIs for

        Returns:
            tuple: (GeoDataFrame of POIs, Path to saved file)
        """
        try:
            # Set boundary if provided
            if boundary_gdf is not None and not boundary_gdf.empty:
                if not self.set_boundary(boundary_gdf):
                    return None, None

            # Ensure we have a boundary polygon
            if self.boundary_polygon is None:
                logger.error("No boundary polygon available for extraction")
                return None, None

            logger.info("Extracting POIs using OSMnx with Overpass API")

            # Define POI tags
            poi_tags = {
                "amenity": True,
                "shop": True,
                "tourism": True,
                "leisure": True,
                "office": True
            }
            # Properties to keep
            poi_keep_tags = set([
                "id", "name", "amenity", "shop", "tourism", "leisure", "office",
                "building", "building:use", "landuse", "man_made", "industrial",
                "craft", "public_transport", "operator:type", "government", "military",
                "description", "addr:street", "addr:housenumber", "addr:city", "name:en"
            ])

            # Extract POIs using OSMnx's current API
            pois = ox.features.features_from_polygon(
                polygon=self.boundary_polygon,
                tags=poi_tags,
            )
            raw_pois_filepath = self.dataset_output_dir / "raw" / "raw_pois.geojson"
            # create the raw directory if it doesn't exist
            raw_pois_filepath.parent.mkdir(parents=True, exist_ok=True)
            pois.to_file(raw_pois_filepath, driver="GeoJSON")

            if pois is None or pois.empty:
                logger.warning("No POIs found in OpenStreetMap")
                return None, None

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

    def extract_landuse(self, boundary_gdf=None):
        """
        Extract land use polygons and classify them as residential, industrial, or public.
        All other landuse types are ignored.

        Returns:
            tuple: (GeoDataFrame of filtered landuse, Path to saved file)
        """
        try:
            # Set boundary if provided
            if boundary_gdf is not None and not boundary_gdf.empty:
                if not self.set_boundary(boundary_gdf):
                    return None, None

            if self.boundary_polygon is None:
                logger.error("No boundary polygon available for extraction")
                return None, None

            logger.info("Extracting land use data using OSMnx with Overpass API")

            relevant_tags = set([
                "landuse",
                "name"
            ])

            # Only extract landuse
            landuse_gdf = ox.features.features_from_polygon(
                polygon=self.boundary_polygon,
                tags={"landuse": True},
            )
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

    def download(self, boundary_gdf=None):
        """
        Extract data from OpenStreetMap using OSMnx.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract data for

        Returns:
            dict: Dictionary containing extracted data
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

        # Extract buildings
        buildings, buildings_filepath = self.extract_buildings(boundary_gdf)
        if buildings is not None:
            results['buildings'] = buildings
            results['buildings_filepath'] = buildings_filepath

        # Extract POIs
        pois, pois_filepath = self.extract_pois(boundary_gdf)
        if pois is not None:
            results['pois'] = pois
            results['pois_filepath'] = pois_filepath

        # Extract power infrastructure
        try:
            power = self.extract_power_infrastructure(boundary_gdf)

            if power is not None and not power.empty:
                logger.info(f"Extracted {len(power)} power infrastructure features")
                power_filepath = self.dataset_output_dir / "power.geojson"
                power.to_file(power_filepath, driver="GeoJSON")
                logger.info(f"Saved power infrastructure to {power_filepath}")

                results['power'] = power
                results['power_filepath'] = power_filepath
            else:
                logger.warning("No power infrastructure found in OpenStreetMap")
        except Exception as e:
            logger.error(f"Error extracting power infrastructure: {e}")
            logger.error(traceback.format_exc())

        # Extract land use data
        try:
            landuse, landuse_filepath = self.extract_landuse(boundary_gdf)
            if landuse is not None:
                results['landuse'] = landuse
                results['landuse_filepath'] = landuse_filepath
        except Exception as e:
            logger.error(f"Error extracting land use data: {e}")
            logger.error(traceback.format_exc())

        return results

    def process(self, boundary_gdf=None):
        """
        Process OSM data for the region using OSMnx.

        Extract buildings, POIs, and power infrastructure from OpenStreetMap
        using the exact boundary polygon.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to extract data for.
                Must be provided for OSMnx extraction.

        Returns:
            dict: Dictionary containing processed data and file paths.
        """
        logger.info(
            f"Processing OSM data for {self.fips_dict['state']} - {self.fips_dict['county']}"
        )

        # Extract all data with boundary filtering
        osm_data = self.download(boundary_gdf)

        return osm_data

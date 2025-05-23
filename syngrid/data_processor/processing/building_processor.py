from pathlib import Path
from typing import Dict, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from syngrid.data_processor.utils import logger


class BuildingHeuristicsProcessor:
    """
    Processes building footprints to create detailed building attributes
    for energy demand modeling and grid infrastructure planning.

    This class implements the building classification pipeline described in
    the building_heuristics.md document. It applies a series of heuristics
    to determine building characteristics when direct data is unavailable.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the building processor.

        Parameters:
        -----------
        output_dir : str
            Path to output directory where processed data will be saved
        """
        self.output_dir = Path(output_dir)

    def process(self, census_data: Dict, osm_data: Dict,
                microsoft_buildings_data: Dict) -> Dict[str, str]:
        """
        Main method that orch§estrates the entire building classification process.

        Parameters:
        -----------
        region_data : dict
            Contains boundary and administrative information
        osm_data : dict
            Contains buildings, POIs, landuse polygons from OpenStreetMap
        census_data : dict
            Contains census blocks with demographic information
        nrel_data : dict
            Contains building typology reference data

        Returns:
        --------
        dict : Paths to output files
        """
        logger.info("Starting building classification process")

        # Step 0: Only process buildings with a floor area greater than 45 sq meters
        logger.info("Step 0: Filtering out small buildings")
        osm_data['buildings'] = self.filter_small_buildings(osm_data['buildings'])

        osm_data['buildings'].to_file(self.output_dir / "buildings_with_floor_area.geojson")

        logger.info(osm_data['buildings'].columns)

        # Step 1: Calculate free walls for all buildings
        # Step 1: Classify building use (residential, commercial, industrial, etc.)
        logger.info("Step 1: Classifying building use")
        all_buildings = self.classify_building_use(
            osm_data.get('buildings'),
            osm_data.get('pois'),
            osm_data.get('landuse'),
        )

        # Step 2: Split buildings by use
        logger.info(f"Found {len(all_buildings)} buildings to process")
        residential = all_buildings[all_buildings['building_use'] == 'Residential'].copy()
        other = all_buildings[all_buildings['building_use'] != 'Residential'].copy()

        # Step 3: Calculate free walls for all buildings
        logger.info("Step 3: Calculating free walls")
        if len(other) > 0:
            other = self.calculate_free_walls(other)

        # Step 4: Process residential buildings
        if len(residential) > 0:
            logger.info(f"Processing {len(residential)} residential buildings")

            # Calculate free walls
            residential = self.calculate_free_walls(residential)

            residential.to_file(self.output_dir / "residential_with_free_walls.geojson")

            # Calculate floors
            residential = self.calculate_floors(
                residential, microsoft_buildings_data.get('ms_buildings'))

            # Building type classification
            residential = self.classify_building_type(
                residential,
                census_data.get('target_region_blocks')
            )

            # Allot occupants based on census
            residential = self.allot_occupants(
                residential,
                census_data.get('target_region_blocks')
            )

        #     # Allot construction year
        #     residential = self.allot_construction_year(
        #         residential,
        #         census_data.get('housing_age')
        #     )

        #     # Optional: Allot refurbishment level
        #     residential = self.allot_refurbishment_level(residential)

        #     # Write residential output
        #     residential_output_path = self.write_buildings_output(
        #         residential,
        #         self.output_dir,
        #         'buildings_residential.shp'
        #     )
        #     logger.info(f"Residential buildings saved to: {residential_output_path}")
        # else:
        #     # Create empty stub file for residential
        #     logger.warning("No residential buildings found")
        #     residential_output_path = self.write_empty_data_stub(
        #         self.output_dir,
        #         'buildings_residential.txt',
        #         "No residential buildings found"
        #     )

        # # Step 5: Process non-residential buildings
        # other_output_path = None
        # if len(other) > 0:
        #     logger.info(f"Processing {len(other)} non-residential buildings")
        #     # Write non-residential output
        #     other_output_path = self.write_buildings_output(
        #         other,
        #         self.output_dir,
        #         'buildings_other.shp'
        #     )
        #     logger.info(f"Non-residential buildings saved to: {other_output_path}")
        # else:
        #     # Create empty stub file for non-residential
        #     logger.warning("No non-residential buildings found")
        #     other_output_path = self.write_empty_data_stub(
        #         self.output_dir,
        #         'buildings_other.txt',
        #         "No non-residential buildings found"
        #     )

        # # Step 6: Generate final classified buildings shapefile by merging
        # classified_output_path = None
        # if residential_output_path and residential_output_path.endswith('.shp') and \
        #    other_output_path and other_output_path.endswith('.shp'):
        #     logger.info("Merging residential and non-residential buildings")
        #     classified_output_path = self.merge_building_layers(
        #         residential_output_path,
        #         other_output_path,
        #         self.output_dir,
        #         'buildings_classified.shp'
        #     )
        # elif residential_output_path and residential_output_path.endswith('.shp'):
        #     classified_output_path = residential_output_path
        # elif other_output_path and other_output_path.endswith('.shp'):
        #     classified_output_path = other_output_path
        # else:
        #     classified_output_path = self.write_empty_data_stub(
        #         self.output_dir,
        #         'buildings_classified.txt',
        #         "No buildings found"
        #     )

        # logger.info(f"Building classification complete. Final output: {classified_output_path}")
        # return {
        #     'residential': residential_output_path,
        #     'other': other_output_path,
        #     'classified': classified_output_path
        # }
        return all_buildings

    def filter_small_buildings(self, buildings: gpd.GeoDataFrame,
                               min_area: int = 45) -> gpd.GeoDataFrame:
        """
        Filters out buildings with a floor area less than 45 sq meters.
        """
        if buildings is None or len(buildings) == 0:
            logger.warning("No buildings provided to filter")
            return gpd.GeoDataFrame()

        # 1. Filter out small buildings and non-buildings
        # Minimum building area threshold (e.g., 45 sq meters)
        logger.info(f"Found {len(buildings)} buildings before small building filter")

        # Calculate accurate areas in square meters
        buildings = self._calculate_floor_area(buildings)

        buildings = buildings[
            buildings['floor_area'] >= min_area
        ]
        logger.info(
            f"{len(buildings)} buildings remain after filtering out buildings "
            f"with floor area less than {min_area} sq meters"
        )

        return buildings

    def classify_building_use(self, buildings: gpd.GeoDataFrame,
                              pois: gpd.GeoDataFrame,
                              landuse: gpd.GeoDataFrame,
                              ) -> gpd.GeoDataFrame:
        """
        Classifies buildings by their use based on OSM tags, POIs, land use data,

        Heuristic priority order:
        1.  Direct OSM tags on the building (building, building:use, amenity, shop, office, etc.)
        2.  POIs inside or very near the building (requires spatial processing).
        3.  Land use zones the building falls within (requires spatial processing).
        4.  Building name keywords.
        5.  Final default classification.

        Parameters:
        -----------
        buildings : GeoDataFrame
            OSM building polygons.
        pois : GeoDataFrame, optional
            Points of Interest.
        landuse : GeoDataFrame, optional
            Land use polygons.

        Returns:
        --------
        GeoDataFrame : Buildings with 'building_use' column added (Residential,
                       Commercial, Industrial, Public).
        """
        if buildings is None or buildings.empty:
            logger.warning("No buildings provided to classify_building_use.")
            return gpd.GeoDataFrame()

        classified_buildings = buildings.copy()
        logger.info(
            f"Starting building use classification for {len(classified_buildings)} buildings.")

        # Initialize 'building_use' column
        classified_buildings['building_use'] = pd.NA

        # --- Step 0: Exclusions ---
        # Filter out common non-building structures or utility infrastructure
        if 'building' in classified_buildings.columns:
            exclude_building_values = [
                'garage',
                'shed',
                'garages',
                'carport',
                'roof',
                'gazebo',
                'service']
            mask = classified_buildings['building'].isin(exclude_building_values)
            classified_buildings.loc[mask, 'building_use'] = 'Excluded - Non Habitable Structure'
            logger.info(
                f"{mask.sum()} buildings marked as 'Excluded - Non Habitable Structure' based on 'building' tag.")

        if 'power' in classified_buildings.columns:
            exclude_power_values = [
                'transformer',
                'substation',
                'pole',
                'tower',
                'portal',
                'catenary_mast']
            mask = classified_buildings['power'].isin(
                exclude_power_values) & classified_buildings['building_use'].isna()
            classified_buildings.loc[mask, 'building_use'] = 'Excluded - Power Infrastructure'
            logger.info(f"{mask.sum()} buildings marked as 'Excluded - Power Infrastructure'.")

        # Consider only buildings not yet excluded for further classification
        candidate_buildings = classified_buildings[classified_buildings['building_use'].isna()].copy(
        )
        if candidate_buildings.empty:
            logger.info("No candidate buildings remaining after initial exclusions.")
            return classified_buildings

        # --- Step 1: Direct OSM Tags on Buildings ---
        # Order within this step reflects assumed specificity/reliability of tags.

        # 1.1 'building' tag (primary OSM building type)
        if 'building' in candidate_buildings.columns:
            # Residential from 'building'
            res_types = ['residential', 'house', 'detached', 'apartments', 'terrace',
                         'dormitory', 'semidetached_house', 'bungalow', 'static_caravan',
                         'hut', 'cabin']  # hut/cabin often residential
            mask = candidate_buildings['building'].isin(res_types)
            candidate_buildings.loc[mask, 'building_use'] = 'Residential'
            logger.info(f"{mask.sum()} buildings classified as Residential from 'building' tag.")

            # Commercial from 'building'
            com_types = ['commercial', 'retail', 'office', 'supermarket', 'kiosk', 'shop',
                         'hotel', 'motel', 'hostel']
            mask = candidate_buildings['building'].isin(
                com_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Commercial'
            logger.info(f"{mask.sum()} buildings classified as Commercial from 'building' tag.")

            # Industrial from 'building'
            ind_types = [
                'industrial',
                'warehouse',
                'factory',
                'manufacture',
                'workshop']  # workshop added
            mask = candidate_buildings['building'].isin(
                ind_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Industrial'
            logger.info(f"{mask.sum()} buildings classified as Industrial from 'building' tag.")

            # Public from 'building'
            pub_types = ['school', 'hospital', 'government', 'university', 'public',
                         'church', 'mosque', 'synagogue', 'temple', 'chapel', 'cathedral',
                         'civic', 'kindergarten', 'college', 'train_station', 'transportation',
                         'public_transport']
            mask = candidate_buildings['building'].isin(
                pub_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Public'
            logger.info(f"{mask.sum()} buildings classified as Public from 'building' tag.")

        # 1.2 'building:use' tag (often more specific than 'building')
        if 'building:use' in candidate_buildings.columns:
            use_map = {
                'residential': 'Residential', 'residental': 'Residential',  # common typo
                'commercial': 'Commercial', 'retail': 'Commercial', 'office': 'Commercial',
                'industrial': 'Industrial', 'warehouse': 'Industrial',
                'public': 'Public', 'civic': 'Public', 'governmental': 'Public',
                'education': 'Public', 'school': 'Public', 'university': 'Public',
                'college': 'Public', 'kindergarten': 'Public',
                'hospital': 'Public', 'clinic': 'Public', 'healthcare': 'Public',
                'place_of_worship': 'Public', 'religious': 'Public',
                'transportation': 'Public',
            }
            for osm_val, syn_val in use_map.items():
                mask = candidate_buildings['building:use'].fillna('').str.lower() == osm_val
                # Overwrite if more specific, or fill if NA. Let's overwrite for building:use
                candidate_buildings.loc[mask, 'building_use'] = syn_val
            logger.info(f"Applied 'building:use' tag classifications. Check specific counts if needed.")

        # 1.3 'amenity' tag
        if 'amenity' in candidate_buildings.columns:
            pub_amenities = [
                'place_of_worship', 'school', 'university', 'college', 'kindergarten', 'library',
                'fire_station', 'police', 'townhall', 'courthouse', 'community_centre',
                'hospital', 'clinic', 'doctors', 'dentist', 'social_facility', 'public_building',
                'post_office', 'government_office', 'arts_centre', 'museum', 'gallery'
            ]
            com_amenities = [
                'restaurant', 'cafe', 'fast_food', 'pub', 'bar', 'nightclub', 'food_court',
                'bank', 'atm', 'pharmacy', 'marketplace', 'fuel', 'car_wash', 'car_rental',
                'theatre', 'cinema', 'studio', 'veterinary', 'money_transfer', 'bureau_de_change',
                'marketplace', 'casino', 'conference_centre', 'events_venue', 'coworking_space'
            ]
            res_amenities = ['shelter']  # e.g., homeless shelter

            mask = candidate_buildings['amenity'].fillna('').str.lower().isin(
                pub_amenities) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Public'
            logger.info(f"{mask.sum()} buildings classified as Public from 'amenity' tag.")

            mask = candidate_buildings['amenity'].fillna('').str.lower().isin(
                com_amenities) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Commercial'
            logger.info(f"{mask.sum()} buildings classified as Commercial from 'amenity' tag.")

            mask = candidate_buildings['amenity'].fillna('').str.lower().isin(
                res_amenities) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Residential'
            logger.info(
                f"{mask.sum()} buildings classified as Residential from 'amenity' (shelter).")

        # 1.4 'shop' tag
        if 'shop' in candidate_buildings.columns:
            mask = candidate_buildings['shop'].notna() & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Commercial'
            logger.info(f"{mask.sum()} buildings classified as Commercial from 'shop' tag.")

        # 1.5 'office' tag
        if 'office' in candidate_buildings.columns:
            # Specific public offices
            public_office_types = [
                'government',
                'administrative',
                'diplomatic',
                'association',
                'ngo']
            mask = candidate_buildings['office'].fillna('').str.lower().isin(
                public_office_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Public'
            logger.info(f"{mask.sum()} buildings classified as Public from specific 'office' types.")

            # General commercial offices
            mask = candidate_buildings['office'].notna() & \
                ~candidate_buildings['office'].fillna('').str.lower().isin(public_office_types) & \
                candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Commercial'
            logger.info(
                f"{mask.sum()} buildings classified as Commercial from general 'office' tag.")

        # 1.6 'building:flats' tag
        if 'building:flats' in candidate_buildings.columns:
            mask = candidate_buildings['building:flats'].notna(
            ) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'Residential'
            logger.info(
                f"{mask.sum()} buildings classified as Residential from 'building:flats' tag.")

        # 1.7 'craft' tag
        if 'craft' in candidate_buildings.columns:
            mask = candidate_buildings['craft'].notna(
            ) & candidate_buildings['building_use'].isna()
            # Or 'Industrial' for some
            candidate_buildings.loc[mask, 'building_use'] = 'Commercial'
            logger.info(f"{mask.sum()} buildings classified as Commercial from 'craft' tag.")

        # Update main dataframe with classifications from candidate_buildings
        classified_buildings.update(candidate_buildings[['building_use']])

        logger.info(
            f"After initial building tag classification: \n{classified_buildings['building_use'].value_counts(dropna=False)}")

        # --- Step 2: Classification by POIs (Spatial Join) ---
        logger.info("Starting POI-based classification for remaining unclassified buildings.")

        # Ensure POIs GeoDataFrame is provided and not empty
        if pois is None or pois.empty:
            logger.warning(
                "POIs GeoDataFrame is missing or empty. Skipping POI-based classification.")
        else:
            # Ensure CRSs match, or warn if they don't. Ideally, reproject beforehand.
            if classified_buildings.crs != pois.crs:
                logger.warning(
                    f"CRS mismatch between buildings ({classified_buildings.crs}) and POIs ({pois.crs}). "
                    f"Spatial join results may be incorrect. Reproject to a common CRS."
                )
                # Example reprojection (consider which CRS is appropriate for your area):
                pois = pois.to_crs(classified_buildings.crs)

            # Get buildings that are still unclassified
            buildings_to_classify_via_poi = classified_buildings[classified_buildings['building_use'].isna(
            )].copy()
            logger.info(
                f"Found {len(buildings_to_classify_via_poi)} buildings to attempt POI classification on.")

            if not buildings_to_classify_via_poi.empty:
                buildings_with_pois = gpd.sjoin(
                    buildings_to_classify_via_poi, pois, how='left', predicate='intersects', lsuffix='', rsuffix='poi'
                )

                poi_index_col_name = 'index_poi'
                if pois.index.name is not None:
                    poi_index_col_name = f"{pois.index.name}_poi"

                if poi_index_col_name not in buildings_with_pois.columns:
                    logger.warning(
                        f"POI index column '{poi_index_col_name}' not found after sjoin. Will try fallback 'index_poi'. Available columns: {buildings_with_pois.columns.tolist()}")
                    if 'index_poi' not in buildings_with_pois.columns:
                        logger.error(
                            f"Critical: Fallback POI index column 'index_poi' also not found. Cannot filter POI matches.")
                        # Skip POI classification if index column can't be identified
                        buildings_with_pois_matches = pd.DataFrame()  # Empty dataframe
                    else:
                        poi_index_col_name = 'index_poi'  # Confirmed fallback

                # Filter out rows where no POI was joined
                if not buildings_with_pois.empty and poi_index_col_name in buildings_with_pois.columns:
                    buildings_with_pois_matches = buildings_with_pois[buildings_with_pois[poi_index_col_name].notna(
                    )].copy()
                    logger.info(
                        f"{len(buildings_with_pois_matches)} building-POI intersections found.")
                else:
                    buildings_with_pois_matches = gpd.GeoDataFrame()  # Ensure it's an empty GeoDataFrame
                    logger.info("No building-POI intersections found or POI index column missing.")

                # Define POI tags for classification
                # Amenity-based (highest priority from POIs)
                poi_public_amenities = [
                    'place_of_worship', 'school', 'university', 'college', 'kindergarten', 'library',
                    'fire_station', 'police', 'townhall', 'courthouse', 'community_centre',
                    'hospital', 'clinic', 'doctors', 'dentist', 'social_facility', 'public_building',
                    'post_office', 'government_office', 'arts_centre', 'museum', 'gallery', 'embassy'
                ]
                poi_commercial_amenities = [
                    'restaurant', 'cafe', 'fast_food', 'pub', 'bar', 'nightclub', 'food_court',
                    'bank', 'atm', 'pharmacy', 'marketplace', 'fuel', 'car_wash', 'car_rental',
                    'theatre', 'cinema', 'studio', 'veterinary', 'money_transfer', 'bureau_de_change',
                    'casino', 'conference_centre', 'events_venue', 'coworking_space', 'hotel'  # hotel is often an amenity
                ]
                poi_residential_amenities = ['shelter']

                # Iterate over unique building indices that had a POI match
                # This is important because a building might match multiple POIs.
                # We process rules in order for each *original* building.
                unique_building_indices_with_poi_match = buildings_with_pois_matches.index.unique()

                for building_idx in unique_building_indices_with_poi_match:
                    # Only proceed if the building in the *original* dataframe is still
                    # unclassified
                    if pd.isna(classified_buildings.loc[building_idx, 'building_use']):
                        # Get all POIs that matched this specific building
                        pois_for_this_building = buildings_with_pois_matches.loc[[building_idx]]

                        classified_this_building = False

                        # Rule 1: POI Amenity for Public
                        if 'amenity_poi' in pois_for_this_building.columns:
                            if pois_for_this_building['amenity_poi'].isin(
                                    poi_public_amenities).any():
                                classified_buildings.loc[building_idx, 'building_use'] = 'Public'
                                classified_this_building = True

                        # Rule 2: POI Amenity for Commercial (if not already Public)
                        if not classified_this_building and 'amenity_poi' in pois_for_this_building.columns:
                            if pois_for_this_building['amenity_poi'].isin(
                                    poi_commercial_amenities).any():
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'Commercial'
                                classified_this_building = True

                        # Rule 3: POI Amenity for Residential (if not already classified)
                        if not classified_this_building and 'amenity_poi' in pois_for_this_building.columns:
                            if pois_for_this_building['amenity_poi'].isin(
                                    poi_residential_amenities).any():
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'Residential'
                                classified_this_building = True

                        # Rule 4: POI Shop tag (if not already classified)
                        if not classified_this_building and 'shop_poi' in pois_for_this_building.columns:
                            if pois_for_this_building['shop_poi'].notna().any():
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'Commercial'
                                classified_this_building = True

                        # Rule 5: POI Office tag (if not already classified)
                        if not classified_this_building and 'office_poi' in pois_for_this_building.columns:
                            poi_public_office_types = [
                                'government', 'administrative', 'diplomatic', 'association', 'ngo']
                            if pois_for_this_building['office_poi'].isin(
                                    poi_public_office_types).any():
                                classified_buildings.loc[building_idx, 'building_use'] = 'Public'
                                classified_this_building = True
                            elif pois_for_this_building['office_poi'].notna().any():  # Any other office
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'Commercial'
                                classified_this_building = True

                        # Add more rules for other POI tags like craft_poi, tourism_poi,
                        # leisure_poi, government_poi if needed

                logger.info(
                    f"After POI-based classification: \n{classified_buildings['building_use'].value_counts(dropna=False)}")

        # --- Step 3: Classification by Land Use Overlay (Spatial Join) ---
        logger.info("Starting Landuse-based classification for remaining unclassified buildings.")
        if landuse is None or landuse.empty:
            logger.warning(
                "Landuse GeoDataFrame is missing or empty. Skipping Landuse-based classification.")
        else:
            if classified_buildings.crs != landuse.crs:
                logger.warning(
                    f"CRS mismatch between buildings ({classified_buildings.crs}) and landuse ({landuse.crs}). "
                    f"Reprojecting landuse to match buildings. Ensure this is the correct approach."
                )
                landuse = landuse.to_crs(classified_buildings.crs)

            buildings_to_classify_via_landuse = classified_buildings[classified_buildings['building_use'].isna(
            )].copy()
            logger.info(
                f"Found {len(buildings_to_classify_via_landuse)} buildings to attempt Landuse classification on.")

            if not buildings_to_classify_via_landuse.empty:
                buildings_with_landuse = gpd.sjoin(
                    buildings_to_classify_via_landuse, landuse, how='left', predicate='within', lsuffix='', rsuffix='landuse'
                )
                buildings_with_landuse.to_file('buildings_with_landuse.geojson', driver='GeoJSON')

                # Dynamically determine the index column from the right GeoDataFrame (landuse)
                landuse_join_index_col_name = 'index_landuse'  # Default if landuse.index was unnamed
                if landuse.index.name is not None:
                    landuse_join_index_col_name = f"{landuse.index.name}_landuse"

                if landuse_join_index_col_name not in buildings_with_landuse.columns:
                    logger.warning(
                        f"Landuse index column '{landuse_join_index_col_name}' not found after sjoin. Will try fallback 'index_landuse'. Available columns: {buildings_with_landuse.columns.tolist()}")
                    if 'index_landuse' not in buildings_with_landuse.columns:
                        logger.error(
                            f"Critical: Fallback Landuse index column 'index_landuse' also not found. Cannot filter landuse matches.")
                        buildings_with_landuse_matches = pd.DataFrame()  # Empty dataframe
                    else:
                        landuse_join_index_col_name = 'index_landuse'  # Confirmed fallback

                if not buildings_with_landuse.empty and landuse_join_index_col_name in buildings_with_landuse.columns:
                    buildings_with_landuse_matches = buildings_with_landuse[buildings_with_landuse[landuse_join_index_col_name].notna(
                    )].copy()
                    logger.info(
                        f"{len(buildings_with_landuse_matches)} building-landuse intersections (within) found.")
                else:
                    buildings_with_landuse_matches = gpd.GeoDataFrame()
                    logger.info(
                        "No building-landuse intersections found or landuse index column missing.")

                # Define Landuse tag mappings (primary column from landuse.geojson seems to be 'landuse')
                # Adjust the column name 'landuse_landuse' if your actual join results in
                # a different suffixed name.
                landuse_col_name = 'landuse_landuse'  # Default if 'landuse' was the key column in landuse gdf
                if landuse_col_name not in buildings_with_landuse_matches.columns and 'landuse' in landuse.columns:
                    # Attempt to find the correct suffixed column if default isn't present
                    potential_cols = [col for col in buildings_with_landuse_matches.columns if col.startswith(
                        'landuse') and col.endswith('_landuse')]
                    if potential_cols:
                        landuse_col_name = potential_cols[0]
                        logger.info(f"Using landuse column: {landuse_col_name}")
                    else:
                        logger.warning(
                            f"Could not identify the correct landuse type column in the spatially joined data. Searched for columns starting with 'landuse' and ending with '_landuse'. Skipping landuse classification.")
                        landuse_col_name = None  # Ensure we skip if not found
                elif landuse_col_name not in buildings_with_landuse_matches.columns:
                    logger.warning(
                        f"Default landuse column '{landuse_col_name}' not found and no alternative identified. Skipping landuse classification.")
                    landuse_col_name = None  # Ensure we skip if not found

                if landuse_col_name:
                    unique_building_indices_with_landuse_match = buildings_with_landuse_matches.index.unique()

                    for building_idx in unique_building_indices_with_landuse_match:
                        if pd.isna(classified_buildings.loc[building_idx, 'building_use']):
                            landuses_for_building = buildings_with_landuse_matches.loc[[
                                building_idx]]
                            # Take the first landuse match if multiple (though 'within' should be
                            # less prone to this for polygons)
                            primary_landuse_type = landuses_for_building[landuse_col_name].iloc[0]

                            current_use = pd.NA
                            if pd.notna(primary_landuse_type):
                                lu_type_lower = str(primary_landuse_type).lower()
                                if lu_type_lower in ['residential']:
                                    current_use = 'Residential'
                                elif lu_type_lower in ['commercial', 'retail']:
                                    current_use = 'Commercial'
                                elif lu_type_lower in ['industrial', 'railway', 'brownfield']:
                                    current_use = 'Industrial'
                                elif lu_type_lower in ['religious', 'cemetery', 'military', 'public', 'civic', 'governmental', 'education', 'school', 'university', 'college', 'kindergarten', 'hospital', 'clinic']:
                                    current_use = 'Public'
                                # 'construction' is often temporary; might be better to leave for default or keyword if no other info

                            if pd.notna(current_use):
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = current_use

                    logger.info(
                        f"After Landuse-based classification: \n{classified_buildings['building_use'].value_counts(dropna=False)}")

        mask = classified_buildings['building_use'].isna()
        classified_buildings.loc[mask, 'building_use'] = 'Residential'  # Default
        logger.info(
            f"{mask.sum()} buildings assigned default use 'Residential'."
        )

        # Final counts
        logger.info("Building use classification complete. Value counts:")
        logger.info(f"\n{classified_buildings['building_use'].value_counts(dropna=False)}")

        return classified_buildings

    def calculate_free_walls(self, B: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculates the number of "free walls" for each building based on
        topological relationships with neighbors.

        Free walls are exterior walls not connected to other buildings,
        which is important for energy modeling and building typology.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Building polygons

        Returns:
        --------
        GeoDataFrame : Buildings with 'free_walls' and 'neighbors' columns added
        """
        Neigh = pd.DataFrame(data=None, columns=['Neighbors'])

        for index, row in B.iterrows():
            neighbors = np.array(B[B.geometry.touches(row['geometry'])].id)
            neigh_i = pd.DataFrame({"Neighbors": [neighbors]})
            neigh_i = neigh_i.set_index(pd.Index([index]))
            Neigh = pd.concat([Neigh, neigh_i])
        B = pd.concat([B, Neigh], axis=1)

        B["Free_walls"] = 4
        for index, row in B.iterrows():
            x = len(B.at[index, "Neighbors"])
            if x < 4:
                B.at[index, 'Free_walls'] = 4 - x
            else:
                B.at[index, 'Free_walls'] = 0

        logger.info(
            f"After calculating free walls: \n{B['Free_walls'].value_counts(dropna=False)}")
        return B

    def classify_building_type(self, buildings: gpd.GeoDataFrame,
                               housing_data: Optional[Dict] = None) -> gpd.GeoDataFrame:
        """
        Classifies residential buildings into specific typologies:
        - SFH (Single Family Home)
        - TH (Townhouse/Row House)
        - MFH (Multi-Family Home)
        - AB (Apartment Building)

        Parameters:
        -----------
        buildings : GeoDataFrame
            Residential building polygons with free_walls calculated
        housing_data : Dict, optional
            Reference housing type distribution

        Returns:
        --------
        GeoDataFrame : Buildings with 'building_type' column added
        """
        if buildings is None or len(buildings) == 0:
            return buildings

        # Create a copy to avoid modifying the original
        classified_buildings = buildings.copy()

        # Initialize building_type column
        classified_buildings['building_type'] = None

        # 1. Identify neighborhood clusters
        logger.debug("Identifying neighborhood clusters")
        # This would require clustering buildings based on proximity
        # For now, use a simplified approach

        # 2. Calculate total area for each neighborhood cluster
        # This would aggregate areas within clusters

        # 3. Initial classification based on geometry and neighbors
        logger.debug("Initial classification based on geometry and neighbors")

        # Large buildings/clusters → AB (Apartment Building)
        large_threshold = 1000  # sq meters
        mask = classified_buildings.geometry.area > large_threshold
        classified_buildings.loc[mask, 'building_type'] = 'AB'

        # Small detached buildings (free_walls = 4) → SFH (Single Family Home)
        mask = (classified_buildings['Free_walls'] == 4) & \
               (classified_buildings.geometry.area < large_threshold) & \
               (classified_buildings['building_type'].isna())
        classified_buildings.loc[mask, 'building_type'] = 'SFH'

        # Small attached buildings in rows (free_walls = 2) → TH (Townhouse)
        mask = (classified_buildings['Free_walls'] == 2) & \
               (classified_buildings.geometry.area < large_threshold) & \
               (classified_buildings['building_type'].isna())
        classified_buildings.loc[mask, 'building_type'] = 'TH'

        # Medium-sized buildings with varied neighbor counts → MFH (Multi-Family Home)
        medium_threshold = 300  # sq meters
        mask = (classified_buildings.geometry.area > medium_threshold) & \
               (classified_buildings.geometry.area <= large_threshold) & \
               (classified_buildings['building_type'].isna())
        classified_buildings.loc[mask, 'building_type'] = 'MFH'

        # Assign remaining buildings as SFH
        mask = classified_buildings['building_type'].isna()
        classified_buildings.loc[mask, 'building_type'] = 'SFH'

        # 4. Balance classification with reference distribution
        logger.debug("Balancing classification with reference distribution")
        # This would compare the current distribution with the expected one
        # and adjust classifications to match regional statistics

        # 5. Handle townhouses/row houses specially
        logger.debug("Special handling for townhouses")
        # This would check for linear arrangement and split if appropriate

        return classified_buildings

    def allot_occupants(self, buildings: gpd.GeoDataFrame,
                        census_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Allocates household occupants to residential buildings based on
        building type, size, and census population data.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Residential buildings with building_type
        population_data : Dict, optional
            Census population and household data

        Returns:
        --------
        GeoDataFrame : Buildings with 'occupants' column added
        """
        if buildings is None or len(buildings) == 0:
            return buildings

        # Create a copy to avoid modifying the original
        buildings_with_occupants = buildings.copy()

        # Initialize occupants column
        buildings_with_occupants['occupants'] = 0

        # 1. Calculate maximum occupant capacity per building type
        logger.debug("Calculating maximum occupant capacity")

        # Default capacity calculation based on building type and area
        for idx, building in buildings_with_occupants.iterrows():
            area = building['floor_area']
            if building['building_type'] == 'SFH':
                # SFH: Typically 2-6 people based on area
                buildings_with_occupants.at[idx, 'occupants'] = min(6, max(2, int(area / 50)))
            elif building['building_type'] == 'TH':
                # TH: Typically 2-4 people per unit
                buildings_with_occupants.at[idx, 'occupants'] = min(4, max(2, int(area / 50)))
            elif building['building_type'] == 'MFH':
                # MFH: Based on number of units estimated from floor area
                units = max(2, int(area / 100))
                avg_occupants_per_unit = 2.5
                buildings_with_occupants.at[idx, 'occupants'] = int(units * avg_occupants_per_unit)
            elif building['building_type'] == 'AB':
                # AB: Based on number of units estimated from floor area
                units = max(4, int(area / 80))
                avg_occupants_per_unit = 2.0
                buildings_with_occupants.at[idx, 'occupants'] = int(units * avg_occupants_per_unit)

        # 2. Distribute census block population to buildings
        logger.debug("Distributing census block population")
        # This would spatially join buildings to census blocks
        # and proportionally distribute population based on capacity

        # 3. Handle buildings with no census block data
        # Already handled by default calculation above

        # 4. Ensure total population matches census totals
        logger.debug("Validating population distribution")
        # This would adjust occupant counts to match census totals

        # Calculate households
        buildings_with_occupants['households'] = buildings_with_occupants.apply(
            lambda x: 1 if x['building_type'] in ['SFH', 'TH'] else
            max(1, int(x['occupants'] / 2.5)), axis=1
        )

        return buildings_with_occupants

    def _calculate_floor_height_from_osm_tags(
            self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extracts height and floor information from OSM building tags and adds them to the GeoDataFrame.

        This function processes the following OSM tags:
        - 'height': Building height in meters (string format)
        - 'building:levels': Number of floors/levels (string format)
        - 'building:min_level': Minimum level if available
        - 'building:levels:underground': Underground levels if available

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings with OSM tag information

        Returns:
        --------
        GeoDataFrame : Buildings with 'height' and 'floors' columns added
        """
        if buildings is None or len(buildings) == 0:
            return buildings

        # Create a copy to avoid modifying the original
        buildings_with_height_floors = buildings.copy()

        # Initialize height and floors columns with default values
        buildings_with_height_floors['height'] = None
        buildings_with_height_floors['floors'] = 1

        logger.debug(
            f"Processing {len(buildings_with_height_floors)} buildings for height and floor information")

        # Extract height information from OSM 'height' tag
        if 'height' in buildings_with_height_floors.columns:
            # Convert height strings to float, handling non-numeric values
            def parse_height(height_val):
                """Parse height value from OSM tag (could be string with units)."""
                if pd.isna(height_val) or height_val is None:
                    return None
                try:
                    # Handle string values that might have units like "21.4 m" or just "21.4"
                    height_str = str(height_val).strip()
                    # Remove common units and extract numeric part
                    height_str = height_str.replace(
                        'm',
                        '').replace(
                        'meters',
                        '').replace(
                        'ft',
                        '').replace(
                        'feet',
                        '').strip()
                    height_float = float(height_str)
                    # Sanity check: building height should be reasonable (1-500 meters)
                    if 1.0 <= height_float <= 500.0:
                        return height_float
                    else:
                        logger.debug(f"Height value {height_float} seems unreasonable, ignoring")
                        return None
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse height value: {height_val}")
                    return None

            # Apply height parsing
            height_mask = buildings_with_height_floors['height'].notna()
            if height_mask.any():
                buildings_with_height_floors.loc[height_mask,
                                                 'height'] = buildings_with_height_floors.loc[height_mask,
                                                                                              'height'].apply(parse_height)
                valid_heights = buildings_with_height_floors['height'].notna().sum()
                logger.debug(f"Successfully extracted {valid_heights} height values from OSM data")

        # Extract floor information from OSM 'building:levels' tag
        if 'building:levels' in buildings_with_height_floors.columns:
            def parse_floors(levels_val):
                """Parse building levels from OSM tag."""
                if pd.isna(levels_val) or levels_val is None:
                    return None
                try:
                    # Handle decimal floors (like "3.5") by rounding up
                    floors_float = float(str(levels_val).strip())
                    # Sanity check: number of floors should be reasonable (1-200)
                    if 1.0 <= floors_float <= 100.0:
                        # Round and ensure at least 1 floor
                        return max(1, int(round(floors_float)))
                    else:
                        logger.debug(f"Floor count {floors_float} seems unreasonable, ignoring")
                        return None
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse building:levels value: {levels_val}")
                    return None

            # Apply floor parsing
            levels_mask = buildings_with_height_floors['building:levels'].notna()
            if levels_mask.any():
                buildings_with_height_floors.loc[levels_mask,
                                                 'floors'] = buildings_with_height_floors.loc[levels_mask,
                                                                                              'building:levels'].apply(parse_floors)
                valid_floors = (buildings_with_height_floors['floors'] > 1).sum()
                logger.debug(
                    f"Successfully extracted {valid_floors} floor count values from OSM data")

        # Handle cases where we have height but no floors - estimate floors from height
        height_no_floors_mask = (
            buildings_with_height_floors['height'].notna()) & (
            buildings_with_height_floors['floors'] == 1)
        if height_no_floors_mask.any():
            # Estimate floors using typical floor height of 3
            estimated_floors = buildings_with_height_floors.loc[height_no_floors_mask, 'height'] / 3
            buildings_with_height_floors.loc[height_no_floors_mask, 'floors'] = estimated_floors.apply(
                lambda x: max(1, int(round(x))))
            logger.debug(
                f"Estimated floors from height for {height_no_floors_mask.sum()} buildings")

        # Handle cases where we have floors but no height - estimate height from floors
        floors_no_height_mask = (
            buildings_with_height_floors['height'].isna()) & (
            buildings_with_height_floors['floors'] > 1)
        if floors_no_height_mask.any():
            # Estimate height using typical floor height of 3
            buildings_with_height_floors.loc[floors_no_height_mask,
                                             'height'] = buildings_with_height_floors.loc[floors_no_height_mask,
                                                                                          'floors'] * 3
            logger.debug(
                f"Estimated height from floors for {floors_no_height_mask.sum()} buildings")

        # Add minimum level adjustment if available
        if 'building:min_level' in buildings_with_height_floors.columns:
            min_level_mask = buildings_with_height_floors['building:min_level'].notna()
            if min_level_mask.any():
                logger.debug(
                    f"Found {min_level_mask.sum()} buildings with minimum level information")

        # Log summary statistics
        height_available = buildings_with_height_floors['height'].notna().sum()
        floors_available = (buildings_with_height_floors['floors'] > 1).sum()
        logger.info(
            f"Height and floor extraction complete: {height_available} buildings with height data, {floors_available} buildings with >1 floor")
        logger.info(
            f"Height: {buildings_with_height_floors['height'].value_counts(dropna=False).sort_index()}")
        logger.info(
            f"# of floors: {buildings_with_height_floors['floors'].value_counts(dropna=False).sort_index()}")
        return buildings_with_height_floors

    def _calculate_floor_height_from_ms_buildings(
        self, buildings: gpd.GeoDataFrame, microsoft_buildings: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Extract height information from Microsoft Buildings data via spatial join.

        This method performs spatial overlays between OSM buildings and Microsoft Buildings
        to assign height values where available and valid (not -1).

        Parameters:
        -----------
        buildings : GeoDataFrame
            OSM buildings to assign height data to
        microsoft_buildings : GeoDataFrame
            Microsoft Buildings data with height information

        Returns:
        --------
        GeoDataFrame : Buildings with 'height' and 'floors' columns populated from MS data
        """
        if buildings is None or len(buildings) == 0:
            logger.debug("No buildings provided for MS Buildings height extraction")
            return buildings

        if microsoft_buildings is None or len(microsoft_buildings) == 0:
            logger.debug("No Microsoft Buildings data available for height extraction")
            # Initialize height and floors columns if they don't exist
            buildings_copy = buildings.copy()
            if 'height' not in buildings_copy.columns:
                buildings_copy['height'] = None
            if 'floors' not in buildings_copy.columns:
                buildings_copy['floors'] = 1
            return buildings_copy

        logger.debug(
            f"Extracting height data for {len(buildings)} buildings from {len(microsoft_buildings)} MS building footprints")
        logger.debug(f"Microsoft Buildings columns: {list(microsoft_buildings.columns)}")

        # Create working copy
        buildings_with_ms_height = buildings.copy()
        ms_buildings_copy = microsoft_buildings.copy()

        # Initialize height and floors columns if they don't exist
        if 'height' not in buildings_with_ms_height.columns:
            buildings_with_ms_height['height'] = None
        if 'floors' not in buildings_with_ms_height.columns:
            buildings_with_ms_height['floors'] = 1

        # Handle nested properties - extract height and confidence if they're nested
        if 'properties' in ms_buildings_copy.columns and 'height' not in ms_buildings_copy.columns:
            logger.debug("Height data appears to be nested in properties column, extracting...")

            # Extract height from properties
            def extract_height(properties):
                if properties is None or not isinstance(properties, dict):
                    return None
                return properties.get('height', None)

            def extract_confidence(properties):
                if properties is None or not isinstance(properties, dict):
                    return None
                return properties.get('confidence', None)

            ms_buildings_copy['height'] = ms_buildings_copy['properties'].apply(extract_height)
            ms_buildings_copy['confidence'] = ms_buildings_copy['properties'].apply(
                extract_confidence)

            logger.debug(
                f"Extracted height for {ms_buildings_copy['height'].notna().sum()} buildings from properties")

        # Check if height column exists now
        if 'height' not in ms_buildings_copy.columns:
            logger.warning(
                "No height column found in Microsoft Buildings data, even after property extraction")
            return buildings_with_ms_height

        # Ensure both datasets are in the same CRS for spatial operations
        # Use a projected CRS for more accurate spatial joins (EPSG:5070 for US)
        target_crs = "EPSG:5070"

        if buildings_with_ms_height.crs != target_crs:
            buildings_projected = buildings_with_ms_height.to_crs(target_crs)
        else:
            buildings_projected = buildings_with_ms_height.copy()

        if ms_buildings_copy.crs != target_crs:
            ms_buildings_projected = ms_buildings_copy.to_crs(target_crs)
        else:
            ms_buildings_projected = ms_buildings_copy.copy()

        # Filter out invalid height values (-1 or negative values)
        height_mask = (ms_buildings_projected['height'].notna()) & \
            (pd.to_numeric(ms_buildings_projected['height'], errors='coerce') > 0) & \
            (pd.to_numeric(ms_buildings_projected['height'], errors='coerce') != -1)

        valid_ms_buildings = ms_buildings_projected[height_mask].copy()

        if len(valid_ms_buildings) == 0:
            logger.debug(
                "No valid height data found in Microsoft Buildings (all heights are -1 or invalid)")
            return buildings_with_ms_height

        logger.debug(
            f"Found {len(valid_ms_buildings)} MS buildings with valid height data (filtered from {len(ms_buildings_projected)})")

        # Convert height to numeric if it's not already
        valid_ms_buildings['height'] = pd.to_numeric(valid_ms_buildings['height'], errors='coerce')

        # Prepare columns for spatial join
        join_columns = ['geometry', 'height']
        if 'confidence' in valid_ms_buildings.columns:
            join_columns.append('confidence')

        # Perform spatial join to find overlapping buildings
        # Use 'intersects' predicate for initial overlap detection
        try:
            joined = gpd.sjoin(
                buildings_projected,
                valid_ms_buildings[join_columns],
                how='left',
                predicate='intersects'
            )

            # Handle multiple matches by selecting the MS building with highest confidence
            # Group by original building index and select best match
            if 'confidence' in joined.columns:
                # Sort by confidence (descending) and take first (highest confidence)
                # match per building
                best_matches = joined.sort_values(
                    'confidence', ascending=False).groupby(
                    level=0).first()
            else:
                # If no confidence column, just take first match
                best_matches = joined.groupby(level=0).first()

            # Assign height values from best matches - handle right-side column naming
            height_col = 'height_right' if 'height_right' in best_matches.columns else 'height'
            height_mask = best_matches[height_col].notna()

            if height_mask.any():
                buildings_with_ms_height.loc[height_mask,
                                             'height'] = best_matches.loc[height_mask, height_col]

                # Estimate floors from height (using 3.0m per floor as typical)
                floors_from_height = (
                    best_matches.loc[height_mask, height_col] / 3.0).round().astype(int)
                floors_from_height = floors_from_height.clip(lower=1)  # Ensure at least 1 floor
                buildings_with_ms_height.loc[height_mask, 'floors'] = floors_from_height

                assigned_count = height_mask.sum()
                logger.info(
                    f"Successfully assigned height data from Microsoft Buildings to {assigned_count} buildings")

                # Log how many buildings now have an assigned height and floors to total
                # number of buildings:
                logger.debug(
                    f"Number of buildings with assigned height and floors: {height_mask.sum()} out of {len(buildings_with_ms_height)}")
            else:
                logger.debug(
                    "No spatial overlaps found between OSM buildings and Microsoft Buildings")

            buildings_with_ms_height.to_file(self.output_dir / "buildings_with_ms_height.geojson")

        except Exception as e:
            logger.warning(f"Error during spatial join with Microsoft Buildings: {e}")
            logger.debug("Continuing without Microsoft Buildings height data")

        return buildings_with_ms_height

    def calculate_floors(self, buildings: gpd.GeoDataFrame,
                         microsoft_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Estimates number of floors for each building based on
        building type, occupants, and area.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings with occupants and building_type

        Returns:
        --------
        GeoDataFrame : Buildings with 'floors' column added
        """
        if buildings is None or len(buildings) == 0:
            return buildings

        # Step 0: Extract height and floor information from Microsoft Buildings data
        logger.debug("Step 0: Extracting height and floors from Microsoft Buildings data")
        print(microsoft_buildings.columns)
        print(microsoft_buildings.head())
        buildings_with_floors = self._calculate_floor_height_from_ms_buildings(
            buildings, microsoft_buildings)

        # # Step 1: Extract height and floor information from OSM tags
        # logger.debug("Step 1: Extracting height and floors from OSM tags")
        # # Filter out buildings that have height and floors from MS data before osm tags extraction to not overwrite:
        # buildings_with_floors = buildings_with_floors[buildings_with_floors['height'].isna() | buildings_with_floors['floors'].isna()]
        # buildings_with_floors = self._calculate_floor_height_from_osm_tags(buildings_with_floors)

        # Step 2: For buildings without OSM data, estimate from occupants and area
        logger.debug("Step 2: Estimating floors from occupants and area for remaining buildings")

        # Only process buildings that still don't have floor data (floors == 1 and no height)
        needs_estimation_mask = (
            buildings_with_floors['floors'] == 1) & (
            buildings_with_floors['height'].isna())

        if needs_estimation_mask.any():
            logger.debug(
                f"Estimating floors for {needs_estimation_mask.sum()} buildings without OSM data")

            for idx, building in buildings_with_floors[needs_estimation_mask].iterrows():
                # Check if required columns exist for estimation
                if 'building_type' not in building or 'occupants' not in building:
                    logger.debug(
                        f"Skipping estimation for building {idx}: missing building_type or occupants")
                    continue

                building_type = building['building_type']
                area = building.geometry.area if hasattr(
                    building, 'geometry') else 100  # fallback area
                occupants = building['occupants'] if pd.notna(
                    building['occupants']) else 2

                if building_type == 'SFH':
                    # SFH: Usually 1-3 floors
                    if area < 100:
                        buildings_with_floors.at[idx, 'floors'] = 1
                    elif area < 200:
                        buildings_with_floors.at[idx, 'floors'] = 2
                    else:
                        buildings_with_floors.at[idx, 'floors'] = min(3, int(occupants / 3) + 1)

                elif building_type == 'TH':
                    # TH: Usually 2-3 floors
                    buildings_with_floors.at[idx, 'floors'] = min(
                        3, max(2, int(occupants / 3) + 1))

                elif building_type == 'MFH':
                    # MFH: Usually 2-5 floors
                    buildings_with_floors.at[idx, 'floors'] = min(
                        5, max(2, int(occupants / 6) + 1))

                elif building_type == 'AB':
                    # AB: Usually 4-10 floors
                    buildings_with_floors.at[idx, 'floors'] = min(
                        10, max(4, int(occupants / 8) + 1))
                else:
                    # Default case for unknown building types
                    buildings_with_floors.at[idx, 'floors'] = max(1, int(occupants / 4) + 1)

        # Step 3: Calculate height for buildings that still don't have height data
        missing_height_mask = buildings_with_floors['height'].isna()
        if missing_height_mask.any():
            logger.debug(
                f"Estimating height for {missing_height_mask.sum()} buildings without height data")
            buildings_with_floors.loc[missing_height_mask,
                                      'height'] = buildings_with_floors.loc[missing_height_mask,
                                                                            'floors'] * 3.5

        # Step 4: Calculate floor area
        logger.debug("Calculating floor area")
        buildings_with_floors['floor_area'] = buildings_with_floors.geometry.area * \
            buildings_with_floors['floors']

        # Step 5: Validate and adjust estimates
        logger.debug("Validating floor and height estimates")

        # Convert height to numeric, handling any string values
        buildings_with_floors['height'] = pd.to_numeric(
            buildings_with_floors['height'], errors='coerce')

        # Ensure minimum values
        buildings_with_floors['floors'] = buildings_with_floors['floors'].clip(lower=1)
        buildings_with_floors['height'] = buildings_with_floors['height'].clip(
            lower=2.5)  # minimum reasonable height

        return buildings_with_floors

    def allot_construction_year(self, buildings: gpd.GeoDataFrame,
                                housing_age_data: Optional[Dict] = None) -> gpd.GeoDataFrame:
        """
        Assigns construction year periods to buildings based on
        available data and statistical distribution.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings with type and other attributes
        housing_age_data : Dict, optional
            Statistical data on building age distribution by region

        Returns:
        --------
        GeoDataFrame : Buildings with 'construction_year' column added
        """
        if buildings is None or len(buildings) == 0:
            return buildings

        # Create a copy to avoid modifying the original
        buildings_with_year = buildings.copy()

        # Initialize construction_year column
        buildings_with_year['construction_year'] = None

        # 1. Use direct OSM data if available
        logger.debug("Using OSM year or start_date tags if available")
        if 'start_date' in buildings_with_year.columns:
            mask = buildings_with_year['start_date'].notna()
            buildings_with_year.loc[mask,
                                    'construction_year'] = buildings_with_year.loc[mask,
                                                                                   'start_date']

        # 2. Use spatial reference data
        logger.debug("Using spatial reference data for construction year")
        # This would spatially join with assessor or historical data

        # 3. Apply neighborhood consistency patterns
        logger.debug("Applying neighborhood consistency patterns")
        # This would cluster buildings and apply consistent ages within neighborhoods

        # 4. Allocate remaining buildings based on statistical distribution
        logger.debug("Allocating construction year based on statistical distribution")

        # Common categories for US buildings
        periods = ['Pre-1950', '1950-1969', '1970-1989', '1990-2009', '2010-present']

        # Default distribution if no reference data provided
        distribution = {'Pre-1950': 0.2, '1950-1969': 0.2, '1970-1989': 0.25,
                        '1990-2009': 0.25, '2010-present': 0.1}

        # If housing_age_data is provided, use it to determine distribution
        if housing_age_data:
            # Parse housing_age_data to update distribution
            pass

        # Apply distribution to buildings with missing construction_year
        mask = buildings_with_year['construction_year'].isna()
        num_to_assign = mask.sum()

        if num_to_assign > 0:
            # Calculate the number of buildings for each period
            period_counts = {period: int(num_to_assign * distribution[period])
                             for period in periods}

            # Adjust to make sure we assign all buildings
            total_assigned = sum(period_counts.values())
            if total_assigned < num_to_assign:
                # Add remaining to most common period
                most_common = max(distribution, key=distribution.get)
                period_counts[most_common] += num_to_assign - total_assigned

            # Create a list of periods to assign
            periods_to_assign = []
            for period, count in period_counts.items():
                periods_to_assign.extend([period] * count)

            # Shuffle the periods
            np.random.shuffle(periods_to_assign)

            # Assign periods to buildings
            buildings_with_year.loc[mask, 'construction_year'] = periods_to_assign

        # 5. Add confidence indicator for the source of the year data
        # This is handled by the default confidence score

        return buildings_with_year

    def allot_refurbishment_level(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assigns refurbishment level indicators to buildings based on
        age, type, and statistical patterns.

        For energy modeling, this indicates upgrades to:
        - Walls (insulation)
        - Roof
        - Windows
        - Basement/Foundation

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings with type, year, and other attributes

        Returns:
        --------
        GeoDataFrame : Buildings with refurbishment indicators added
        """
        if buildings is None or len(buildings) == 0:
            return buildings

        # Create a copy to avoid modifying the original
        buildings_with_refurbishment = buildings.copy()

        # Initialize refurbishment columns
        buildings_with_refurbishment['refurb_walls'] = 0
        buildings_with_refurbishment['refurb_roof'] = 0
        buildings_with_refurbishment['refurb_windows'] = 0
        buildings_with_refurbishment['refurb_basement'] = 0

        # 1. Assign probability of refurbishment based on building age
        logger.debug("Assigning refurbishment probabilities based on age")

        for idx, building in buildings_with_refurbishment.iterrows():
            refurb_prob = 0.0

            # Base probability on construction year
            if building['construction_year'] == 'Pre-1950':
                refurb_prob = 0.9  # Older buildings very likely to have been refurbished
            elif building['construction_year'] == '1950-1969':
                refurb_prob = 0.8
            elif building['construction_year'] == '1970-1989':
                refurb_prob = 0.6
            elif building['construction_year'] == '1990-2009':
                refurb_prob = 0.3
            elif building['construction_year'] == '2010-present':
                refurb_prob = 0.0  # New buildings unlikely to be refurbished

            # Adjust based on building type
            if building['building_type'] == 'SFH':
                refurb_prob *= 1.1  # SFH slightly more likely to be refurbished
            elif building['building_type'] == 'AB':
                refurb_prob *= 0.9  # AB slightly less likely to be refurbished

            # Cap probability at 1.0
            refurb_prob = min(1.0, refurb_prob)

            # Assign specific refurbishment components
            buildings_with_refurbishment.at[idx,
                                            'refurb_walls'] = 1 if np.random.random() < refurb_prob else 0
            buildings_with_refurbishment.at[idx,
                                            'refurb_roof'] = 1 if np.random.random() < refurb_prob else 0
            buildings_with_refurbishment.at[idx,
                                            'refurb_windows'] = 1 if np.random.random() < refurb_prob else 0
            buildings_with_refurbishment.at[idx, 'refurb_basement'] = 1 if np.random.random(
            ) < refurb_prob * 0.7 else 0

        # 2. Consider neighborhood effects
        logger.debug("Considering neighborhood effects on refurbishment")
        # This would cluster buildings and apply consistency within neighborhoods

        return buildings_with_refurbishment

    def write_buildings_output(self, buildings: gpd.GeoDataFrame,
                               output_dir: Union[str, Path],
                               filename: str) -> str:
        """
        Writes processed building data to shapefile.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Processed buildings
        output_dir : str or Path
            Path to output directory
        filename : str
            Output filename

        Returns:
        --------
        str : Path to output file
        """
        # Create output path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / filename)

        # Ensure all required columns from schema are present
        from syngrid.data_processor.processing.building_schema import BuildingOutputSchema
        schema_fields = BuildingOutputSchema.get_schema_fields()

        # Add missing columns with None values
        for field in schema_fields:
            if field not in buildings.columns and field != 'geometry':
                buildings[field] = None

        # Write shapefile
        logger.info(f"Writing {len(buildings)} buildings to {output_path}")
        buildings.to_file(output_path)

        return output_path

    def write_empty_data_stub(self, output_dir: Union[str, Path],
                              filename: str,
                              message: str) -> str:
        """
        Creates a stub text file when no buildings of a category exist.

        Parameters:
        -----------
        output_dir : str or Path
            Path to output directory
        filename : str
            Output filename
        message : str
            Message explaining the absence of data

        Returns:
        --------
        str : Path to output file
        """
        # Create output path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / filename)

        # Write text file with message
        with open(output_path, 'w') as f:
            f.write(message + '\n')

        logger.info(f"Created empty data stub: {output_path}")
        return output_path

    def merge_building_layers(self, residential_path: str,
                              other_path: str,
                              output_dir: Union[str, Path],
                              filename: str) -> str:
        """
        Merges residential and non-residential building layers into a single shapefile.

        Parameters:
        -----------
        residential_path : str
            Path to residential buildings shapefile
        other_path : str
            Path to non-residential buildings shapefile
        output_dir : str or Path
            Path to output directory
        filename : str
            Output filename

        Returns:
        --------
        str : Path to merged output file
        """
        # Create output path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / filename)

        # Read input shapefiles
        residential = gpd.read_file(residential_path)
        other = gpd.read_file(other_path)

        # Merge GeoDataFrames
        merged = gpd.GeoDataFrame(pd.concat([residential, other], ignore_index=True))

        # Write output shapefile
        logger.info(f"Writing merged buildings ({len(merged)} features) to {output_path}")
        merged.to_file(output_path)

        return output_path

    def _get_local_crs(self, gdf):
        """Get appropriate local CRS for accurate area calculations"""
        # Calculate centroid of all geometries
        lon, lat = gdf.unary_union.centroid.x, gdf.unary_union.centroid.y

        # For North America, determine UTM zone
        utm_zone = int(np.floor((lon + 180) / 6) + 1)
        hemisphere = 'north' if lat >= 0 else 'south'

        # Return UTM CRS (EPSG code format)
        if hemisphere == 'north':
            return f"EPSG:{32600+utm_zone}"
        else:
            return f"EPSG:{32700+utm_zone}"

    def _calculate_floor_area(self, buildings):
        """
        Add floor_area column in square meters and ensure data is in EPSG:4326

        Parameters:
        -----------
        buildings : GeoDataFrame
            Building polygons in any CRS

        Returns:
        --------
        GeoDataFrame : Buildings with floor_area column added, in EPSG:4326
        """
        # First ensure data is in EPSG:5070 (US metric)
        if buildings.crs != "EPSG:5070":
            buildings_projected = buildings.to_crs(epsg=5070)
        else:
            buildings_projected = buildings

        # Calculate area in square meters
        buildings['floor_area'] = buildings_projected.geometry.area

        return buildings

    def ensure_wgs84(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Ensures any GeoDataFrame is in WGS84 (EPSG:4326) projection

        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe in any CRS

        Returns:
        --------
        GeoDataFrame : Input geodataframe reprojected to EPSG:4326 if needed
        """
        if gdf is None or len(gdf) == 0:
            return gdf

        if gdf.crs != "EPSG:4326":
            logger.debug(f"Reprojecting from {gdf.crs} to EPSG:4326")
            return gdf.to_crs("EPSG:4326")

        return gdf

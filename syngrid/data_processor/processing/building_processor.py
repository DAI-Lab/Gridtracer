from pathlib import Path
from typing import Dict, List, Optional, Union

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
        osm_data['buildings'] = self._filter_small_buildings(osm_data['buildings'])

        # Step 1: Classify building use (residential, commercial, industrial, etc.)
        logger.info("Step 1: Classifying building use")
        all_buildings = self.classify_building_use(
            osm_data.get('buildings'),
            osm_data.get('pois'),
            osm_data.get('landuse'),
        )

        # Step 2: Split buildings by use
        logger.info(f"Found {len(all_buildings)} buildings to process")

        # Step 3: Calculate free walls for all buildings
        logger.info("Step 3: Calculating free walls")
        all_buildings = self.calculate_free_walls(all_buildings)
        all_buildings.to_file(self.output_dir / "03_all_buildings_with_free_walls.geojson")

        # Step 4: Calculate floors
        logger.info("Step 4: Calculating floors")
        all_buildings = self.calculate_floors(
            all_buildings, microsoft_buildings_data.get('ms_buildings'))
        all_buildings.to_file(self.output_dir / "04_all_buildings_with_floors.geojson")

        # Step 5: Assign building IDs
        logger.info("Step 5: Assigning building IDs")
        all_buildings = self._assign_building_id(
            all_buildings, census_data.get('target_region_blocks'))

        all_buildings.to_file(self.output_dir / "05_all_buildings_with_building_id.geojson")

        # Split buildings by use after ID assignment
        residential = all_buildings[all_buildings['building_use'] == 'residential'].copy()
        # other = all_buildings[all_buildings['building_use'] != 'residential'].copy()

        if len(residential) > 0:
            logger.info(
                f"6. Processing {len(residential)} residential buildings to determine building type")

            # Building type classification
            residential = self.classify_building_type(
                residential,
                census_data.get('target_region_blocks')
            )
            residential.to_file(self.output_dir
                                / "06_residential_buildings_with_building_type.geojson")
            # Allot occupants based on census
            residential = self._allot_occupants(
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

    def _filter_small_buildings(self, buildings: gpd.GeoDataFrame,
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
        GeoDataFrame : Buildings with 'building_use' column added (residential,
                       commercial, industrial, public).
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
            candidate_buildings.loc[mask, 'building_use'] = 'residential'
            logger.info(f"{mask.sum()} buildings classified as residential from 'building' tag.")

            # Commercial from 'building'
            com_types = ['commercial', 'retail', 'office', 'supermarket', 'kiosk', 'shop',
                         'hotel', 'motel', 'hostel']
            mask = candidate_buildings['building'].isin(
                com_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'commercial'
            logger.info(f"{mask.sum()} buildings classified as commercial from 'building' tag.")

            # Industrial from 'building'
            ind_types = [
                'industrial',
                'warehouse',
                'factory',
                'manufacture',
                'workshop']  # workshop added
            mask = candidate_buildings['building'].isin(
                ind_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'industrial'
            logger.info(f"{mask.sum()} buildings classified as industrial from 'building' tag.")

            # Public from 'building'
            pub_types = ['school', 'hospital', 'government', 'university', 'public',
                         'church', 'mosque', 'synagogue', 'temple', 'chapel', 'cathedral',
                         'civic', 'kindergarten', 'college', 'train_station', 'transportation',
                         'public_transport']
            mask = candidate_buildings['building'].isin(
                pub_types) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'public'
            logger.info(f"{mask.sum()} buildings classified as public from 'building' tag.")

        # 1.2 'building:use' tag (often more specific than 'building')
        if 'building:use' in candidate_buildings.columns:
            use_map = {
                'residential': 'residential', 'residental': 'residential',  # common typo
                'commercial': 'commercial', 'retail': 'commercial', 'office': 'commercial',
                'industrial': 'industrial', 'warehouse': 'industrial',
                'public': 'public', 'civic': 'public', 'governmental': 'public',
                'education': 'public', 'school': 'public', 'university': 'public',
                'college': 'public', 'kindergarten': 'public',
                'hospital': 'public', 'clinic': 'public', 'healthcare': 'public',
                'place_of_worship': 'public', 'religious': 'public',
                'transportation': 'public',
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
            candidate_buildings.loc[mask, 'building_use'] = 'public'
            logger.info(f"{mask.sum()} buildings classified as public from 'amenity' tag.")

            mask = candidate_buildings['amenity'].fillna('').str.lower().isin(
                com_amenities) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'commercial'
            logger.info(f"{mask.sum()} buildings classified as commercial from 'amenity' tag.")

            mask = candidate_buildings['amenity'].fillna('').str.lower().isin(
                res_amenities) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'residential'
            logger.info(
                f"{mask.sum()} buildings classified as residential from 'amenity' (shelter).")

        # 1.4 'shop' tag
        if 'shop' in candidate_buildings.columns:
            mask = candidate_buildings['shop'].notna() & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'commercial'
            logger.info(f"{mask.sum()} buildings classified as commercial from 'shop' tag.")

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
            candidate_buildings.loc[mask, 'building_use'] = 'public'
            logger.info(f"{mask.sum()} buildings classified as Public from specific 'office' types.")

            # General commercial offices
            mask = candidate_buildings['office'].notna() & \
                ~candidate_buildings['office'].fillna('').str.lower().isin(public_office_types) & \
                candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'commercial'
            logger.info(
                f"{mask.sum()} buildings classified as commercial from general 'office' tag.")

        # 1.6 'building:flats' tag
        if 'building:flats' in candidate_buildings.columns:
            mask = candidate_buildings['building:flats'].notna(
            ) & candidate_buildings['building_use'].isna()
            candidate_buildings.loc[mask, 'building_use'] = 'residential'
            logger.info(
                f"{mask.sum()} buildings classified as Residential from 'building:flats' tag.")

        # 1.7 'craft' tag
        if 'craft' in candidate_buildings.columns:
            mask = candidate_buildings['craft'].notna(
            ) & candidate_buildings['building_use'].isna()
            # Or 'Industrial' for some
            candidate_buildings.loc[mask, 'building_use'] = 'commercial'
            logger.info(f"{mask.sum()} buildings classified as commercial from 'craft' tag.")

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
                                                         'building_use'] = 'commercial'
                                classified_this_building = True

                        # Rule 3: POI Amenity for Residential (if not already classified)
                        if not classified_this_building and 'amenity_poi' in pois_for_this_building.columns:
                            if pois_for_this_building['amenity_poi'].isin(
                                    poi_residential_amenities).any():
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'residential'
                                classified_this_building = True

                        # Rule 4: POI Shop tag (if not already classified)
                        if not classified_this_building and 'shop_poi' in pois_for_this_building.columns:
                            if pois_for_this_building['shop_poi'].notna().any():
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'commercial'
                                classified_this_building = True

                        # Rule 5: POI Office tag (if not already classified)
                        if not classified_this_building and 'office_poi' in pois_for_this_building.columns:
                            poi_public_office_types = [
                                'government', 'administrative', 'diplomatic', 'association', 'ngo']
                            if pois_for_this_building['office_poi'].isin(
                                    poi_public_office_types).any():
                                classified_buildings.loc[building_idx, 'building_use'] = 'public'
                                classified_this_building = True
                            elif pois_for_this_building['office_poi'].notna().any():  # Any other office
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = 'commercial'
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
                                    current_use = 'residential'
                                elif lu_type_lower in ['commercial', 'retail']:
                                    current_use = 'commercial'
                                elif lu_type_lower in ['industrial', 'railway', 'brownfield']:
                                    current_use = 'Industrial'
                                elif lu_type_lower in ['religious', 'cemetery', 'military', 'public', 'civic', 'governmental', 'education', 'school', 'university', 'college', 'kindergarten', 'hospital', 'clinic']:
                                    current_use = 'public'
                                # 'construction' is often temporary; might be better to leave for default or keyword if no other info

                            if pd.notna(current_use):
                                classified_buildings.loc[building_idx,
                                                         'building_use'] = current_use

                    logger.info(
                        f"After Landuse-based classification: \n{classified_buildings['building_use'].value_counts(dropna=False)}")

        mask = classified_buildings['building_use'].isna()
        classified_buildings.loc[mask, 'building_use'] = 'residential'  # Default
        logger.info(
            f"{mask.sum()} buildings assigned default use 'Residential'."
        )

        # ---Step 4: Remove irrelevant properties
        classified_buildings = self._cleaning_osm_data(classified_buildings)

        # Final counts
        logger.info("Building use classification complete. Value counts:")
        logger.info(f"\n{classified_buildings['building_use'].value_counts(dropna=False)}")

        return classified_buildings

    def _cleaning_osm_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Cleans a GeoDataFrame extracted from OSM by:
        - Dropping unnecessary top-level columns.
        - Removing specified keys from nested 'tags' dict.
        - Flattening remaining 'tags' dict into top-level columns.

        Parameters:
        -----------
        gdf : GeoDataFrame
            Input GeoDataFrame with OSM-extracted data including a 'tags' column.

        Returns:
        --------
        GeoDataFrame
            Cleaned GeoDataFrame with flattened and filtered attributes.
        """
        # Unwanted top-level columns
        columns_to_drop = [
            "addr:city", "addr:country", "email", "opening_hours", "operator", "phone",
            "ref", "visible", "website", "internet_access", "source", "start_date", "wikipedia"
        ]

        # Unwanted keys within the 'tags' dict
        tag_keys_to_drop = [
            "addr:state", "air_conditioning", "alt_name", "check_date", "contact:facebook",
            "contact:instagram", "contact:linkedin", "contact:tiktok", "contact:twitter",
            "contact:youtube", "ele", "facebook", "fee", "gnis:feature_id", "layer", "museum",
            "opening_hours:url", "tourism", "twitter", "wheelchair", "wikidata"
        ]

        # Drop top-level columns if present
        gdf = gdf.drop(columns=[col for col in columns_to_drop if col in gdf.columns])

        # Process tags if present
        if 'tags' in gdf.columns:
            # Remove unwanted keys from each 'tags' dict
            gdf['tags'] = gdf['tags'].apply(
                lambda d: {k: v for k, v in d.items() if k not in tag_keys_to_drop}
                if isinstance(d, dict) else {}
            )

            # Flatten 'tags' into top-level columns
            tags_df = pd.json_normalize(gdf['tags'])
            tags_df.columns = [f"tag:{col}" for col in tags_df.columns]
            tags_df.index = gdf.index  # ensure alignment
            gdf = gdf.drop(columns='tags').join(tags_df)

        return gdf

    def calculate_free_walls(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
        if buildings is None or len(buildings) == 0:
            logger.warning("No buildings provided for free walls calculation")
            return buildings

        # Create a copy to avoid modifying the original
        buildings_with_walls = buildings.copy()

        # Reset index to ensure consistent indexing
        buildings_with_walls = buildings_with_walls.reset_index(drop=True)

        # Use the efficient neighbor detection method
        logger.info("Finding neighbors for free walls calculation")
        neighbors_dict = self._find_direct_neighbors(buildings_with_walls)

        # Convert neighbors dict to the format expected by the original logic
        buildings_with_walls['neighbors'] = buildings_with_walls.index.map(
            lambda idx: neighbors_dict.get(idx, [])
        )

        # TODO: Delete later Log neighbor distribution analysis
        neighbor_counts = buildings_with_walls['neighbors'].apply(len)
        buildings_with_neighbors = (neighbor_counts > 0).sum()
        total_buildings = len(buildings_with_walls)

        logger.info(f"Neighbor Analysis:")
        logger.info(f"  Total buildings: {total_buildings}")
        logger.info(
            f"  Buildings with neighbors: {buildings_with_neighbors} ({buildings_with_neighbors/total_buildings*100:.1f}%)")
        logger.info(
            f"  Buildings without neighbors: {total_buildings - buildings_with_neighbors} ({(total_buildings - buildings_with_neighbors)/total_buildings*100:.1f}%)")

        # Distribution breakdown
        neighbor_distribution = neighbor_counts.value_counts().sort_index()
        logger.info(f"  Neighbor count distribution:")
        for neighbor_count, building_count in neighbor_distribution.items():
            percentage = building_count / total_buildings * 100
            logger.info(
                f"    {neighbor_count} neighbors: {building_count} buildings ({percentage:.1f}%)")

        # Summary statistics
        avg_neighbors = neighbor_counts.mean()
        max_neighbors = neighbor_counts.max()
        logger.info(f"  Average neighbors per building: {avg_neighbors:.2f}")
        logger.info(f"  Maximum neighbors for any building: {max_neighbors}")

        # Calculate free walls (assuming max 4 walls per building)
        buildings_with_walls["free_walls"] = 4
        for index, row in buildings_with_walls.iterrows():
            neighbor_count = len(row["neighbors"])
            if neighbor_count < 4:
                buildings_with_walls.at[index, 'free_walls'] = 4 - neighbor_count
            else:
                buildings_with_walls.at[index, 'free_walls'] = 0

        logger.info(
            f"Free walls calculation complete:\n{buildings_with_walls['free_walls'].value_counts(dropna=False)}")

        return buildings_with_walls

    def classify_building_type(self, buildings: gpd.GeoDataFrame,
                               housing_data: Optional[Dict] = None) -> gpd.GeoDataFrame:
        """
        Classifies residential buildings into specific typologies based on
        neighborhood clustering and geometric relationships:
        - SFH (Single Family Home): Small isolated buildings
        - TH (Townhouse/Row House): Buildings in linear arrangements
        - MFH (Multi-Family Home): Medium-sized connected buildings
        - AB (Apartment Building): Large building clusters

        The classification uses a graph-based clustering approach where
        touching buildings are grouped together.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Residential building polygons with floor_area calculated
        housing_data : Dict, optional
            Reference housing type distribution for statistical adjustment

        Returns:
        --------
        GeoDataFrame : Buildings with 'building_type' and 'total_cluster_area' columns added
        """
        if buildings is None or len(buildings) == 0:
            logger.warning("No buildings provided for classification")
            return buildings

        # Create a copy to avoid modifying the original
        classified_buildings = buildings.copy()

        # Ensure we have necessary columns
        if 'floor_area' not in classified_buildings.columns:
            logger.error("Buildings must have 'floor_area' column")
            raise ValueError("Missing required 'floor_area' column")

        # Reset index to ensure consistent indexing
        classified_buildings = classified_buildings.reset_index(drop=True)

        # Step 1: Find direct neighbors (touching buildings)
        logger.info("Step 1: Finding direct neighbors for each building")
        neighbors_dict = self._find_direct_neighbors(classified_buildings)
        classified_buildings['neighbors'] = classified_buildings.index.map(
            lambda idx: neighbors_dict.get(idx, [])
        )

        # Step 2: Expand to full clusters (all connected buildings)
        logger.info("Step 2: Expanding to full building clusters")
        clusters_dict = self._expand_to_clusters(neighbors_dict)
        classified_buildings['cluster'] = classified_buildings.index.map(
            lambda idx: sorted(list(clusters_dict.get(idx, {idx})))
        )

        # Step 3: Calculate total cluster area
        logger.info("Step 3: Calculating total cluster areas")
        classified_buildings['total_cluster_area'] = classified_buildings.apply(
            lambda row: classified_buildings.loc[
                classified_buildings.index.isin(row['cluster']), 'floor_area'
            ].sum(),
            axis=1
        )

        # Step 4: Initial classification based on cluster characteristics
        logger.info("Step 4: Classifying building types based on cluster characteristics")
        classified_buildings = self._assign_building_types(classified_buildings)

        # Step 5: Statistical adjustment to match regional distribution
        # TODO: Implement statistical adjustment based on housing_data
        # This would involve:
        # 1. Calculate current distribution of building types
        # 2. Compare with target distribution from housing_data
        # 3. Reclassify borderline cases to match target percentages
        #
        # Example from legacy code:
        # res_types = classified_buildings.groupby('building_type').size()
        # res_types_percent = (res_types / res_types.sum() * 100).round()
        #
        # # Compare with target distribution
        # if housing_data:
        #     target_dist = housing_data.get('building_type_distribution', {})
        #     # Adjust classifications to match target...

        # Log final distribution
        type_counts = classified_buildings['building_type'].value_counts()
        logger.info(f"Building type distribution:\n{type_counts}")

        return classified_buildings

    def _find_direct_neighbors(self, buildings: gpd.GeoDataFrame) -> Dict[int, List[int]]:
        """
        Find all buildings that touch each other geometrically.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Building polygons with consistent indexing

        Returns:
        --------
        dict : Mapping of building index to list of neighbor indices
        """
        neighbors_dict = {}

        # Use spatial index for efficient neighbor detection
        spatial_index = buildings.sindex

        for idx, building in buildings.iterrows():
            # Get potential neighbors from spatial index
            possible_matches_index = list(spatial_index.intersection(building.geometry.bounds))
            possible_matches = buildings.iloc[possible_matches_index]

            # Check which buildings actually touch
            touching = possible_matches[
                possible_matches.geometry.touches(building.geometry)
            ].index.tolist()

            # Remove self from neighbors
            neighbors_dict[idx] = [n for n in touching if n != idx]

        return neighbors_dict

    def _expand_to_clusters(self, neighbors_dict: Dict[int, List[int]]) -> Dict[int, set]:
        """
        Expand direct neighbors to full clusters using graph traversal.
        Each building should know about all buildings in its connected component.

        Parameters:
        -----------
        neighbors_dict : dict
            Mapping of building index to list of direct neighbor indices

        Returns:
        --------
        dict : Mapping of building index to set of all connected building indices
        """
        clusters_dict = {}
        visited = set()

        def dfs_cluster(start_idx: int, current_cluster: set):
            """Depth-first search to find all connected buildings"""
            if start_idx in visited:
                return

            visited.add(start_idx)
            current_cluster.add(start_idx)

            # Recursively visit all neighbors
            for neighbor_idx in neighbors_dict.get(start_idx, []):
                if neighbor_idx not in visited:
                    dfs_cluster(neighbor_idx, current_cluster)

        # Find all clusters
        for building_idx in neighbors_dict:
            if building_idx not in visited:
                current_cluster = set()
                dfs_cluster(building_idx, current_cluster)

                # Assign this cluster to all buildings in it
                for idx in current_cluster:
                    clusters_dict[idx] = current_cluster

        return clusters_dict

    def _assign_building_types(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign building types based on cluster characteristics and geometric properties.

        Type assignment rules:
        - AB: Clusters with total area > 2000 m²
        - TH: Linear arrangements with 2 neighbors of similar size
        - SFH: Small isolated buildings or small clusters
        - MFH: Everything else

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings with cluster and neighbor information

        Returns:
        --------
        GeoDataFrame : Buildings with 'building_type' column added
        """
        buildings['building_type'] = None

        # Rule 1: Large clusters (> 2000 m²) are Apartment Buildings (AB)
        large_cluster_mask = buildings['total_cluster_area'] > 2000
        buildings.loc[large_cluster_mask, 'building_type'] = 'AB'

        # Propagate AB classification to all buildings in the same cluster
        ab_clusters = buildings[buildings['building_type'] == 'AB']['cluster'].tolist()
        for cluster in ab_clusters:
            cluster_mask = buildings['cluster'].apply(lambda x: x == cluster)
            buildings.loc[cluster_mask, 'building_type'] = 'AB'

        # Rule 2: Small isolated buildings are Single Family Homes (SFH)
        # - Area < 200 m² and no neighbors
        sfh_mask1 = (
            (buildings['floor_area'] < 200)
            & (buildings['neighbors'].apply(len) == 0)
            & (buildings['building_type'].isna())
        )
        buildings.loc[sfh_mask1, 'building_type'] = 'SFH'

        # - Area < 200 m² and only 2 buildings in cluster with total < 400 m²
        sfh_mask2 = (
            (buildings['floor_area'] < 200)
            & (buildings['cluster'].apply(len) == 2)
            & (buildings['total_cluster_area'] < 400)
            & (buildings['building_type'].isna())
        )
        buildings.loc[sfh_mask2, 'building_type'] = 'SFH'

        # Rule 3: Terraced houses (TH) - linear arrangements with similar-sized neighbors
        # Buildings with exactly 2 neighbors of similar size (within 10% difference)
        for idx, row in buildings[buildings['building_type'].isna()].iterrows():
            if len(row['neighbors']) == 2 and row['floor_area'] < 270:
                neighbor_areas = buildings.loc[row['neighbors'], 'floor_area'].values
                building_area = row['floor_area']

                # Check if neighbors are similar in size (within 10%)
                similar_size = all(
                    0.9 <= building_area / area <= 1.1 or 0.9 <= area / building_area <= 1.1
                    for area in neighbor_areas
                )

                if similar_size:
                    buildings.at[idx, 'building_type'] = 'TH'

        # Propagate TH classification to neighbors
        th_indices = buildings[buildings['building_type'] == 'TH'].index
        for idx in th_indices:
            neighbors = buildings.at[idx, 'neighbors']
            neighbor_mask = buildings.index.isin(neighbors) & buildings['building_type'].isna()
            buildings.loc[neighbor_mask, 'building_type'] = 'TH'

        # Rule 4: Everything else is Multi-Family Home (MFH)
        mfh_mask = buildings['building_type'].isna()
        buildings.loc[mfh_mask, 'building_type'] = 'MFH'

        # Additional refinement: Check for linear MFH that should be TH
        # This handles row houses that might have been missed
        for idx, row in buildings[buildings['building_type'] == 'MFH'].iterrows():
            if row['total_cluster_area'] < 1000 and len(row['cluster']) >= 3:
                # Check if buildings in cluster form a linear arrangement
                cluster_buildings = buildings.loc[
                    buildings.index.isin(row['cluster'])
                ]

                # Simple heuristic: if most buildings have exactly 2 neighbors, it's likely linear
                two_neighbor_count = (cluster_buildings['neighbors'].apply(len) == 2).sum()
                if two_neighbor_count >= len(cluster_buildings) * 0.6:
                    buildings.loc[
                        buildings.index.isin(row['cluster']), 'building_type'
                    ] = 'TH'

        return buildings

    def _allot_occupants(self, buildings: gpd.GeoDataFrame,
                         census_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Allocates household occupants to residential buildings based on
        building type, size, and census population data.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Residential buildings with building_type
        census_blocks : GeoDataFrame
            Census blocks with population data

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

    def _assign_building_id(self, buildings: gpd.GeoDataFrame,
                            census_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assigns unique building IDs based on a two-step census block assignment.
        Step 1: Assigns buildings fully 'within' a single census block.
        Step 2: For remaining buildings, assigns to the first block they 'intersect'.

        Building ID format: {STATEFP20}{COUNTYFP20}{BLOCKCE20}{SEQUENTIAL_NUMBER}
        """
        if buildings is None or len(buildings) == 0:
            logger.warning("No buildings provided for building ID assignment")
            buildings_copy = buildings.copy() if buildings is not None else gpd.GeoDataFrame()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            return buildings_copy

        if census_blocks is None or len(census_blocks) == 0:
            logger.warning(
                "No census blocks provided for building ID assignment. All buildings will be unassigned.")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            if not buildings_copy.empty:
                # Ensure output_dir exists before trying to save
                self.output_dir.mkdir(parents=True, exist_ok=True)
                buildings_copy.to_file(
                    self.output_dir
                    / "unassigned_buildings.geojson",
                    driver="GeoJSON")
            return buildings_copy

        required_cols = ['STATEFP20', 'COUNTYFP20', 'BLOCKCE20', 'geometry']
        missing_cols = [col for col in required_cols if col not in census_blocks.columns]
        if missing_cols:
            logger.error(
                f"Census blocks missing required columns: {missing_cols}. Cannot assign IDs.")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            if not buildings_copy.empty:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                buildings_copy.to_file(
                    self.output_dir
                    / "unassigned_buildings.geojson",
                    driver="GeoJSON")
            return buildings_copy

        logger.info(f"Original buildings CRS: {buildings.crs}")
        logger.info(f"Original census_blocks CRS: {census_blocks.crs}")

        target_crs = "EPSG:5070"
        try:
            buildings_proj = buildings.to_crs(target_crs)
            census_blocks_proj = census_blocks.to_crs(target_crs)
        except Exception as e:
            logger.error(f"Error during CRS projection to {target_crs}: {e}. Cannot assign IDs.")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            if not buildings_copy.empty:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                buildings_copy.to_file(
                    self.output_dir
                    / "unassigned_buildings.geojson",
                    driver="GeoJSON")
            return buildings_copy

        logger.info(
            f"Projected to {target_crs}. Buildings bounds: {buildings_proj.total_bounds}, Census blocks bounds: {census_blocks_proj.total_bounds}")

        # Use original buildings DataFrame index for tracking and final assignment
        buildings_with_ids = buildings.copy()
        buildings_with_ids['building_id'] = pd.NA
        buildings_with_ids['census_block_id'] = pd.NA

        # Store assignments as (original_building_index, state_fp, county_fp, block_ce)
        all_assignments_list = []  # Using a list of tuples/dicts first

        # --- Step 1: Assign buildings 'within' a census block ---
        logger.info("Step 1: Attempting to assign buildings 'within' census blocks.")
        # Use sjoin without suffixes since there are no duplicate column names
        sjoined_within = gpd.sjoin(
            buildings_proj,
            census_blocks_proj[['STATEFP20', 'COUNTYFP20', 'BLOCKCE20', 'geometry']],
            how='left',
            predicate='within'
        )

        assigned_in_step1_indices = set()
        # A building is 'within' if index_right is notna. Group by original building index.
        for original_building_idx, group in sjoined_within[sjoined_within['index_right'].notna()].groupby(
                level=0):
            if not group.empty:
                # If a building is somehow 'within' multiple blocks (geometrically unlikely for valid, non-overlapping blocks),
                # sjoin would create multiple rows. We take the first one.
                first_match = group.iloc[0]
                all_assignments_list.append({
                    'original_building_idx': original_building_idx,
                    'STATEFP20': first_match['STATEFP20'],
                    'COUNTYFP20': first_match['COUNTYFP20'],
                    'BLOCKCE20': first_match['BLOCKCE20']
                })
                assigned_in_step1_indices.add(original_building_idx)

        logger.info(f"{len(assigned_in_step1_indices)} buildings assigned using 'within' predicate.")

        # --- Identify unassigned after Step 1 and output ---
        # These are indices from the original `buildings` DataFrame
        unassigned_after_step1_original_indices = buildings.index.difference(
            assigned_in_step1_indices)
        unassigned_after_within_gdf = buildings.loc[unassigned_after_step1_original_indices].copy()

        if not unassigned_after_within_gdf.empty:
            logger.info(f"{len(unassigned_after_within_gdf)} buildings were not assigned in Step 1 ('within'). "
                        "Saving to 'unassigned_after_within_join.geojson'.")
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Save the original geometry and attributes, not projected
                unassigned_after_within_gdf.to_file(
                    self.output_dir / "04_unassigned_after_within_join.geojson", driver="GeoJSON")
            except Exception as e:
                logger.error(f"Failed to save 'unassigned_after_within_join.geojson': {e}")
        elif not buildings.empty:  # Only log if there were buildings to process
            logger.info("All buildings were assigned in Step 1 ('within').")

        # --- Step 2: Assign remaining buildings using 'intersects' (first match) ---
        if not unassigned_after_step1_original_indices.empty:
            # Work with the projected geometries of the unassigned buildings
            buildings_for_step2_proj = buildings_proj.loc[unassigned_after_step1_original_indices]
            logger.info(
                f"Step 2: Attempting to assign {len(buildings_for_step2_proj)} remaining buildings using 'intersects' (first match).")

            sjoined_intersects = gpd.sjoin(
                buildings_for_step2_proj,
                census_blocks_proj[['STATEFP20', 'COUNTYFP20', 'BLOCKCE20', 'geometry']],
                how='left',
                predicate='intersects'
            )
            # Print to file the sjoined_intersects:
            # sjoined_intersects.to_file(
            #     self.output_dir
            #     / "sjoined_intersects.geojson",
            #     driver="GeoJSON")

            assigned_in_step2_indices = set()
            for original_building_idx, group in sjoined_intersects[sjoined_intersects['index_right'].notna(
            )].groupby(level=0):
                if not group.empty:
                    first_intersect_match = group.iloc[0]
                    all_assignments_list.append({
                        'original_building_idx': original_building_idx,
                        'STATEFP20': first_intersect_match['STATEFP20'],
                        'COUNTYFP20': first_intersect_match['COUNTYFP20'],
                        'BLOCKCE20': first_intersect_match['BLOCKCE20']
                    })
                    assigned_in_step2_indices.add(original_building_idx)

            logger.info(
                f"{len(assigned_in_step2_indices)} buildings assigned using 'intersects' (first match) predicate in Step 2.")
        else:
            logger.info("No buildings remaining for Step 2 ('intersects' join).")

        # --- Consolidate all assignments and generate sequential IDs ---
        if all_assignments_list:
            assignments_df = pd.DataFrame(all_assignments_list)

            # Sort by original building index within each block for deterministic ID generation
            assignments_df = assignments_df.sort_values(
                by=['STATEFP20', 'COUNTYFP20', 'BLOCKCE20', 'original_building_idx'])
            assignments_df['sequential_id'] = assignments_df.groupby(
                ['STATEFP20', 'COUNTYFP20', 'BLOCKCE20']).cumcount() + 1

            for _, row in assignments_df.iterrows():
                original_idx = row['original_building_idx']
                state_fp = row['STATEFP20']
                county_fp = row['COUNTYFP20']
                block_ce = row['BLOCKCE20']
                seq_id = row['sequential_id']

                building_id_val = f"{state_fp}{county_fp}{block_ce}{seq_id:04d}"
                # Assign to the copy of the original DataFrame that we are modifying
                buildings_with_ids.loc[original_idx, 'building_id'] = building_id_val
                buildings_with_ids.loc[original_idx, 'census_block_id'] = block_ce
            logger.info(f"Generated building IDs for {len(assignments_df)} buildings.")
        else:
            logger.info("No assignments made in either step.")

        # --- Final removal of any buildings that are still unassigned ---
        final_unassigned_mask = buildings_with_ids['building_id'].isna()
        final_unassigned_count = final_unassigned_mask.sum()

        if final_unassigned_count > 0:
            logger.warning(f"Found {final_unassigned_count} buildings that could not be assigned a census block "
                           "after both 'within' and 'intersects' attempts. These will be removed.")
            # Save the original geometry and attributes of finally unassigned buildings
            unassigned_buildings_final = buildings_with_ids[final_unassigned_mask].copy()
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                unassigned_buildings_final.to_file(
                    self.output_dir / "unassigned_buildings.geojson", driver="GeoJSON")
                logger.info(
                    f"Saved {len(unassigned_buildings_final)} finally unassigned buildings to 'unassigned_buildings.geojson'.")
            except Exception as e:
                logger.error(f"Failed to save 'unassigned_buildings.geojson': {e}")

            buildings_with_ids = buildings_with_ids[~final_unassigned_mask].copy()

        total_assigned = len(buildings_with_ids)
        if len(buildings) > 0:
            success_rate = total_assigned / len(buildings) * 100
            logger.info(
                f"Building ID assignment complete: {total_assigned}/{len(buildings)} buildings assigned ({success_rate:.1f}% success).")
        else:
            logger.info("Building ID assignment complete: 0/0 buildings assigned.")

        return buildings_with_ids

    def _calculate_floor_height_from_osm_tags(
            self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract height information from OSM 'building:levels' tag.

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

        logger.info(
            f"Processing {len(buildings_with_height_floors)} buildings for OSM height and floor information")

        # Initial counts
        initial_height_count = buildings_with_height_floors['height'].notna().sum()
        initial_floors_count = buildings_with_height_floors['floors'].notna().sum()

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

        # Log after height parsing
        after_height_parsing = buildings_with_height_floors['height'].notna().sum()
        logger.info(
            f"After height parsing: {after_height_parsing} buildings have height data (+{after_height_parsing - initial_height_count})")

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

        # Log after floors parsing
        after_floors_parsing = buildings_with_height_floors['floors'].notna().sum()
        logger.info(
            f"After floors parsing: {after_floors_parsing} buildings have floor data (+{after_floors_parsing - initial_floors_count})")

        # Handle cases where we have height but no floors - estimate floors from
        # height (2.5m per floor)
        height_no_floors_mask = (
            buildings_with_height_floors['height'].notna()) & (
            buildings_with_height_floors['floors'].isna())  # Fixed mask logic

        if height_no_floors_mask.any():
            # Estimate floors using 2.5m per floor
            estimated_floors = buildings_with_height_floors.loc[height_no_floors_mask, 'height'] / 2.5
            buildings_with_height_floors.loc[height_no_floors_mask, 'floors'] = estimated_floors.apply(
                lambda x: max(1, int(round(x))))
            logger.info(
                f"Estimated floors from height for {height_no_floors_mask.sum()} buildings")

        # Handle cases where we have floors but no height - estimate height from
        # floors (2.5m per floor)
        floors_no_height_mask = (
            buildings_with_height_floors['height'].isna()) & (
            buildings_with_height_floors['floors'].notna())  # Fixed mask logic

        if floors_no_height_mask.any():
            # Estimate height using 2.5m per floor
            buildings_with_height_floors.loc[floors_no_height_mask,
                                             'height'] = buildings_with_height_floors.loc[floors_no_height_mask,
                                                                                          'floors'] * 2.5
            logger.info(
                f"Estimated height from floors for {floors_no_height_mask.sum()} buildings")

        # Add minimum level adjustment if available
        if 'building:min_level' in buildings_with_height_floors.columns:
            min_level_mask = buildings_with_height_floors['building:min_level'].notna()
            if min_level_mask.any():
                logger.debug(
                    f"Found {min_level_mask.sum()} buildings with minimum level information")

        # Final summary statistics (no spam logs)
        final_height_count = buildings_with_height_floors['height'].notna().sum()
        final_floors_count = buildings_with_height_floors['floors'].notna().sum()
        total_buildings = len(buildings_with_height_floors)

        logger.info(f"OSM tag processing complete:")
        logger.info(
            f"  Height data: {final_height_count}/{total_buildings} buildings (+{final_height_count - initial_height_count})")
        logger.info(
            f"  Floor data: {final_floors_count}/{total_buildings} buildings (+{final_floors_count - initial_floors_count})")

        return buildings_with_height_floors

    def _calculate_floor_height_from_ms_buildings(
        self, buildings: gpd.GeoDataFrame, microsoft_buildings: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Extract height information from Microsoft Buildings data via centroid-based spatial join.

        This method uses MS building centroids to find which OSM building they fall within,
        providing a more robust one-to-many matching approach.

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
                buildings_copy['floors'] = None
            return buildings_copy

        logger.debug(
            f"Extracting height data for {len(buildings)} buildings from {len(microsoft_buildings)} MS building footprints")
        logger.debug(f"Microsoft Buildings columns: {list(microsoft_buildings.columns)}")

        # Create working copy
        buildings_with_ms_height = buildings.copy()
        ms_buildings_copy = microsoft_buildings.copy()

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

        # Project ms_building to osm_building crs
        ms_buildings_projected = ms_buildings_copy.to_crs(buildings_with_ms_height.crs)
        buildings_projected = buildings_with_ms_height.copy()

        # Filter out invalid height values (na or negative values)
        height_mask = (ms_buildings_projected['height'].notna()) & \
            (pd.to_numeric(ms_buildings_projected['height'], errors='coerce') > 0)

        valid_ms_buildings = ms_buildings_projected[height_mask].copy()

        if len(valid_ms_buildings) == 0:
            logger.debug(
                "No valid height data found in Microsoft Buildings (all heights are -1 or invalid)")
            return buildings_with_ms_height

        logger.info(
            f"Found {len(valid_ms_buildings)} MS buildings with valid height data (filtered from {len(ms_buildings_projected)})")

        # Convert height to numeric if it's not already
        valid_ms_buildings['height'] = pd.to_numeric(valid_ms_buildings['height'], errors='coerce')

        # Calculate centroids of MS buildings
        logger.debug("Calculating centroids of Microsoft Buildings for robust spatial matching")
        ms_buildings_centroids = valid_ms_buildings.copy()
        ms_buildings_centroids['centroid'] = ms_buildings_centroids.geometry.centroid

        # Create a new GeoDataFrame with centroids as geometry
        ms_centroids_gdf = gpd.GeoDataFrame(
            ms_buildings_centroids.drop('geometry', axis=1),
            geometry=ms_buildings_centroids['centroid'],
            crs=ms_buildings_centroids.crs
        )

        # Keep original MS building geometry for visualization
        ms_centroids_gdf['ms_geometry'] = ms_buildings_centroids.geometry

        # Apply confidence threshold before spatial join
        if 'confidence' in ms_centroids_gdf.columns:
            confidence_threshold = 0.5
            high_confidence_mask = pd.to_numeric(
                ms_centroids_gdf['confidence'],
                errors='coerce') >= confidence_threshold
            ms_centroids_filtered = ms_centroids_gdf[high_confidence_mask]
            logger.info(
                f"After confidence filtering (>{confidence_threshold}): {len(ms_centroids_filtered)} high-confidence MS buildings")
        else:
            ms_centroids_filtered = ms_centroids_gdf

        # Perform centroid-based spatial join
        try:
            logger.debug("Performing centroid-based spatial join")
            joined = gpd.sjoin(
                ms_centroids_filtered,  # MS building centroids (left)
                buildings_projected,    # OSM buildings (right)
                how='inner',
                predicate='within',
                lsuffix='ms',          # MS building columns get _ms suffix
                rsuffix='osm'          # OSM building columns get _osm suffix
            )
            logger.info(f"Length of ms_centroids_filtered: {len(ms_centroids_filtered)}")
            logger.info(f"Length of buildings_projected: {len(buildings_projected)}")
            logger.info(
                f"Centroid-based spatial join results: {len(joined)} MS building centroids matched with OSM buildings")

            # Debug: Check what columns are actually in the joined result
            logger.info(f"Joined DataFrame columns: {list(joined.columns)}")
            logger.info(f"Joined DataFrame index name: {joined.index.name}")
            logger.info(f"Sample of joined data:\n{joined.head()}")

            if len(joined) == 0:
                logger.debug("No MS building centroids fall within OSM buildings")
                return buildings_with_ms_height

            # Calculate statistics per OSM building using MS height data
            agg_dict = {'height_ms': ['mean', 'count']}
            if 'confidence_ms' in joined.columns:
                agg_dict['confidence_ms'] = 'mean'

            # Use index_osm to group by OSM building indices (not index_right due to rsuffix='osm')
            osm_stats = joined.groupby('index_osm').agg(agg_dict).round(2)
            osm_stats.columns = ['avg_height', 'ms_count'] + \
                (['avg_confidence'] if 'confidence_ms' in joined.columns else [])

            logger.info(f"Calculated height data for {len(osm_stats)} OSM buildings from MS data")
            logger.info(
                f"Average MS buildings per OSM building: {osm_stats['ms_count'].mean():.1f}")

            # Now use the correct OSM building indices
            osm_building_indices = osm_stats.index  # These are now OSM building indices!
            buildings_with_ms_height.loc[osm_building_indices, 'height'] = osm_stats['avg_height']

            # Calculate floors with validation
            floors = (
                osm_stats['avg_height']
                / 2.5).round().astype(int).clip(
                lower=1)  # Using 2.5m as requested
            buildings_with_ms_height.loc[osm_building_indices, 'floors'] = floors

            assigned_count = len(osm_stats)
            logger.info(f"Successfully assigned MS height data to {assigned_count} OSM buildings")

        except KeyError as e:
            logger.warning(f"Missing expected column: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in MS Buildings processing: {e}")

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

        # Initialize height and floors columns if they don't exist
        if 'height' not in buildings.columns:
            buildings['height'] = None
        if 'floors' not in buildings.columns:
            buildings['floors'] = None

        # Step 1: Extract height and floor information from Microsoft Buildings data
        logger.info("Step 1: Extracting height and floors from Microsoft Buildings data")

        # Only process buildings which do not have height data yet:
        buildings_without_height = buildings[buildings['height'].isna()]

        buildings_with_floors = self._calculate_floor_height_from_ms_buildings(
            buildings_without_height, microsoft_buildings)
        buildings_with_floors.to_file(
            self.output_dir
            / "ms_buildings_with_floors_fucntion_output.geojson",
            driver='GeoJSON')

        # Step 1: Extract height and floor information from OSM tags
        logger.info("Step 2: Extracting height and floors from OSM tags")
        # Filter out buildings that have height and floors from MS data before osm
        # tags extraction to not overwrite:
        undetermined_buildings = buildings_with_floors[buildings_with_floors['height'].isna(
        ) | buildings_with_floors['floors'].isna()]
        logger.info(f"Buildings needing OSM tag processing: {len(undetermined_buildings)}")

        if len(undetermined_buildings) > 0:
            processed_undetermined = self._calculate_floor_height_from_osm_tags(
                undetermined_buildings)
            # Update only the processed buildings back into the main dataset
            buildings_with_floors.loc[processed_undetermined.index] = processed_undetermined

        # Log final counts after OSM processing
        height_count = buildings_with_floors['height'].notna().sum()
        floors_count = buildings_with_floors['floors'].notna().sum()
        total_buildings = len(buildings_with_floors)
        logger.info(
            f"After OSM processing: {height_count}/{total_buildings} buildings have height data, {floors_count}/{total_buildings} have floor data")

        # # TODO: Move this to separate function after population count is allocated.
        # Step 2: For buildings without OSM data, estimate from occupants and area
        # logger.debug("Step 2: Estimating floors from occupants and area for remaining buildings")

        # # Only process buildings that still don't have floor data (floors == 1 and no height)
        # needs_estimation_mask = (
        #     buildings_with_floors['floors'].isna()) | (
        #     buildings_with_floors['height'].isna())

        # logger.info(f"Length of buildings_with_floors: {len(buildings_with_floors)}")

        # if needs_estimation_mask.any():
        #     logger.debug(
        # f"Estimating floors for {needs_estimation_mask.sum()} buildings without
        # OSM data")

        #     for idx, building in buildings_with_floors[needs_estimation_mask].iterrows():
        #         # Check if required columns exist for estimation
        #         if 'building_type' not in building or 'occupants' not in building:
        #             logger.debug(
        #                 f"Skipping estimation for building {idx}: missing building_type or occupants")
        #             continue

        #         building_type = building['building_type']
        #         if pd.isna(building['floor_area']):
        #             raise ValueError(f"Building {idx} has no floor area")
        #         else:
        #             area = building['floor_area']
        #         occupants = building['occupants'] if pd.notna(
        #             building['occupants']) else 2

        #         if building_type == 'SFH':
        #             # SFH: Usually 1-3 floors
        #             if area < 100:
        #                 buildings_with_floors.at[idx, 'floors'] = 1
        #             elif area < 200:
        #                 buildings_with_floors.at[idx, 'floors'] = 2
        #             else:
        #                 buildings_with_floors.at[idx, 'floors'] = min(3, int(occupants / 3) + 1)

        #         elif building_type == 'TH':
        #             # TH: Usually 2-3 floors
        #             buildings_with_floors.at[idx, 'floors'] = min(
        #                 3, max(2, int(occupants / 3) + 1))

        #         elif building_type == 'MFH':
        #             # MFH: Usually 2-5 floors
        #             buildings_with_floors.at[idx, 'floors'] = min(
        #                 5, max(2, int(occupants / 6) + 1))

        #         elif building_type == 'AB':
        #             # AB: Usually 4-10 floors
        #             buildings_with_floors.at[idx, 'floors'] = min(
        #                 10, max(4, int(occupants / 8) + 1))
        #         else:
        #             # Default case for unknown building types
        #             buildings_with_floors.at[idx, 'floors'] = max(1, int(occupants / 4) + 1)

        # # Step 3: Calculate height for buildings that still don't have height data
        # missing_height_mask = buildings_with_floors['height'].isna()
        # if missing_height_mask.any():
        #     logger.debug(
        #         f"Estimating height for {missing_height_mask.sum()} buildings without height data")
        #     buildings_with_floors.loc[missing_height_mask,
        #                               'height'] = buildings_with_floors.loc[missing_height_mask,
        #                                                                     'floors'] * 2.5

        # # Convert height to numeric, handling any string values
        # buildings_with_floors['height'] = pd.to_numeric(
        #     buildings_with_floors['height'], errors='coerce')

        # # Log buildings with valid height measurements before clipping
        # valid_height_before = buildings_with_floors['height'].notna().sum()
        # buildings_below_min_floors = (buildings_with_floors['floors'] < 1).sum()
        # buildings_below_min_height = (buildings_with_floors['height'] < 2.5).sum()

        # logger.info(f"Buildings with valid height measurements: {valid_height_before}")
        # logger.info(f"Buildings with floors < 1 (will be clipped): {buildings_below_min_floors}")
        # logger.info(f"Buildings with height < 2.5m (will be clipped): {buildings_below_min_height}")

        # # Ensure minimum values
        # buildings_with_floors['floors'] = buildings_with_floors['floors'].clip(lower=1)
        # buildings_with_floors['height'] = buildings_with_floors['height'].clip(
        #     lower=2.5)  # minimum reasonable height

        # # Log final statistics
        # final_valid_height = buildings_with_floors['height'].notna().sum()
        # final_avg_floors = buildings_with_floors['floors'].mean()
        # final_avg_height = buildings_with_floors['height'].mean()

        # logger.info(f"Final buildings with height data: {final_valid_height}/{len(buildings_with_floors)}")
        # logger.info(f"Average floors after processing: {final_avg_floors:.2f}")
        # logger.info(f"Average height after processing: {final_avg_height:.2f}m")

        return buildings_with_floors

    # def allot_construction_year(self, buildings: gpd.GeoDataFrame,
    #                             housing_age_data: Optional[Dict] = None) -> gpd.GeoDataFrame:
    #     """
    #     Assigns construction year periods to buildings based on
    #     available data and statistical distribution.

    #     Parameters:
    #     -----------
    #     buildings : GeoDataFrame
    #         Buildings with type and other attributes
    #     housing_age_data : Dict, optional
    #         Statistical data on building age distribution by region

    #     Returns:
    #     --------
    #     GeoDataFrame : Buildings with 'construction_year' column added
    #     """
    #     if buildings is None or len(buildings) == 0:
    #         return buildings

    #     # Create a copy to avoid modifying the original
    #     buildings_with_year = buildings.copy()

    #     # Initialize construction_year column
    #     buildings_with_year['construction_year'] = None

    #     # 1. Use direct OSM data if available
    #     logger.debug("Using OSM year or start_date tags if available")
    #     if 'start_date' in buildings_with_year.columns:
    #         mask = buildings_with_year['start_date'].notna()
    #         buildings_with_year.loc[mask,
    #                                 'construction_year'] = buildings_with_year.loc[mask,
    #                                                                                'start_date']

    #     # 2. Use spatial reference data
    #     logger.debug("Using spatial reference data for construction year")
    #     # This would spatially join with assessor or historical data

    #     # 3. Apply neighborhood consistency patterns
    #     logger.debug("Applying neighborhood consistency patterns")
    #     # This would cluster buildings and apply consistent ages within neighborhoods

    #     # 4. Allocate remaining buildings based on statistical distribution
    #     logger.debug("Allocating construction year based on statistical distribution")

    #     # Common categories for US buildings
    #     periods = ['Pre-1950', '1950-1969', '1970-1989', '1990-2009', '2010-present']

    #     # Default distribution if no reference data provided
    #     distribution = {'Pre-1950': 0.2, '1950-1969': 0.2, '1970-1989': 0.25,
    #                     '1990-2009': 0.25, '2010-present': 0.1}

    #     # If housing_age_data is provided, use it to determine distribution
    #     if housing_age_data:
    #         # Parse housing_age_data to update distribution
    #         pass

    #     # Apply distribution to buildings with missing construction_year
    #     mask = buildings_with_year['construction_year'].isna()
    #     num_to_assign = mask.sum()

    #     if num_to_assign > 0:
    #         # Calculate the number of buildings for each period
    #         period_counts = {period: int(num_to_assign * distribution[period])
    #                          for period in periods}

    #         # Adjust to make sure we assign all buildings
    #         total_assigned = sum(period_counts.values())
    #         if total_assigned < num_to_assign:
    #             # Add remaining to most common period
    #             most_common = max(distribution, key=distribution.get)
    #             period_counts[most_common] += num_to_assign - total_assigned

    #         # Create a list of periods to assign
    #         periods_to_assign = []
    #         for period, count in period_counts.items():
    #             periods_to_assign.extend([period] * count)

    #         # Shuffle the periods
    #         np.random.shuffle(periods_to_assign)

    #         # Assign periods to buildings
    #         buildings_with_year.loc[mask, 'construction_year'] = periods_to_assign

    #     # 5. Add confidence indicator for the source of the year data
    #     # This is handled by the default confidence score

    #     return buildings_with_year

    # def allot_refurbishment_level(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    #     """
    #     Assigns refurbishment level indicators to buildings based on
    #     age, type, and statistical patterns.

    #     For energy modeling, this indicates upgrades to:
    #     - Walls (insulation)
    #     - Roof
    #     - Windows
    #     - Basement/Foundation

    #     Parameters:
    #     -----------
    #     buildings : GeoDataFrame
    #         Buildings with type, year, and other attributes

    #     Returns:
    #     --------
    #     GeoDataFrame : Buildings with refurbishment indicators added
    #     """
    #     if buildings is None or len(buildings) == 0:
    #         return buildings

    #     # Create a copy to avoid modifying the original
    #     buildings_with_refurbishment = buildings.copy()

    #     # Initialize refurbishment columns
    #     buildings_with_refurbishment['refurb_walls'] = 0
    #     buildings_with_refurbishment['refurb_roof'] = 0
    #     buildings_with_refurbishment['refurb_windows'] = 0
    #     buildings_with_refurbishment['refurb_basement'] = 0

    #     # 1. Assign probability of refurbishment based on building age
    #     logger.debug("Assigning refurbishment probabilities based on age")

    #     for idx, building in buildings_with_refurbishment.iterrows():
    #         refurb_prob = 0.0

    #         # Base probability on construction year
    #         if building['construction_year'] == 'Pre-1950':
    #             refurb_prob = 0.9  # Older buildings very likely to have been refurbished
    #         elif building['construction_year'] == '1950-1969':
    #             refurb_prob = 0.8
    #         elif building['construction_year'] == '1970-1989':
    #             refurb_prob = 0.6
    #         elif building['construction_year'] == '1990-2009':
    #             refurb_prob = 0.3
    #         elif building['construction_year'] == '2010-present':
    #             refurb_prob = 0.0  # New buildings unlikely to be refurbished

    #         # Adjust based on building type
    #         if building['building_type'] == 'SFH':
    #             refurb_prob *= 1.1  # SFH slightly more likely to be refurbished
    #         elif building['building_type'] == 'AB':
    #             refurb_prob *= 0.9  # AB slightly less likely to be refurbished

    #         # Cap probability at 1.0
    #         refurb_prob = min(1.0, refurb_prob)

    #         # Assign specific refurbishment components
    #         buildings_with_refurbishment.at[idx,
    #                                         'refurb_walls'] = 1 if np.random.random() < refurb_prob else 0
    #         buildings_with_refurbishment.at[idx,
    #                                         'refurb_roof'] = 1 if np.random.random() < refurb_prob else 0
    #         buildings_with_refurbishment.at[idx,
    #                                         'refurb_windows'] = 1 if np.random.random() < refurb_prob else 0
    #         buildings_with_refurbishment.at[idx, 'refurb_basement'] = 1 if np.random.random(
    #         ) < refurb_prob * 0.7 else 0

    #     # 2. Consider neighborhood effects
    #     logger.debug("Considering neighborhood effects on refurbishment")
    #     # This would cluster buildings and apply consistency within neighborhoods

    #     return buildings_with_refurbishment

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

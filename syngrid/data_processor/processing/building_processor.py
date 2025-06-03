from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from syngrid.data_processor.processing.building_schema import (
    NonResidentialBuildingOutput, ResidentialBuildingOutput,)
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
                microsoft_buildings_data: Dict, nrel_vintage_distribution: Dict) -> Dict[str, str]:
        """
        Main method that orchestrates the entire building classification process.

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
        other = all_buildings[all_buildings['building_use'] != 'residential'].copy()

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

            # Allot construction year
            residential = self._allot_construction_year(
                residential,
                nrel_vintage_distribution
            )
            residential.to_file(self.output_dir
                                / "07_residential_buildings_with_construction_year.geojson")

            if len(residential) > 0:
                # Write residential output
                residential_output_path = self.write_buildings_output(
                    residential,
                    self.output_dir,
                    'residential_buildings.shp',
                    'residential'
                )
                logger.info(f"Residential buildings saved to: {residential_output_path}")
            else:
                logger.warning("No residential buildings found")

        if len(other) > 0:
            logger.info(f"Processing {len(other)} non-residential buildings")

            if len(other) > 0:
                # Write non-residential output
                other_output_path = self.write_buildings_output(
                    other,
                    self.output_dir,
                    'non_residential_buildings.shp',
                    'non_residential'
                )
                logger.info(f"Non-residential buildings saved to: {other_output_path}")
            else:
                logger.warning("No non-residential buildings found")

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

        # --- Step 0: Exclusions ---
        # Filter out common non-building structures or utility infrastructure
        total_excluded = 0

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
            excluded_count = mask.sum()
            total_excluded += excluded_count
            classified_buildings = classified_buildings[~mask]
            logger.info(
                f"Removed {excluded_count} non-habitable structures based on 'building' tag.")

        if 'power' in classified_buildings.columns:
            exclude_power_values = [
                'transformer',
                'substation',
                'pole',
                'tower',
                'portal',
                'catenary_mast']
            mask = classified_buildings['power'].isin(exclude_power_values)
            excluded_count = mask.sum()
            total_excluded += excluded_count
            classified_buildings = classified_buildings[~mask]
            logger.info(f"Removed {excluded_count} power infrastructure buildings.")

        logger.info(f"Total excluded buildings removed: {total_excluded}")
        logger.info(f"Remaining buildings for classification: {len(classified_buildings)}")

        # Initialize 'building_use' column for remaining buildings
        classified_buildings['building_use'] = pd.NA

        # Check if any buildings remain for classification
        if classified_buildings.empty:
            logger.info("No buildings remaining after exclusions.")
            return classified_buildings

        # All remaining buildings are candidates for classification
        candidate_buildings = classified_buildings.copy()

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
            f"{mask.sum()} buildings assigned default use 'residential'."
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
        Allocates household occupants and housing units to residential buildings
        based on building type, size, and census population data.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Residential buildings with columns: ['building_type', 'census_block_id', 'floor_area']
        census_blocks : GeoDataFrame
            Census blocks with columns: ['GEOID20', 'POP20', 'HOUSING20']

        Returns:
        --------
        GeoDataFrame : Same as input, with 'occupants' and 'housing_units' columns filled.
        """

        if buildings is None or buildings.empty:
            return buildings

        if census_blocks is None or census_blocks.empty:
            logger.warning("No census blocks provided for occupant allocation.")
            buildings_copy = buildings.copy()
            buildings_copy['occupants'] = 0
            buildings_copy['housing_units'] = 0.0
            return buildings_copy

        # Copy to avoid modifying original
        buildings_with_occupants = buildings.copy()

        # Initialize output columns
        buildings_with_occupants['occupants'] = 0
        buildings_with_occupants['housing_units'] = 0.0

        logger.info("Starting census-based occupant allocation")

        for _, census_block in census_blocks.iterrows():
            geoid = census_block['GEOID20']
            total_pop = census_block.get('POP20', 0)
            total_units = census_block.get('HOUSING20', 0)

            # Get buildings in this census block
            mask = buildings_with_occupants['census_block_id'] == geoid
            block_buildings = buildings_with_occupants[mask]

            if block_buildings.empty:
                logger.debug(f"No buildings found for census block {geoid}, skipping.")
                continue

            # Handle special case where population is missing/invalid
            if total_pop <= 0:
                logger.debug(f"No population data for census block {geoid}, skipping.")
                continue

            # Calculate initial capacity for each building
            def calculate_initial_capacity(row):
                t = row['building_type']
                a = row['floor_area']
                if t == 'SFH':
                    return 2 * max(6, (2 * a / 50))
                elif t == 'TH':
                    return 2 * max(3, (2 * a / 50))
                elif t == 'MFH':
                    return 2 * max(36, (3 * a / 50))
                elif t == 'AB':
                    return 2 * max(1000, (10 * a / 50))
                else:
                    return max(1, a / 100)  # fallback

            block_buildings = block_buildings.copy()
            block_buildings['initial_capacity'] = block_buildings.apply(
                calculate_initial_capacity, axis=1)
            block_buildings['max_capacity'] = block_buildings['initial_capacity'].copy()
            block_buildings['occupants'] = 0
            block_buildings['housing_units'] = 0

            # Count building types
            type_counts = block_buildings['building_type'].value_counts()
            ab_count = type_counts.get('AB', 0)
            mfh_count = type_counts.get('MFH', 0)
            th_count = type_counts.get('TH', 0)
            sfh_count = type_counts.get('SFH', 0)

            # Sort buildings by area (largest first) within each type for priority allocation
            building_priority_order = []
            for building_type in ['AB', 'MFH', 'TH', 'SFH']:
                type_buildings = block_buildings[block_buildings['building_type'] == building_type]
                type_buildings_sorted = type_buildings.sort_values('floor_area', ascending=False)
                building_priority_order.extend(type_buildings_sorted.index.tolist())

            # Calculate total initial capacity
            total_initial_capacity = block_buildings['max_capacity'].sum()
            remaining_population = int(total_pop)

            logger.debug(f"Block {geoid}: {remaining_population} people, "
                         f"{total_initial_capacity:.0f} initial capacity, "
                         f"AB:{ab_count}, MFH:{mfh_count}, TH:{th_count}, SFH:{sfh_count}")

            # Phase 1: Capacity Adjustment (if needed)
            if remaining_population > total_initial_capacity and ab_count == 0:
                logger.debug(
                    f"Block {geoid}: Population exceeds capacity, adjusting building capacities")

                # Increase MFH capacity first
                if mfh_count > 0:
                    while total_initial_capacity < remaining_population:
                        for idx in block_buildings[block_buildings['building_type']
                                                   == 'MFH'].index:
                            block_buildings.at[idx, 'max_capacity'] += 1
                        total_initial_capacity = block_buildings['max_capacity'].sum()
                        if total_initial_capacity >= remaining_population:
                            break

                # Then increase SFH capacity if still needed
                if sfh_count > 0 and total_initial_capacity < remaining_population:
                    while total_initial_capacity < remaining_population:
                        for idx in block_buildings[block_buildings['building_type']
                                                   == 'SFH'].index:
                            block_buildings.at[idx, 'max_capacity'] += 1
                        total_initial_capacity = block_buildings['max_capacity'].sum()
                        if total_initial_capacity >= remaining_population:
                            break

                # Finally increase TH capacity if still needed
                if th_count > 0 and total_initial_capacity < remaining_population:
                    while total_initial_capacity < remaining_population:
                        for idx in block_buildings[block_buildings['building_type'] == 'TH'].index:
                            block_buildings.at[idx, 'max_capacity'] += 1
                        total_initial_capacity = block_buildings['max_capacity'].sum()
                        if total_initial_capacity >= remaining_population:
                            break

            # Phase 2: Incremental Population Allocation
            logger.debug(
                f"Block {geoid}: Starting incremental allocation of {remaining_population} people")

            # Allocate population incrementally, one person at a time
            iteration_count = 0
            max_iterations = remaining_population * 2  # Safety limit

            while remaining_population > 0 and iteration_count < max_iterations:
                allocated_this_round = False

                # Go through buildings in priority order
                for building_idx in building_priority_order:
                    if remaining_population <= 0:
                        break

                    current_occupants = block_buildings.at[building_idx, 'occupants']
                    max_capacity = block_buildings.at[building_idx, 'max_capacity']

                    if current_occupants < max_capacity:
                        block_buildings.at[building_idx, 'occupants'] += 1
                        remaining_population -= 1
                        allocated_this_round = True

                if not allocated_this_round:
                    logger.warning(
                        f"Block {geoid}: Could not allocate remaining {remaining_population} people")
                    break

                iteration_count += 1

            # Phase 3: Calculate Housing Units
            # Use proportional allocation for housing units based on occupants
            total_allocated_population = block_buildings['occupants'].sum()
            if total_allocated_population > 0 and total_units > 0:
                for idx in block_buildings.index:
                    occupant_ratio = block_buildings.at[idx,
                                                        'occupants'] / total_allocated_population
                    block_buildings.at[idx, 'housing_units'] = int(
                        round(occupant_ratio * total_units))
            else:
                # Fallback: estimate housing units from occupants
                block_buildings['housing_units'] = (
                    block_buildings['occupants'] / 2.6).round().astype(int)

            # Update the master dataframe
            buildings_with_occupants.loc[mask, 'occupants'] = block_buildings['occupants']
            buildings_with_occupants.loc[mask,
                                         'housing_units'] = block_buildings['housing_units'].astype(int)

            # Log allocation results
            final_allocated = block_buildings['occupants'].sum()
            logger.debug(f"Block {geoid}: Allocated {final_allocated}/{total_pop} people "
                         f"({final_allocated/total_pop*100:.1f}%)")

        # Phase 4: Statistical Analysis and Remaining Population Allocation
        logger.info("Performing statistical analysis and final allocation adjustments")

        # Calculate statistics by building type for buildings with occupants
        building_stats = {}
        for building_type in ['SFH', 'TH', 'MFH', 'AB']:
            type_buildings = buildings_with_occupants[
                (buildings_with_occupants['building_type'] == building_type)
                & (buildings_with_occupants['occupants'] > 0)
            ]
            if len(type_buildings) > 0:
                mean_occupants = type_buildings['occupants'].mean()
                std_occupants = type_buildings['occupants'].std()
                if pd.isna(std_occupants):
                    std_occupants = 0
                max_occupants = mean_occupants + 2 * std_occupants
                building_stats[building_type] = {
                    'mean': mean_occupants,
                    'std': std_occupants,
                    'max_statistical': max(max_occupants, mean_occupants)
                }
            else:
                # Default values if no buildings of this type have occupants
                building_stats[building_type] = {
                    'mean': 2.0,
                    'std': 1.0,
                    'max_statistical': 4.0
                }

        # Allocate remaining population to buildings with 0 occupants
        zero_occupant_buildings = buildings_with_occupants[buildings_with_occupants['occupants'] == 0]
        if len(zero_occupant_buildings) > 0:
            logger.info(
                f"Allocating population to {len(zero_occupant_buildings)} buildings with 0 occupants")

            for idx in zero_occupant_buildings.index:
                building_type = buildings_with_occupants.at[idx, 'building_type']
                if building_type in building_stats:
                    estimated_occupants = max(
                        1, int(building_stats[building_type]['max_statistical'] / 4))
                    buildings_with_occupants.at[idx, 'occupants'] = estimated_occupants
                    # Estimate housing units and cast to int
                    buildings_with_occupants.at[idx, 'housing_units'] = max(
                        1, int(round(estimated_occupants / 2.6)))

        # Final capacity check - ensure no building exceeds statistical maximum
        for idx in buildings_with_occupants.index:
            building_type = buildings_with_occupants.at[idx, 'building_type']
            if building_type in building_stats:
                max_allowed = int(building_stats[building_type]['max_statistical'])
                current_occupants = buildings_with_occupants.at[idx, 'occupants']
                if current_occupants > max_allowed:
                    buildings_with_occupants.at[idx, 'occupants'] = max_allowed
                    buildings_with_occupants.at[idx, 'housing_units'] = max(
                        1, int(round(max_allowed / 2.6)))

        logger.info("Completed census-based occupant allocation.")
        return buildings_with_occupants

    def evaluate_census_block_allocation(self, buildings: gpd.GeoDataFrame,
                                         geoid20: str) -> pd.DataFrame:
        """
        Evaluate and summarize building allocation results for a specific census block.

        This function provides detailed analysis of how population and housing units
        were allocated to buildings within a given census block, useful for
        validation and debugging.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings with allocation results (must include census_block_id column)
        geoid20 : str
            Census block GEOID20 to analyze

        Returns:
        --------
        DataFrame : Summary of buildings in the census block with key metrics
        """
        if buildings is None or buildings.empty:
            logger.warning("No buildings provided for evaluation")
            return pd.DataFrame()

        if 'census_block_id' not in buildings.columns:
            logger.error("Buildings must have 'census_block_id' column for evaluation")
            return pd.DataFrame()

        # Filter buildings for the specified census block
        block_buildings = buildings[buildings['census_block_id'] == geoid20].copy()

        if block_buildings.empty:
            logger.warning(f"No buildings found for census block {geoid20}")
            return pd.DataFrame()

        # Select relevant columns for evaluation
        eval_columns = [
            'building_id',
            'building_use',
            'building_type',
            'floor_area',
            'floors',
            'height',
            'occupants',
            'housing_units',
            'free_walls'
        ]

        # Only include columns that exist in the data
        available_columns = [col for col in eval_columns if col in block_buildings.columns]

        # Create evaluation dataframe
        eval_df = block_buildings[available_columns].copy()

        # Add derived metrics
        if 'occupants' in eval_df.columns and 'housing_units' in eval_df.columns:
            eval_df['people_per_unit'] = eval_df['occupants'] / \
                eval_df['housing_units'].replace(0, pd.NA)

        if 'occupants' in eval_df.columns and 'floor_area' in eval_df.columns:
            eval_df['people_per_sqm'] = eval_df['occupants'] / eval_df['floor_area']

        # Reset index and sort by building type and size
        eval_df = eval_df.reset_index(drop=True)
        if 'building_type' in eval_df.columns and 'floor_area' in eval_df.columns:
            eval_df = eval_df.sort_values(['building_type', 'floor_area'], ascending=[True, False])

        # Log summary statistics
        logger.info(f"Census Block {geoid20} Evaluation Summary:")
        logger.info(f"  Total buildings: {len(eval_df)}")

        if 'building_type' in eval_df.columns:
            type_counts = eval_df['building_type'].value_counts()
            logger.info(f"  Building types: {type_counts.to_dict()}")

        if 'occupants' in eval_df.columns:
            total_occupants = eval_df['occupants'].sum()
            logger.info(f"  Total occupants: {total_occupants:.0f}")
            logger.info(f"  Average occupants per building: {eval_df['occupants'].mean():.1f}")

        if 'housing_units' in eval_df.columns:
            total_units = eval_df['housing_units'].sum()
            logger.info(f"  Total housing units: {total_units:.0f}")

        if 'floor_area' in eval_df.columns:
            total_area = eval_df['floor_area'].sum()
            logger.info(f"  Total floor area: {total_area:.0f} sq m")

        return eval_df

    def _assign_building_id(self, buildings: gpd.GeoDataFrame,
                            census_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assigns unique building IDs based on centroid-to-census-block assignment.
        Uses building centroids for robust spatial matching while preserving original polygon geometries.

        Building ID format: {GEOID20}{SEQUENTIAL_NUMBER}
        """
        if buildings is None or len(buildings) == 0:
            logger.warning("No buildings provided for building ID assignment")
            buildings_copy = buildings.copy() if buildings is not None else gpd.GeoDataFrame()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            return buildings_copy

        if census_blocks is None or len(census_blocks) == 0:
            logger.warning(
                "No census blocks provided for building ID assignment. "
                "All buildings will be unassigned.")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            if not buildings_copy.empty:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                buildings_copy.to_file(
                    self.output_dir / "unassigned_buildings.geojson", driver="GeoJSON")
            return buildings_copy

        required_cols = ['GEOID20', 'geometry']
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
                    self.output_dir / "unassigned_buildings.geojson", driver="GeoJSON")
            return buildings_copy

        logger.info("Starting centroid-based building ID assignment")
        logger.info(f"Input: {len(buildings)} buildings, {len(census_blocks)} census blocks")
        logger.info(f"Original buildings CRS: {buildings.crs}")
        logger.info(f"Original census_blocks CRS: {census_blocks.crs}")

        # Project to target CRS for accurate centroid calculation
        target_crs = "EPSG:5070"
        try:
            buildings_proj = buildings.to_crs(target_crs)
            census_blocks_proj = census_blocks.to_crs(target_crs)
            logger.info(f"Projected to {target_crs} for accurate centroid calculation")
        except Exception as e:
            logger.error(f"Error during CRS projection to {target_crs}: {e}. Cannot assign IDs.")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            if not buildings_copy.empty:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                buildings_copy.to_file(
                    self.output_dir / "unassigned_buildings.geojson", driver="GeoJSON")
            return buildings_copy

        # Phase 2: Calculate building centroids for spatial matching
        logger.info("Phase 2: Calculating building centroids for spatial matching")
        try:
            # Create centroids while preserving original building indices and attributes
            building_centroids = buildings_proj.copy()
            building_centroids['centroid'] = buildings_proj.geometry.centroid
            building_centroids['original_geometry'] = buildings_proj.geometry

            # Replace geometry with centroids for spatial join
            building_centroids = building_centroids.set_geometry('centroid')

            centroid_success_count = building_centroids['centroid'].notna().sum()
            logger.info(f"Successfully calculated {centroid_success_count}/{len(buildings)} "
                        f"building centroids ({centroid_success_count/len(buildings)*100:.1f}% success)")

            if centroid_success_count == 0:
                logger.error("No valid centroids calculated. Cannot proceed with assignment.")
                buildings_copy = buildings.copy()
                buildings_copy['building_id'] = None
                buildings_copy['census_block_id'] = None
                return buildings_copy

        except Exception as e:
            logger.error(f"Error calculating building centroids: {e}")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            return buildings_copy

        # Phase 3: Single-step spatial join using centroids
        logger.info("Phase 3: Performing centroid-based spatial join with census blocks")
        try:
            # Spatial join: centroid WITHIN census_block
            joined = gpd.sjoin(
                building_centroids,
                census_blocks_proj[['GEOID20', 'geometry']],
                how='left',
                predicate='within'
            )

            # Count successful assignments
            assigned_mask = joined['index_right'].notna()
            assigned_count = assigned_mask.sum()
            success_rate = assigned_count / len(buildings) * 100

            logger.info(f"Spatial join results: {assigned_count}/{len(buildings)} buildings assigned "
                        f"({success_rate:.1f}% success)")

            # Log assignment distribution by census block
            if assigned_count > 0:
                block_distribution = joined[assigned_mask].groupby('GEOID20').size()
                logger.info(
                    f"Assignment distribution across {len(block_distribution)} census blocks:")
                logger.info(f"  Average buildings per block: {block_distribution.mean():.1f}")
                logger.info(f"  Max buildings in single block: {block_distribution.max()}")
                logger.info(f"  Min buildings in single block: {block_distribution.min()}")

                # Log top 5 blocks by building count
                top_blocks = block_distribution.nlargest(5)
                logger.info("Top 5 blocks by building count:")
                for geoid, count in top_blocks.items():
                    logger.info(f"  Block {geoid}: {count} buildings")

        except Exception as e:
            logger.error(f"Error during spatial join: {e}")
            buildings_copy = buildings.copy()
            buildings_copy['building_id'] = None
            buildings_copy['census_block_id'] = None
            return buildings_copy

        # Phase 4: Generate building IDs and assign back to original buildings
        logger.info("Phase 4: Generating building IDs and mapping back to original geometries")

        # Prepare result DataFrame with original polygon geometries
        buildings_with_ids = buildings.copy()
        buildings_with_ids['building_id'] = pd.NA
        buildings_with_ids['census_block_id'] = pd.NA

        # Process assignments and generate IDs
        all_assignments_list = []
        assigned_buildings = joined[assigned_mask]

        if len(assigned_buildings) > 0:
            # Create assignments list for ID generation
            for idx, row in assigned_buildings.iterrows():
                all_assignments_list.append({
                    'original_building_idx': idx,
                    'GEOID20': row['GEOID20']
                })

            # Generate sequential IDs within each block
            assignments_df = pd.DataFrame(all_assignments_list)
            assignments_df = assignments_df.sort_values(
                by=['GEOID20', 'original_building_idx'])
            assignments_df['sequential_id'] = assignments_df.groupby('GEOID20').cumcount() + 1

            # Assign IDs back to original buildings
            for _, row in assignments_df.iterrows():
                original_idx = row['original_building_idx']
                geoid = row['GEOID20']
                seq_id = row['sequential_id']

                building_id_val = f"{geoid}{seq_id:04d}"
                buildings_with_ids.loc[original_idx, 'building_id'] = building_id_val
                buildings_with_ids.loc[original_idx, 'census_block_id'] = geoid

            logger.info(f"Generated building IDs for {len(assignments_df)} buildings")
        else:
            logger.warning("No buildings were successfully assigned to census blocks")

        # Phase 5: Handle unassigned buildings
        unassigned_mask = buildings_with_ids['building_id'].isna()
        unassigned_count = unassigned_mask.sum()

        if unassigned_count > 0:
            logger.warning(f"Found {unassigned_count} buildings whose centroids fall outside "
                           f"all census blocks ({unassigned_count/len(buildings)*100:.1f}%)")

            # Save unassigned buildings for debugging
            unassigned_buildings = buildings_with_ids[unassigned_mask].copy()
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                unassigned_buildings.to_file(
                    self.output_dir / "unassigned_buildings.geojson", driver="GeoJSON")
                logger.info(f"Saved {len(unassigned_buildings)} unassigned buildings "
                            "to 'unassigned_buildings.geojson' for debugging")
            except Exception as e:
                logger.error(f"Failed to save unassigned buildings: {e}")

            # Remove unassigned buildings from final result
            buildings_with_ids = buildings_with_ids[~unassigned_mask].copy()

        # Final summary
        total_assigned = len(buildings_with_ids)
        if len(buildings) > 0:
            final_success_rate = total_assigned / len(buildings) * 100
            logger.info(f"Building ID assignment complete: {total_assigned}/{len(buildings)} "
                        f"buildings assigned ({final_success_rate:.1f}% success)")
        else:
            logger.info("Building ID assignment complete: 0/0 buildings assigned")

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

        return buildings_with_floors

    def _allot_construction_year(self, buildings: gpd.GeoDataFrame,
                                 nrel_vintage_distribution: Dict[str, float]) -> gpd.GeoDataFrame:
        """
        Allot construction year for each building based on NREL vintage distribution.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Buildings to assign construction years to
        nrel_vintage_distribution : Dict[str, float]
            Distribution of vintage bins with keys like '<1940', '1940s', etc. and values as percentages (0.0-1.0)

        Returns:
        --------
        GeoDataFrame : Buildings with added 'construction_year' column
        """
        if buildings.empty:
            logger.info("No buildings to process for construction year assignment")
            buildings['construction_year'] = None
            return buildings

        if not nrel_vintage_distribution or sum(nrel_vintage_distribution.values()) == 0:
            logger.warning("Empty or invalid vintage distribution, assigning 'Unknown'")
            buildings['construction_year'] = 'Unknown'
            return buildings

        # Extract vintage categories and their probabilities
        vintage_bins = list(nrel_vintage_distribution.keys())
        probabilities = list(nrel_vintage_distribution.values())

        # Normalize probabilities to ensure they sum to 1.0
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback to uniform distribution if all probabilities are 0
            probabilities = [1.0 / len(vintage_bins)] * len(vintage_bins)

        assigned_vintages = np.random.choice(
            vintage_bins,
            size=len(buildings),
            p=probabilities
        )

        buildings = buildings.copy()
        buildings['construction_year'] = assigned_vintages

        logger.info(
            f"Assigned construction years to {len(buildings)} buildings. "
            f"Distribution: {dict(zip(*np.unique(assigned_vintages, return_counts=True)))}"
        )

        return buildings

    def write_buildings_output(self, buildings: gpd.GeoDataFrame,
                               output_dir: Union[str, Path],
                               filename: str,
                               building_type: str = 'residential',
                               pylovo_mapping: bool = False) -> str:
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
        building_type : str
            Type of buildings ('residential' or 'non_residential')
        pylovo_mapping : bool
            If True, applies PyLOVO format (osm_id, Area, Use, etc.).
            If False (default), preserves original column names (building_id, floor_area, etc.).

        Returns:
        --------
        str : Path to output file
        """
        # Create output path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / filename)

        # Prepare output data based on mapping requirements
        if pylovo_mapping:
            # Use schema classes for PyLOVO-specific formatting
            if building_type == 'residential':
                output_buildings = ResidentialBuildingOutput.prepare_pylovo_output(buildings)
                logger.info("Applied PyLOVO mapping for residential buildings")
            else:  # non_residential
                output_buildings = NonResidentialBuildingOutput.prepare_pylovo_output(buildings)
                logger.info("Applied PyLOVO mapping for non-residential buildings")
        else:
            # Use schema classes to filter and organize columns (default behavior)
            if building_type == 'residential':
                output_buildings = ResidentialBuildingOutput.prepare_default_output(buildings)
                logger.info("Applied residential schema filtering with original column names")
            else:  # non_residential
                output_buildings = NonResidentialBuildingOutput.prepare_default_output(buildings)
                logger.info("Applied non-residential schema filtering with original column names")

        # Write shapefile
        logger.info(f"Writing {len(output_buildings)} {building_type} buildings to {output_path}")
        logger.info(f"Output columns: {list(output_buildings.columns)}")
        output_buildings.to_file(output_path)

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

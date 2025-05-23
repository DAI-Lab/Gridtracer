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

    def process(self, osm_data: Dict,
                census_data: Dict, nrel_data: Dict) -> Dict[str, str]:
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

        # Step 1: Classify building use (residential, commercial, industrial, etc.)
        logger.info("Step 1: Classifying building use")
        all_buildings = self.classify_building_use(
            osm_data.get('buildings'),
            osm_data.get('pois'),
            osm_data.get('landuse'),
            census_data.get('blocks')
        )

        # If no buildings found, return early
        if all_buildings is None or len(all_buildings) == 0:
            logger.warning("No buildings found to process")
            no_buildings_path = self.write_empty_data_stub(
                self.output_dir,
                'buildings_classified.txt',
                "No buildings found in the region"
            )
            return {
                'residential': no_buildings_path,
                'other': no_buildings_path,
                'classified': no_buildings_path
            }

        # Step 2: Split buildings by use
        logger.info(f"Found {len(all_buildings)} buildings to process")
        residential = all_buildings[all_buildings['building_use'] == 'Residential'].copy()
        other = all_buildings[all_buildings['building_use'] != 'Residential'].copy()

        # Step 3: Calculate free walls for all buildings
        logger.info("Step 3: Calculating free walls")
        if len(other) > 0:
            other = self.calculate_free_walls(other)

        # Step 4: Process residential buildings
        residential_output_path = None
        if len(residential) > 0:
            logger.info(f"Processing {len(residential)} residential buildings")

            # Calculate free walls
            residential = self.calculate_free_walls(residential)

            # Classify building types (SFH, MFH, etc.)
            residential = self.classify_building_type(
                residential,
                census_data.get('housing')
            )

            # Allot occupants based on census
            residential = self.allot_occupants(
                residential,
                census_data.get('population')
            )

            # Calculate floors
            residential = self.calculate_floors(residential)

            # Allot construction year
            residential = self.allot_construction_year(
                residential,
                census_data.get('housing_age')
            )

            # Optional: Allot refurbishment level
            residential = self.allot_refurbishment_level(residential)

            # Write residential output
            residential_output_path = self.write_buildings_output(
                residential,
                self.output_dir,
                'buildings_residential.shp'
            )
            logger.info(f"Residential buildings saved to: {residential_output_path}")
        else:
            # Create empty stub file for residential
            logger.warning("No residential buildings found")
            residential_output_path = self.write_empty_data_stub(
                self.output_dir,
                'buildings_residential.txt',
                "No residential buildings found"
            )

        # Step 5: Process non-residential buildings
        other_output_path = None
        if len(other) > 0:
            logger.info(f"Processing {len(other)} non-residential buildings")
            # Write non-residential output
            other_output_path = self.write_buildings_output(
                other,
                self.output_dir,
                'buildings_other.shp'
            )
            logger.info(f"Non-residential buildings saved to: {other_output_path}")
        else:
            # Create empty stub file for non-residential
            logger.warning("No non-residential buildings found")
            other_output_path = self.write_empty_data_stub(
                self.output_dir,
                'buildings_other.txt',
                "No non-residential buildings found"
            )

        # Step 6: Generate final classified buildings shapefile by merging
        classified_output_path = None
        if residential_output_path and residential_output_path.endswith('.shp') and \
           other_output_path and other_output_path.endswith('.shp'):
            logger.info("Merging residential and non-residential buildings")
            classified_output_path = self.merge_building_layers(
                residential_output_path,
                other_output_path,
                self.output_dir,
                'buildings_classified.shp'
            )
        elif residential_output_path and residential_output_path.endswith('.shp'):
            classified_output_path = residential_output_path
        elif other_output_path and other_output_path.endswith('.shp'):
            classified_output_path = other_output_path
        else:
            classified_output_path = self.write_empty_data_stub(
                self.output_dir,
                'buildings_classified.txt',
                "No buildings found"
            )

        logger.info(f"Building classification complete. Final output: {classified_output_path}")
        return {
            'residential': residential_output_path,
            'other': other_output_path,
            'classified': classified_output_path
        }

    def classify_building_use(self, buildings: gpd.GeoDataFrame,
                              pois: gpd.GeoDataFrame,
                              landuse: gpd.GeoDataFrame,
                              census_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Classifies buildings by their use based on OSM tags, POIs, land use data,
        and census block information.

        Heuristic priority order:
        1. Direct OSM tags (e.g., building=residential)
        2. POIs inside the building
        3. Land use zones the building falls within
        4. Census block properties (e.g., residential zones)
        5. Building name keywords
        6. Final default classification (typically residential)

        Parameters:
        -----------
        buildings : GeoDataFrame
            OSM building polygons
        pois : GeoDataFrame
            Points of Interest
        landuse : GeoDataFrame
            Land use polygons
        census_blocks : GeoDataFrame
            Census blocks with demographic data

        Returns:
        --------
        GeoDataFrame : Buildings with 'building_use' column added
        """
        if buildings is None or len(buildings) == 0:
            logger.warning("No buildings provided to classify")
            return gpd.GeoDataFrame()

        # Ensure all data is in EPSG:4326 (WGS84)
        buildings = self.ensure_wgs84(buildings)
        census_blocks = self.ensure_wgs84(census_blocks)
        landuse = self.ensure_wgs84(landuse)

        # Create a copy to avoid modifying the original
        classified_buildings = buildings.copy()

        # Initialize the building_use column
        classified_buildings['building_use'] = None

        # 1. Filter out small buildings and non-buildings
        # Minimum building area threshold (e.g., 45 sq meters)
        logger.debug("Filtering out small buildings and non-buildings")

        # Calculate accurate areas in square meters
        classified_buildings = self.add_floor_area(classified_buildings)

        # Filter by properly calculated area
        min_area = 45  # sq meters

        classified_buildings = classified_buildings[
            classified_buildings['floor_area'] >= min_area
        ]

        # Filter out gazebos, garages, etc.
        if 'building' in classified_buildings.columns:
            exclude_types = ['garage', 'shed', 'garages', 'carport', 'roof']
            mask = ~classified_buildings['building'].isin(exclude_types)
            classified_buildings = classified_buildings[mask]

        # 2. Direct classification from OSM tags
        logger.debug("Classifying buildings from OSM tags")
        if 'building' in classified_buildings.columns:
            # Residential buildings
            residential_types = ['residential', 'house', 'detached', 'apartments',
                                 'terrace', 'dormitory', 'semidetached_house']
            mask = classified_buildings['building'].isin(residential_types)
            classified_buildings.loc[mask, 'building_use'] = 'Residential'

            # Commercial buildings
            commercial_types = ['commercial', 'retail', 'supermarket', 'kiosk', 'shop']
            mask = classified_buildings['building'].isin(commercial_types)
            classified_buildings.loc[mask, 'building_use'] = 'Commercial'

            # Industrial buildings
            industrial_types = ['industrial', 'warehouse', 'manufacture', 'factory']
            mask = classified_buildings['building'].isin(industrial_types)
            classified_buildings.loc[mask, 'building_use'] = 'Industrial'

            # Public buildings
            public_types = ['school', 'hospital', 'government', 'university',
                            'public', 'church', 'mosque', 'synagogue', 'temple']
            mask = classified_buildings['building'].isin(public_types)
            classified_buildings.loc[mask, 'building_use'] = 'Public'

        # 3. Keyword-based classification from building names
        logger.debug("Classifying buildings from name keywords")
        if 'name' in classified_buildings.columns:
            # Apply classification based on name keywords
            # This would be implemented with regex or string matching
            pass

        # 4. Classification by land use overlay
        logger.debug("Classifying buildings by land use overlay")
        if landuse is not None and len(landuse) > 0:
            # Spatial join with land use polygons
            # This would use spatial join operations from geopandas
            pass

        # 5. Classification by POIs
        logger.debug("Classifying buildings by contained POIs")
        if pois is not None and len(pois) > 0:
            # Spatial join with POIs
            # This would use spatial join operations from geopandas
            pass

        # 6. Classification by census block characteristics
        logger.debug("Classifying buildings by census block characteristics")
        if census_blocks is not None and len(census_blocks) > 0:
            # Spatial join with census blocks
            # This would use spatial join operations from geopandas
            pass

        # 7. Default classification strategy
        logger.debug("Applying default classification")
        # Assign remaining unclassified buildings to Residential
        mask = classified_buildings['building_use'].isna()
        classified_buildings.loc[mask, 'building_use'] = 'Residential'

        # 8. Add confidence score for each classification
        classified_buildings['confidence'] = 0.7  # Default confidence

        return classified_buildings

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
            return buildings

        # Create a copy to avoid modifying the original
        buildings_with_free_walls = buildings.copy()

        # Initialize free_walls and neighbors columns
        buildings_with_free_walls['free_walls'] = 4  # Default to 4 free walls
        buildings_with_free_walls['neighbors'] = None

        # 1. Find topologically connected neighbors for each building
        # This would require spatial operations to find buildings that touch
        # For a proper implementation, we would use spatial index and topology operations
        # For now, use a simplified approach:

        # Find neighboring buildings (touching)
        logger.debug("Finding neighboring buildings")
        # This is a simplified placeholder - actual implementation would use spatial operations
        # neighbors = {i: [] for i in range(len(buildings_with_free_walls))}
        # for i in range(len(buildings_with_free_walls)):
        #     for j in range(len(buildings_with_free_walls)):
        #         if i != j and buildings_with_free_walls.iloc[i].geometry.touches(
        #                 buildings_with_free_walls.iloc[j].geometry):
        #             neighbors[i].append(j)

        # For demonstration, assign random free walls
        # In a real implementation, use proper spatial operations
        buildings_with_free_walls['free_walls'] = np.random.randint(
            1, 5, len(buildings_with_free_walls))

        # 2. Default to 4 free walls (assumes rectangular buildings)
        # Already done in initialization

        # 3. Subtract number of neighbors from default (max 4)
        # This would be implemented using the neighbors dictionary
        # For buildings with more than 4 neighbors, set free_walls to 0

        # 4. Add both neighbors array and free_walls count to buildings
        # This would store the neighbors list in the GeoDataFrame

        return buildings_with_free_walls

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
        mask = (classified_buildings['free_walls'] == 4) & \
               (classified_buildings.geometry.area < large_threshold) & \
               (classified_buildings['building_type'].isna())
        classified_buildings.loc[mask, 'building_type'] = 'SFH'

        # Small attached buildings in rows (free_walls = 2) → TH (Townhouse)
        mask = (classified_buildings['free_walls'] == 2) & \
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
                        population_data: Optional[Dict] = None) -> gpd.GeoDataFrame:
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
            area = building.geometry.area
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

    def calculate_floors(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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

        # Create a copy to avoid modifying the original
        buildings_with_floors = buildings.copy()

        # Initialize floors column
        buildings_with_floors['floors'] = 1

        # 1. Use direct OSM data if available
        logger.debug("Using OSM building:levels data when available")
        if 'building:levels' in buildings_with_floors.columns:
            mask = buildings_with_floors['building:levels'].notna()
            buildings_with_floors.loc[mask,
                                      'floors'] = buildings_with_floors.loc[mask,
                                                                            'building:levels']

        # 2. Estimate from occupants and area
        logger.debug("Estimating floors from occupants and area")
        for idx, building in buildings_with_floors.iterrows():
            if building['floors'] == 1:  # Only process if not already set
                building_type = building['building_type']
                area = building.geometry.area
                occupants = building['occupants']

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

                # Apply typical constraints by building type (already handled above)

        # 4. Calculate floor area
        logger.debug("Calculating floor area")
        buildings_with_floors['floor_area'] = buildings_with_floors.geometry.area * \
            buildings_with_floors['floors']

        # 5. Calculate building height (simple estimate: 3m per floor)
        buildings_with_floors['height'] = buildings_with_floors['floors'] * 3.0

        # 6. Validate and adjust estimates
        logger.debug("Validating floor estimates")
        # This would check for consistency with building type and regional patterns

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

    def add_floor_area(self, buildings):
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
        # First ensure data is in EPSG:4326 (WGS84)
        if buildings.crs != "EPSG:4326":
            buildings = buildings.to_crs("EPSG:4326")

        # Reproject to appropriate local CRS for area calculation
        local_crs = self._get_local_crs(buildings)
        buildings_projected = buildings.to_crs(local_crs)

        # Calculate area in square meters
        buildings['floor_area'] = buildings_projected.geometry.area

        return buildings

    def ensure_wgs84(self, gdf):
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

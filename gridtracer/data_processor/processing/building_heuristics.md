## 0  Quick-Glance Flow

extract_building_data()
├─ classify_building_use()              # tag every building polygon with a "Use"
├─ split Residential / Other
│   ├─ calculate_free_walls()
│   └─ classify_building_type()
│       ├─ allot_occupants()
│       ├─ calculate_floors()
│       ├─ allot_construction_year()
│       └─ allot_refurbishment_level()  # optional
└─ write shapefiles / "no-data" txt stubs

## 1  Building Classification Pipeline Pseudocode

The building classification pipeline processes building footprints to create detailed building attributes suitable for energy demand modeling and grid infrastructure planning. Each step applies heuristics to determine building characteristics when direct data is unavailable.

### 1.1  Main Processing Function

```python
def extract_building_data(region_data, osm_data, census_data, nrel_data, output_dir):
    """
    Main function that orchestrates the entire building classification process.
    
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
    output_dir : str
        Path to output directory
    
    Returns:
    --------
    dict : Paths to output files
    """
    
    # Step 1: Classify building use (residential, commercial, industrial, etc.)
    all_buildings = classify_building_use(
        osm_data['buildings'], 
        osm_data['pois'], 
        osm_data['landuse'],
        census_data['blocks']
    )
    
    # Step 2: Split buildings by use
    residential = all_buildings[all_buildings['use'] == 'Residential']
    other = all_buildings[all_buildings['use'] != 'Residential']
    
    # Step 3: Calculate free walls for all buildings
    other = calculate_free_walls(other)
    
    # Step 4: Process residential buildings
    if len(residential) > 0:
        # Calculate free walls
        residential = calculate_free_walls(residential)
        
        # Classify building types (SFH, MFH, etc.)
        residential = classify_building_type(residential, census_data['housing'])
        
        # Allot occupants based on census
        residential = allot_occupants(residential, census_data['population'])
        
        # Calculate floors
        residential = calculate_floors(residential)
        
        # Allot construction year
        residential = allot_construction_year(residential, census_data['housing_age'])
        
        # Optional: Allot refurbishment level
        residential = allot_refurbishment_level(residential)
        
        # Write residential output
        residential_output_path = write_buildings_output(
            residential, 
            output_dir, 
            'buildings_residential.shp'
        )
    else:
        # Create empty stub file for residential
        residential_output_path = write_empty_data_stub(
            output_dir, 
            'buildings_residential.txt', 
            "No residential buildings found"
        )
    
    # Step 5: Process non-residential buildings
    if len(other) > 0:
        # Write non-residential output
        other_output_path = write_buildings_output(
            other, 
            output_dir, 
            'buildings_other.shp'
        )
    else:
        # Create empty stub file for non-residential
        other_output_path = write_empty_data_stub(
            output_dir, 
            'buildings_other.txt', 
            "No non-residential buildings found"
        )
    
    # Step 6: Generate final classified buildings shapefile by merging
    if residential_output_path.endswith('.shp') and other_output_path.endswith('.shp'):
        classified_output_path = merge_building_layers(
            residential_output_path,
            other_output_path,
            output_dir,
            'buildings_classified.shp'
        )
    elif residential_output_path.endswith('.shp'):
        classified_output_path = residential_output_path
    elif other_output_path.endswith('.shp'):
        classified_output_path = other_output_path
    else:
        classified_output_path = write_empty_data_stub(
            output_dir, 
            'buildings_classified.txt', 
            "No buildings found"
        )
    
    return {
        'residential': residential_output_path,
        'other': other_output_path,
        'classified': classified_output_path
    }
```

### 1.2  Building Use Classification

```python
def classify_building_use(buildings, pois, landuse, census_blocks):
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
    GeoDataFrame : Buildings with 'use' column added
    """
    
    # 1. Filter out small buildings and non-buildings
    # Minimum building area threshold (e.g., 45 sq meters)
    # Filter out gazebos, garages, etc.
    
    # 2. Direct classification from OSM tags
    # Identify buildings with explicit type tags
    # - Residential: house, residential, apartments, etc.
    # - Commercial: commercial, retail, supermarket, etc.
    # - Industrial: industrial, warehouse, etc.
    # - Public: school, hospital, government, etc.
    
    # 3. Keyword-based classification from building names
    # Check for keywords in name that indicate building use
    
    # 4. Classification by land use overlay
    # Assign building use based on underlying land use polygons
    
    # 5. Classification by POIs
    # Check for POIs contained within buildings to determine use
    
    # 6. Classification by census block characteristics
    # Use residential vs. non-residential census block designations
    
    # 7. Default classification strategy
    # Assign remaining unclassified buildings (typically to Residential)
    
    # 8. Add confidence score for each classification

    return classified_buildings
```

### 1.3  Free Walls Calculation

```python
def calculate_free_walls(buildings):
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
    
    # 1. Find topologically connected neighbors for each building
    # Use spatial operations to find buildings that touch each other
    
    # 2. Default to 4 free walls (assumes rectangular buildings)
    
    # 3. Subtract number of neighbors from default (max 4)
    # For buildings with more than 4 neighbors, set free_walls to 0
    
    # 4. Add both neighbors array and free_walls count to buildings
    
    return buildings_with_free_walls
```

### 1.4  Building Type Classification

```python
def classify_building_type(buildings, housing_data):
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
    housing_data : DataFrame
        Reference housing type distribution
    
    Returns:
    --------
    GeoDataFrame : Buildings with 'building_type' column added
    """
    
    # 1. Identify neighborhood clusters
    # Group nearby buildings that form a building complex
    
    # 2. Calculate total area for each neighborhood cluster
    
    # 3. Initial classification based on geometry and neighbors:
    # - Large buildings/clusters (area > threshold) → AB
    # - Small detached buildings (free_walls = 4) → SFH
    # - Small attached buildings in rows (free_walls = 2) with similar areas → TH
    # - Medium-sized buildings with varied neighbor counts → MFH
    
    # 4. Balance classification with reference distribution
    # Compare classified counts with expected housing type distribution
    # Adjust classifications to match regional statistics
    # Most common adjustments:
    # - Reclassify smallest MFH → SFH if SFH is underrepresented
    # - Reclassify largest SFH → MFH if MFH is underrepresented
    # - Reclassify MFH → AB or AB → MFH based on size
    
    # 5. Handle townhouses/row houses specially
    # Check for linear arrangement and split if appropriate
    
    return classified_buildings
```

### 1.5  Occupant Allocation

```python
def allot_occupants(buildings, population_data):
    """
    Allocates household occupants to residential buildings based on
    building type, size, and census population data.
    
    Parameters:
    -----------
    buildings : GeoDataFrame
        Residential buildings with building_type
    population_data : GeoDataFrame
        Census population and household data
    
    Returns:
    --------
    GeoDataFrame : Buildings with 'occupants' column added
    """
    
    # 1. Calculate maximum occupant capacity per building type
    # Based on building area, typical unit sizes, and occupancy rates:
    # - SFH: Typically 2-6 people based on area/50 sq m per person
    # - TH: Typically 2-4 people based on area/50 sq m per person
    # - MFH: Based on number of units estimated from floor area
    # - AB: Based on number of units estimated from floor area
    
    # 2. Distribute census block population to buildings
    # Spatially join buildings to census blocks
    # Proportionally distribute population based on capacity
    
    # 3. Handle buildings with no census block data
    # Assign typical occupancy based on building type and area
    
    # 4. Ensure total population matches census totals
    # Make adjustments if necessary without exceeding building capacity
    
    return buildings_with_occupants
```

### 1.6  Floor Calculation

```python
def calculate_floors(buildings):
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
    
    # 1. Use direct OSM data if available
    # Check for building:levels tag
    
    # 2. Estimate from occupants and area
    # Calculate typical area per person for each building type
    # Estimate floors needed to house all occupants
    
    # 3. Apply typical constraints by building type:
    # - SFH/TH: Usually 1-3 floors (max 4)
    # - MFH: Usually 2-5 floors (max 6)
    # - AB: Usually 4-10 floors
    
    # 4. Calculate floor area
    # floor_area = footprint_area * floors
    
    # 5. Validate and adjust estimates
    # Check for consistency with building type and regional patterns
    
    return buildings_with_floors
```

### 1.7  Construction Year Allocation

```python
def allot_construction_year(buildings, housing_age_data):
    """
    Assigns construction year periods to buildings based on
    available data and statistical distribution.
    
    Parameters:
    -----------
    buildings : GeoDataFrame
        Buildings with type and other attributes
    housing_age_data : DataFrame
        Statistical data on building age distribution by region
    
    Returns:
    --------
    GeoDataFrame : Buildings with 'construction_year' column added
    """
    
    # 1. Use direct OSM data if available
    # Check for year or start_date tags
    
    # 2. Use spatial reference data
    # Spatially join with any available assessor or historical data
    
    # 3. Apply neighborhood consistency patterns
    # Buildings in the same neighborhood often have similar ages
    
    # 4. Allocate remaining buildings based on statistical distribution
    # Assign construction year periods to maintain statistical distribution
    # Common categories for US buildings:
    # - Pre-1950
    # - 1950-1969
    # - 1970-1989
    # - 1990-2009
    # - 2010-present
    
    # 5. Add confidence indicator for the source of the year data
    
    return buildings_with_year
```

### 1.8  Refurbishment Level Allocation (Optional)

```python
def allot_refurbishment_level(buildings):
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
    
    # 1. Assign probability of refurbishment based on:
    # - Building age (older buildings have higher probability)
    # - Building type (different patterns for SFH vs. MFH)
    # - Regional factors (climate zone, affluence)
    
    # 2. Assign specific refurbishment components:
    # - Wall insulation
    # - Roof insulation
    # - Window replacement
    # - Basement insulation
    
    # 3. Consider neighborhood effects
    # Buildings near each other often have similar refurbishment patterns
    
    return buildings_with_refurbishment
```

### 1.9  Output Handling

```python
def write_buildings_output(buildings, output_dir, filename):
    """
    Writes processed building data to shapefile.
    
    Parameters:
    -----------
    buildings : GeoDataFrame
        Processed buildings
    output_dir : str
        Path to output directory
    filename : str
        Output filename
    
    Returns:
    --------
    str : Path to output file
    """
    # Create output path
    # Write shapefile with all attributes
    # Return path to the output file
    
    return output_path

def write_empty_data_stub(output_dir, filename, message):
    """
    Creates a stub text file when no buildings of a category exist.
    
    Parameters:
    -----------
    output_dir : str
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
    # Write text file with message
    # Return path to the output file
    
    return output_path

def merge_building_layers(residential_path, other_path, output_dir, filename):
    """
    Merges residential and non-residential building layers into a single shapefile.
    
    Parameters:
    -----------
    residential_path : str
        Path to residential buildings shapefile
    other_path : str
        Path to non-residential buildings shapefile
    output_dir : str
        Path to output directory
    filename : str
        Output filename
    
    Returns:
    --------
    str : Path to merged output file
    """
    # Read input shapefiles
    # Merge GeoDataFrames
    # Write output shapefile
    # Return path to the output file
    
    return output_path
```

## 2  Output Schema

Final classified buildings will have the following attributes:

| Attribute              | Description                                | Source/Method                                 |
|------------------------|--------------------------------------------|----------------------------------------------|
| `osm_id`               | Unique identifier                          | OSM                                          |
| `geometry`             | Building footprint geometry                | OSM or external footprint extractor          |
| `building_use`         | Primary use (e.g., residential, commercial)| OSM tags + zoning overlays + classification  |
| `building_type`        | Building typology (SFH, TH, MFH, AB)       | Heuristic on area + levels + landuse + free walls |
| `floor_number`         | Number of floors/stories                   | OSM building:levels or height ÷ 3 or estimated |
| `floor_area`           | Total floor area (m²)                      | OSM footprint area × floors                  |
| `height`               | Building height (m)                        | LiDAR or floor_number × 3 m                  |
| `occupants`            | Number of occupants                        | Census household size × units                |
| `households`           | Number of household units                  | Floor area ÷ unit size or modeled            |
| `construction_year`    | Construction period                        | Census age bins or assessor data             |
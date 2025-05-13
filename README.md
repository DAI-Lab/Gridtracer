<p align="left">
  <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab Logo" />
  <i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Travis CI Shield](https://travis-ci.org/DAI-Lab/syngrid.svg?branch=master)](https://travis-ci.org/DAI-Lab/syngrid)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/syngrid/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/syngrid)

# SynGrid

This repository is part of a **research project** developed for a master's thesis. The goal is to create a **data preprocessing pipeline** for generating georeferenced datasets that serve as the foundation for building synthetic **low-voltage grid infrastructure schemata** in the United States.

**Project Focus:** This repository *solely* focuses on the data preprocessing pipeline. It takes a user-defined US region and generates specific, georeferenced datasets suitable as input for downstream synthetic grid modeling tasks (handled in a separate project).

---

## Overview

The preprocessing pipeline handles the collection, processing, and output of regional data, specifically targeting:
- Building footprints with detailed classifications.
- Points of Interest (POIs).
- Land use data.
- A routable road network suitable for `pgRouting`.
- Land cover data.
- Electrical transformer locations.

The final output consists of organized, geospatially referenced files tailored for grid synthesis inputs.

---

## Configuration Input

The pipeline is initialized through a YAML-based configuration file. This file specifies the geographic scope for data collection:

```yaml
region:
  state: "MA"                     # Required: State abbreviation (e.g., "MA")
  county: "Middlesex County"      # Required: Full county name
  county_subdivision: "Cambridge city" # Optional: Full county subdivision name. If omitted, processes the entire county.
  # Optional: URL to fetch county subdivision codes/names if needed for lookup. Defaults may be provided.
  lookup_url: "https://www2.census.gov/geo/docs/reference/codes/files/national_cousub.txt"
```
Input validation ensures correct state abbreviations and county names are used.

---

## Data Sources

| Source                | Data Extracted                                            | Purpose                                    |
|----------------------|-----------------------------------------------------------|--------------------------------------------|
| **OpenStreetMap**    | Buildings, POIs, Land Use, Roads, Power Infrastructure    | Base geometry, network, feature extraction |
| **NREL**             | Residential building typology datasets                    | Building classification for energy demand  |
| **NLCD**             | Land cover raster data                                    | Land use classification (residential, commercial, etc.) |
| **US Census TIGER**  | Administrative boundaries (state, county, subdivision)    | Defining regional scope, clipping          |
| **US Census Data**   | Demographic data (population density, potentially others) | Building classification heuristics         |
| **(External Data)**  | TABULA rules  (TBD)                                       | Heuristics for building classification     |

All datasets are clipped to the defined region (county or subdivision) and stored with geospatial reference (projected CRS might be needed, e.g., state plane or UTM zone).

---

## Pipeline Outputs

The primary outputs generated for the specified region are:

1.  **Routable Road Network:** An `.sql` file containing the road network processed and formatted for direct import into a PostgreSQL/PostGIS database with the `pgRouting` extension (`roads_pgr.sql`).
2.  **Classified Building Footprints:** A shapefile (`buildings_classified.shp`) containing building polygons with attributes derived from the detailed classification heuristic (see Workflow below).
3.  **Transformer Network:** A GeoJSON file (`transformers.geojson`) containing points representing electrical transformers extracted from OSM.

---

## Folder Structure

All outputs are organized in the following directory tree, ensuring clear separation by region:

```
/output/
  └── [STATE_ABBREVIATION]/                 # e.g., MA
      └── [COUNTY_NAME]/                    # e.g., Middlesex County
          └── [Optional_Subdivision_Name]/  # e.g., Cambridge city (only if specified in config)
              ├── boundaries.geojson        # Clipped boundary of the processed region
              ├── buildings_classified.shp  # Classified building footprints
              ├── roads_pgr.sql             # Routable road network for pgRouting
              ├── transformers.geojson      # Extracted transformer locations
              ├── nlcd_landcover.tif        # Clipped land cover raster
              ├── pois.geojson              # Clipped Points of Interest
              ├── landuse.geojson           # Clipped land use polygons
              └── ...                       # Other intermediate or raw data files might be stored here
```

---

## Pipeline Workflow

### Step 1: Regional Data Extraction & Preparation
1. Parse the YAML config, validate inputs, and determine the precise geographic boundary (county or specific subdivision).
2. Download/load administrative boundaries (state, county, subdivision) from Census TIGER/Line data.
3. Use the defined boundary to query and download data from OSM (buildings, roads, POIs, power tags) and NLCD (land cover).
4. Clip all downloaded datasets precisely to the target boundary.
5. Project data to a suitable projected Coordinate Reference System (CRS) if necessary for accurate geometric operations.
6. Store initial processed/clipped files in the designated output subfolder.

### Step 2: Building Classification Heuristics
Apply the detailed heuristic strategy (as outlined by the user) to classify building footprints. This involves multiple sub-steps:
- Clean base geometry (filtering).
- Core typology classification (OSM tags, POIs, land use, census).
- Residential type refinement (shape, neighbors).
- Structural attribute estimation (walls, floors).
- Demographic assignment (occupants, year, refurbishment - using Census data).
- Contextual reinforcement using neighbor analysis.
- Apply confidence prioritization rules.
- Output the final `buildings_classified.shp`.

### Step 3: Road Network Processing
1. Clean the topology of the extracted road network.
2. Prepare the network data structure for `pgRouting` (e.g., creating vertices, assigning costs).
3. Export the prepared network as an `.sql` file (`roads_pgr.sql`).

### Step 4: Transformer Extraction
1. Filter OSM data for relevant power infrastructure tags (e.g., `power=transformer`).
2. Extract the geometries (points) for these features.
3. Output as `transformers.geojson`.




**1. Core Data Ingestion & Regional Scoping:**
- [ ] Finalize YAML configuration schema (state, county, subdivision, lookup URL) and implement robust parsing and validation.
- [ ] Implement logic to fetch and use Census county subdivision boundaries based on the lookup URL and config.
- [ ] Implement downloader for US Census TIGER/Line shapefiles for administrative boundaries (state, county, subdivisions).
- [ ] Implement downloader for OpenStreetMap data (buildings, roads, POIs, power infrastructure) for the specified region (e.g., using `osmnx` or Overpass API).
- [ ] Implement downloader for NLCD land cover raster data.
- [ ] Implement robust clipping logic to ensure all downloaded datasets precisely match the specified regional boundaries (county or subdivision).
- [ ] Establish and enforce the standardized output directory structure: `output/state/county/[subdivision]/`.
- [ ] Write initial raw/clipped data (boundaries, buildings, roads, POIs, land cover) to the appropriate output folders.

**2. Building Classification Pipeline:**
- [ ] Implement Step 1: Clean Base Geometry (filter by area, exclude irrelevant types).
- [ ] Implement Step 2.1: Core Typology Classification (using OSM tags, POIs, land use, census data).
- [ ] Implement Step 2.2: Residential Building Type Classification (using area, shape ratios, neighbor topology - requires spatial indexing/analysis).
- [ ] Implement Step 3: Structural Attributes Estimation (free walls, number of floors - requires neighbor analysis and potentially external data like TABULA rules/defaults).
- [ ] Implement Step 4: Demographic Assignment (occupants, construction year, refurbishment - requires integrating Census demographic data, potentially external lookup tables like Bayern). Define data sources/placeholders for external lookups.
- [ ] Implement Step 5: Contextual Reinforcement (using neighbor analysis `NeighborsALL` - requires spatial clustering/analysis).
- [ ] Define and implement the "Meta-Heuristic: Confidence Prioritization" logic for resolving conflicting information.
- [ ] Output the final classified buildings as a shapefile (`buildings_classified.shp` or similar).

**3. Routable Road Network Generation:**
- [ ] Process raw OSM road network data.
- [ ] Clean road network topology (e.g., ensure connectivity, handle intersections).
- [ ] Convert cleaned network into a format suitable for `pgRouting` (e.g., using `osmnx` functions or custom SQL generation).
- [ ] Generate the final `.sql` file (`roads_pgr.sql` or similar) for direct import into PostgreSQL/PostGIS.

**4. Transformer Network Extraction:**
- [ ] Identify relevant OSM tags for electrical transformers and substations (`power=transformer`, `power=substation`, etc.).
- [ ] Extract features matching these tags within the specified region.
- [ ] Output the extracted features as a GeoJSON file (`transformers.geojson`).

**5. Integration, Refinement & Documentation:**
- [ ] Ensure all pipeline steps integrate correctly and data flows smoothly.
- [ ] Implement comprehensive error handling and logging throughout the pipeline.
- [ ] Add data caching mechanisms to avoid redundant downloads and processing on repeated runs (as mentioned in original TODO).
- [ ] Thoroughly document the configuration, data sources, heuristic logic, and output formats within the code and potentially separate documentation files.
- [ ] Update the main `README.md` to reflect the final implementation details.

---

## Documentation

- 📄 Homepage: https://github.com/DAI-Lab/syngrid
- 📚 Documentation: https://DAI-Lab.github.io/syngrid

<p align="left">
  <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab Logo" />
  <i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Travis CI Shield](https://travis-ci.org/DAI-Lab/gridtracer.svg?branch=master)](https://travis-ci.org/DAI-Lab/gridtracer)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/gridtracer/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/gridtracer)

# Gridtracer

This repository is part of a **research project** developed for a master's thesis. The goal is to create a **data preprocessing pipeline** for generating georeferenced datasets that serve as the foundation for building synthetic **low-voltage grid infrastructure schemata** in the United States.

**Project Focus:** This repository *solely* focuses on the data preprocessing pipeline. It takes a user-defined US region and generates specific, georeferenced datasets suitable as input for downstream synthetic grid modeling tasks (handled in a separate project).

---

## Overview

This pipeline collects and processes geospatial data for any US region (state, county, or subdivision) to create comprehensive building-level datasets.

**Key Outputs:**
- Classified building footprints with energy-relevant attributes
- Routable road networks for transportation analysis  
- Points of Interest and land use data
- Regional boundaries and demographic information

All outputs are georeferenced and organized by administrative hierarchy for seamless integration into energy system modeling, urban planning, or technoeconomic analysis workflows.

---

## Configuration Input

The pipeline is initialized through a YAML-based configuration file. This file specifies the geographic scope for data collection:

```yaml
region:
  state: "MA"                     # Required: State abbreviation (e.g., "MA")
  county: "Middlesex County"      # Required: Full county name
  county_subdivision: "Cambridge city" # Optional: Full county subdivision name. If omitted, processes the entire county.
```
Input validation ensures correct state abbreviations and county names are used.

---

## Data Sources

| Source                | Data Extracted                                            | Purpose                                    |
|----------------------|-----------------------------------------------------------|--------------------------------------------|
| **OpenStreetMap**    | Buildings, POIs, Land Use, Roads, Power Infrastructure    | Base geometry, network, feature extraction |
| **NREL**             | Residential building typology datasets                    | Building classification for energy demand  |
| **US Census **  | Administrative boundaries (state, county, subdivision)    | Defining regional scope, clipping          |
| **US Census Data**   | Demographic data (population density, potentially others) | Building classification heuristics         |

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
              ├── target_region_boundary.geojson        # Clipped boundary of the processed region
              ├── buildings_classified.shp  # Classified building footprints
              ├── roads_pgr.sql             # Routable road network for pgRouting
              ├── transformers.geojson      # Extracted transformer locations
              ├── pois.geojson              # Clipped Points of Interest
              ├── landuse.geojson           # Clipped land use polygons
              └── ...                       # Other intermediate or raw data files might be stored here
```

---

## Pipeline Workflow

### Step 1: Regional Data Extraction & Preparation
- Parse YAML configuration and establish precise geographic boundaries using Census TIGER/Line data
- Create authoritative target region boundary and set up output directory structure

### Step 2: NREL Data Processing
- Download and process NREL residential building typology datasets
- Extract vintage distribution data for building classification heuristics

### Step 3: OpenStreetMap Data Extraction
- Query and download OSM data (buildings, roads, POIs, power infrastructure) for the target region
- Clip and store processed OSM datasets for subsequent analysis steps

### Step 3.5: Microsoft Buildings Data Processing
- Download Microsoft Buildings footprint data as additional building source
- Integrate and prepare data for building classification pipeline

### Step 4: Building Classification Heuristics
- Apply comprehensive classification strategy using OSM, Census, NREL, and Microsoft Buildings data
- Generate final classified building footprints with typology, structural, and demographic attributes

### Step 5: Routable Road Network Generation
- Process and clean OSM road network topology for routing applications
- Export prepared network as GeoJSON/GPKG files for integration with routing systems


## Documentation

- 📄 Homepage: https://github.com/DAI-Lab/gridtracer
- 📚 Documentation: https://DAI-Lab.github.io/gridtracer

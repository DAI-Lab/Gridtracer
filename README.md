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

## Installation

```bash
make install
```

## Configuration

The pipeline is initialized through a YAML-based configuration file (`gridtracer/config/config.yaml`). This file specifies the geographic scope for data collection:

```yaml
REGION:
  STATE: "MA"                     # Required: State abbreviation (e.g., "MA")
  COUNTY: "Middlesex County"      # Required: Full county name
  COUNTY_SUBDIVISION: "Cambridge city" # Optional: Full county subdivision name. If omitted, processes the entire county.
```

Input validation ensures correct state abbreviations and county names are used via FIPS code lookup.

---

## Data Sources

| Source                     | Data Extracted                                            | Purpose                                    |
|----------------------------|-----------------------------------------------------------|--------------------------------------------|
| **OpenStreetMap**          | Buildings, POIs, Land Use, Roads, Power Infrastructure    | Base geometry, network, feature extraction |
| **NREL**                   | Residential building typology datasets                    | Building vintage distributions for energy modeling |
| **US Census TIGER**        | Administrative boundaries (state, county, subdivision)    | Defining regional scope, FIPS code resolution |
| **US Census Data**         | Demographic data (population density, housing units)      | Building classification heuristics         |
| **Microsoft Buildings**    | ML-derived building footprints with height data          | Enhanced building geometry and attributes  |

---

## Pipeline Outputs

The primary outputs generated for the specified region are:

1.  **Routable Road Network:** An `.sql` file containing the road network processed and formatted for direct import into a PostgreSQL/PostGIS database with the `pgRouting` extension (`roads_pgr.sql`).
2.  **Classified Building Footprints:** A shapefile (`buildings_classified.shp`) containing building polygons with attributes derived from the detailed classification heuristic (see Workflow below).
3.  **Transformer Network:** A GeoJSON file (`transformers.geojson`) containing points representing electrical transformers extracted from OSM.

---

## Output Structure

All outputs are organized in a hierarchical directory structure by administrative region:

```text
output/
└── [STATE]/                        # e.g., MA
    └── [COUNTY]/                   # e.g., Middlesex_County
        └── [SUBDIVISION]/          # e.g., Cambridge_city (optional)
            ├── CENSUS/             # Administrative boundaries and census data
            ├── NREL/               # Building typology distributions
            ├── OSM/                # OpenStreetMap extracts
            ├── MICROSOFT_BUILDINGS/ # ML-derived building footprints
            ├── BUILDINGS_OUTPUT/   # Final classified buildings
            ├── ROAD_NETWORK/       # Routable road networks
            └── PLOTS/              # Visualization outputs
```

---

## Pipeline Workflow

The pipeline processes data through seven sequential stages:

### Step 1: Census Boundary Definition

- Parse YAML configuration and resolve FIPS codes for target region
- Establish precise geographic boundaries using Census TIGER/Line data
- Create output directory structure for all data products

### Step 2: Census Subdivision Segmentation

- Generate comprehensive subdivision datasets for the target region
- Extract population and housing unit metrics for classification heuristics

### Step 3: NREL Data Processing

- Process NREL residential building typology datasets
- Extract vintage distribution data for energy modeling parameters

### Step 4: OpenStreetMap Data Extraction

- Query and download OSM data (buildings, roads, POIs, power infrastructure)
- Clip and store processed OSM datasets for subsequent analysis steps

### Step 5: Microsoft Buildings Integration

- Download ML-derived building footprints with height information
- Integrate with existing building data for enhanced geometry

### Step 6: Building Classification

- Apply energy-focused classification heuristics using all data sources
- Generate final classified building footprints with typology and structural attributes

### Step 7: Road Network Generation

- Process OSM road network topology for routing applications
- Export pgRouting-compatible SQL files for database integration
## Usage

### Running the Pipeline

```bash
# Run the complete data processing pipeline
python -m gridtracer.scripts.main

# Or run directly from the scripts directory
python gridtracer/scripts/main.py
```

### Testing

```bash
# Run tests with coverage
make test

# Run specific test file
python -m pytest tests/path/to/test_file.py

# Check code style
make lint

# Auto-fix code style issues
make fix-lint
```

## Documentation

- 📄 Homepage: [https://github.com/DAI-Lab/gridtracer](https://github.com/DAI-Lab/gridtracer)
- 📚 Documentation: [https://DAI-Lab.github.io/gridtracer](https://DAI-Lab.github.io/gridtracer)

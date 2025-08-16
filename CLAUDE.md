# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Setup and Installation
```bash
# Install package in development mode
make install-develop

# Install package for production
make install

# Install test dependencies only
make install-test
```

### Testing and Quality Assurance
```bash
# Run tests
make test

# Run single test file or test function
python -m pytest tests/path/to/test_file.py
python -m pytest tests/path/to/test_file.py::test_function_name

# Run tests on all Python versions
make test-all

# Check code style and imports
make lint

# Auto-fix code style issues
make fix-lint

# Check test coverage
make coverage
```

### Development Tools
```bash
# Build documentation
make docs

# View documentation in browser
make view-docs

# Create distribution packages
make dist

# Clean all artifacts
make clean
```

### Running the Pipeline
```bash
# Run the complete data processing pipeline
python -m gridtracer.scripts.main

# Or run directly from the scripts directory
python gridtracer/scripts/main.py
```

## Architecture Overview

GridTracer is a geospatial data preprocessing pipeline for synthetic electrical grid modeling. The codebase follows a modular, orchestrated architecture:

### Core Components

1. **Configuration Management** (`gridtracer/config/`)
   - `config_loader.py`: Singleton configuration loader using YAML
   - `config.yaml`: Main configuration file defining regions, data paths, and processing parameters
   - Centralized configuration accessed via `from gridtracer.config import config`

2. **Workflow Orchestration** (`gridtracer/data/workflow.py`)
   - `WorkflowOrchestrator`: Central coordinator managing the entire pipeline
   - Handles FIPS code resolution, output directory creation, and data flow between components
   - Manages region boundaries and OSM parser initialization

3. **Data Import Modules** (`gridtracer/data/imports/`)
   - `base.py`: Abstract base classes for data handlers
   - `census.py`: US Census data processing (administrative boundaries, demographics)
   - `nrel.py`: NREL building typology data processing
   - `osm/osm_data_handler.py`: OpenStreetMap data extraction (buildings, roads, power infrastructure)
   - `osm/road_network_builder.py`: Road network topology generation for pgRouting
   - `msft_building_footprints.py`: Microsoft Buildings footprint data integration

4. **Data Processing** (`gridtracer/data/processing/`)
   - `building_processor.py`: Building classification and energy modeling heuristics
   - `building_schema.py`: Data schemas and validation for building attributes

5. **Main Pipeline** (`gridtracer/scripts/main.py`)
   - Entry point orchestrating the 7-step pipeline process
   - Sequential execution: Census → Subdivision → NREL → OSM → Microsoft Buildings → Building Classification → Road Networks

### Pipeline Flow

The pipeline processes data in this sequence:
1. **Census Boundary Definition**: Resolve FIPS codes and define target boundaries
2. **Census Subdivision Segmentation**: Generate subdivision datasets with population metrics
3. **NREL Processing**: Extract building vintage distributions for energy modeling
4. **OSM Data Extraction**: Download buildings, roads, POIs, and power infrastructure
5. **Microsoft Buildings Integration**: Enrich building data with ML-derived footprints
6. **Building Classification**: Apply energy-focused heuristics to classify and attribute buildings
7. **Road Network Generation**: Create routable networks for pgRouting

### Configuration-Driven Design

The system is heavily configuration-driven via `config.yaml`:
- **Region Selection**: State, county, and optional subdivision using FIPS codes
- **Data Sources**: File paths for OSM PBF files and NREL datasets
- **Processing Parameters**: Building classification thresholds and spatial settings
- **Output Structure**: Hierarchical output directories by administrative region

### Output Structure

All outputs follow a standardized hierarchy:
```
output/
└── [STATE]/
    └── [COUNTY]/
        └── [SUBDIVISION]/  # Optional
            ├── CENSUS/
            ├── NREL/
            ├── OSM/
            ├── MICROSOFT_BUILDINGS/
            ├── BUILDINGS_OUTPUT/
            ├── ROAD_NETWORK/
            └── PLOTS/
```

### Key Patterns

- **Singleton Configuration**: Use `from gridtracer.config import config` for consistent access
- **Orchestrator Pattern**: All data handlers receive a `WorkflowOrchestrator` instance
- **Lazy Initialization**: OSM parser and other heavy resources are initialized on-demand
- **EPSG Consistency**: All spatial data uses EPSG:5070 (NAD83 Conus Albers) for US regions
- **Logging**: Centralized logging configuration via `gridtracer.utils.create_logger()`

### Dependencies

The project relies on key geospatial libraries:
- **geopandas/shapely**: Geometric operations and spatial data handling
- **pyrosm**: High-performance OSM data parsing
- **osmnx**: Road network analysis and routing preparation
- **rasterio**: Raster data processing
- **requests**: API data retrieval

### Testing

Tests are organized by module structure under `tests/`:
- Unit tests for individual data handlers and processors
- Integration tests for workflow orchestration
- Configuration via pytest with coverage reporting
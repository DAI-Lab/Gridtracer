"""Shared test fixtures for data processor tests."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from gridtracer.data.workflow import WorkflowOrchestrator


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Fixture providing a complete sample configuration dictionary matching actual config.yaml."""
    return {
        "region": {
            "STATE": "MA",
            "COUNTY": "Middlesex County",
            "COUNTY_SUBDIVISION": "Cambridge city",
            "LOOKUP_URL": "https://www2.census.gov/geo/docs/reference/codes/files/national_cousub.txt",
        },
        "input_data": {
            "OSM_PBF_FILE": "/path/to/test.pbf",
            "NREL_FILE": "/path/to/nrel.tsv",
        },
        "output_dir": "test_output/",
        "epsg": 5070,
        "census_urls": {
            "BASE_URL": "https://www2.census.gov/geo/tiger/TIGER2020",
            "YEAR": "2020",
        },
        "log_level": "INFO",
        "log_file": "logs/gridtracer.log",
        "building_type_thresholds": {
            "AB_MIN_FLOOR_AREA": 600,
            "AB_MIN_CLUSTER_AREA": 1000,
            "TH_MAX_FLOOR_AREA": 270,
            "TH_MAX_HEIGHT": 12,
            "SFH_MAX_FLOOR_AREA": 200,
            "SFH_MAX_HEIGHT": 12,
            "SFH_MAX_CLUSTER_SIZE": 2,
            "TH_MIN_CLUSTER_SIZE": 3,
            "TH_MIN_NEIGHBORS": 2,
            "MFH_MAX_FLOOR_AREA": 600,
        },
    }


@pytest.fixture
def sample_fips_csv_content() -> str:
    """Fixture providing sample FIPS lookup CSV content for testing."""
    return """STATE,STATEFP,COUNTYFP,COUNTYNAME,COUSUB FP,COUSUBNAME,FUNCSTAT
MA,25,017,Middlesex County,11000,Cambridge city,A
MA,25,017,Middlesex County,22500,Somerville city,A
MA,25,017,Middlesex County,33000,Arlington town,A
MA,25,025,Norfolk County,12345,Boston city,A
CA,06,001,Alameda County,54321,Oakland city,A
"""


@pytest.fixture
def sample_boundary_gdf() -> gpd.GeoDataFrame:
    """Fixture providing a sample boundary GeoDataFrame."""
    boundary_data = {
        "GEOID": ["25017"],
        "NAME": ["Middlesex County"],
        "geometry": [Polygon([(-71.5, 42.3), (-71.5, 42.4), (-71.4, 42.4), (-71.4, 42.3)])],
    }
    return gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config_loader(sample_config, temp_output_dir):
    """Fixture providing a mocked ConfigLoader."""
    # Mock the config object for census handler and other modules that import it
    with patch("gridtracer.config.config") as mock_config:
        mock_config.get_region.return_value = sample_config["region"]
        mock_config.get_output_dir.return_value = temp_output_dir
        mock_config.get_input_data_paths.return_value = sample_config["input_data"]
        mock_config.get_census_urls.return_value = sample_config["census_urls"]
        mock_config.log_level = logging.INFO
        mock_config.log_file = "test.log"
        yield mock_config


@pytest.fixture
def mock_workflow_config(sample_config, temp_output_dir):
    """Fixture providing mocked workflow configuration constants."""
    with patch("gridtracer.data.workflow.REGION", sample_config["region"]), patch(
        "gridtracer.data.workflow.OUTPUT_DIR", temp_output_dir
    ), patch("gridtracer.data.workflow.INPUT_DATA", sample_config["input_data"]), patch(
        "gridtracer.data.workflow.EPSG", sample_config["epsg"]
    ), patch("gridtracer.data.workflow.LOG_LEVEL", logging.INFO), patch(
        "gridtracer.data.workflow.LOG_FILE", sample_config["log_file"]
    ):
        yield


def create_mock_fips_file(filepath: Path, content: str) -> None:
    """Helper function to create mock FIPS file."""
    with open(filepath, "w", encoding="latin-1") as f:
        f.write(content)


@pytest.fixture
def mock_osm_parser():
    """Fixture providing a mocked OSM parser."""
    with patch("gridtracer.data.workflow.OSM") as mock_osm_class:
        mock_osm_instance = Mock()
        mock_osm_class.return_value = mock_osm_instance
        yield mock_osm_instance


@pytest.fixture
def sample_region_boundary() -> gpd.GeoDataFrame:
    """Fixture providing a sample region boundary GeoDataFrame."""
    boundary_data = {
        "GEOID": ["2501792500"],
        "NAME": ["Cambridge city"],
        "geometry": [Polygon([(-71.15, 42.35), (-71.15, 42.40), (-71.10, 42.40), (-71.10, 42.35)])],
    }
    return gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")


@pytest.fixture
def orchestrator_with_fips(mock_workflow_config, sample_fips_csv_content, temp_output_dir):
    """Fixture providing a fully initialized WorkflowOrchestrator with FIPS data."""
    with patch("urllib.request.urlretrieve") as mock_urlretrieve:
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )
        orchestrator = WorkflowOrchestrator()
        yield orchestrator

"""Shared test fixtures for data processor tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from syngrid.data_processor.config import ConfigLoader
from syngrid.data_processor.workflow import WorkflowOrchestrator


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Fixture providing a complete sample configuration dictionary."""
    return {
        'region': {
            'state': 'MA',
            'county': 'Middlesex County',
            'county_subdivision': 'Cambridge city',
            'lookup_url': (
                'https://www2.census.gov/geo/docs/reference/codes/'
                'files/national_cousub.txt'
            )
        },
        'output_dir': 'test_output/',
        'input_data': {
            'osm_pbf_file': '/path/to/test.pbf',
            'nrel_data': '/path/to/nrel.tsv',
            'nlcd_landuse': '/path/to/landuse.tif'
        },
        'overpass': {
            'api_url': 'http://overpass-api.de/api/interpreter',
            'timeout': 500
        },
        'processing': {
            'distance_threshold_meters': 10,
            'crs': 'epsg:4326'
        },
        'api_keys': {
            'census_api_key': 'test_key'
        }
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
        'GEOID': ['25017'],
        'NAME': ['Middlesex County'],
        'geometry': [
            Polygon([(-71.5, 42.3), (-71.5, 42.4), (-71.4, 42.4), (-71.4, 42.3)])
        ]
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
    with patch('syngrid.data_processor.workflow.ConfigLoader') as mock_loader_class:
        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.get_region.return_value = sample_config['region']
        mock_loader.get_output_dir.return_value = temp_output_dir
        mock_loader.get_input_data_paths.return_value = sample_config['input_data']
        mock_loader.get_overpass_config.return_value = sample_config['overpass']
        mock_loader_class.return_value = mock_loader
        yield mock_loader


def create_mock_fips_file(filepath: Path, content: str) -> None:
    """Helper function to create mock FIPS file."""
    with open(filepath, 'w', encoding='latin-1') as f:
        f.write(content)


@pytest.fixture
def orchestrator_with_fips(mock_config_loader, sample_fips_csv_content, temp_output_dir):
    """Fixture providing a fully initialized WorkflowOrchestrator with FIPS data."""
    with patch('urllib.request.urlretrieve') as mock_urlretrieve:
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )
        orchestrator = WorkflowOrchestrator()
        yield orchestrator

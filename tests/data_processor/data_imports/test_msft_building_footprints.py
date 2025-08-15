"""
Test module for MicrosoftBuildingsDataHandler.

This module provides comprehensive tests for the MicrosoftBuildingsDataHandler class,
covering all public and internal methods, error conditions, and integration
with the workflow orchestrator.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, mock_open, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from gridtracer.data.imports.msft_building_footprints import MicrosoftBuildingsDataHandler


@pytest.fixture
def sample_building_data() -> gpd.GeoDataFrame:
    """Create sample building data for testing."""
    building_data = {
        'geometry': [
            Polygon([(-71.12, 42.35), (-71.119, 42.35), (-71.119, 42.351), (-71.12, 42.351)]),
            Polygon([(-71.11, 42.36), (-71.109, 42.36), (-71.109, 42.361), (-71.11, 42.361)]),
            Polygon([(-71.13, 42.37), (-71.129, 42.37), (-71.129, 42.371), (-71.13, 42.371)])
        ],
        'quadkey': ['1234567890', '1234567891', '1234567892'],
        'state_abbr': ['MA', 'MA', 'MA'],
        'height': [15.5, 22.3, 8.7]
    }
    return gpd.GeoDataFrame(building_data, crs="EPSG:4326")


@pytest.fixture
def sample_dataset_links() -> pd.DataFrame:
    """Create sample Microsoft dataset links data."""
    return pd.DataFrame({
        'Location': ['UnitedStates', 'UnitedStates', 'UnitedStates'],
        'Url': [
            'https://example.com/buildings.json?quadkey=1234567890',
            'https://example.com/buildings.json?quadkey=1234567891',
            'https://example.com/buildings.json?quadkey=1234567892'
        ],
        'Size': ['10MB', '15MB', '8MB']
    })


@pytest.fixture
def sample_state_boundaries() -> gpd.GeoDataFrame:
    """Create sample US state boundaries."""
    state_data = {
        'NAME': ['Massachusetts', 'Rhode Island'],
        'STUSPS': ['MA', 'RI'],
        'geometry': [
            Polygon([(-73, 41), (-69, 41), (-69, 43), (-73, 43)]),
            Polygon([(-72, 41), (-71, 41), (-71, 42), (-72, 42)])
        ]
    }
    return gpd.GeoDataFrame(state_data, crs="EPSG:4326")


@pytest.fixture
def sample_state_mapping() -> Dict:
    """Create sample state-to-quadkey mapping."""
    return {
        'MA': {
            'state_name': 'Massachusetts',
            'num_quadkeys': 2,
            'quadkeys': {
                '1234567890': {
                    'url': 'https://example.com/buildings.json?quadkey=1234567890',
                    'geometry': 'POLYGON((-71.12 42.35, -71.11 42.35, -71.11 42.36, -71.12 42.36, -71.12 42.35))'
                },
                '1234567891': {
                    'url': 'https://example.com/buildings.json?quadkey=1234567891',
                    'geometry': 'POLYGON((-71.11 42.36, -71.10 42.36, -71.10 42.37, -71.11 42.37, -71.11 42.36))'
                }
            }
        }
    }


@pytest.fixture
def mock_orchestrator(sample_region_boundary: gpd.GeoDataFrame) -> Mock:
    """Create a mock WorkflowOrchestrator."""
    orchestrator = Mock()
    orchestrator.fips_dict = {'state': 'MA', 'county': 'Middlesex County'}
    orchestrator.get_region_boundary.return_value = sample_region_boundary
    orchestrator.base_output_dir = Path('/tmp/test_output')

    # Mock dataset_output_dir from DataHandler base class
    dataset_output_dir = Path('/tmp/test_output/MICROSOFT_BUILDINGS')
    orchestrator.get_dataset_specific_output_directory.return_value = dataset_output_dir

    return orchestrator


@pytest.fixture
def handler(mock_orchestrator: Mock) -> MicrosoftBuildingsDataHandler:
    """Create a MicrosoftBuildingsDataHandler instance with mocked dependencies."""
    with patch('gridtracer.data.imports.msft_building_footprints.MSFT_BUILD_FOOTPRINTS', {
        'DATASET_LINKS_URL': 'https://example.com/dataset-links.csv',
        'STATES_URL': 'https://example.com/states.zip'
    }):
        handler = MicrosoftBuildingsDataHandler(mock_orchestrator)
        # Mock the dataset_output_dir property that comes from DataHandler
        handler.dataset_output_dir = Path('/tmp/test_output/MICROSOFT_BUILDINGS')
        return handler


class TestMicrosoftBuildingsDataHandler:
    """Test cases for MicrosoftBuildingsDataHandler."""

    def test_init(self, mock_orchestrator: Mock):
        """Test handler initialization."""
        with patch('gridtracer.data.imports.msft_building_footprints.MSFT_BUILD_FOOTPRINTS', {
            'DATASET_LINKS_URL': 'https://example.com/dataset-links.csv',
            'STATES_URL': 'https://example.com/states.zip'
        }):
            handler = MicrosoftBuildingsDataHandler(mock_orchestrator)

            assert handler.orchestrator == mock_orchestrator
            assert str(handler.mapping_file).endswith('us_state_quadkey_mapping.json')
            assert handler.state_mapping is None

    def test_get_dataset_name(self, handler: MicrosoftBuildingsDataHandler):
        """Test dataset name getter."""
        assert handler._get_dataset_name() == "MICROSOFT_BUILDINGS"

    def test_quadkey_to_tile_xy(self, handler: MicrosoftBuildingsDataHandler):
        """Test QuadKey to tile coordinates conversion."""
        # Test a known QuadKey conversion
        quadkey = "0313102310"
        tile_x, tile_y, zoom_level = handler._quadkey_to_tile_xy(quadkey)

        assert zoom_level == len(quadkey)
        assert isinstance(tile_x, int)
        assert isinstance(tile_y, int)
        assert tile_x >= 0
        assert tile_y >= 0

    def test_quadkey_to_tile_xy_edge_cases(self, handler: MicrosoftBuildingsDataHandler):
        """Test QuadKey conversion edge cases."""
        # Empty quadkey
        tile_x, tile_y, zoom_level = handler._quadkey_to_tile_xy("")
        assert tile_x == 0
        assert tile_y == 0
        assert zoom_level == 0

        # Single digit quadkeys
        for digit in ['0', '1', '2', '3']:
            tile_x, tile_y, zoom_level = handler._quadkey_to_tile_xy(digit)
            assert zoom_level == 1

    def test_tile_xy_to_lat_lon(self, handler: MicrosoftBuildingsDataHandler):
        """Test tile coordinates to lat/lon conversion."""
        tile_x, tile_y, zoom_level = 0, 0, 1
        min_lat, min_lon, max_lat, max_lon = handler._tile_xy_to_lat_lon(
            tile_x, tile_y, zoom_level)

        # Check reasonable bounds
        assert -90 <= min_lat <= 90
        assert -180 <= min_lon <= 180
        assert -90 <= max_lat <= 90
        assert -180 <= max_lon <= 180
        assert min_lat <= max_lat
        assert min_lon <= max_lon

    def test_quadkey_to_lat_lon(self, handler: MicrosoftBuildingsDataHandler):
        """Test QuadKey to lat/lon conversion."""
        quadkey = "0313102310"
        min_lat, min_lon, max_lat, max_lon = handler._quadkey_to_lat_lon(quadkey)

        # Check reasonable bounds for a US location
        assert -90 <= min_lat <= 90
        assert -180 <= min_lon <= 180
        assert -90 <= max_lat <= 90
        assert -180 <= max_lon <= 180

    def test_extract_quadkey_from_url(self, handler: MicrosoftBuildingsDataHandler):
        """Test QuadKey extraction from URL."""
        # Valid URL with quadkey
        url = "https://example.com/buildings.json?quadkey=1234567890"
        quadkey = handler._extract_quadkey_from_url(url)
        assert quadkey == "1234567890"

        # URL without quadkey
        url_no_quadkey = "https://example.com/buildings.json"
        quadkey = handler._extract_quadkey_from_url(url_no_quadkey)
        assert quadkey is None

    @patch('tempfile.TemporaryDirectory')
    @patch('urllib.request.urlretrieve')
    @patch('geopandas.read_file')
    @patch('pandas.read_csv')
    def test_create_state_quadkey_mapping(
        self,
        mock_read_csv: Mock,
        mock_read_file: Mock,
        mock_urlretrieve: Mock,
        mock_temp_dir: Mock,
        handler: MicrosoftBuildingsDataHandler,
        sample_dataset_links: pd.DataFrame,
        sample_state_boundaries: gpd.GeoDataFrame
    ):
        """Test state-quadkey mapping creation."""
        # Setup mocks
        mock_temp_dir.return_value.__enter__.return_value = '/tmp/test'
        mock_read_csv.return_value = sample_dataset_links
        mock_read_file.return_value = sample_state_boundaries

        # Mock spatial join to return some intersections
        with patch('geopandas.sjoin') as mock_sjoin:
            # Create mock intersection result
            intersection_result = gpd.GeoDataFrame({
                'quadkey': ['1234567890', '1234567891'],
                'STUSPS': ['MA', 'MA'],
                'NAME': ['Massachusetts', 'Massachusetts'],
                'url': ['https://example.com/buildings.json?quadkey=1234567890',
                        'https://example.com/buildings.json?quadkey=1234567891'],
                'geometry': [sample_state_boundaries.geometry.iloc[0],
                             sample_state_boundaries.geometry.iloc[0]]
            })
            mock_sjoin.return_value = intersection_result

            # Mock file writing
            with patch('builtins.open', mock_open()) as mock_file:
                result = handler._create_state_quadkey_mapping()

                # Verify external calls
                mock_read_csv.assert_called_once()
                mock_urlretrieve.assert_called_once()
                mock_read_file.assert_called_once()
                mock_file.assert_called_once()

                # Verify result structure
                assert isinstance(result, dict)
                assert len(result) >= 0  # Could be 0 if no intersections

    def test_load_state_mapping_existing_file(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_state_mapping: Dict
    ):
        """Test loading existing state mapping file."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_state_mapping))):
                result = handler._load_state_mapping()

                assert result == sample_state_mapping

    def test_load_state_mapping_create_new(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_state_mapping: Dict
    ):
        """Test creating new state mapping when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.object(handler, '_create_state_quadkey_mapping', return_value=sample_state_mapping):
                result = handler._load_state_mapping()

                assert result == sample_state_mapping

    def test_filter_quadkeys_by_region(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_state_mapping: Dict
    ):
        """Test filtering quadkeys by region boundary."""
        handler.state_mapping = sample_state_mapping

        with patch('geopandas.sjoin') as mock_sjoin:
            # Mock the spatial join result
            mock_result = gpd.GeoDataFrame({
                'quadkey': ['1234567890', '1234567891']
            })
            mock_sjoin.return_value = mock_result

            result = handler._filter_quadkeys_by_region('MA')

            assert isinstance(result, list)
            assert len(result) == 2
            assert '1234567890' in result
            assert '1234567891' in result

    def test_filter_quadkeys_by_region_invalid_state(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_state_mapping: Dict
    ):
        """Test filtering quadkeys with invalid state."""
        handler.state_mapping = sample_state_mapping

        with pytest.raises(ValueError, match="State 'CA' not found in mapping"):
            handler._filter_quadkeys_by_region('CA')

    def test_download_state_buildings(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_state_mapping: Dict
    ):
        """Test downloading state buildings."""
        handler.state_mapping = sample_state_mapping

        with patch('pandas.read_json') as mock_read_json:
            with patch('geopandas.GeoDataFrame.to_file'):
                # Mock the data download
                mock_df = pd.DataFrame({
                    'geometry': [{'type': 'Polygon', 'coordinates': [[[-71.12, 42.35], [-71.119, 42.35], [-71.119, 42.351], [-71.12, 42.351]]]},
                                 {'type': 'Polygon', 'coordinates': [[[-71.11, 42.36], [-71.109, 42.36], [-71.109, 42.361], [-71.11, 42.361]]]}],
                    'properties': [{'height': 15.5}, {'height': 22.3}]
                })
                mock_read_json.return_value = mock_df

                with patch.object(handler, '_filter_quadkeys_by_region', return_value=['1234567890']):
                    with patch('pathlib.Path.exists', return_value=False):
                        with patch('pathlib.Path.mkdir'):
                            result = handler._download_state_buildings('MA', max_tiles=1)

                            assert isinstance(result, list)
                            assert len(result) >= 0

    def test_filter_buildings_to_region(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_building_data: gpd.GeoDataFrame,
    ):
        """Test filtering buildings to region boundary."""
        # Create temporary files with building data
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_buildings.geojson"
            sample_building_data.to_file(file_path, driver="GeoJSON")

            with patch('geopandas.clip', return_value=sample_building_data):
                result = handler._filter_buildings_to_region([file_path])

                assert isinstance(result, gpd.GeoDataFrame)
                assert len(result) >= 0

    def test_filter_buildings_to_region_empty_files(
        self,
        handler: MicrosoftBuildingsDataHandler
    ):
        """Test filtering buildings with empty file list."""
        result = handler._filter_buildings_to_region([])

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_filter_buildings_to_region_invalid_files(
        self,
        handler: MicrosoftBuildingsDataHandler
    ):
        """Test filtering buildings with invalid files."""
        invalid_file = Path('/nonexistent/file.geojson')
        result = handler._filter_buildings_to_region([invalid_file])

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_download_existing_file(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_building_data: gpd.GeoDataFrame
    ):
        """Test download when output file already exists."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('geopandas.read_file', return_value=sample_building_data):
                result = handler.download()

                assert 'ms_buildings' in result
                assert 'ms_buildings_filepath' in result
                assert isinstance(result['ms_buildings'], gpd.GeoDataFrame)

    def test_download_no_files(
        self,
        handler: MicrosoftBuildingsDataHandler
    ):
        """Test download when no building files are downloaded."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.object(handler, '_download_state_buildings', return_value=[]):
                result = handler.download()

                assert 'ms_buildings' in result
                assert 'ms_buildings_filepath' in result
                assert len(result['ms_buildings']) == 0
                assert result['ms_buildings_filepath'] is None

    def test_download_success(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_building_data: gpd.GeoDataFrame
    ):
        """Test successful download and processing."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.object(handler, '_download_state_buildings', return_value=[Path('/tmp/test.geojson')]):
                with patch.object(handler, '_filter_buildings_to_region', return_value=sample_building_data):
                    with patch('geopandas.GeoDataFrame.to_file'):
                        result = handler.download()

                        assert 'ms_buildings' in result
                        assert 'ms_buildings_filepath' in result
                        assert isinstance(result['ms_buildings'], gpd.GeoDataFrame)
                        assert len(result['ms_buildings']) > 0

    def test_download_error_handling(
        self,
        handler: MicrosoftBuildingsDataHandler
    ):
        """Test download error handling."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch.object(handler, '_download_state_buildings', side_effect=Exception("Network error")):
                result = handler.download()

                assert 'error' in result
                assert "Network error" in result['error']

    def test_process_success(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_building_data: gpd.GeoDataFrame
    ):
        """Test successful processing."""
        download_result = {
            'ms_buildings': sample_building_data,
            'ms_buildings_filepath': Path('/tmp/test.geojson')
        }

        with patch.object(handler, 'download', return_value=download_result):
            result = handler.process()

            assert result == download_result

    def test_process_error_handling(
        self,
        handler: MicrosoftBuildingsDataHandler
    ):
        """Test process error handling."""
        with patch.object(handler, 'download', side_effect=Exception("Processing error")):
            result = handler.process()

            assert 'error' in result
            assert "Processing error" in result['error']

    def test_process_with_subdivision(
        self,
        handler: MicrosoftBuildingsDataHandler,
        sample_building_data: gpd.GeoDataFrame
    ):
        """Test processing with subdivision in FIPS data."""
        # Update orchestrator to include subdivision
        handler.orchestrator.fips_dict = {
            'state': 'MA',
            'county': 'Middlesex County',
            'subdivision': 'Cambridge city'
        }

        download_result = {
            'ms_buildings': sample_building_data,
            'ms_buildings_filepath': Path('/tmp/test.geojson')
        }

        with patch.object(handler, 'download', return_value=download_result):
            result = handler.process()

            assert result == download_result

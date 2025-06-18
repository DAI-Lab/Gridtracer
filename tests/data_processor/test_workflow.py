"""
Comprehensive test suite for the WorkflowOrchestrator class.

This module tests the core orchestration functionality including:
- Configuration loading and validation
- FIPS code resolution and error handling
- Directory structure creation and management
- Region boundary management
- OSM parser integration
- Error handling and edge cases
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from gridtracer.data_processor.config import ConfigLoader
from gridtracer.data_processor.workflow import ALL_DATASETS, WorkflowOrchestrator


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
def sample_subdivision_boundary_gdf() -> gpd.GeoDataFrame:
    """Fixture providing a sample subdivision boundary GeoDataFrame."""
    boundary_data = {
        'GEOID': ['2501711000'],
        'NAME': ['Cambridge city'],
        'geometry': [
            Polygon([(-71.15, 42.35), (-71.15, 42.40), (-71.10, 42.40), (-71.10, 42.35)])
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
    with patch('gridtracer.data_processor.workflow.ConfigLoader') as mock_loader_class:
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


class TestWorkflowOrchestratorInitialization:
    """Test suite for WorkflowOrchestrator initialization."""

    @patch('urllib.request.urlretrieve')
    def test_initialization_success(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test successful initialization with valid configuration."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        assert orchestrator.fips_dict is not None
        assert orchestrator.fips_dict['state'] == 'MA'
        assert orchestrator.fips_dict['county'] == 'Middlesex County'
        assert orchestrator.fips_dict['subdivision'] == 'Cambridge city'
        assert orchestrator.is_county_subdivision is True

        # Check that output directories were created
        assert orchestrator.regional_base_output_dir.exists()
        for dataset in ALL_DATASETS:
            dataset_path = orchestrator.regional_base_output_dir / dataset
            assert dataset_path.exists(), f"Dataset directory {dataset} was not created"

    def test_initialization_without_subdivision(
        self,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test initialization without county subdivision."""
        # Modify config to not include subdivision
        region_config = mock_config_loader.get_region()
        del region_config['county_subdivision']

        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), sample_fips_csv_content
            )

            orchestrator = WorkflowOrchestrator()

            assert orchestrator.fips_dict['subdivision'] is None
            assert orchestrator.is_county_subdivision is False

    def test_initialization_missing_config_parameters(self, temp_output_dir):
        """Test initialization with missing required configuration parameters."""
        incomplete_config = {
            'state': 'MA',
            # Missing county and lookup_url
        }

        with patch('gridtracer.data_processor.workflow.ConfigLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_region.return_value = incomplete_config
            mock_loader.get_output_dir.return_value = temp_output_dir
            mock_loader_class.return_value = mock_loader

            with pytest.raises(
                ValueError,
                match="State, county, and lookup_url must be provided"
            ):
                WorkflowOrchestrator()


class TestFIPSCodeResolution:
    """Test suite for FIPS code resolution functionality."""

    @patch('urllib.request.urlretrieve')
    def test_fips_resolution_success(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test successful FIPS code resolution."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        expected_fips = {
            'state': 'MA',
            'state_fips': '25',
            'county': 'Middlesex County',
            'county_fips': '017',
            'subdivision': 'Cambridge city',
            'subdivision_fips': '11000',
            'funcstat': 'A'
        }

        assert orchestrator.fips_dict == expected_fips

    @patch('urllib.request.urlretrieve')
    def test_fips_resolution_invalid_state(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test FIPS resolution with invalid state."""
        # Modify config to use invalid state
        region_config = mock_config_loader.get_region()
        region_config['state'] = 'XX'  # Invalid state

        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        with pytest.raises(ValueError, match="State abbreviation 'XX' not found"):
            WorkflowOrchestrator()

    @patch('urllib.request.urlretrieve')
    def test_fips_resolution_invalid_county(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test FIPS resolution with invalid county."""
        # Modify config to use invalid county
        region_config = mock_config_loader.get_region()
        region_config['county'] = 'Nonexistent County'

        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        with pytest.raises(ValueError, match="County 'Nonexistent County' not found"):
            WorkflowOrchestrator()

    @patch('urllib.request.urlretrieve')
    def test_fips_resolution_invalid_subdivision(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test FIPS resolution with invalid subdivision."""
        # Modify config to use invalid subdivision
        region_config = mock_config_loader.get_region()
        region_config['county_subdivision'] = 'Nonexistent city'

        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        with pytest.raises(ValueError, match="Subdivision 'Nonexistent city' not found"):
            WorkflowOrchestrator()

    @patch('urllib.request.urlretrieve')
    def test_fips_file_download_failure(
        self,
        mock_urlretrieve,
        mock_config_loader,
        temp_output_dir
    ):
        """Test handling of FIPS file download failure."""
        mock_urlretrieve.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            WorkflowOrchestrator()

    @patch('urllib.request.urlretrieve')
    def test_fips_csv_parsing_with_inconsistent_rows(
        self,
        mock_urlretrieve,
        mock_config_loader,
        temp_output_dir
    ):
        """Test FIPS CSV parsing with inconsistent row lengths."""
        # CSV content with some 8-column rows that need merging
        inconsistent_csv = (
            "STATE,STATEFP,COUNTYFP,COUNTYNAME,COUSUB FP,COUSUBNAME,FUNCSTAT\n"
            "MA,25,017,Middlesex County,11000,Cambridge,city,A\n"
            "MA,25,017,Middlesex County,22500,Somerville city,A\n"
        )

        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), inconsistent_csv
        )

        # Should handle the 8-column row by merging columns 5 and 6
        orchestrator = WorkflowOrchestrator()
        assert orchestrator.fips_dict['subdivision'] == 'Cambridge city'


class TestDirectoryManagement:
    """Test suite for directory management functionality."""

    @patch('urllib.request.urlretrieve')
    def test_output_directory_creation(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test creation of output directory structure."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        # Check regional directory structure
        expected_path = temp_output_dir / "MA" / "Middlesex_County" / "Cambridge_city"
        assert orchestrator.regional_base_output_dir == expected_path
        assert expected_path.exists()

        # Check that all dataset directories were created
        for dataset in ALL_DATASETS:
            dataset_dir = expected_path / dataset
            assert dataset_dir.exists(), f"Dataset directory {dataset} missing"

    @patch('urllib.request.urlretrieve')
    def test_get_dataset_specific_directory(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test getting dataset-specific output directories."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        # Test valid dataset names
        for dataset in ALL_DATASETS:
            dataset_dir = orchestrator.get_dataset_specific_output_directory(dataset)
            expected_path = orchestrator.regional_base_output_dir / dataset
            assert dataset_dir == expected_path
            assert dataset_dir.exists()

    @patch('urllib.request.urlretrieve')
    def test_get_dataset_specific_directory_invalid(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test error handling for invalid dataset names."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        with pytest.raises(ValueError, match="Unknown dataset name: INVALID"):
            orchestrator.get_dataset_specific_output_directory("INVALID")


class TestRegionBoundaryManagement:
    """Test suite for region boundary management."""

    @patch('urllib.request.urlretrieve')
    def test_set_and_get_region_boundary(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        sample_boundary_gdf,
        temp_output_dir
    ):
        """Test setting and getting region boundary."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        # Initially should raise error
        with pytest.raises(ValueError, match="Region boundary has not been set yet"):
            orchestrator.get_region_boundary()

        # Set boundary
        orchestrator.set_region_boundary(sample_boundary_gdf)

        # Should now return the boundary
        retrieved_boundary = orchestrator.get_region_boundary()
        assert isinstance(retrieved_boundary, gpd.GeoDataFrame)
        assert len(retrieved_boundary) == len(sample_boundary_gdf)

    @patch('urllib.request.urlretrieve')
    def test_region_boundary_subdivision_status(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test subdivision processing status."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        assert orchestrator.is_subdivision_processing() is True

        # Test without subdivision
        region_config = mock_config_loader.get_region()
        del region_config['county_subdivision']

        orchestrator2 = WorkflowOrchestrator()
        assert orchestrator2.is_subdivision_processing() is False


class TestOSMParserIntegration:
    """Test suite for OSM parser integration."""

    @patch('pathlib.Path.exists')
    @patch('gridtracer.data_processor.workflow.OSM')
    def test_osm_parser_initialization_success(
        self,
        mock_osm_class,
        mock_path_exists,
        mock_config_loader,
        sample_fips_csv_content,
        sample_boundary_gdf,
        temp_output_dir
    ):
        """Test successful OSM parser initialization."""
        # Pre-create the FIPS file so no download is needed
        fips_file_path = temp_output_dir / "national_cousub.txt"
        create_mock_fips_file(fips_file_path, sample_fips_csv_content)

        # Mock Path.exists to return True for all files during test
        mock_path_exists.return_value = True

        mock_osm_instance = Mock()
        mock_osm_class.return_value = mock_osm_instance

        orchestrator = WorkflowOrchestrator()
        orchestrator.set_region_boundary(sample_boundary_gdf)

        # Get OSM parser (should initialize lazily)
        parser = orchestrator.get_osm_parser()

        assert parser is not None
        assert parser == mock_osm_instance
        mock_osm_class.assert_called_once()

    @patch('urllib.request.urlretrieve')
    @patch('pathlib.Path.exists')
    def test_osm_parser_missing_pbf_file(
        self,
        mock_path_exists,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        sample_boundary_gdf,
        temp_output_dir
    ):
        """Test OSM parser initialization with missing PBF file."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )
        mock_path_exists.return_value = False  # PBF file doesn't exist

        orchestrator = WorkflowOrchestrator()
        orchestrator.set_region_boundary(sample_boundary_gdf)

        # Should return None when PBF file doesn't exist
        parser = orchestrator.get_osm_parser()
        assert parser is None

    @patch('pathlib.Path.exists')
    @patch('gridtracer.data_processor.workflow.OSM')
    def test_osm_parser_boundary_projection(
        self,
        mock_osm_class,
        mock_path_exists,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test OSM parser with boundary projection to WGS84."""
        # Pre-create the FIPS file so no download is needed
        fips_file_path = temp_output_dir / "national_cousub.txt"
        create_mock_fips_file(fips_file_path, sample_fips_csv_content)

        # Mock Path.exists to return True for all files during test
        mock_path_exists.return_value = True

        # Create boundary in different CRS
        boundary_data = {
            'GEOID': ['25017'],
            'NAME': ['Middlesex County'],
            'geometry': [
                Polygon([
                    (200000, 900000), (200000, 950000),
                    (250000, 950000), (250000, 900000)
                ])
            ]
        }
        boundary_gdf = gpd.GeoDataFrame(boundary_data, crs="EPSG:3857")  # Web Mercator

        mock_osm_instance = Mock()
        mock_osm_class.return_value = mock_osm_instance

        orchestrator = WorkflowOrchestrator()
        orchestrator.set_region_boundary(boundary_gdf)

        parser = orchestrator.get_osm_parser()

        # Should have called OSM with projected geometry
        assert parser is not None
        mock_osm_class.assert_called_once()

    @patch('pathlib.Path.exists')
    @patch('gridtracer.data_processor.workflow.OSM')
    def test_osm_parser_multiple_geometries(
        self,
        mock_osm_class,
        mock_path_exists,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test OSM parser with multiple boundary geometries."""
        # Pre-create the FIPS file so no download is needed
        fips_file_path = temp_output_dir / "national_cousub.txt"
        create_mock_fips_file(fips_file_path, sample_fips_csv_content)

        # Mock Path.exists to return True for all files during test
        mock_path_exists.return_value = True

        # Create boundary with multiple geometries
        boundary_data = {
            'GEOID': ['25017A', '25017B'],
            'NAME': ['Part A', 'Part B'],
            'geometry': [
                Polygon([(-71.5, 42.3), (-71.5, 42.35), (-71.45, 42.35), (-71.45, 42.3)]),
                Polygon([(-71.45, 42.3), (-71.45, 42.35), (-71.4, 42.35), (-71.4, 42.3)])
            ]
        }
        boundary_gdf = gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")

        mock_osm_instance = Mock()
        mock_osm_class.return_value = mock_osm_instance

        orchestrator = WorkflowOrchestrator()
        orchestrator.set_region_boundary(boundary_gdf)

        parser = orchestrator.get_osm_parser()

        assert parser is not None
        mock_osm_class.assert_called_once()


class TestConfigurationAccess:
    """Test suite for configuration access methods."""

    @patch('urllib.request.urlretrieve')
    def test_configuration_access_methods(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test all configuration access methods."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        # Test region config access
        region_config = orchestrator.get_region_config()
        assert region_config['state'] == 'MA'
        assert region_config['county'] == 'Middlesex County'

        # Test input data paths access
        input_paths = orchestrator.get_input_data_paths()
        assert 'osm_pbf_file' in input_paths
        assert 'nrel_data' in input_paths

        # Test overpass config access
        overpass_config = orchestrator.get_overpass_config()
        assert 'api_url' in overpass_config
        assert 'timeout' in overpass_config

        # Test FIPS dict access
        fips_dict = orchestrator.get_fips_dict()
        assert fips_dict is not None
        assert fips_dict['state'] == 'MA'

    @patch('urllib.request.urlretrieve')
    def test_base_output_directory_access(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test base output directory access."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        base_dir = orchestrator.get_base_output_directory()
        assert isinstance(base_dir, Path)
        assert base_dir.exists()


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_uninitialized_boundary_access(self, mock_config_loader, temp_output_dir):
        """Test error when accessing unset region boundary."""
        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.side_effect = Exception("Should not be called")

            # Mock the entire initialization process
            with patch.object(WorkflowOrchestrator, '_initialize_orchestrator'):
                orchestrator = WorkflowOrchestrator()
                # Manually set the attributes that would be set during initialization
                orchestrator.fips_dict = {
                    'state': 'MA', 'county': 'Test', 'subdivision': None
                }
                orchestrator.regional_base_output_dir = temp_output_dir
                orchestrator.is_county_subdivision = False
                orchestrator.region_boundary_gdf = None  # This should remain None for the test

                with pytest.raises(ValueError, match="Region boundary has not been set yet"):
                    orchestrator.get_region_boundary()

    @patch('urllib.request.urlretrieve')
    def test_deprecated_path_method_warning(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test deprecated path construction method issues warning."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        with pytest.warns(DeprecationWarning, match="get_path_in_output_dir is deprecated"):
            path = orchestrator.get_path_in_output_dir("test", "subdir")
            assert isinstance(path, Path)


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    @patch('urllib.request.urlretrieve')
    def test_full_workflow_county_level(
        self,
        mock_urlretrieve,
        temp_output_dir,
        sample_boundary_gdf
    ):
        """Test complete workflow for county-level processing."""
        # Create config without subdivision
        config_without_subdivision = {
            'region': {
                'state': 'MA',
                'county': 'Middlesex County',
                'lookup_url': 'https://test.com/fips.txt'
            },
            'output_dir': str(temp_output_dir),
            'input_data': {
                'osm_pbf_file': '/test/file.pbf'
            },
            'overpass': {'api_url': 'http://test.com', 'timeout': 300}
        }

        fips_content = (
            "STATE,STATEFP,COUNTYFP,COUNTYNAME,COUSUB FP,COUSUBNAME,FUNCSTAT\n"
            "MA,25,017,Middlesex County,11000,Cambridge city,A\n"
        )

        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), fips_content
        )

        with patch('gridtracer.data_processor.workflow.ConfigLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_region.return_value = config_without_subdivision['region']
            mock_loader.get_output_dir.return_value = Path(
                config_without_subdivision['output_dir']
            )
            mock_loader.get_input_data_paths.return_value = (
                config_without_subdivision['input_data']
            )
            mock_loader.get_overpass_config.return_value = (
                config_without_subdivision['overpass']
            )
            mock_loader_class.return_value = mock_loader

            orchestrator = WorkflowOrchestrator()

            # Verify county-level setup
            assert orchestrator.is_subdivision_processing() is False
            assert orchestrator.fips_dict['subdivision'] is None

            # Test boundary management
            orchestrator.set_region_boundary(sample_boundary_gdf)
            retrieved = orchestrator.get_region_boundary()
            assert len(retrieved) == len(sample_boundary_gdf)

            # Test directory structure
            expected_path = temp_output_dir / "MA" / "Middlesex_County"
            assert orchestrator.regional_base_output_dir == expected_path

    @patch('urllib.request.urlretrieve')
    def test_full_workflow_subdivision_level(
        self,
        mock_urlretrieve,
        mock_config_loader,
        sample_fips_csv_content,
        sample_subdivision_boundary_gdf,
        temp_output_dir
    ):
        """Test complete workflow for subdivision-level processing."""
        mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
            Path(filepath), sample_fips_csv_content
        )

        orchestrator = WorkflowOrchestrator()

        # Verify subdivision-level setup
        assert orchestrator.is_subdivision_processing() is True
        assert orchestrator.fips_dict['subdivision'] == 'Cambridge city'
        assert orchestrator.fips_dict['subdivision_fips'] == '11000'

        # Test boundary management
        orchestrator.set_region_boundary(sample_subdivision_boundary_gdf)
        retrieved = orchestrator.get_region_boundary()
        assert len(retrieved) == len(sample_subdivision_boundary_gdf)

        # Test directory structure includes subdivision
        expected_path = temp_output_dir / "MA" / "Middlesex_County" / "Cambridge_city"
        assert orchestrator.regional_base_output_dir == expected_path

        # Test all dataset directories exist
        for dataset in ALL_DATASETS:
            dataset_dir = orchestrator.get_dataset_specific_output_directory(dataset)
            assert dataset_dir.exists()
            assert dataset_dir == expected_path / dataset

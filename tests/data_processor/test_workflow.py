"""
Comprehensive test suite for the WorkflowOrchestrator class.

This module tests the core orchestration functionality based on the current
implementation including:
- FIPS code resolution and validation
- Directory structure creation and management
- Region boundary management
- OSM parser integration
- Error handling and edge cases
"""

from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from gridtracer.data.workflow import ALL_DATASETS, WorkflowOrchestrator
from tests.conftest import create_mock_fips_file


class TestWorkflowOrchestratorInitialization:
    """Test suite for WorkflowOrchestrator initialization and FIPS resolution."""

    def test_initialization_success(
        self,
        orchestrator_with_fips: WorkflowOrchestrator
    ):
        """Test successful initialization with valid configuration."""
        orchestrator = orchestrator_with_fips

        # Check FIPS data was resolved correctly
        assert orchestrator.fips_dict is not None
        assert orchestrator.fips_dict['state'] == 'MA'
        assert orchestrator.fips_dict['county'] == 'Middlesex County'
        assert orchestrator.fips_dict['subdivision'] == 'Cambridge city'
        assert orchestrator.fips_dict['state_fips'] == '25'
        assert orchestrator.fips_dict['county_fips'] == '017'
        assert orchestrator.fips_dict['subdivision_fips'] == '11000'

        # Check subdivision processing flag
        assert orchestrator.is_county_subdivision is True
        assert orchestrator.is_subdivision_processing() is True

        # Check output directories were created
        assert orchestrator.regional_base_output_dir.exists()
        expected_path = orchestrator.base_output_dir / "MA" / "Middlesex_County" / "Cambridge_city"
        assert orchestrator.regional_base_output_dir == expected_path

        # Check all dataset directories exist
        for dataset in ALL_DATASETS:
            dataset_path = orchestrator.regional_base_output_dir / dataset
            assert dataset_path.exists(), f"Dataset directory {dataset} was not created"

    def test_initialization_county_only(
        self,
        mock_workflow_config,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test initialization without county subdivision."""
        # Create config without subdivision
        region_without_subdivision = {
            'STATE': 'MA',
            'COUNTY': 'Middlesex County',
            'LOOKUP_URL': 'https://test.com/fips.txt'
            # No COUNTY_SUBDIVISION
        }

        with patch('gridtracer.data.workflow.REGION', region_without_subdivision), \
                patch('urllib.request.urlretrieve') as mock_urlretrieve:

            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), sample_fips_csv_content
            )

            orchestrator = WorkflowOrchestrator()

            # Should be county-level processing
            assert orchestrator.fips_dict['subdivision'] is None
            assert orchestrator.fips_dict['subdivision_fips'] is None
            assert orchestrator.is_county_subdivision is False
            assert orchestrator.is_subdivision_processing() is False

            # Directory should be county-level
            expected_path = temp_output_dir / "MA" / "Middlesex_County"
            assert orchestrator.regional_base_output_dir == expected_path

    def test_initialization_missing_required_config(
        self,
        mock_workflow_config,
        temp_output_dir
    ):
        """Test initialization with missing required configuration parameters."""
        incomplete_region = {
            'STATE': 'MA',
            # Missing COUNTY and LOOKUP_URL
        }

        with patch('gridtracer.data.workflow.REGION', incomplete_region):
            with pytest.raises(ValueError, match="State, county, and lookup_url must be provided"):
                WorkflowOrchestrator()

    def test_fips_resolution_invalid_state(
        self,
        mock_workflow_config,
        temp_output_dir
    ):
        """Test FIPS resolution with invalid state."""
        fips_content = (
            "STATE,STATEFP,COUNTYFP,COUNTYNAME,COUSUB FP,COUSUBNAME,FUNCSTAT\n"
            "MA,25,017,Middlesex County,11000,Cambridge city,A\n"
        )

        invalid_region = {
            'STATE': 'XX',  # Invalid state
            'COUNTY': 'Middlesex County',
            'LOOKUP_URL': 'https://test.com/fips.txt'
        }

        with patch('gridtracer.data.workflow.REGION', invalid_region), \
                patch('urllib.request.urlretrieve') as mock_urlretrieve:

            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), fips_content
            )

            with pytest.raises(ValueError, match="State abbreviation 'XX' not found"):
                WorkflowOrchestrator()

    def test_fips_resolution_invalid_county(
        self,
        mock_workflow_config,
        temp_output_dir
    ):
        """Test FIPS resolution with invalid county."""
        fips_content = (
            "STATE,STATEFP,COUNTYFP,COUNTYNAME,COUSUB FP,COUSUBNAME,FUNCSTAT\n"
            "MA,25,017,Middlesex County,11000,Cambridge city,A\n"
        )

        invalid_region = {
            'STATE': 'MA',
            'COUNTY': 'Nonexistent County',  # Invalid county
            'LOOKUP_URL': 'https://test.com/fips.txt'
        }

        with patch('gridtracer.data.workflow.REGION', invalid_region), \
                patch('urllib.request.urlretrieve') as mock_urlretrieve:

            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), fips_content
            )

            with pytest.raises(ValueError, match="County 'Nonexistent County' not found"):
                WorkflowOrchestrator()

    def test_fips_file_download_failure(
        self,
        mock_workflow_config
    ):
        """Test handling of FIPS file download failure."""
        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.side_effect = Exception("Download failed")

            with pytest.raises(Exception, match="Download failed"):
                WorkflowOrchestrator()

    def test_fips_csv_parsing_inconsistent_rows(
        self,
        mock_workflow_config,
        temp_output_dir
    ):
        """Test FIPS CSV parsing with inconsistent row lengths."""
        # CSV with 8-column row that needs merging
        inconsistent_csv = (
            "STATE,STATEFP,COUNTYFP,COUNTYNAME,COUSUB FP,COUSUBNAME,FUNCSTAT\n"
            "MA,25,017,Middlesex County,11000,Cambridge,city,A\n"
        )

        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), inconsistent_csv
            )

            orchestrator = WorkflowOrchestrator()
            # Should handle the 8-column row by merging columns 5 and 6
            assert orchestrator.fips_dict['subdivision'] == 'Cambridge city'


class TestDirectoryManagement:
    """Test suite for directory management functionality."""

    def test_dataset_directory_creation(
        self,
        orchestrator_with_fips: WorkflowOrchestrator
    ):
        """Test creation of dataset-specific directories."""
        orchestrator = orchestrator_with_fips

        # Test all valid dataset names
        for dataset in ALL_DATASETS:
            dataset_dir = orchestrator.get_dataset_specific_output_directory(dataset)

            expected_path = orchestrator.regional_base_output_dir / dataset
            assert dataset_dir == expected_path
            assert dataset_dir.exists()

    def test_invalid_dataset_name(
        self,
        orchestrator_with_fips: WorkflowOrchestrator
    ):
        """Test error handling for invalid dataset names."""
        orchestrator = orchestrator_with_fips

        with pytest.raises(ValueError, match="Unknown dataset name: INVALID"):
            orchestrator.get_dataset_specific_output_directory("INVALID")

    def test_directory_structure_county_level(
        self,
        mock_workflow_config,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test directory structure for county-level processing."""
        region_county_only = {
            'STATE': 'MA',
            'COUNTY': 'Middlesex County',
            'LOOKUP_URL': 'https://test.com/fips.txt'
        }

        with patch('gridtracer.data.workflow.REGION', region_county_only), \
                patch('urllib.request.urlretrieve') as mock_urlretrieve:

            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), sample_fips_csv_content
            )

            orchestrator = WorkflowOrchestrator()

            # Should create county-level path structure
            expected_path = temp_output_dir / "MA" / "Middlesex_County"
            assert orchestrator.regional_base_output_dir == expected_path
            assert expected_path.exists()

            # All dataset directories should exist
            for dataset in ALL_DATASETS:
                dataset_dir = expected_path / dataset
                assert dataset_dir.exists()


class TestRegionBoundaryManagement:
    """Test suite for region boundary management."""

    def test_boundary_not_set_initially(
        self,
        orchestrator_with_fips: WorkflowOrchestrator
    ):
        """Test that boundary is not set initially."""
        orchestrator = orchestrator_with_fips

        assert orchestrator.has_region_boundary() is False

        with pytest.raises(ValueError, match="Region boundary has not been set yet"):
            orchestrator.get_region_boundary()

    def test_set_and_get_region_boundary(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        sample_region_boundary: gpd.GeoDataFrame
    ):
        """Test setting and getting region boundary."""
        orchestrator = orchestrator_with_fips

        # Set boundary
        orchestrator.set_region_boundary(sample_region_boundary)

        # Should now be available
        assert orchestrator.has_region_boundary() is True

        retrieved_boundary = orchestrator.get_region_boundary()
        assert isinstance(retrieved_boundary, gpd.GeoDataFrame)
        assert len(retrieved_boundary) == len(sample_region_boundary)
        assert retrieved_boundary.crs == sample_region_boundary.crs

    def test_boundary_overwrite(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        sample_region_boundary: gpd.GeoDataFrame
    ):
        """Test overwriting existing boundary."""
        orchestrator = orchestrator_with_fips

        # Set initial boundary
        orchestrator.set_region_boundary(sample_region_boundary)

        # Create a different boundary
        new_boundary_data = {
            'GEOID': ['different'],
            'NAME': ['Different Region'],
            'geometry': [
                Polygon([(-72.0, 43.0), (-72.0, 43.1), (-71.9, 43.1), (-71.9, 43.0)])
            ]
        }
        new_boundary = gpd.GeoDataFrame(new_boundary_data, crs="EPSG:4326")

        # Overwrite boundary
        orchestrator.set_region_boundary(new_boundary)

        # Should return new boundary
        retrieved = orchestrator.get_region_boundary()
        assert len(retrieved) == 1
        assert retrieved.iloc[0]['NAME'] == 'Different Region'


class TestOSMParserIntegration:
    """Test suite for OSM parser integration."""

    def test_osm_parser_initialization_with_boundary(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        sample_region_boundary: gpd.GeoDataFrame,
        mock_osm_parser
    ):
        """Test OSM parser initialization with boundary."""
        orchestrator = orchestrator_with_fips
        orchestrator.set_region_boundary(sample_region_boundary)

        # Mock Path.exists to return True for PBF file
        with patch('pathlib.Path.exists', return_value=True):
            parser = orchestrator.get_osm_parser()

            assert parser is not None
            assert parser == mock_osm_parser

    def test_osm_parser_missing_pbf_file(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        sample_region_boundary: gpd.GeoDataFrame
    ):
        """Test OSM parser when PBF file is missing."""
        orchestrator = orchestrator_with_fips
        orchestrator.set_region_boundary(sample_region_boundary)

        # Mock Path.exists to return False for PBF file
        with patch('pathlib.Path.exists', return_value=False):
            parser = orchestrator.get_osm_parser()

            assert parser is None

    def test_osm_parser_without_boundary(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        mock_osm_parser
    ):
        """Test OSM parser initialization without boundary."""
        orchestrator = orchestrator_with_fips

        # Mock Path.exists to return True for PBF file
        with patch('pathlib.Path.exists', return_value=True):
            parser = orchestrator.get_osm_parser()

            assert parser is not None
            assert parser == mock_osm_parser

    def test_osm_parser_lazy_loading(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        mock_osm_parser
    ):
        """Test that OSM parser is lazily loaded."""
        orchestrator = orchestrator_with_fips

        # Initially should be None
        assert orchestrator._osm_parser is None

        with patch('pathlib.Path.exists', return_value=True):
            # First call should initialize
            parser1 = orchestrator.get_osm_parser()
            assert parser1 is not None

            # Second call should return same instance
            parser2 = orchestrator.get_osm_parser()
            assert parser2 is parser1

    def test_osm_parser_initialization_error(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        sample_region_boundary: gpd.GeoDataFrame
    ):
        """Test OSM parser initialization error handling."""
        orchestrator = orchestrator_with_fips
        orchestrator.set_region_boundary(sample_region_boundary)

        with patch('pathlib.Path.exists', return_value=True), \
                patch('gridtracer.data.workflow.OSM', side_effect=Exception("OSM init failed")):

            parser = orchestrator.get_osm_parser()
            assert parser is None


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_malformed_fips_file(
        self,
        mock_workflow_config,
        temp_output_dir
    ):
        """Test handling of malformed FIPS file."""
        malformed_csv = "invalid,csv,content\n"

        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), malformed_csv
            )

            with pytest.raises(ValueError, match="Failed to lookup FIPS codes"):
                WorkflowOrchestrator()

    def test_empty_fips_file(
        self,
        mock_workflow_config,
        temp_output_dir
    ):
        """Test handling of empty FIPS file."""
        empty_csv = ""

        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), empty_csv
            )

            with pytest.raises(ValueError, match="Failed to lookup FIPS codes"):
                WorkflowOrchestrator()


class TestIntegrationScenarios:
    """Test suite for integration scenarios mimicking real usage."""

    def test_typical_data_handler_workflow(
        self,
        orchestrator_with_fips: WorkflowOrchestrator,
        sample_region_boundary: gpd.GeoDataFrame
    ):
        """Test typical workflow as used by data handlers."""
        orchestrator = orchestrator_with_fips

        # 1. Check FIPS data is available (as CensusDataHandler would)
        assert orchestrator.fips_dict is not None
        assert 'state_fips' in orchestrator.fips_dict
        assert 'county_fips' in orchestrator.fips_dict

        # 2. Get dataset-specific directory (as any DataHandler would)
        census_dir = orchestrator.get_dataset_specific_output_directory("CENSUS")
        assert census_dir.exists()
        assert census_dir.name == "CENSUS"

        # 3. Set region boundary (as CensusDataHandler would)
        orchestrator.set_region_boundary(sample_region_boundary)
        assert orchestrator.has_region_boundary()

        # 4. Get OSM parser (as OSMDataHandler would)
        with patch('pathlib.Path.exists', return_value=True), \
                patch('gridtracer.data.workflow.OSM') as mock_osm:
            mock_osm.return_value = Mock()
            parser = orchestrator.get_osm_parser()
            assert parser is not None

    def test_subdivision_vs_county_processing(
        self,
        mock_workflow_config,
        sample_fips_csv_content,
        temp_output_dir
    ):
        """Test the difference between subdivision and county processing."""

        # Test subdivision processing
        region_with_subdivision = {
            'STATE': 'MA',
            'COUNTY': 'Middlesex County',
            'COUNTY_SUBDIVISION': 'Cambridge city',
            'LOOKUP_URL': 'https://test.com/fips.txt'
        }

        with patch('gridtracer.data.workflow.REGION', region_with_subdivision), \
                patch('urllib.request.urlretrieve') as mock_urlretrieve:

            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), sample_fips_csv_content
            )

            orchestrator_subdivision = WorkflowOrchestrator()

            assert orchestrator_subdivision.is_subdivision_processing() is True
            assert orchestrator_subdivision.fips_dict['subdivision'] == 'Cambridge city'

            expected_path = temp_output_dir / "MA" / "Middlesex_County" / "Cambridge_city"
            assert orchestrator_subdivision.regional_base_output_dir == expected_path

        # Test county-only processing
        region_county_only = {
            'STATE': 'MA',
            'COUNTY': 'Middlesex County',
            'LOOKUP_URL': 'https://test.com/fips.txt'
        }

        with patch('gridtracer.data.workflow.REGION', region_county_only), \
                patch('urllib.request.urlretrieve') as mock_urlretrieve:

            mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                Path(filepath), sample_fips_csv_content
            )

            orchestrator_county = WorkflowOrchestrator()

            assert orchestrator_county.is_subdivision_processing() is False
            assert orchestrator_county.fips_dict['subdivision'] is None

            expected_path = temp_output_dir / "MA" / "Middlesex_County"
            assert orchestrator_county.regional_base_output_dir == expected_path

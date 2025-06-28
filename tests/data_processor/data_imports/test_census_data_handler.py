"""
Test module for Census data handler.

This module contains tests to verify the Census data handler works correctly
with the gridtracer pipeline, including boundary processing, census blocks extraction,
and data visualization functionality.
"""

from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from gridtracer.data.imports.census import CensusDataHandler


@pytest.fixture
def sample_subdivision_gdf() -> gpd.GeoDataFrame:
    """Create a sample subdivision GeoDataFrame for testing."""
    subdivision_data = {
        'GEOID': ['2501792500'],
        'STATEFP': ['25'],
        'COUNTYFP': ['017'],
        'COUSUBFP': ['92500'],
        'NAME': ['Cambridge'],
        'geometry': [
            Polygon([(-71.15, 42.3), (-71.05, 42.3), (-71.05, 42.4), (-71.15, 42.4)])
        ]
    }
    return gpd.GeoDataFrame(subdivision_data, crs="EPSG:4326")


@pytest.fixture
def sample_census_blocks_gdf() -> gpd.GeoDataFrame:
    """Create a sample census blocks GeoDataFrame for testing."""
    blocks_data = {
        'GEOID20': ['250170001001000', '250170001001001', '250170001001002'],
        'STATEFP20': ['25', '25', '25'],
        'COUNTYFP20': ['017', '017', '017'],
        'TRACTCE20': ['000100', '000100', '000100'],
        'BLOCKCE20': ['1000', '1001', '1002'],
        'HOUSING20': [45, 23, 67],
        'POP20': [120, 78, 145],
        'geometry': [
            Polygon([(-71.12, 42.35), (-71.11, 42.35), (-71.11, 42.36), (-71.12, 42.36)]),
            Polygon([(-71.11, 42.35), (-71.10, 42.35), (-71.10, 42.36), (-71.11, 42.36)]),
            Polygon([(-71.12, 42.36), (-71.11, 42.36), (-71.11, 42.37), (-71.12, 42.37)])
        ]
    }
    return gpd.GeoDataFrame(blocks_data, crs="EPSG:4326")


@pytest.fixture
def sample_fips_dict() -> Dict[str, str]:
    """Create a sample FIPS dictionary for testing."""
    return {
        'state': 'MA',
        'state_fips': '25',
        'county': 'Middlesex County',
        'county_fips': '017',
        'subdivision': 'Cambridge',
        'subdivision_fips': '92500'
    }


@pytest.fixture
def census_data_handler(orchestrator_with_fips) -> CensusDataHandler:
    """Create a census data handler with a configured orchestrator."""
    return CensusDataHandler(orchestrator_with_fips)


class TestCensusDataHandlerInitialization:
    """Test suite for Census data handler initialization."""

    def test_initialization_and_dataset_name(
        self,
        census_data_handler: CensusDataHandler,
        orchestrator_with_fips
    ) -> None:
        """Test initialization and dataset name method."""
        assert census_data_handler is not None
        assert census_data_handler.orchestrator == orchestrator_with_fips
        assert census_data_handler._get_dataset_name() == "CENSUS"
        assert census_data_handler.dataset_output_dir.exists()

        # Verify FIPS data is accessible
        fips = census_data_handler.orchestrator.get_fips_dict()
        assert fips is not None
        assert fips['state'] == 'MA'


class TestShapefileOperations:
    """Test suite for shapefile download and reading functionality."""

    def test_download_and_read_new_file(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test downloading and reading a new shapefile."""
        test_url = "https://example.com/test_shapefile.zip"
        test_prefix = "test_blocks"

        with patch('geopandas.read_file', return_value=sample_census_blocks_gdf):
            result = census_data_handler._download_and_read_census_shp(test_url, test_prefix)

            assert result is not None
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == len(sample_census_blocks_gdf)

            # Check that file was saved
            expected_path = census_data_handler.dataset_output_dir / f"{test_prefix}.geojson"
            assert expected_path.exists()

    def test_download_and_read_existing_file(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test reading an existing shapefile."""
        test_url = "https://example.com/test_shapefile.zip"
        test_prefix = "existing_blocks"

        # Create existing file
        existing_path = census_data_handler.dataset_output_dir / f"{test_prefix}.geojson"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        sample_census_blocks_gdf.to_file(existing_path, driver="GeoJSON")

        result = census_data_handler._download_and_read_census_shp(test_url, test_prefix)

        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_census_blocks_gdf)

    def test_download_error_handling(
        self,
        census_data_handler: CensusDataHandler
    ) -> None:
        """Test handling of download and file reading errors."""
        test_url = "https://example.com/nonexistent_shapefile.zip"
        test_prefix = "error_test"

        with patch('geopandas.read_file', side_effect=Exception("Download failed")):
            result = census_data_handler._download_and_read_census_shp(test_url, test_prefix)
            assert result is None


class TestDataProcessing:
    """Test suite for main data processing functionality."""

    def test_county_processing(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_fips_dict: Dict[str, str]
    ) -> None:
        """Test processing data for county-only (no subdivision)."""
        # Mock orchestrator methods
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=sample_fips_dict)
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)
        census_data_handler.orchestrator.set_region_boundary = Mock()

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_census_blocks_gdf
        ):
            result = census_data_handler.download_and_process_data()

            assert isinstance(result, dict)
            expected_keys = [
                'target_region_blocks', 'target_region_blocks_filepath',
                'target_region_boundary', 'target_region_boundary_filepath'
            ]
            for key in expected_keys:
                assert key in result
                assert result[key] is not None

            # Verify files were created
            assert Path(result['target_region_blocks_filepath']).exists()
            assert Path(result['target_region_boundary_filepath']).exists()

            # Verify orchestrator method was called
            census_data_handler.orchestrator.set_region_boundary.assert_called_once()

    def test_subdivision_processing(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_subdivision_gdf: gpd.GeoDataFrame,
        sample_fips_dict: Dict[str, str]
    ) -> None:
        """Test processing data for subdivision."""
        # Mock orchestrator methods
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=sample_fips_dict)
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)
        census_data_handler.orchestrator.set_region_boundary = Mock()

        def mock_download_shp(url, filename_prefix):
            if 'cousub' in url:
                return sample_subdivision_gdf
            elif 'tabblock' in url:
                return sample_census_blocks_gdf
            return None

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            side_effect=mock_download_shp
        ):
            result = census_data_handler.download_and_process_data()

            assert isinstance(result, dict)
            assert result['target_region_blocks'] is not None
            assert result['target_region_boundary'] is not None
            census_data_handler.orchestrator.set_region_boundary.assert_called_once()

    def test_error_conditions(
        self,
        census_data_handler: CensusDataHandler,
        sample_fips_dict: Dict[str, str]
    ) -> None:
        """Test various error conditions in data processing."""
        # Test missing FIPS
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=None)
        with pytest.raises(ValueError, match="FIPS dictionary is missing"):
            census_data_handler.download_and_process_data()

        # Test failed block loading
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=sample_fips_dict)
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=None
        ):
            with pytest.raises(ValueError, match="County blocks.*could not be loaded"):
                census_data_handler.download_and_process_data()

        # Test missing county column
        blocks_without_county = gpd.GeoDataFrame({
            'GEOID20': ['123456'],
            'geometry': [Polygon([(-71.1, 42.3), (-71.0, 42.3), (-71.0, 42.4), (-71.1, 42.4)])]
        }, crs="EPSG:4326")

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=blocks_without_county
        ):
            with pytest.raises(ValueError, match="County FIPS column not found"):
                census_data_handler.download_and_process_data()


class TestVisualization:
    """Test suite for census data visualization functionality."""

    def test_visualization_success(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_boundary_gdf: gpd.GeoDataFrame,
        sample_fips_dict: Dict[str, str]
    ) -> None:
        """Test successful census data visualization."""
        # Mock orchestrator methods
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=sample_fips_dict)
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)
        census_data_handler.orchestrator.get_dataset_specific_output_directory = Mock(
            return_value=census_data_handler.dataset_output_dir / "plots"
        )

        # Create plots directory
        plots_dir = census_data_handler.dataset_output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()

        with patch('matplotlib.pyplot.subplots', return_value=(mock_fig, mock_ax)), \
                patch('matplotlib.pyplot.savefig') as mock_savefig, \
                patch('matplotlib.pyplot.close'), \
                patch('contextily.add_basemap'):

            result = census_data_handler._visualize_census_data(
                sample_census_blocks_gdf,
                sample_boundary_gdf
            )

            assert result is not None
            assert isinstance(result, str)
            mock_savefig.assert_called_once()

    def test_visualization_edge_cases(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test visualization edge cases and error handling."""
        # Test with empty blocks
        empty_blocks = gpd.GeoDataFrame()
        result = census_data_handler._visualize_census_data(empty_blocks)
        assert result is None

        # Test with None blocks
        result = census_data_handler._visualize_census_data(None)
        assert result is None

        # Test with missing FIPS
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=None)
        result = census_data_handler._visualize_census_data(sample_census_blocks_gdf)
        assert result is None


class TestPublicInterface:
    """Test suite for public interface methods."""

    def test_download_method(self, census_data_handler: CensusDataHandler) -> None:
        """Test the download method."""
        result = census_data_handler.download()

        assert isinstance(result, dict)
        assert "status" in result
        assert "download_and_process_data" in result["status"]

    def test_process_without_plotting(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test process method without plotting."""
        expected_result = {
            'target_region_blocks': sample_census_blocks_gdf,
            'target_region_blocks_filepath': 'test_blocks.geojson',
            'target_region_boundary': sample_census_blocks_gdf,
            'target_region_boundary_filepath': 'test_boundary.geojson'
        }

        with patch.object(
            census_data_handler,
            'download_and_process_data',
            return_value=expected_result
        ):
            result = census_data_handler.process(plot=False)
            assert result == expected_result

    def test_process_with_plotting(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test process method with plotting enabled."""
        expected_result = {
            'target_region_blocks': sample_census_blocks_gdf,
            'target_region_blocks_filepath': 'test_blocks.geojson',
            'target_region_boundary': sample_census_blocks_gdf,
            'target_region_boundary_filepath': 'test_boundary.geojson'
        }

        with patch.object(
            census_data_handler,
            'download_and_process_data',
            return_value=expected_result
        ), patch.object(
            census_data_handler,
            '_visualize_census_data',
            return_value="test_plot.png"
        ) as mock_visualize:
            result = census_data_handler.process(plot=True)

            assert result == expected_result
            mock_visualize.assert_called_once_with(
                blocks_gdf=sample_census_blocks_gdf,
                boundary_to_plot_gdf=sample_census_blocks_gdf
            )

    def test_process_error_handling(
        self,
        census_data_handler: CensusDataHandler
    ) -> None:
        """Test process method exception handling."""
        with patch.object(
            census_data_handler,
            'download_and_process_data',
            side_effect=Exception("Processing failed")
        ):
            with pytest.raises(Exception, match="Processing failed"):
                census_data_handler.process()


class TestIntegrationWorkflows:
    """Test suite for complete integration workflows."""

    def test_complete_county_workflow(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_fips_dict: Dict[str, str]
    ) -> None:
        """Test complete workflow for county-level processing with plotting."""
        # Setup orchestrator mocks
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=sample_fips_dict)
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)
        census_data_handler.orchestrator.set_region_boundary = Mock()
        census_data_handler.orchestrator.get_dataset_specific_output_directory = Mock(
            return_value=census_data_handler.dataset_output_dir / "plots"
        )

        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_census_blocks_gdf
        ), patch('matplotlib.pyplot.subplots', return_value=(mock_fig, mock_ax)), \
                patch('matplotlib.pyplot.savefig'), \
                patch('matplotlib.pyplot.close'), \
                patch('contextily.add_basemap'):

            result = census_data_handler.process(plot=True)

            # Verify all expected outputs
            assert result['target_region_blocks'] is not None
            assert result['target_region_boundary'] is not None
            assert Path(result['target_region_blocks_filepath']).exists()
            assert Path(result['target_region_boundary_filepath']).exists()

            # Verify orchestrator interactions
            census_data_handler.orchestrator.set_region_boundary.assert_called_once()

    def test_complete_subdivision_workflow(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_subdivision_gdf: gpd.GeoDataFrame,
        sample_fips_dict: Dict[str, str]
    ) -> None:
        """Test complete workflow for subdivision-level processing."""
        # Setup orchestrator mocks
        census_data_handler.orchestrator.get_fips_dict = Mock(return_value=sample_fips_dict)
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)
        census_data_handler.orchestrator.set_region_boundary = Mock()

        def mock_download_shp(url, filename_prefix):
            if 'cousub' in url:
                return sample_subdivision_gdf
            elif 'tabblock' in url:
                return sample_census_blocks_gdf
            return None

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            side_effect=mock_download_shp
        ):
            result = census_data_handler.process(plot=False)

            # Verify all expected outputs
            assert result['target_region_blocks'] is not None
            assert result['target_region_boundary'] is not None
            assert Path(result['target_region_blocks_filepath']).exists()
            assert Path(result['target_region_boundary_filepath']).exists()

            # Verify orchestrator interactions
            census_data_handler.orchestrator.set_region_boundary.assert_called_once()

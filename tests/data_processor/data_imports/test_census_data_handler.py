"""
Test module for CensusDataHandler.

This module provides comprehensive tests for the CensusDataHandler class,
covering all public and internal methods, error conditions, and integration
with the workflow orchestrator.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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
def census_data_handler(orchestrator_with_fips) -> CensusDataHandler:
    """Create a census data handler with a configured orchestrator."""
    return CensusDataHandler(orchestrator_with_fips)


class TestCensusDataHandlerInitialization:
    """Test suite for CensusDataHandler initialization and basic properties."""

    def test_initialization(self, census_data_handler: CensusDataHandler, orchestrator_with_fips):
        """Test proper initialization of CensusDataHandler."""
        assert census_data_handler is not None
        assert census_data_handler.orchestrator == orchestrator_with_fips
        assert census_data_handler.dataset_output_dir.exists()

    def test_dataset_name(self, census_data_handler: CensusDataHandler):
        """Test that dataset name returns correct value."""
        assert census_data_handler._get_dataset_name() == "CENSUS"

    def test_fips_data_access(self, census_data_handler: CensusDataHandler):
        """Test access to FIPS data through orchestrator."""
        fips = census_data_handler.orchestrator.fips_dict
        assert fips is not None
        assert fips['state'] == 'MA'
        assert fips['county'] == 'Middlesex County'
        assert 'state_fips' in fips
        assert 'county_fips' in fips


class TestShapefileDownloadAndCaching:
    """Test suite for shapefile download and caching functionality."""

    @patch('urllib.request.urlretrieve')
    @patch('geopandas.read_file')
    def test_download_new_shapefile(
        self,
        mock_read_file: Mock,
        mock_urlretrieve: Mock,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test downloading and processing a new shapefile."""
        mock_read_file.return_value = sample_census_blocks_gdf
        test_url = "https://example.com/test_shapefile.zip"
        test_prefix = "test_blocks"

        result = census_data_handler._download_and_read_census_shp(test_url, test_prefix)

        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_census_blocks_gdf)
        mock_urlretrieve.assert_called_once()
        mock_read_file.assert_called_once()

        # Verify file was saved
        expected_path = census_data_handler.dataset_output_dir / f"{test_prefix}.geojson"
        assert expected_path.exists()

    def test_load_existing_shapefile(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test loading an existing cached shapefile."""
        test_prefix = "existing_blocks"
        existing_path = census_data_handler.dataset_output_dir / f"{test_prefix}.geojson"

        # Create the cached file
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        sample_census_blocks_gdf.to_file(existing_path, driver="GeoJSON")

        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            result = census_data_handler._download_and_read_census_shp(
                "https://example.com/test.zip", test_prefix
            )

            # Should not download when file exists
            mock_urlretrieve.assert_not_called()
            assert result is not None
            assert len(result) == len(sample_census_blocks_gdf)

    @patch('urllib.request.urlretrieve')
    @patch('geopandas.read_file')
    def test_download_error_handling(
        self,
        mock_read_file: Mock,
        mock_urlretrieve: Mock,
        census_data_handler: CensusDataHandler
    ):
        """Test proper error handling during shapefile download."""
        mock_urlretrieve.side_effect = Exception("Network error")

        result = census_data_handler._download_and_read_census_shp(
            "https://example.com/bad_url.zip", "error_test"
        )

        assert result is None

    def test_read_error_handling(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test error handling when reading corrupted cached files."""
        test_prefix = "corrupted_file"
        corrupted_path = census_data_handler.dataset_output_dir / f"{test_prefix}.geojson"

        # Create a corrupted file
        corrupted_path.parent.mkdir(parents=True, exist_ok=True)
        corrupted_path.write_text("invalid geojson content")

        result = census_data_handler._download_and_read_census_shp(
            "https://example.com/test.zip", test_prefix
        )

        assert result is None


class TestSubdivisionDownload:
    """Test suite for subdivision boundary download functionality."""

    def test_subdivision_download_success(
        self,
        census_data_handler: CensusDataHandler,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test successful subdivision download and filtering."""
        fips = {
            'state_fips': '25',
            'county_fips': '017',
            'subdivision_fips': '92500'
        }

        # Mock orchestrator methods
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_subdivision_gdf
        ) as mock_download:
            result = census_data_handler.download_subdivisions(fips)

            assert result is not None
            assert len(result) == 1
            mock_download.assert_called_once()

            # Verify URL construction
            call_args = mock_download.call_args[0]
            assert 'COUSUB' in call_args[0]
            assert '25' in call_args[0]  # state_fips

    def test_subdivision_download_no_subdivision_processing(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test that subdivision download is skipped when not needed."""
        fips = {
            'state_fips': '25',
            'county_fips': '017',
            'subdivision_fips': '92500'
        }

        # Mock orchestrator to return False
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        result = census_data_handler.download_subdivisions(fips)
        assert result is None

    def test_subdivision_download_no_fips(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test subdivision download when subdivision_fips is None."""
        fips = {
            'state_fips': '25',
            'county_fips': '017',
            'subdivision_fips': None
        }

        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)

        result = census_data_handler.download_subdivisions(fips)
        assert result is None

    def test_subdivision_download_not_found(
        self,
        census_data_handler: CensusDataHandler,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test subdivision download when target subdivision is not found."""
        fips = {
            'state_fips': '25',
            'county_fips': '017',
            'subdivision_fips': '99999'  # Non-existent
        }

        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_subdivision_gdf
        ):
            result = census_data_handler.download_subdivisions(fips)
            assert result is None


class TestBlocksDownload:
    """Test suite for census blocks download functionality."""

    def test_blocks_download_success(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test successful blocks download and filtering."""
        fips = {
            'state_fips': '25',
            'county_fips': '017'
        }

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_census_blocks_gdf
        ) as mock_download:
            result = census_data_handler.download_blocks(fips)

            assert result is not None
            assert len(result) == 3  # All blocks match county_fips
            mock_download.assert_called_once()

            # Verify URL construction
            call_args = mock_download.call_args[0]
            assert 'TABBLOCK20' in call_args[0]
            assert '25' in call_args[0]  # state_fips

    def test_blocks_download_no_data(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test blocks download when no data is returned."""
        fips = {
            'state_fips': '25',
            'county_fips': '017'
        }

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=None
        ):
            with pytest.raises(ValueError, match="County blocks.*could not be loaded"):
                census_data_handler.download_blocks(fips)

    def test_blocks_download_missing_county_column(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test blocks download with missing county FIPS column."""
        fips = {
            'state_fips': '25',
            'county_fips': '017'
        }

        # Create blocks without county column
        blocks_no_county = gpd.GeoDataFrame({
            'GEOID20': ['123456'],
            'geometry': [Polygon([(-71.1, 42.3), (-71.0, 42.3), (-71.0, 42.4), (-71.1, 42.4)])]
        }, crs="EPSG:4326")

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=blocks_no_county
        ):
            with pytest.raises(ValueError, match="County FIPS column not found"):
                census_data_handler.download_blocks(fips)

    def test_blocks_download_no_matching_county(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test blocks download when no blocks match the county FIPS."""
        fips = {
            'state_fips': '25',
            'county_fips': '999'  # Non-matching county
        }

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_census_blocks_gdf
        ):
            with pytest.raises(ValueError, match="No blocks found for county FIPS"):
                census_data_handler.download_blocks(fips)


class TestConfigurationIntegration:
    """Test suite for configuration integration."""

    @patch('gridtracer.data.imports.census.config')
    def test_census_urls_usage(
        self,
        mock_config: Mock,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test that Census URLs are retrieved from configuration."""
        mock_config.get_census_urls.return_value = {
            'BASE_URL': 'https://custom.census.gov/data',
            'YEAR': '2021'
        }

        fips = {
            'state_fips': '25',
            'county_fips': '017'
        }

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_census_blocks_gdf
        ) as mock_download:
            census_data_handler.download_blocks(fips)

            # Verify custom URL was used
            call_args = mock_download.call_args[0]
            assert 'custom.census.gov' in call_args[0]
            assert '2021' in call_args[0]

    @patch('gridtracer.data.imports.census.config')
    def test_census_urls_fallback(
        self,
        mock_config: Mock,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test fallback to default URLs when config is incomplete."""
        mock_config.get_census_urls.return_value = {}  # Empty config

        fips = {
            'state_fips': '25',
            'county_fips': '017'
        }

        with patch.object(
            census_data_handler,
            '_download_and_read_census_shp',
            return_value=sample_census_blocks_gdf
        ) as mock_download:
            census_data_handler.download_blocks(fips)

            # Verify fallback URLs were used
            call_args = mock_download.call_args[0]
            assert 'https://www2.census.gov' in call_args[0]
            assert '2020' in call_args[0]


class TestDataProcessing:
    """Test suite for data processing and clipping functionality."""

    def test_clip_and_filter_data_with_subdivision(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test clipping blocks to subdivision boundary."""
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)

        with patch('geopandas.clip') as mock_clip:
            mock_clip.return_value = sample_census_blocks_gdf

            result = census_data_handler.clip_and_filter_data(
                sample_census_blocks_gdf, sample_subdivision_gdf
            )

            mock_clip.assert_called_once()
            assert result is not None

    def test_clip_and_filter_data_no_subdivision(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test processing without subdivision clipping."""
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        result = census_data_handler.clip_and_filter_data(
            sample_census_blocks_gdf, None
        )

        # Should return filtered data without clipping (but still copies due to filtering)
        assert result is not None
        assert len(result) == len(sample_census_blocks_gdf)  # Same number of polygons
        assert list(result.columns) == list(sample_census_blocks_gdf.columns)

    def test_geometry_filtering(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test that non-polygon geometries are filtered out."""
        from shapely.geometry import LineString, Point

        mixed_geometry_gdf = gpd.GeoDataFrame({
            'GEOID20': ['1', '2', '3'],
            'geometry': [
                Polygon([(-71.1, 42.3), (-71.0, 42.3), (-71.0, 42.4), (-71.1, 42.4)]),
                Point(-71.05, 42.35),  # Point - should be filtered out
                LineString([(-71.1, 42.3), (-71.0, 42.4)])  # Line - should be filtered out
            ]
        }, crs="EPSG:4326")

        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        result = census_data_handler.clip_and_filter_data(mixed_geometry_gdf, None)

        # Should only keep the polygon
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == 'Polygon'


class TestBoundaryProcessing:
    """Test suite for boundary processing functionality."""

    def test_process_boundaries_with_subdivision(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test boundary processing when subdivision is available."""
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)

        result = census_data_handler.process_boundaries(
            sample_census_blocks_gdf, sample_subdivision_gdf
        )

        # Should use subdivision as authoritative boundary
        assert result is sample_subdivision_gdf

    def test_process_boundaries_from_blocks(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test boundary creation from block union."""
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        result = census_data_handler.process_boundaries(sample_census_blocks_gdf, None)

        assert result is not None
        assert len(result) == 1
        assert result.crs == sample_census_blocks_gdf.crs

    def test_process_boundaries_no_data(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test boundary processing when no data is available."""
        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        with pytest.raises(ValueError, match="Failed to determine.*boundary"):
            census_data_handler.process_boundaries(None, None)

    def test_process_boundaries_missing_crs(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test boundary processing when blocks have no CRS."""
        blocks_no_crs = gpd.GeoDataFrame({
            'GEOID20': ['123'],
            'geometry': [Polygon([(-71.1, 42.3), (-71.0, 42.3), (-71.0, 42.4), (-71.1, 42.4)])]
        })  # No CRS set

        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=False)

        with pytest.raises(ValueError, match="CRS missing"):
            census_data_handler.process_boundaries(blocks_no_crs, None)


class TestMainWorkflow:
    """Test suite for main download and process workflow."""

    def test_download_method(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test the download method returns proper intermediate data."""
        with patch.object(
            census_data_handler,
            'download_subdivisions',
            return_value=sample_subdivision_gdf
        ), patch.object(
            census_data_handler,
            'download_blocks',
            return_value=sample_census_blocks_gdf
        ):
            result = census_data_handler.download()

            assert isinstance(result, dict)
            assert 'subdivision_gdf' in result
            assert 'county_blocks_gdf' in result
            assert 'fips' in result
            assert result['subdivision_gdf'] is sample_subdivision_gdf
            assert result['county_blocks_gdf'] is sample_census_blocks_gdf

    def test_process_method_complete_workflow(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test the complete process method workflow."""
        census_data_handler.orchestrator.set_region_boundary = Mock()

        with patch.object(
            census_data_handler,
            'download',
            return_value={
                'subdivision_gdf': sample_subdivision_gdf,
                'county_blocks_gdf': sample_census_blocks_gdf,
                'fips': {'state': 'MA', 'county': 'Middlesex County'}
            }
        ), patch.object(
            census_data_handler,
            'clip_and_filter_data',
            return_value=sample_census_blocks_gdf
        ), patch.object(
            census_data_handler,
            'process_boundaries',
            return_value=sample_subdivision_gdf
        ):
            result = census_data_handler.process()

            # Verify output structure
            assert isinstance(result, dict)
            expected_keys = [
                'target_region_blocks', 'target_region_blocks_filepath',
                'target_region_boundary', 'target_region_boundary_filepath'
            ]
            for key in expected_keys:
                assert key in result

            # Verify files were created
            assert result['target_region_blocks'] is not None
            assert result['target_region_boundary'] is not None
            assert Path(result['target_region_blocks_filepath']).exists()
            assert Path(result['target_region_boundary_filepath']).exists()

            # Verify orchestrator interaction
            census_data_handler.orchestrator.set_region_boundary.assert_called_once()

    def test_process_method_error_handling(
        self,
        census_data_handler: CensusDataHandler
    ):
        """Test process method error handling and propagation."""
        with patch.object(
            census_data_handler,
            'download',
            side_effect=Exception("Download failed")
        ):
            with pytest.raises(Exception, match="Download failed"):
                census_data_handler.process()

    def test_process_method_no_blocks_warning(
        self,
        census_data_handler: CensusDataHandler,
        sample_subdivision_gdf: gpd.GeoDataFrame
    ):
        """Test process method handles empty blocks gracefully."""
        empty_blocks = gpd.GeoDataFrame()

        with patch.object(
            census_data_handler,
            'download',
            return_value={
                'subdivision_gdf': sample_subdivision_gdf,
                'county_blocks_gdf': empty_blocks,
                'fips': {'state': 'MA', 'county': 'Middlesex County'}
            }
        ), patch.object(
            census_data_handler,
            'clip_and_filter_data',
            return_value=empty_blocks
        ), patch.object(
            census_data_handler,
            'process_boundaries',
            return_value=sample_subdivision_gdf
        ):
            # Should not raise an error, just log a warning
            result = census_data_handler.process()

            assert result['target_region_blocks'] is None
            assert result['target_region_blocks_filepath'] is None


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and comprehensive error handling."""

    def test_crs_mismatch_handling(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test handling of CRS mismatches between blocks and subdivision."""
        # Create subdivision with different CRS
        subdivision_different_crs = sample_census_blocks_gdf.copy()
        subdivision_different_crs = subdivision_different_crs.to_crs("EPSG:3857")

        census_data_handler.orchestrator.is_subdivision_processing = Mock(return_value=True)

        with patch('geopandas.clip') as mock_clip:
            mock_clip.return_value = sample_census_blocks_gdf

            census_data_handler.clip_and_filter_data(
                sample_census_blocks_gdf, subdivision_different_crs
            )

            # Should have reprojected before clipping
            mock_clip.assert_called_once()

    def test_temporary_file_cleanup(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test that temporary files are properly cleaned up."""
        with patch('tempfile.NamedTemporaryFile') as mock_temp, \
                patch('urllib.request.urlretrieve') as mock_retrieve, \
                patch('geopandas.read_file', return_value=sample_census_blocks_gdf), \
                patch('pathlib.Path.unlink') as mock_unlink:

            mock_temp_file = MagicMock()
            mock_temp_file.name = '/tmp/test_file.zip'
            mock_temp.return_value.__enter__.return_value = mock_temp_file

            census_data_handler._download_and_read_census_shp(
                "https://example.com/test.zip", "test_prefix"
            )

            # Verify cleanup was called
            mock_unlink.assert_called_once()

    def test_file_save_error_handling(
        self,
        census_data_handler: CensusDataHandler,
        sample_census_blocks_gdf: gpd.GeoDataFrame
    ):
        """Test handling of file save errors during shapefile processing."""
        with patch('urllib.request.urlretrieve') as mock_retrieve, \
                patch('geopandas.read_file', return_value=sample_census_blocks_gdf), \
                patch.object(sample_census_blocks_gdf, 'to_file', side_effect=Exception("Disk full")):

            # Should handle the error gracefully and return None
            result = census_data_handler._download_and_read_census_shp(
                "https://example.com/test.zip", "error_prefix"
            )

            # Should return None when file save fails
            assert result is None

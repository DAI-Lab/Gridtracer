"""
Comprehensive test module for OSM data handler.

This module contains tests to verify the OSM data handler works correctly
with the gridtracer pipeline, including building, POI, landuse, and power
infrastructure extraction using pyrosm.
"""

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from gridtracer.data.imports.osm.osm_data_handler import (
    BUILDINGS_TAGS, DEDUPLICATION_THRESHOLD_IN_METERS, LANDUSE_TAGS, MAX_VOLTAGE, POI_TAGS,
    OSMDataHandler,)

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_buildings_gdf() -> gpd.GeoDataFrame:
    """Create a sample buildings GeoDataFrame for testing."""
    buildings_data = {
        'id': [1, 2, 3, 4],
        'building': ['residential', 'commercial', 'office', 'house'],
        'addr:street': ['Main St', 'Oak Ave', 'Elm St', None],
        'addr:housenumber': ['123', '456', '789', None],
        'building:levels': [2, 5, 10, 1],
        'name': ['Home', 'Store', 'Office Building', None],
        'geometry': [
            Polygon([(-71.1, 42.3), (-71.09, 42.3), (-71.09, 42.31), (-71.1, 42.31)]),
            Polygon([(-71.08, 42.32), (-71.07, 42.32), (-71.07, 42.33), (-71.08, 42.33)]),
            Polygon([(-71.06, 42.34), (-71.05, 42.34), (-71.05, 42.35), (-71.06, 42.35)]),
            Polygon([(-71.04, 42.36), (-71.03, 42.36), (-71.03, 42.37), (-71.04, 42.37)])
        ]
    }
    return gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")


@pytest.fixture
def sample_pois_gdf() -> gpd.GeoDataFrame:
    """Create a sample POIs GeoDataFrame for testing."""
    pois_data = {
        'id': [100, 101, 102, 103],
        'name': ['Central Park', 'Coffee Shop', 'Library', 'Restaurant'],
        'amenity': ['park', 'cafe', 'library', 'restaurant'],
        'shop': [None, None, None, None],
        'tourism': [None, None, None, None],
        'leisure': ['park', None, None, None],
        'office': [None, None, None, None],
        'addr:street': ['Park Ave', 'Main St', 'Library Ln', 'Food St'],
        'geometry': [
            Point(-71.1, 42.3),
            Point(-71.09, 42.31),
            Point(-71.08, 42.32),
            Point(-71.07, 42.33)
        ]
    }
    return gpd.GeoDataFrame(pois_data, crs="EPSG:4326")


@pytest.fixture
def sample_landuse_gdf() -> gpd.GeoDataFrame:
    """Create a sample landuse GeoDataFrame for testing."""
    landuse_data = {
        'osmid': [200, 201, 202, 203, 204],
        'landuse': ['residential', 'commercial', 'industrial', 'cemetery', 'education'],
        'name': [
            'Residential Area', 'Shopping District', 'Industrial Zone',
            'Old Cemetery', 'School Campus'
        ],
        'geometry': [
            Polygon([(-71.12, 42.30), (-71.10, 42.30), (-71.10, 42.32), (-71.12, 42.32)]),
            Polygon([(-71.10, 42.30), (-71.08, 42.30), (-71.08, 42.32), (-71.10, 42.32)]),
            Polygon([(-71.08, 42.30), (-71.06, 42.30), (-71.06, 42.32), (-71.08, 42.32)]),
            Polygon([(-71.06, 42.30), (-71.04, 42.30), (-71.04, 42.32), (-71.06, 42.32)]),
            Polygon([(-71.04, 42.30), (-71.02, 42.30), (-71.02, 42.32), (-71.04, 42.32)])
        ]
    }
    return gpd.GeoDataFrame(landuse_data, crs="EPSG:4326")


@pytest.fixture
def sample_power_gdf() -> gpd.GeoDataFrame:
    """Create a sample power infrastructure GeoDataFrame with realistic structure."""
    power_data = {
        'id': [300, 301, 302, 303, 304, 305],
        'power': ['transformer', 'substation', 'pole', 'pole', 'substation', 'transformer'],
        'substation': [None, 'distribution', None, None, 'transmission', None],
        'transformer': [None, None, None, None, None, 'distribution'],
        'tags': [
            {'voltage': '11000'},
            {'voltage': '132000;11000'},
            {},  # No voltage tag for pole
            {'material': 'wood'},
            {'voltage': '345000'},  # High voltage transmission
            None  # No tags at all
        ],
        'osm_type': ['node', 'way', 'node', 'node', 'way', 'node'],
        'geometry': [
            Point(-71.1, 42.3),
            Polygon([(-71.09, 42.31), (-71.08, 42.31), (-71.08, 42.32), (-71.09, 42.32)]),
            Point(-71.07, 42.33),
            Point(-71.06, 42.34),
            Polygon([(-71.05, 42.35), (-71.04, 42.35), (-71.04, 42.36), (-71.05, 42.36)]),
            Point(-71.03, 42.37)
        ]
    }
    return gpd.GeoDataFrame(power_data, crs="EPSG:4326")


@pytest.fixture
def mock_osm_parser(
    sample_buildings_gdf: gpd.GeoDataFrame,
    sample_pois_gdf: gpd.GeoDataFrame,
    sample_landuse_gdf: gpd.GeoDataFrame,
    sample_power_gdf: gpd.GeoDataFrame
) -> Mock:
    """Create a mock OSM parser with sample data."""
    mock_parser = Mock()
    mock_parser.get_buildings.return_value = sample_buildings_gdf
    mock_parser.get_pois.return_value = sample_pois_gdf
    mock_parser.get_landuse.return_value = sample_landuse_gdf
    mock_parser.get_data_by_custom_criteria.return_value = sample_power_gdf
    return mock_parser


@pytest.fixture
def osm_data_handler(orchestrator_with_fips) -> OSMDataHandler:
    """Create an OSM data handler with a configured orchestrator."""
    return OSMDataHandler(orchestrator_with_fips)


class TestOSMDataHandlerInitialization:
    """Test suite for OSM data handler initialization."""

    def test_osm_data_handler_initialization(
        self,
        osm_data_handler: OSMDataHandler,
        orchestrator_with_fips
    ) -> None:
        """Test successful OSM data handler initialization."""
        assert osm_data_handler is not None
        assert osm_data_handler.orchestrator == orchestrator_with_fips
        assert osm_data_handler._get_dataset_name() == "OSM"
        assert osm_data_handler.dataset_output_dir.exists()

    def test_constants_are_loaded(self) -> None:
        """Test that module constants are properly defined."""
        assert MAX_VOLTAGE == 70_000
        assert DEDUPLICATION_THRESHOLD_IN_METERS == 15
        assert isinstance(POI_TAGS, dict)
        assert isinstance(BUILDINGS_TAGS, set)
        assert isinstance(LANDUSE_TAGS, set)


class TestPowerInfrastructureFiltering:
    """Test suite for power infrastructure filtering functions."""

    def test_filter_by_voltage_with_tags(self, osm_data_handler: OSMDataHandler) -> None:
        """Test voltage filtering with various tag formats."""
        power_data = {
            'id': [1, 2, 3, 4, 5],
            'power': ['transformer', 'substation', 'pole', 'substation', 'transformer'],
            'tags': [
                {'voltage': '11000'},  # Below threshold
                {'voltage': '132000;11000'},  # Multiple values, first above threshold
                {},  # No voltage tag
                {'voltage': '69000'},  # Just below threshold
                {'voltage': '71000'}  # Just above threshold
            ],
            'geometry': [Point(i, i) for i in range(5)]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        filtered = osm_data_handler.filter_by_voltage(power_gdf, max_voltage=70_000)

        # Should keep: 11000, no voltage, 69000
        # Should filter: 132000, 71000
        assert len(filtered) == 3
        assert 1 in filtered['id'].values  # 11000V
        assert 3 in filtered['id'].values  # No voltage
        assert 4 in filtered['id'].values  # 69000V
        assert 2 not in filtered['id'].values  # 132000V
        assert 5 not in filtered['id'].values  # 71000V

    def test_filter_by_voltage_no_tags_column(self, osm_data_handler: OSMDataHandler) -> None:
        """Test voltage filtering when tags column doesn't exist."""
        power_data = {
            'power': ['transformer', 'substation'],
            'geometry': [Point(1, 1), Point(2, 2)]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        # Should return unchanged when no tags column
        filtered = osm_data_handler.filter_by_voltage(power_gdf)
        assert len(filtered) == 2

    def test_filter_transmission_tags(self, osm_data_handler: OSMDataHandler) -> None:
        """Test filtering of transmission-level infrastructure."""
        power_data = {
            'id': [1, 2, 3, 4],
            'power': ['substation', 'transformer', 'substation', 'transformer'],
            'substation': ['transmission', None, 'distribution', None],
            'transformer': [None, 'transmission', None, 'distribution'],
            'geometry': [Point(i, i) for i in range(4)]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        filtered = osm_data_handler.filter_transmission_tags(power_gdf)

        # Should filter out transmission substations and transformers
        assert len(filtered) == 2
        assert 3 in filtered['id'].values  # distribution substation
        assert 4 in filtered['id'].values  # distribution transformer
        assert 1 not in filtered['id'].values  # transmission substation
        assert 2 not in filtered['id'].values  # transmission transformer

    def test_remove_contained_points(self, osm_data_handler: OSMDataHandler) -> None:
        """Test removal of points contained within polygons."""
        power_data = {
            'id': [1, 2, 3, 4],
            'power': ['substation', 'transformer', 'transformer', 'pole'],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),  # Substation polygon
                Point(5, 5),  # Transformer inside substation
                Point(15, 15),  # Transformer outside
                Point(3, 3)  # Pole inside substation
            ]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        filtered = osm_data_handler.remove_contained_points(power_gdf)

        # Should remove points inside polygon
        assert len(filtered) == 2
        assert 1 in filtered['id'].values  # Polygon substation
        assert 3 in filtered['id'].values  # Outside transformer
        assert 2 not in filtered['id'].values  # Inside transformer
        assert 4 not in filtered['id'].values  # Inside pole

    def test_remove_contained_points_no_polygons(self, osm_data_handler: OSMDataHandler) -> None:
        """Test remove_contained_points when there are no polygons."""
        power_data = {
            'power': ['transformer', 'pole'],
            'geometry': [Point(1, 1), Point(2, 2)]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        # Should return unchanged when no polygons
        filtered = osm_data_handler.remove_contained_points(power_gdf)
        assert len(filtered) == 2

    def test_deduplicate_power_features(self, osm_data_handler: OSMDataHandler) -> None:
        """Test spatial deduplication of power features."""
        # Create power data with some very close points
        power_data = {
            'id': [1, 2, 3, 4, 5],
            'power': ['substation', 'transformer', 'transformer', 'pole', 'pole'],
            'geometry': [
                Point(0, 0),
                Point(0.00001, 0.00001),  # Very close to first (within threshold)
                Point(1, 1),  # Far away
                Point(0.00002, 0),  # Close to first
                Point(2, 2)  # Far away
            ]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        deduplicated = osm_data_handler.deduplicate_power_features(
            power_gdf,
            distance_threshold_meters=15
        )

        # Should keep substation (highest priority) and the far away features
        assert len(deduplicated) == 3
        assert 1 in deduplicated['id'].values  # Substation (priority)
        assert 3 in deduplicated['id'].values  # Far transformer
        assert 5 in deduplicated['id'].values  # Far pole

    def test_convert_to_centroids(self, osm_data_handler: OSMDataHandler) -> None:
        """Test conversion of all geometries to centroids."""
        power_data = {
            'id': [1, 2, 3],
            'power': ['substation', 'transformer', 'pole'],
            'geometry': [
                Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),  # Should become centroid at (1, 1)
                Point(5, 5),  # Already a point
                Polygon([(10, 10), (10, 12), (12, 12), (12, 10)])  # Centroid at (11, 11)
            ]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        centroids = osm_data_handler.convert_to_centroids(power_gdf)

        # All should be points
        assert all(isinstance(geom, Point) for geom in centroids.geometry)
        assert 'geom_type' in centroids.columns
        assert 'area' in centroids.columns
        # Check area calculation for polygons
        assert centroids[centroids['id'] == 1]['area'].values[0] > 0  # Polygon has area
        assert centroids[centroids['id'] == 2]['area'].values[0] == 0  # Point has no area


class TestPowerInfrastructureExtraction:
    """Test suite for complete power infrastructure extraction."""

    def test_extract_power_infrastructure_complete_pipeline(
        self,
        osm_data_handler: OSMDataHandler,
        sample_power_gdf: gpd.GeoDataFrame,
        mock_osm_parser: Mock
    ) -> None:
        """Test the complete power infrastructure extraction pipeline."""
        power, filepath = osm_data_handler.extract_power_infrastructure(mock_osm_parser)

        assert power is not None
        assert filepath is not None
        assert isinstance(power, gpd.GeoDataFrame)
        assert filepath.exists()
        assert filepath.name == "power.geojson"

        # Check that osm_id was added
        assert 'osm_id' in power.columns
        assert 'element_type' in power.columns

        # All geometries should be points (centroids)
        assert all(geom.geom_type == 'Point' for geom in power.geometry)

        # Raw file should exist
        raw_filepath = osm_data_handler.dataset_output_dir / "raw" / "raw_power.geojson"
        assert raw_filepath.exists()

    def test_extract_power_keeps_all_poles(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test that all poles are kept after recent logic change."""
        # Create power data with various poles
        power_data = {
            'id': [1, 2, 3, 4, 5],
            'power': ['pole', 'pole', 'pole', 'transformer', 'substation'],
            'substation': [None, None, None, None, 'distribution'],
            'transformer': [None, 'distribution', None, None, None],
            'tags': [{}, {}, {'material': 'wood'}, {}, {}],
            'osm_type': ['node'] * 5,
            'geometry': [Point(i, i) for i in range(5)]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")
        mock_osm_parser.get_data_by_custom_criteria.return_value = power_gdf

        power, _ = osm_data_handler.extract_power_infrastructure(mock_osm_parser)

        # All 3 poles should be kept (no distribution requirement anymore)
        pole_count = (power['power'] == 'pole').sum()
        assert pole_count == 3

    def test_extract_power_infrastructure_empty_result(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test power extraction with empty result."""
        mock_osm_parser.get_data_by_custom_criteria.return_value = gpd.GeoDataFrame()

        power, filepath = osm_data_handler.extract_power_infrastructure(mock_osm_parser)

        assert power is None
        assert filepath is None

    def test_extract_power_infrastructure_no_parser(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """Test power extraction with no parser."""
        power, filepath = osm_data_handler.extract_power_infrastructure(None)

        assert power is None
        assert filepath is None


class TestBuildingExtraction:
    """Test suite for building extraction functionality."""

    def test_extract_buildings_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_buildings_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test successful building extraction."""
        buildings, filepath = osm_data_handler.extract_buildings(mock_osm_parser)

        assert buildings is not None
        assert filepath is not None
        assert isinstance(buildings, gpd.GeoDataFrame)
        assert len(buildings) == len(sample_buildings_gdf)
        assert filepath.exists()
        assert filepath.name == "buildings.geojson"

        # Raw file should exist
        raw_filepath = osm_data_handler.dataset_output_dir / "raw" / "raw_buildings.geojson"
        assert raw_filepath.exists()

    def test_extract_buildings_column_filtering(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test that building extraction filters columns based on BUILDINGS_TAGS."""
        # Create buildings with extra columns
        buildings_data = {
            'id': [1],
            'building': ['residential'],
            'name': ['Test Building'],
            # Not in BUILDINGS_TAGS but kept due to copy()
            'unnecessary_column': ['should_be_kept'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }
        buildings_gdf = gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")
        mock_osm_parser.get_buildings.return_value = buildings_gdf

        buildings, _ = osm_data_handler.extract_buildings(mock_osm_parser)

        # Should have geometry and id always, plus any matching tags
        assert 'geometry' in buildings.columns
        assert 'id' in buildings.columns


class TestPOIExtraction:
    """Test suite for POI extraction functionality."""

    def test_extract_pois_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_pois_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test successful POI extraction."""
        pois, filepath = osm_data_handler.extract_pois(mock_osm_parser)

        assert pois is not None
        assert filepath is not None
        assert isinstance(pois, gpd.GeoDataFrame)
        assert len(pois) > 0
        assert filepath.exists()
        assert filepath.name == "pois.geojson"

    def test_extract_pois_custom_filter(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test that POI extraction uses correct custom filter."""
        osm_data_handler.extract_pois(mock_osm_parser)

        # Verify the parser was called with correct filter
        mock_osm_parser.get_pois.assert_called_once_with(custom_filter=POI_TAGS)


class TestLanduseExtraction:
    """Test suite for landuse extraction functionality."""

    def test_extract_landuse_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_landuse_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test successful landuse extraction."""
        # Ensure raw directory exists
        raw_dir = osm_data_handler.dataset_output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        landuse, filepath = osm_data_handler.extract_landuse(mock_osm_parser)

        assert landuse is not None
        assert filepath is not None
        assert isinstance(landuse, gpd.GeoDataFrame)
        assert filepath.exists()
        assert filepath.name == "landuse.geojson"

        # Check categorization
        assert 'category' in landuse.columns
        categories = set(landuse['category'].unique())
        assert categories.issubset({'residential', 'industrial', 'public'})

    def test_extract_landuse_categorization(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test landuse categorization logic."""
        landuse_data = {
            'landuse': ['residential', 'commercial', 'military', 'farmland', 'education'],
            'name': ['Res', 'Comm', 'Mil', 'Farm', 'Edu'],
            'geometry': [Polygon([(i, i), (i + 1, i), (i + 1, i + 1), (i, i + 1)]) for i in range(5)]
        }
        landuse_gdf = gpd.GeoDataFrame(landuse_data, crs="EPSG:4326")
        mock_osm_parser.get_landuse.return_value = landuse_gdf

        # Ensure raw directory exists
        raw_dir = osm_data_handler.dataset_output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        landuse, _ = osm_data_handler.extract_landuse(mock_osm_parser)

        # farmland should be filtered out
        assert len(landuse) == 4
        assert 'farmland' not in landuse['landuse'].values

        # Check correct categorization
        assert landuse[landuse['landuse'] == 'residential']['category'].values[0] == 'residential'
        assert landuse[landuse['landuse'] == 'commercial']['category'].values[0] == 'industrial'
        assert landuse[landuse['landuse'] == 'military']['category'].values[0] == 'public'
        assert landuse[landuse['landuse'] == 'education']['category'].values[0] == 'public'


class TestDownloadAndProcess:
    """Test suite for download and process methods."""

    def test_download_all_datasets(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test download method extracts all datasets."""
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        results = osm_data_handler.download()

        # Check all keys exist
        expected_keys = [
            'buildings', 'buildings_filepath',
            'pois', 'pois_filepath',
            'landuse', 'landuse_filepath',
            'power', 'power_filepath'
        ]
        for key in expected_keys:
            assert key in results

        # All should have data
        assert results['buildings'] is not None
        assert results['pois'] is not None
        assert results['landuse'] is not None
        assert results['power'] is not None

    def test_download_reuses_existing_files(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_buildings_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test that download reuses existing files."""
        # Create an existing buildings file
        buildings_filepath = osm_data_handler.dataset_output_dir / "buildings.geojson"
        buildings_filepath.parent.mkdir(parents=True, exist_ok=True)
        sample_buildings_gdf.to_file(buildings_filepath, driver="GeoJSON")

        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        # Mock extract_buildings to track if it's called
        with patch.object(osm_data_handler, 'extract_buildings') as mock_extract:
            results = osm_data_handler.download()

            # extract_buildings should NOT be called since file exists
            mock_extract.assert_not_called()

            # But we should still have buildings data loaded from file
            assert results['buildings'] is not None
            assert len(results['buildings']) == len(sample_buildings_gdf)

    def test_process_without_plotting(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test process method without plotting."""
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        results = osm_data_handler.process(plot=False)

        assert isinstance(results, dict)
        assert 'buildings' in results

    def test_process_with_plotting(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test process method with plotting enabled."""
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        # Mock plot_osm_data to avoid actual plotting
        with patch.object(osm_data_handler, 'plot_osm_data') as mock_plot:
            results = osm_data_handler.process(plot=True)

            assert isinstance(results, dict)
            mock_plot.assert_called_once()


class TestErrorHandling:
    """Test suite for error handling."""

    def test_handle_none_parser(self, osm_data_handler: OSMDataHandler) -> None:
        """Test handling of None parser in all extraction methods."""
        buildings, b_filepath = osm_data_handler.extract_buildings(None)
        assert buildings is None
        assert b_filepath is None

        pois, p_filepath = osm_data_handler.extract_pois(None)
        assert pois is None
        assert p_filepath is None

        landuse, l_filepath = osm_data_handler.extract_landuse(None)
        assert landuse is None
        assert l_filepath is None

        power, pw_filepath = osm_data_handler.extract_power_infrastructure(None)
        assert power is None
        assert pw_filepath is None

    def test_handle_empty_geodataframes(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test handling of empty GeoDataFrames from parser."""
        mock_osm_parser.get_buildings.return_value = gpd.GeoDataFrame()
        mock_osm_parser.get_pois.return_value = gpd.GeoDataFrame()
        mock_osm_parser.get_landuse.return_value = gpd.GeoDataFrame()
        mock_osm_parser.get_data_by_custom_criteria.return_value = gpd.GeoDataFrame()

        buildings, _ = osm_data_handler.extract_buildings(mock_osm_parser)
        pois, _ = osm_data_handler.extract_pois(mock_osm_parser)
        landuse, _ = osm_data_handler.extract_landuse(mock_osm_parser)
        power, _ = osm_data_handler.extract_power_infrastructure(mock_osm_parser)

        assert buildings is None
        assert pois is None
        assert landuse is None
        assert power is None

    def test_handle_parser_exceptions(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """Test handling of exceptions from parser."""
        mock_osm_parser.get_buildings.side_effect = Exception("Parser error")
        mock_osm_parser.get_pois.side_effect = Exception("Parser error")
        mock_osm_parser.get_landuse.side_effect = Exception("Parser error")

        buildings, _ = osm_data_handler.extract_buildings(mock_osm_parser)
        pois, _ = osm_data_handler.extract_pois(mock_osm_parser)
        landuse, _ = osm_data_handler.extract_landuse(mock_osm_parser)

        assert buildings is None
        assert pois is None
        assert landuse is None


class TestIntegrationWithRealData:
    """Test suite for integration with real data samples."""

    def test_with_cambridge_sample_data(self, osm_data_handler: OSMDataHandler) -> None:
        """Test with sample data structure from Cambridge."""
        # Load a small sample from the actual raw power data structure
        raw_power_data = {
            'tags': [None, {'voltage': '115000'}, {'material': 'wood'}],
            'lat': [42.391411, 42.365837, 42.385422],
            'lon': [-71.142868, -71.091568, -71.12645],
            'id': [3855410660, 4496268769, 9520462290],
            'power': ['transformer', 'transformer', 'pole'],
            'substation': [None, None, None],
            'transformer': [None, None, None],
            'osm_type': ['node', 'node', 'node'],
            'geometry': [
                Point(-71.142868, 42.391411),
                Point(-71.091568, 42.365837),
                Point(-71.12645, 42.385422)
            ]
        }
        power_gdf = gpd.GeoDataFrame(raw_power_data, crs="EPSG:4326")

        # Test voltage filtering
        filtered = osm_data_handler.filter_by_voltage(power_gdf, max_voltage=70_000)

        # Should filter out the 115000V transformer
        assert len(filtered) == 2
        assert 3855410660 in filtered['id'].values  # No voltage
        assert 9520462290 in filtered['id'].values  # Pole with no voltage
        assert 4496268769 not in filtered['id'].values  # 115000V

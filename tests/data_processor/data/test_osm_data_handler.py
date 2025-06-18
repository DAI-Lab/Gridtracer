"""
Test module for OSM data handler.

This module contains tests to verify the OSM data handler works correctly
with the SynGrid pipeline, including building, POI, landuse, and power
infrastructure extraction using pyrosm.
"""

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from syngrid.data_processor.data.osm.osm_data_handler import OSMDataHandler

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_buildings_gdf() -> gpd.GeoDataFrame:
    """
    Create a sample buildings GeoDataFrame for testing.

    Returns:
        gpd.GeoDataFrame: Sample buildings data.
    """
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
    """
    Create a sample POIs GeoDataFrame for testing.

    Returns:
        gpd.GeoDataFrame: Sample POIs data.
    """
    pois_data = {
        'osmid': [100, 101, 102, 103],
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
    """
    Create a sample landuse GeoDataFrame for testing.

    Returns:
        gpd.GeoDataFrame: Sample landuse data.
    """
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
    """
    Create a sample power infrastructure GeoDataFrame for testing.

    Returns:
        gpd.GeoDataFrame: Sample power infrastructure data.
    """
    power_data = {
        'osmid': [300, 301, 302],
        'power': ['transformer', 'substation', 'pole'],
        'transformer': [None, None, 'distribution'],
        'voltage': ['11000', '132000', '11000'],
        'geometry': [
            Point(-71.1, 42.3),
            Polygon([(-71.09, 42.31), (-71.08, 42.31), (-71.08, 42.32), (-71.09, 42.32)]),
            Point(-71.07, 42.33)
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
    """
    Create a mock OSM parser with sample data.

    Args:
        sample_buildings_gdf: Sample buildings data.
        sample_pois_gdf: Sample POIs data.
        sample_landuse_gdf: Sample landuse data.
        sample_power_gdf: Sample power data.

    Returns:
        Mock: Mocked OSM parser.
    """
    mock_parser = Mock()
    mock_parser.get_buildings.return_value = sample_buildings_gdf
    mock_parser.get_pois.return_value = sample_pois_gdf
    mock_parser.get_landuse.return_value = sample_landuse_gdf
    mock_parser.get_data_by_custom_criteria.return_value = sample_power_gdf
    return mock_parser


@pytest.fixture
def osm_data_handler(orchestrator_with_fips) -> OSMDataHandler:
    """
    Create an OSM data handler with a configured orchestrator.

    Args:
        orchestrator_with_fips: Orchestrator fixture with FIPS data.

    Returns:
        OSMDataHandler: Configured handler instance.
    """
    return OSMDataHandler(orchestrator_with_fips)


class TestOSMDataHandlerInitialization:
    """Test suite for OSM data handler initialization."""

    def test_osm_data_handler_initialization_success(
        self,
        osm_data_handler: OSMDataHandler,
        orchestrator_with_fips
    ) -> None:
        """
        Test successful OSM data handler initialization.

        Args:
            osm_data_handler: Handler fixture.
            orchestrator_with_fips: Orchestrator fixture.
        """
        assert osm_data_handler is not None
        assert osm_data_handler.orchestrator == orchestrator_with_fips
        assert osm_data_handler._get_dataset_name() == "OSM"

        # Verify output directory exists
        assert osm_data_handler.dataset_output_dir.exists()

        # Verify FIPS data is accessible
        fips = osm_data_handler.orchestrator.get_fips_dict()
        assert fips is not None
        assert fips['state'] == 'MA'


class TestBoundaryManagement:
    """Test suite for boundary management functionality."""

    def test_set_boundary_success(
        self,
        osm_data_handler: OSMDataHandler,
        sample_boundary_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test successful boundary setting.

        Args:
            osm_data_handler: Handler fixture.
            sample_boundary_gdf: Sample boundary data.
        """
        result = osm_data_handler.set_boundary(sample_boundary_gdf)
        assert result is True

        # Verify boundary polygon was set
        assert hasattr(osm_data_handler, 'boundary_polygon_for_filtering')
        assert osm_data_handler.boundary_polygon_for_filtering is not None

    def test_set_boundary_with_multiple_geometries(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """
        Test boundary setting with multiple geometries.

        Args:
            osm_data_handler: Handler fixture.
        """
        # Create boundary with multiple geometries
        multi_boundary_data = {
            'GEOID': ['25017A', '25017B'],
            'NAME': ['Part A', 'Part B'],
            'geometry': [
                Polygon([(-71.5, 42.3), (-71.5, 42.35), (-71.45, 42.35), (-71.45, 42.3)]),
                Polygon([(-71.45, 42.3), (-71.45, 42.35), (-71.4, 42.35), (-71.4, 42.3)])
            ]
        }
        multi_boundary_gdf = gpd.GeoDataFrame(multi_boundary_data, crs="EPSG:4326")

        result = osm_data_handler.set_boundary(multi_boundary_gdf)
        assert result is True

    def test_set_boundary_with_crs_conversion(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """
        Test boundary setting with CRS conversion.

        Args:
            osm_data_handler: Handler fixture.
        """
        # Create boundary in different CRS
        boundary_data = {
            'GEOID': ['25017'],
            'NAME': ['Test'],
            'geometry': [
                Polygon([
                    (200000, 900000), (200000, 950000),
                    (250000, 950000), (250000, 900000)
                ])
            ]
        }
        boundary_gdf = gpd.GeoDataFrame(boundary_data, crs="EPSG:3857")  # Web Mercator

        result = osm_data_handler.set_boundary(boundary_gdf)
        assert result is True

    def test_set_boundary_none_input(self, osm_data_handler: OSMDataHandler) -> None:
        """
        Test boundary setting with None input.

        Args:
            osm_data_handler: Handler fixture.
        """
        result = osm_data_handler.set_boundary(None)
        assert result is True

    def test_set_boundary_empty_gdf(self, osm_data_handler: OSMDataHandler) -> None:
        """
        Test boundary setting with empty GeoDataFrame.

        Args:
            osm_data_handler: Handler fixture.
        """
        empty_gdf = gpd.GeoDataFrame()
        result = osm_data_handler.set_boundary(empty_gdf)
        assert result is True


class TestBuildingExtraction:
    """Test suite for building extraction functionality."""

    def test_extract_buildings_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_buildings_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test successful building extraction.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
            sample_buildings_gdf: Sample buildings data.
        """
        buildings, filepath = osm_data_handler.extract_buildings(mock_osm_parser)

        assert buildings is not None
        assert filepath is not None
        assert isinstance(buildings, gpd.GeoDataFrame)
        assert len(buildings) == len(sample_buildings_gdf)
        assert filepath.exists()
        assert filepath.name == "buildings.geojson"

        # Verify raw file was also saved
        raw_filepath = osm_data_handler.dataset_output_dir / "raw" / "raw_buildings.geojson"
        assert raw_filepath.exists()

    def test_extract_buildings_no_parser(self, osm_data_handler: OSMDataHandler) -> None:
        """
        Test building extraction with no parser.

        Args:
            osm_data_handler: Handler fixture.
        """
        buildings, filepath = osm_data_handler.extract_buildings(None)

        assert buildings is None
        assert filepath is None

    def test_extract_buildings_empty_result(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test building extraction with empty result.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_buildings.return_value = gpd.GeoDataFrame()

        buildings, filepath = osm_data_handler.extract_buildings(mock_osm_parser)

        assert buildings is None
        assert filepath is None

    def test_extract_buildings_parser_returns_none(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test building extraction when parser returns None.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_buildings.return_value = None

        buildings, filepath = osm_data_handler.extract_buildings(mock_osm_parser)

        assert buildings is None
        assert filepath is None

    def test_extract_buildings_parser_exception(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test building extraction when parser raises exception.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_buildings.side_effect = Exception("Parser error")

        buildings, filepath = osm_data_handler.extract_buildings(mock_osm_parser)

        assert buildings is None
        assert filepath is None


class TestPOIExtraction:
    """Test suite for POI extraction functionality."""

    def test_extract_pois_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_pois_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test successful POI extraction.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
            sample_pois_gdf: Sample POIs data.
        """
        pois, filepath = osm_data_handler.extract_pois(mock_osm_parser)

        assert pois is not None
        assert filepath is not None
        assert isinstance(pois, gpd.GeoDataFrame)
        assert len(pois) == len(sample_pois_gdf)
        assert filepath.exists()
        assert filepath.name == "pois.geojson"

        # Verify raw file was also saved
        raw_filepath = osm_data_handler.dataset_output_dir / "raw" / "raw_pois.geojson"
        assert raw_filepath.exists()

    def test_extract_pois_empty_result(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test POI extraction with empty result.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_pois.return_value = gpd.GeoDataFrame()

        pois, filepath = osm_data_handler.extract_pois(mock_osm_parser)

        assert pois is None
        assert filepath is None

    def test_extract_pois_parser_returns_none(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test POI extraction when parser returns None.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_pois.return_value = None

        pois, filepath = osm_data_handler.extract_pois(mock_osm_parser)

        assert pois is None
        assert filepath is None

    def test_extract_pois_exception_handling(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test POI extraction exception handling.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_pois.side_effect = Exception("POI extraction error")

        pois, filepath = osm_data_handler.extract_pois(mock_osm_parser)

        assert pois is None
        assert filepath is None


class TestLanduseExtraction:
    """Test suite for landuse extraction functionality."""

    def test_extract_landuse_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_landuse_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test successful landuse extraction.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
            sample_landuse_gdf: Sample landuse data.
        """
        # Ensure raw directory exists
        raw_dir = osm_data_handler.dataset_output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        landuse, filepath = osm_data_handler.extract_landuse(mock_osm_parser)

        assert landuse is not None
        assert filepath is not None
        assert isinstance(landuse, gpd.GeoDataFrame)
        assert filepath.exists()
        assert filepath.name == "landuse.geojson"

        # Verify categories were added
        assert 'category' in landuse.columns
        expected_categories = {'residential', 'industrial', 'public'}
        actual_categories = set(landuse['category'].unique())
        assert actual_categories.issubset(expected_categories)

    def test_extract_landuse_filtering(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test landuse extraction with filtering.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Ensure raw directory exists
        raw_dir = osm_data_handler.dataset_output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Create landuse data with some irrelevant categories
        landuse_data = {
            'osmid': [200, 201, 202, 203],
            # farmland/unknown should be filtered
            'landuse': ['residential', 'farmland', 'commercial', 'unknown'],
            'name': ['Residential', 'Farm', 'Commercial', 'Unknown'],
            'geometry': [
                Polygon([(-71.1, 42.3), (-71.09, 42.3), (-71.09, 42.31), (-71.1, 42.31)]),
                Polygon([(-71.08, 42.32), (-71.07, 42.32), (-71.07, 42.33), (-71.08, 42.33)]),
                Polygon([(-71.06, 42.34), (-71.05, 42.34), (-71.05, 42.35), (-71.06, 42.35)]),
                Polygon([(-71.04, 42.36), (-71.03, 42.36), (-71.03, 42.37), (-71.04, 42.37)])
            ]
        }
        mixed_landuse_gdf = gpd.GeoDataFrame(landuse_data, crs="EPSG:4326")
        mock_osm_parser.get_landuse.return_value = mixed_landuse_gdf

        landuse, filepath = osm_data_handler.extract_landuse(mock_osm_parser)

        assert landuse is not None
        # Should filter out 'farmland' and 'unknown'
        assert len(landuse) == 2
        assert set(landuse['landuse'].values) == {'residential', 'commercial'}

    def test_extract_landuse_empty_result(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test landuse extraction with empty result.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_landuse.return_value = gpd.GeoDataFrame()

        landuse, filepath = osm_data_handler.extract_landuse(mock_osm_parser)

        assert landuse is None
        assert filepath is None

    def test_extract_landuse_exception_handling(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test landuse extraction exception handling.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        mock_osm_parser.get_landuse.side_effect = Exception("Landuse extraction error")

        landuse, filepath = osm_data_handler.extract_landuse(mock_osm_parser)

        assert landuse is None
        assert filepath is None


class TestPowerInfrastructureExtraction:
    """Test suite for power infrastructure extraction functionality."""

    def test_extract_power_infrastructure_success(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_power_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test successful power infrastructure extraction.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
            sample_power_gdf: Sample power data.
        """
        power, filepath = osm_data_handler.extract_power_infrastructure(mock_osm_parser)

        assert power is not None
        assert filepath is not None
        assert isinstance(power, gpd.GeoDataFrame)
        assert filepath.exists()
        assert filepath.name == "power.geojson"

        # Verify raw file was also saved
        raw_filepath = osm_data_handler.dataset_output_dir / "raw" / "raw_power.geojson"
        assert raw_filepath.exists()

    def test_extract_power_infrastructure_filtering(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test power infrastructure extraction with filtering logic.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Create power data to test filtering for valid features
        power_data = {
            'osmid': [300, 302, 303],
            'power': ['transformer', 'pole', 'pole'],
            'transformer': [None, 'distribution', None],
            'tags': [
                {'power': 'transformer'},
                {'power': 'pole', 'transformer': 'distribution'},
                {'power': 'pole'}
            ],
            'geometry': [
                Point(-71.1, 42.3),
                Point(-71.08, 42.32),
                # This pole should be filtered out as it's not a distribution pole
                Point(-71.07, 42.33)
            ]
        }
        filtered_power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")
        mock_osm_parser.get_data_by_custom_criteria.return_value = filtered_power_gdf

        power, filepath = osm_data_handler.extract_power_infrastructure(mock_osm_parser)

        assert power is not None
        # Should keep the transformer and the pole with a distribution tag (2 features)
        assert len(power) == 2
        assert 'transformer' in power['power'].values
        assert 'pole' in power['power'].values
        # Ensure the plain pole was filtered
        assert 303 not in power['osmid'].values

    def test_extract_power_infrastructure_empty_result(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test power infrastructure extraction with empty result.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Create an empty GeoDataFrame with at least the required columns
        empty_power_gdf = gpd.GeoDataFrame(columns=['osmid', 'power', 'geometry'], crs="EPSG:4326")
        mock_osm_parser.get_data_by_custom_criteria.return_value = empty_power_gdf

        # Should return (None, None) when no power features found
        result = osm_data_handler.extract_power_infrastructure(mock_osm_parser)
        assert result == (None, None)

    def test_deduplicate_power_features(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """
        Test power features deduplication.

        Args:
            osm_data_handler: Handler fixture.
        """
        # Create power data with duplicates (close points)
        power_data = {
            'osmid': [300, 301, 302],
            'power': ['transformer', 'transformer', 'substation'],
            'tags': [{'power': 'transformer'}, {'power': 'transformer'}, {'power': 'substation'}],
            'geometry': [
                Point(-71.1, 42.3),
                # Very close to first point (should be deduplicated)
                Point(-71.100001, 42.300001),
                Point(-71.08, 42.32)  # Far enough away
            ]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")

        deduplicated = osm_data_handler.deduplicate_power_features(
            power_gdf, distance_threshold_meters=10)

        assert deduplicated is not None
        # Should remove the close duplicate, keeping the substation and one transformer
        assert len(deduplicated) == 2
        assert 'substation' in deduplicated['power'].values
        assert 'transformer' in deduplicated['power'].values

    def test_deduplicate_power_features_empty_input(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """
        Test power features deduplication with empty input.

        Args:
            osm_data_handler: Handler fixture.
        """
        empty_gdf = gpd.GeoDataFrame()
        result = osm_data_handler.deduplicate_power_features(empty_gdf)

        assert result.empty

    def test_deduplicate_power_features_none_input(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """
        Test power features deduplication with None input.

        Args:
            osm_data_handler: Handler fixture.
        """
        result = osm_data_handler.deduplicate_power_features(None)
        assert result is None

    def test_filter_by_voltage(self, osm_data_handler: OSMDataHandler) -> None:
        """Test filtering by voltage."""
        power_data = {
            'power': ['transformer', 'substation', 'pole'],
            'tags': [
                {'voltage': '150000'},
                {'voltage': '11000'},
                {}  # No voltage tag
            ],
            'geometry': [Point(1, 1), Point(2, 2), Point(3, 3)]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")
        print(power_gdf)

        filtered_gdf = osm_data_handler.filter_by_voltage(power_gdf)
        print(f"Filtered power gdf: {filtered_gdf}")
        assert len(filtered_gdf) == 2
        assert '150000' not in [t.get('voltage') for t in filtered_gdf['tags']]

    def test_filter_transmission_tags(self, osm_data_handler: OSMDataHandler) -> None:
        """Test filtering of transmission tags."""
        power_data = {
            'power': ['substation', 'transformer'],
            'substation': ['transmission', None],
            'transformer': [None, 'distribution']
        }
        power_gdf = gpd.GeoDataFrame(power_data)
        filtered_gdf = osm_data_handler.filter_transmission_tags(power_gdf)
        assert len(filtered_gdf) == 1
        assert 'transmission' not in filtered_gdf['substation'].values

    def test_remove_contained_points(self, osm_data_handler: OSMDataHandler) -> None:
        """Test removal of points contained within polygons."""
        power_data = {
            'power': ['substation', 'transformer'],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),  # Substation polygon
                Point(5, 5)  # Transformer point inside the substation
            ]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")
        filtered_gdf = osm_data_handler.remove_contained_points(power_gdf)
        assert len(filtered_gdf) == 1
        assert filtered_gdf.iloc[0].geometry.geom_type == 'Polygon'

    def test_convert_to_centroids(self, osm_data_handler: OSMDataHandler) -> None:
        """Test conversion of geometries to centroids."""
        power_data = {
            'power': ['substation', 'transformer'],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
                Point(1, 1)
            ]
        }
        power_gdf = gpd.GeoDataFrame(power_data, crs="EPSG:4326")
        centroids_gdf = osm_data_handler.convert_to_centroids(power_gdf)
        assert len(centroids_gdf) == 2
        assert all(isinstance(geom, Point) for geom in centroids_gdf.geometry)


class TestDownloadMethod:
    """Test suite for the download method."""

    def test_download_success_all_extractions(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test successful download with all extractions.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Mock the orchestrator to return the parser
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        results = osm_data_handler.download()

        assert isinstance(results, dict)

        # Check all expected keys exist
        expected_keys = [
            'buildings', 'buildings_filepath',
            'pois', 'pois_filepath',
            'landuse', 'landuse_filepath',
            'power', 'power_filepath'
        ]
        for key in expected_keys:
            assert key in results

        # Verify non-None results for successful extractions
        assert results['buildings'] is not None
        assert results['buildings_filepath'] is not None
        assert results['pois'] is not None
        assert results['pois_filepath'] is not None
        assert results['landuse'] is not None
        assert results['landuse_filepath'] is not None
        assert results['power'] is not None
        assert results['power_filepath'] is not None

    def test_download_existing_files_reuse(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_buildings_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test download method reuses existing files.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
            sample_buildings_gdf: Sample buildings data.
        """
        # Create existing files
        buildings_filepath = osm_data_handler.dataset_output_dir / "buildings.geojson"
        buildings_filepath.parent.mkdir(parents=True, exist_ok=True)
        sample_buildings_gdf.to_file(buildings_filepath, driver="GeoJSON")

        # Mock the orchestrator to return the parser
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        results = osm_data_handler.download()

        # Verify existing file is loaded
        assert results['buildings'] is not None
        assert results['buildings_filepath'] == buildings_filepath
        assert len(results['buildings']) == len(sample_buildings_gdf)

    def test_download_mixed_existing_and_new_files(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock,
        sample_pois_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test download with some existing files and some new extractions.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
            sample_pois_gdf: Sample POIs data.
        """
        # Create only POIs file as existing
        pois_filepath = osm_data_handler.dataset_output_dir / "pois.geojson"
        pois_filepath.parent.mkdir(parents=True, exist_ok=True)
        sample_pois_gdf.to_file(pois_filepath, driver="GeoJSON")

        # Mock the orchestrator
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        results = osm_data_handler.download()

        # POIs should be loaded from existing file
        assert results['pois_filepath'] == pois_filepath
        assert len(results['pois']) == len(sample_pois_gdf)

        # Other datasets should be extracted fresh
        assert results['buildings'] is not None
        assert results['landuse'] is not None
        assert results['power'] is not None


class TestProcessMethod:
    """Test suite for the process method."""

    def test_process_without_plotting(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test process method without plotting.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Mock the orchestrator
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        results = osm_data_handler.process(plot=False)

        assert isinstance(results, dict)
        # Should contain all the same keys as download
        expected_keys = [
            'buildings', 'buildings_filepath',
            'pois', 'pois_filepath',
            'landuse', 'landuse_filepath',
            'power', 'power_filepath'
        ]
        for key in expected_keys:
            assert key in results

    def test_process_with_plotting(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test process method with plotting enabled.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Mock the orchestrator
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=mock_osm_parser)

        # Mock the plot_osm_data method to avoid actual plotting
        with patch.object(osm_data_handler, 'plot_osm_data') as mock_plot:
            results = osm_data_handler.process(plot=True)

            assert isinstance(results, dict)
            mock_plot.assert_called_once()


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_process_with_no_osm_parser(
        self,
        osm_data_handler: OSMDataHandler
    ) -> None:
        """
        Test process method when OSM parser is not available.

        Args:
            osm_data_handler: Handler fixture.
        """
        # Mock orchestrator to return None for OSM parser
        osm_data_handler.orchestrator.get_osm_parser = Mock(return_value=None)

        results = osm_data_handler.download()

        # All results should be None when parser is unavailable except for
        # buildings which logs error
        assert results['buildings'] is None
        assert results['buildings_filepath'] is None
        assert results['pois'] is None
        assert results['pois_filepath'] is None
        assert results['landuse'] is None
        assert results['landuse_filepath'] is None
        assert results['power'] is None
        assert results['power_filepath'] is None

    def test_plot_osm_data_with_valid_data(
        self,
        osm_data_handler: OSMDataHandler,
        sample_buildings_gdf: gpd.GeoDataFrame,
        sample_pois_gdf: gpd.GeoDataFrame,
        sample_landuse_gdf: gpd.GeoDataFrame,
        sample_power_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Test plotting OSM data with valid data.

        Args:
            osm_data_handler: Handler fixture.
            sample_buildings_gdf: Sample buildings data.
            sample_pois_gdf: Sample POIs data.
            sample_landuse_gdf: Sample landuse data.
            sample_power_gdf: Sample power data.
        """
        osm_data = {
            'buildings': sample_buildings_gdf,
            'pois': sample_pois_gdf,
            'landuse': sample_landuse_gdf,
            'power': sample_power_gdf
        }

        # Mock matplotlib to avoid actual plotting
        with patch('matplotlib.pyplot.savefig'):
            # Should not raise any exceptions
            osm_data_handler.plot_osm_data(osm_data)

    def test_dataset_name_method(self, osm_data_handler: OSMDataHandler) -> None:
        """
        Test the _get_dataset_name method.

        Args:
            osm_data_handler: Handler fixture.
        """
        dataset_name = osm_data_handler._get_dataset_name()
        assert dataset_name == "OSM"

    def test_file_operations_with_directory_creation(
        self,
        osm_data_handler: OSMDataHandler,
        mock_osm_parser: Mock
    ) -> None:
        """
        Test that necessary directories are created during file operations.

        Args:
            osm_data_handler: Handler fixture.
            mock_osm_parser: Mocked OSM parser.
        """
        # Ensure output directory exists
        assert osm_data_handler.dataset_output_dir.exists()

        # Extract buildings to test directory creation for raw files
        buildings, filepath = osm_data_handler.extract_buildings(mock_osm_parser)

        # Verify raw directory was created
        raw_dir = osm_data_handler.dataset_output_dir / "raw"
        assert raw_dir.exists()

        # Verify raw file exists
        raw_buildings_file = raw_dir / "raw_buildings.geojson"
        assert raw_buildings_file.exists()

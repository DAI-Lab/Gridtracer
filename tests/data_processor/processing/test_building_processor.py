
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from syngrid.data_processor.processing.building_processor import BuildingHeuristicsProcessor


@pytest.fixture
def building_processor() -> BuildingHeuristicsProcessor:
    """Fixture to create a BuildingHeuristicsProcessor instance."""
    return BuildingHeuristicsProcessor(output_dir="test_output")


def test_calculate_floor_area_wgs84(building_processor: BuildingHeuristicsProcessor) -> None:
    """Test _calculate_floor_area with WGS84 input."""
    # Create a sample GeoDataFrame in WGS84 (EPSG:4326)
    # A square of approx 100m x 100m near the equator for simplicity
    # (0,0) to (0.001, 0.001) is roughly 111m x 111m
    data = {
        'id': [1],
        'geometry': [Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)])]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    processed_gdf = building_processor._calculate_floor_area(gdf)

    assert 'floor_area' in processed_gdf.columns
    # Area should be positive. Exact value depends on projection details and shapely/geos versions.
    # We expect an area around 12300 sq meters for a 0.001x0.001 degree square at the equator.
    # The key is that it's calculated and reasonable.
    assert processed_gdf['floor_area'].iloc[0] > 12000
    assert processed_gdf['floor_area'].iloc[0] < 13000
    # Check original gdf is not modified if it was copied, or that it has the new column
    assert 'floor_area' in gdf.columns  # _calculate_floor_area modifies in place


def test_calculate_floor_area_epsg5070(building_processor: BuildingHeuristicsProcessor) -> None:
    """Test _calculate_floor_area with EPSG:5070 input."""
    # Create a sample GeoDataFrame in EPSG:5070 (a metric CRS)
    # A 100m x 100m square
    data = {
        'id': [1],
        'geometry': [Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:5070")

    processed_gdf = building_processor._calculate_floor_area(gdf)

    assert 'floor_area' in processed_gdf.columns
    pd.testing.assert_series_equal(
        processed_gdf['floor_area'],
        pd.Series([10000.0]),
        check_dtype=False,
        check_names=False,  # Add this to ignore differing series names
        rtol=1e-3  # Allow for small floating point inaccuracies
    )
    # Check original gdf is not modified if it was copied, or that it has the new column
    assert 'floor_area' in gdf.columns  # _calculate_floor_area modifies in place


def test_calculate_floor_area_empty_input(building_processor: BuildingHeuristicsProcessor) -> None:
    """Test _calculate_floor_area with an empty GeoDataFrame."""
    empty_gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
    processed_gdf = building_processor._calculate_floor_area(empty_gdf)
    assert 'floor_area' in processed_gdf.columns
    assert len(processed_gdf) == 0

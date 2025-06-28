"""
Test module for Microsoft Buildings data handler integration.

This module contains tests to verify the Microsoft Buildings data handler
works correctly with the gridtracer pipeline, including spatial pre-filtering
and region-specific building downloads.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pytest

from gridtracer.data_processor.data_imports.microsoft_buildings import (
    MicrosoftBuildingsDataHandler,)
from gridtracer.data_processor.workflow import WorkflowOrchestrator

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def orchestrator_with_boundary() -> WorkflowOrchestrator:
    """
    Create a WorkflowOrchestrator with Cambridge city boundary loaded.

    Returns:
        WorkflowOrchestrator: Configured orchestrator with region boundary set.
    """
    orchestrator = WorkflowOrchestrator()

    # Load pre-existing census boundary for testing
    boundary_path = Path(
        "gridtracer/data_processor/output/MA/Middlesex_County/"
        "Cambridge_city_old/Census/25_017_11000_blocks_boundary.geojson"
    )

    if not boundary_path.exists():
        pytest.skip(f"Test boundary file not found: {boundary_path}")

    boundary_gdf = gpd.read_file(boundary_path)
    orchestrator.set_region_boundary(boundary_gdf)

    return orchestrator


@pytest.fixture
def microsoft_buildings_handler(
    orchestrator_with_boundary: WorkflowOrchestrator
) -> MicrosoftBuildingsDataHandler:
    """
    Create a Microsoft Buildings handler with configured orchestrator.

    Args:
        orchestrator_with_boundary: Orchestrator fixture with boundary set.

    Returns:
        MicrosoftBuildingsDataHandler: Configured handler instance.
    """
    return MicrosoftBuildingsDataHandler(orchestrator_with_boundary)


def test_microsoft_buildings_handler_initialization(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler,
    orchestrator_with_boundary: WorkflowOrchestrator
) -> None:
    """
    Test that the Microsoft Buildings handler initializes correctly.

    Args:
        microsoft_buildings_handler: Handler fixture.
        orchestrator_with_boundary: Orchestrator fixture.
    """
    # Verify handler is properly initialized
    assert microsoft_buildings_handler is not None
    assert microsoft_buildings_handler.orchestrator == orchestrator_with_boundary
    assert microsoft_buildings_handler.dataset_output_dir.exists()

    # Verify FIPS data is available
    fips = orchestrator_with_boundary.get_fips_dict()
    assert fips is not None
    assert fips['state'] == 'MA'
    assert fips['county'] == 'Middlesex County'
    assert fips['subdivision'] == 'Cambridge city'


def test_state_mapping_creation_and_loading(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler
) -> None:
    """
    Test state mapping creation and loading functionality.

    Args:
        microsoft_buildings_handler: Handler fixture.
    """
    # Load or create state mapping
    state_mapping = microsoft_buildings_handler._load_state_mapping()

    # Verify mapping structure
    assert isinstance(state_mapping, dict)
    assert len(state_mapping) > 0

    # Verify MA is in mapping
    assert 'MA' in state_mapping

    # Verify MA mapping structure
    ma_data = state_mapping['MA']
    assert 'state_name' in ma_data
    assert 'num_quadkeys' in ma_data
    assert 'quadkeys' in ma_data

    assert ma_data['state_name'] == 'Massachusetts'
    assert ma_data['num_quadkeys'] > 0

    # Verify quadkeys structure (new format with geometry)
    quadkeys = ma_data['quadkeys']
    assert isinstance(quadkeys, dict)

    # Check first quadkey has required fields
    if quadkeys:
        first_quadkey = next(iter(quadkeys.values()))
        assert 'url' in first_quadkey
        assert 'geometry' in first_quadkey
        assert isinstance(first_quadkey['url'], str)
        assert isinstance(first_quadkey['geometry'], str)  # WKT format


def test_spatial_quadkey_filtering(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler
) -> None:
    """
    Test spatial filtering of QuadKeys based on region boundary.

    Args:
        microsoft_buildings_handler: Handler fixture.
    """
    # Get state abbreviation
    fips = microsoft_buildings_handler.orchestrator.get_fips_dict()
    state_abbr = fips['state']

    # Load state mapping first
    state_mapping = microsoft_buildings_handler._load_state_mapping()
    total_quadkeys = state_mapping[state_abbr]['num_quadkeys']

    # Filter QuadKeys spatially
    filtered_quadkeys = microsoft_buildings_handler._filter_quadkeys_by_region(state_abbr)

    # Verify filtering results
    assert isinstance(filtered_quadkeys, list)
    assert len(filtered_quadkeys) > 0
    assert len(filtered_quadkeys) <= total_quadkeys  # Should be equal or fewer

    # For Cambridge, should significantly reduce the number of QuadKeys
    assert len(filtered_quadkeys) < total_quadkeys


def test_building_download_with_spatial_filtering(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler
) -> None:
    """
    Test building download with spatial pre-filtering (limited to 1 tile).

    Args:
        microsoft_buildings_handler: Handler fixture.
    """
    # Get state abbreviation
    fips = microsoft_buildings_handler.orchestrator.get_fips_dict()
    state_abbr = fips['state']

    # Download buildings (limit to 1 tile for testing)
    building_files = microsoft_buildings_handler._download_state_buildings(
        state_abbr, max_tiles=1
    )

    # Verify download results
    assert isinstance(building_files, list)
    assert len(building_files) >= 0  # Could be 0 if no intersection

    # If files were downloaded, verify they exist
    for file_path in building_files:
        assert file_path.exists()
        assert file_path.suffix == '.geojson'


def test_building_filtering_to_region(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler
) -> None:
    """
    Test filtering buildings to the specific region boundary.

    Args:
        microsoft_buildings_handler: Handler fixture.
    """
    # Get state abbreviation
    fips = microsoft_buildings_handler.orchestrator.get_fips_dict()
    state_abbr = fips['state']

    # Download buildings (limit to 1 tile for testing)
    building_files = microsoft_buildings_handler._download_state_buildings(
        state_abbr, max_tiles=1
    )

    if not building_files:
        pytest.skip("No building files downloaded for region filtering test")

    # Filter buildings to region
    filtered_buildings = microsoft_buildings_handler._filter_buildings_to_region(
        building_files
    )

    # Verify filtering results
    assert isinstance(filtered_buildings, gpd.GeoDataFrame)

    if len(filtered_buildings) > 0:
        # Verify required columns exist
        expected_cols = ['quadkey', 'state_abbr', 'geometry']
        for col in expected_cols:
            assert col in filtered_buildings.columns

        # Verify state abbreviation is correct
        assert all(filtered_buildings['state_abbr'] == state_abbr)

        # Verify geometry column
        assert filtered_buildings.geometry is not None


def test_full_microsoft_buildings_process(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler
) -> None:
    """
    Test the complete Microsoft Buildings processing workflow.

    Args:
        microsoft_buildings_handler: Handler fixture.
    """
    # Run the full process
    result = microsoft_buildings_handler.process()

    # Verify process results
    assert isinstance(result, dict)
    assert 'ms_buildings' in result
    assert 'ms_buildings_filepath' in result

    # Check for error handling
    if 'error' in result:
        pytest.fail(f"Process failed with error: {result['error']}")

    # Verify building count is reasonable
    total_buildings = len(result['ms_buildings'])
    assert isinstance(total_buildings, int)
    assert total_buildings >= 0

    # If buildings were found, verify output file exists
    if total_buildings > 0:
        assert 'ms_buildings_filepath' in result
        buildings_filepath = result['ms_buildings_filepath']
        if buildings_filepath:
            assert Path(buildings_filepath).exists()


def test_state_mapping_file_persistence(
    microsoft_buildings_handler: MicrosoftBuildingsDataHandler
) -> None:
    """
    Test that state mapping file is properly created and persisted.

    This test is marked as slow since mapping creation takes time.

    Args:
        microsoft_buildings_handler: Handler fixture.
    """
    # Ensure mapping is created
    microsoft_buildings_handler._load_state_mapping()

    # Verify mapping file exists
    mapping_file = microsoft_buildings_handler.mapping_file
    assert mapping_file.exists()

    # Verify file contains valid JSON
    import json
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)

    assert isinstance(mapping_data, dict)
    assert len(mapping_data) > 0
    assert 'MA' in mapping_data

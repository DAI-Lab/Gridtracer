"""
Tests for the Road_network_builder class.

These tests verify the functionality of the Road_network_builder class
for creating routable road networks from OSM data.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from syngrid.data_processor.data.osm.road_network_builder import RoadNetworkBuilder

# Import WorkflowOrchestrator for type hinting if needed, or mock it directly
# from syngrid.data_processor.workflow import WorkflowOrchestrator


@pytest.fixture
def fips_dict():
    """Create a mock FIPS dict for testing."""
    return {
        'state': 'MA',
        'state_fips': '25',
        'county': 'Suffolk',
        'county_fips': '025',
    }


@pytest.fixture
def mock_osm_data():
    """Create mock OSM data for testing."""
    # Create a simple mock network with two nodes and an edge
    geometries = [LineString([(0, 0), (1, 1)])]
    data = {
        'osm_id': [12345],
        'highway': ['residential'],
        'name': ['Test Road'],
        'oneway': ['no'],
        'u': [1],  # source node id
        'v': [2],  # target node id
        'length': [1000]  # 1 km in meters
    }
    edges = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")

    # Create nodes
    nodes_data = {
        'id': [1, 2],
        'lat': [0, 1],
        'lon': [0, 1]
    }
    nodes = pd.DataFrame(nodes_data)

    return nodes, edges


@pytest.fixture
def mock_orchestrator(fips_dict, tmp_path, mock_osm_data):
    """Creates a mock WorkflowOrchestrator."""
    orchestrator = MagicMock()
    orchestrator.get_fips_dict.return_value = fips_dict
    # Make dataset_output_dir a Path object as expected by the builder
    orchestrator.get_dataset_specific_output_directory.return_value = Path(
        tmp_path) / "STREET_NETWORK"
    # Setup the mock OSM parser to be returned by the orchestrator
    mock_osm_parser = MagicMock()
    nodes, edges = mock_osm_data
    mock_osm_parser.get_network.return_value = (nodes, edges)
    orchestrator.get_osm_parser.return_value = mock_osm_parser
    return orchestrator


@pytest.fixture
def road_network_builder(mock_orchestrator, tmp_path):
    """Create a RoadNetworkBuilder with a mocked orchestrator and config."""
    with patch('syngrid.data_processor.data.osm.road_network_builder.yaml.safe_load') as mock_yaml:
        # Mock the YAML config for osm2po_config.yaml
        mock_yaml.return_value = {
            'way_tag_resolver': {
                'tags': {
                    'residential': {'clazz': 41, 'maxspeed': 40, 'flags': ['car', 'bike']}
                },
                'flag_list': ['car', 'bike', 'foot']
            },
            'network_type': 'driving'  # Default network type
        }

        # The RoadNetworkBuilder now takes the orchestrator directly
        builder = RoadNetworkBuilder(orchestrator=mock_orchestrator)

        # Ensure dataset_output_dir is set correctly for tests if DataHandler doesn't set it
        # based on orchestrator.get_dataset_specific_output_directory during super().__init__
        # This might be redundant if DataHandler's __init__ correctly uses the orchestrator
        builder.dataset_output_dir = mock_orchestrator.get_dataset_specific_output_directory(
            "STREET_NETWORK")
        builder.dataset_output_dir.mkdir(parents=True, exist_ok=True)

        return builder


def test_load_config(road_network_builder):
    """Test that configuration loads correctly."""
    config = road_network_builder.config

    # Check key config sections
    assert 'way_tag_resolver' in config
    assert 'tags' in config['way_tag_resolver']

    # Check tag configuration
    tags = config['way_tag_resolver']['tags']
    assert 'residential' in tags
    assert tags['residential']['clazz'] == 41
    assert tags['residential']['maxspeed'] == 40
    assert 'car' in tags['residential']['flags']
    assert 'bike' in tags['residential']['flags']


def test_resolve_way_tags(road_network_builder):
    """Test that way tags are resolved correctly."""
    # Test resolving residential road
    clazz, maxspeed, flags = road_network_builder._resolve_way_tags({'highway': 'residential'})
    assert clazz == 41
    assert maxspeed == 40
    assert 'car' in flags
    assert 'bike' in flags

    # Test resolving unknown highway type
    clazz, maxspeed, flags = road_network_builder._resolve_way_tags(
        {'highway': 'non_existent_type'}
    )
    assert clazz == 0
    assert maxspeed == 0
    assert len(flags) == 0


def test_flags_to_int(road_network_builder):
    """Test conversion of flags to bitmask integers."""
    # Test with no flags
    assert road_network_builder._flags_to_int(set()) == 0

    # Test with one flag
    assert road_network_builder._flags_to_int({'car'}) == 1  # 2^0

    # Test with multiple flags
    assert road_network_builder._flags_to_int({'car', 'bike'}) == 3  # 2^0 + 2^1

    # Test with all flags
    assert road_network_builder._flags_to_int({'car', 'bike', 'foot'}) == 7  # 2^0 + 2^1 + 2^2


# No need to patch OSM globally if the orchestrator provides the mock parser
def test_build_network(road_network_builder, mock_osm_data, tmp_path):
    """Test the build_network method."""
    # The mock_orchestrator in road_network_builder fixture already provides a mock OSM parser
    # that returns mock_osm_data.

    # Build the network
    results = road_network_builder.build_network(
        network_type='driving'  # network_type is still passed
        # osm_pbf_file is no longer passed directly
    )

    # Verify the orchestrator's OSM parser was used
    road_network_builder.orchestrator.get_osm_parser.assert_called_once()
    # Verify the mock OSM parser's get_network was called
    mock_osm_parser_instance = road_network_builder.orchestrator.get_osm_parser()
    mock_osm_parser_instance.get_network.assert_called_once_with(
        network_type='driving', nodes=True)

    # Check results - nodes might not be explicitly returned by build_network
    # Depending on the RoadNetworkBuilder.build_network implementation,
    # 'nodes' key might not be in results. Update as per actual implementation.
    # assert results['nodes'] is not None
    assert results['edges'] is not None
    assert results['sql_file'] is not None
    assert results['sql_file'].name == "osm2po_routing_network.sql"
    assert results['geojson_file'] is not None  # Changed from raw_edges_file

    # Verify files were created
    assert os.path.exists(results['sql_file'])
    assert os.path.exists(results['geojson_file'])  # Changed from raw_edges_file

    # Check SQL file content
    with open(results['sql_file'], 'r') as f:
        sql_content = f.read()
        # Check for key SQL elements
        assert "CREATE TABLE public_2po_4pgr" in sql_content
        assert "INSERT INTO public_2po_4pgr" in sql_content
        assert "CREATE INDEX" in sql_content


def test_process_method(road_network_builder, mock_osm_data, tmp_path):
    """Test the process method."""
    # The mock_orchestrator already provides the mock OSM parser

    # Create a boundary for testing clipping
    boundary_data = {
        'geometry': [LineString([(0, 0), (1, 1), (1, 0)]).buffer(0.1)]
    }
    boundary_gdf = gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")

    # No need to set osm_pbf_file in config, orchestrator handles it.

    # Call process method
    results = road_network_builder.process(boundary_gdf=boundary_gdf)

    # Verify the orchestrator's OSM parser was used and its get_network was called
    road_network_builder.orchestrator.get_osm_parser.assert_called_once()
    mock_osm_parser_instance = road_network_builder.orchestrator.get_osm_parser()
    # The process method calls build_network, which calls get_network
    mock_osm_parser_instance.get_network.assert_called_once_with(
        network_type='driving', nodes=True)

    # Verify results
    # assert results['nodes'] is not None # Check if 'nodes' is expected
    assert results['edges'] is not None
    assert results['sql_file'] is not None
    assert 'edges' in results  # Changed from 'clipped_edges'


def test_process_and_write_edges(road_network_builder, mock_osm_data):
    """Test the _process_and_write_edges method."""
    # Get mock edge data
    _, edges = mock_osm_data

    # Process the edges
    insert_value_tuples = road_network_builder._process_and_write_edges(edges, "TEST")

    # Check that we got SQL value tuple strings back
    assert len(insert_value_tuples) == 1
    assert isinstance(insert_value_tuples[0], str)

    # Check the SQL tuple content
    sql_tuple = insert_value_tuples[0]
    assert "12345" in sql_tuple  # osm_id
    assert "Test Road" in sql_tuple  # osm_name
    assert "41" in sql_tuple  # clazz
    assert "40" in sql_tuple  # kmh (maxspeed)

    # Test with empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame([], columns=edges.columns, geometry='geometry', crs=edges.crs)
    assert road_network_builder._process_and_write_edges(empty_gdf, "EMPTY") == []


def test_get_dataset_name(road_network_builder):
    """Test the _get_dataset_name method."""
    assert road_network_builder._get_dataset_name() == "STREET_NETWORK"


@patch('syngrid.data_processor.data.osm.road_network_builder.logger')
def test_download_method_not_implemented(mock_logger, road_network_builder):
    """Test that download method raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        road_network_builder.download()

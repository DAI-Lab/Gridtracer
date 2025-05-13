"""
Tests for the Road_network_builder class.

These tests verify the functionality of the Road_network_builder class
for creating routable road networks from OSM data.
"""

import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from syngrid.data_processor.data.osm.road_network_builder import Road_network_builder


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
def road_network_builder(fips_dict, tmp_path):
    """Create a Road_network_builder with mock config."""
    # Patch both DataHandler initialization and yaml loading
    with patch('syngrid.data_processor.data.base.DataHandler._validate_fips_dict'), \
            patch('syngrid.data_processor.data.base.DataHandler._create_dataset_output_dir'), \
            patch('syngrid.data_processor.data.osm.road_network_builder.yaml.safe_load') as mock_yaml:

        # Mock the YAML config
        mock_yaml.return_value = {
            'way_tag_resolver': {
                'tags': {
                    'residential': {'clazz': 41, 'maxspeed': 40, 'flags': ['car', 'bike']}
                },
                'flag_list': ['car', 'bike', 'foot']
            }
        }

        # Create a builder with our mocked fips_dict and a temp directory
        builder = Road_network_builder(fips_dict=fips_dict, output_dir=tmp_path)

        # Mock the dataset_output_dir property
        builder.dataset_output_dir = tmp_path

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


@patch('syngrid.data_processor.data.osm.road_network_builder.OSM')
def test_build_network(mock_osm_class, road_network_builder, mock_osm_data, tmp_path):
    """Test the build_network method."""
    # Configure the mock OSM instance
    mock_osm_instance = MagicMock()
    mock_osm_class.return_value = mock_osm_instance

    # Set up the mock to return our test data
    nodes, edges = mock_osm_data
    mock_osm_instance.get_network.return_value = (nodes, edges)

    # Build the network
    results = road_network_builder.build_network(
        osm_pbf_file="test.osm.pbf",
        network_type='driving'
    )

    # Verify OSM was called correctly
    mock_osm_class.assert_called_once_with("test.osm.pbf")
    mock_osm_instance.get_network.assert_called_once_with(network_type='driving', nodes=True)

    # Check results
    assert results['nodes'] is not None
    assert results['edges'] is not None
    assert results['sql_file'] is not None
    assert results['sql_file'].name == "osm2po_routing_network.sql"
    assert results['raw_edges_file'] is not None

    # Verify files were created
    assert os.path.exists(results['sql_file'])
    assert os.path.exists(results['raw_edges_file'])

    # Check SQL file content
    with open(results['sql_file'], 'r') as f:
        sql_content = f.read()
        # Check for key SQL elements
        assert "CREATE TABLE public_2po_4pgr" in sql_content
        assert "INSERT INTO public_2po_4pgr" in sql_content
        assert "CREATE INDEX" in sql_content


@patch('syngrid.data_processor.data.osm.road_network_builder.OSM')
def test_process_method(mock_osm_class, road_network_builder, mock_osm_data, tmp_path):
    """Test the process method."""
    # Configure the mock OSM instance
    mock_osm_instance = MagicMock()
    mock_osm_class.return_value = mock_osm_instance

    # Set up the mock to return our test data
    nodes, edges = mock_osm_data
    mock_osm_instance.get_network.return_value = (nodes, edges)

    # Create a boundary for testing clipping
    boundary_data = {
        'geometry': [LineString([(0, 0), (1, 1), (1, 0)]).buffer(0.1)]
    }
    boundary_gdf = gpd.GeoDataFrame(boundary_data, crs="EPSG:4326")

    # Set the OSM PBF file in the config
    road_network_builder.config['osm_pbf_file'] = "test.osm.pbf"

    # Mock the clip_to_boundary method to simply return the input data
    with patch.object(road_network_builder, 'clip_to_boundary', return_value=edges):
        results = road_network_builder.process(boundary_gdf=boundary_gdf)

    # Verify results
    assert results['nodes'] is not None
    assert results['edges'] is not None
    assert results['sql_file'] is not None
    assert 'clipped_edges' in results


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
    assert road_network_builder._get_dataset_name() == "street_network"


@patch('syngrid.data_processor.data.osm.road_network_builder.logger')
def test_download_method_not_implemented(mock_logger, road_network_builder):
    """Test that download method raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        road_network_builder.download()

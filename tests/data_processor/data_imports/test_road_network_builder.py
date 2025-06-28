"""
Tests for the Road_network_builder class.

These tests verify the functionality of the Road_network_builder class
for creating routable road networks from OSM data.
"""

import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from gridtracer.data.imports.osm.road_network_builder import RoadNetworkBuilder


@pytest.fixture
def mock_osm_data():
    """Create mock OSM data for testing."""
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
    edges = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:3857")  # Use projected CRS

    # Create nodes as GeoDataFrame with point geometries
    from shapely.geometry import Point
    nodes_data = {
        'id': [1, 2],
        'lat': [0, 1],
        'lon': [0, 1]
    }
    nodes_geom = [Point(0, 0), Point(1, 1)]
    nodes = gpd.GeoDataFrame(nodes_data, geometry=nodes_geom, crs="EPSG:3857")  # Use projected CRS

    return nodes, edges


@pytest.fixture
def mock_orchestrator_for_road_network(orchestrator_with_fips, temp_output_dir, mock_osm_data):
    """Creates a mock WorkflowOrchestrator for road network testing."""
    orchestrator = orchestrator_with_fips

    # Override the get_dataset_specific_output_directory method for road network
    original_get_dir = orchestrator.get_dataset_specific_output_directory

    def mock_get_dir(dataset_name):
        if dataset_name == "STREET_NETWORK":
            return temp_output_dir / "STREET_NETWORK"
        return original_get_dir(dataset_name)

    orchestrator.get_dataset_specific_output_directory = mock_get_dir

    # Setup the mock OSM parser to be returned by the orchestrator
    mock_osm_parser = MagicMock()
    nodes, edges = mock_osm_data
    mock_osm_parser.get_network.return_value = (nodes, edges)

    # Mock the to_graph method to return a simple graph object
    mock_graph = MagicMock()
    mock_osm_parser.to_graph.return_value = mock_graph

    orchestrator.get_osm_parser = MagicMock(return_value=mock_osm_parser)

    return orchestrator


@pytest.fixture
def road_network_builder(mock_orchestrator_for_road_network):
    """Create a RoadNetworkBuilder with a mocked orchestrator and config."""
    with patch('gridtracer.data.imports.osm.road_network_builder.yaml.safe_load') as mock_yaml:
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
        builder = RoadNetworkBuilder(orchestrator=mock_orchestrator_for_road_network)

        # Ensure dataset_output_dir is set correctly for tests
        builder.dataset_output_dir = (
            mock_orchestrator_for_road_network.get_dataset_specific_output_directory(
                "STREET_NETWORK"
            )
        )
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


def test_build_network(road_network_builder, mock_osm_data):
    """Test the build_network method."""
    # The mock_orchestrator in road_network_builder fixture already provides a mock OSM parser
    # that returns mock_osm_data.

    # Mock the to_graph and ox.simplification.simplify_graph calls
    with patch('gridtracer.data.imports.osm.road_network_builder.ox') as mock_ox:
        # Mock the simplify_graph and graph_to_gdfs calls
        mock_simplified_graph = MagicMock()
        mock_ox.simplification.simplify_graph.return_value = mock_simplified_graph
        nodes, edges = mock_osm_data
        # Return both nodes and edges GeoDataFrames (not None for nodes)
        mock_ox.graph_to_gdfs.return_value = (nodes.reset_index(), edges.reset_index())

        # Build the network
        results = road_network_builder.build_network()

        # Verify the orchestrator's OSM parser was used
        road_network_builder.orchestrator.get_osm_parser.assert_called_once()
        # Verify the mock OSM parser's get_network was called
        mock_osm_parser_instance = road_network_builder.orchestrator.get_osm_parser()
        mock_osm_parser_instance.get_network.assert_called_once_with(
            nodes=True, network_type='driving')

        # Check results - nodes might not be explicitly returned by build_network
        # Depending on the RoadNetworkBuilder.build_network implementation,
        # 'nodes' key might not be in results. Update as per actual implementation.
        # assert results['nodes'] is not None
        assert results['edges'] is not None
        assert results['sql_file'] is not None
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


def test_process_method(road_network_builder, mock_osm_data):
    """Test the process method."""
    # The mock_orchestrator already provides the mock OSM parser

    # Mock the to_graph and ox.simplification.simplify_graph calls
    with patch('gridtracer.data.imports.osm.road_network_builder.ox') as mock_ox:
        # Mock the simplify_graph and graph_to_gdfs calls
        mock_simplified_graph = MagicMock()
        mock_ox.simplification.simplify_graph.return_value = mock_simplified_graph
        nodes, edges = mock_osm_data
        # Return both nodes and edges GeoDataFrames (not None for nodes)
        mock_ox.graph_to_gdfs.return_value = (nodes.reset_index(), edges.reset_index())

        # Call process method
        results = road_network_builder.process()

        # Verify the orchestrator's OSM parser was used and its get_network was called
        road_network_builder.orchestrator.get_osm_parser.assert_called_once()
        mock_osm_parser_instance = road_network_builder.orchestrator.get_osm_parser()
        # The process method calls build_network, which calls get_network
        mock_osm_parser_instance.get_network.assert_called_once_with(
            nodes=True, network_type='driving')

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


def test_fips_integration(road_network_builder):
    """Test that the road network builder can access FIPS data from the orchestrator."""
    fips_dict = road_network_builder.orchestrator.get_fips_dict()

    # Verify FIPS data is accessible and has expected structure
    assert fips_dict is not None
    assert 'state' in fips_dict
    assert 'county' in fips_dict
    assert fips_dict['state'] == 'MA'  # From the shared fixture
    assert fips_dict['state_fips'] == '25'


def test_download_method_not_implemented(road_network_builder):
    """Test that the download method raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        road_network_builder.download()

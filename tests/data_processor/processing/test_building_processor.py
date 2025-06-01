import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from syngrid.data_processor.processing.building_processor import BuildingHeuristicsProcessor


@pytest.fixture
def building_processor() -> BuildingHeuristicsProcessor:
    """Fixture to create a BuildingHeuristicsProcessor instance."""
    return BuildingHeuristicsProcessor(output_dir="test_output")


@pytest.fixture
def sample_census_blocks() -> gpd.GeoDataFrame:
    """Create sample census blocks for testing."""
    # Two adjacent 100x100 meter blocks
    blocks_data = {
        'STATEFP20': ['12', '12'],
        'COUNTYFP20': ['345', '345'],
        'BLOCKCE20': ['6789', '6790'],
        'geometry': [
            Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]),  # Block 1
            Polygon([(100, 0), (100, 100), (200, 100), (200, 0)])  # Block 2 (adjacent)
        ]
    }
    return gpd.GeoDataFrame(blocks_data, crs="EPSG:5070")


@pytest.fixture
def sample_isolated_buildings() -> gpd.GeoDataFrame:
    """Create sample isolated buildings (no neighbors) for testing."""
    # Small isolated buildings that should be classified as SFH
    buildings_data = {
        'id': [1, 2, 3],
        'floor_area': [150, 180, 190],
        'geometry': [
            Polygon([(10, 10), (10, 20), (20, 20), (20, 10)]),      # 10x10 building
            Polygon([(40, 10), (40, 22), (52, 22), (52, 10)]),      # 12x12 building
            Polygon([(70, 10), (70, 23), (83, 23), (83, 10)])       # 13x13 building
        ]
    }
    return gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")


@pytest.fixture
def sample_townhouse_cluster() -> gpd.GeoDataFrame:
    """Create a linear arrangement of touching buildings (townhouses)."""
    # 5 connected buildings in a row, each 8x15 meters (120 m²)
    buildings_data = {
        'id': [1, 2, 3, 4, 5],
        'floor_area': [120, 120, 120, 120, 120],
        'geometry': [
            Polygon([(0, 0), (0, 15), (8, 15), (8, 0)]),       # Building 1
            Polygon([(8, 0), (8, 15), (16, 15), (16, 0)]),     # Building 2 (touches 1 and 3)
            Polygon([(16, 0), (16, 15), (24, 15), (24, 0)]),   # Building 3 (touches 2 and 4)
            Polygon([(24, 0), (24, 15), (32, 15), (32, 0)]),   # Building 4 (touches 3 and 5)
            Polygon([(32, 0), (32, 15), (40, 15), (40, 0)])    # Building 5 (touches 4)
        ]
    }
    return gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")


@pytest.fixture
def sample_apartment_cluster() -> gpd.GeoDataFrame:
    """Create a large cluster of connected buildings (apartment complex)."""
    # Large connected buildings forming a U-shape, total > 2000 m²
    buildings_data = {
        'id': [1, 2, 3, 4],
        'floor_area': [600, 600, 600, 400],
        'geometry': [
            # Left wing
            Polygon([(0, 0), (0, 30), (20, 30), (20, 0)]),
            # Bottom connector
            Polygon([(20, 0), (20, 20), (50, 20), (50, 0)]),
            # Right wing
            Polygon([(50, 0), (50, 30), (70, 30), (70, 0)]),
            # Small attachment
            Polygon([(70, 10), (70, 30), (80, 30), (80, 10)])
        ]
    }
    return gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")


@pytest.fixture
def sample_mixed_neighborhood() -> gpd.GeoDataFrame:
    """Create a mixed neighborhood with different building types."""
    buildings_data = {
        'id': list(range(1, 11)),
        'floor_area': [
            150, 180,  # Isolated SFH
            120, 120, 120,  # Townhouse row
            400, 300,  # MFH cluster
            800, 800, 500  # Large AB cluster
        ],
        'geometry': [
            # Isolated SFH buildings
            Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
            Polygon([(20, 0), (20, 12), (32, 12), (32, 0)]),

            # Townhouse row (3 connected)
            Polygon([(0, 30), (0, 45), (8, 45), (8, 30)]),
            Polygon([(8, 30), (8, 45), (16, 45), (16, 30)]),
            Polygon([(16, 30), (16, 45), (24, 45), (24, 30)]),

            # MFH cluster (2 connected)
            Polygon([(40, 0), (40, 20), (60, 20), (60, 0)]),
            Polygon([(60, 0), (60, 15), (75, 15), (75, 0)]),

            # Large AB cluster (3 connected, total > 2000)
            Polygon([(40, 30), (40, 50), (70, 50), (70, 30)]),
            Polygon([(70, 35), (70, 55), (90, 55), (90, 35)]),
            Polygon([(90, 40), (90, 55), (100, 55), (100, 40)])
        ]
    }
    return gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")


class TestBuildingClassification:
    """Test suite for building type classification functionality."""

    def test_classify_isolated_buildings_as_sfh(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_isolated_buildings: gpd.GeoDataFrame
    ) -> None:
        """Test that small isolated buildings are classified as SFH."""
        result = building_processor.classify_building_type(sample_isolated_buildings)

        # All buildings should be classified as SFH
        assert all(result['building_type'] == 'SFH')

        # Check that neighbor detection worked (no neighbors)
        assert all(result['neighbors'].apply(len) == 0)

        # Check cluster formation (each building is its own cluster)
        assert all(result['cluster'].apply(len) == 1)

        # Total cluster area should equal individual building area
        assert all(result['total_cluster_area'] == result['floor_area'])

    def test_classify_townhouse_cluster(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_townhouse_cluster: gpd.GeoDataFrame
    ) -> None:
        """Test that linear arrangements of similar buildings are classified as TH."""
        result = building_processor.classify_building_type(sample_townhouse_cluster)

        # All buildings should be classified as TH
        assert all(result['building_type'] == 'TH')

        # Check neighbor counts
        neighbor_counts = result['neighbors'].apply(len)
        assert neighbor_counts.iloc[0] == 1  # First building has 1 neighbor
        assert neighbor_counts.iloc[1] == 2  # Middle buildings have 2 neighbors
        assert neighbor_counts.iloc[2] == 2
        assert neighbor_counts.iloc[3] == 2
        assert neighbor_counts.iloc[4] == 1  # Last building has 1 neighbor

        # All buildings should be in the same cluster
        assert all(result['cluster'].apply(len) == 5)

        # Total cluster area should be 600 m² (5 x 120)
        assert all(result['total_cluster_area'] == 600)

    def test_classify_apartment_cluster(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_apartment_cluster: gpd.GeoDataFrame
    ) -> None:
        """Test that large connected clusters are classified as AB."""
        result = building_processor.classify_building_type(sample_apartment_cluster)

        # All buildings should be classified as AB
        assert all(result['building_type'] == 'AB')

        # All buildings should be in the same cluster
        assert all(result['cluster'].apply(len) == 4)

        # Total cluster area should be 2200 m² (600+600+600+400)
        assert all(result['total_cluster_area'] == 2200)

    def test_classify_mixed_neighborhood(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_mixed_neighborhood: gpd.GeoDataFrame
    ) -> None:
        """Test classification of a mixed neighborhood with all building types."""
        result = building_processor.classify_building_type(sample_mixed_neighborhood)

        # Check building type distribution
        type_counts = result['building_type'].value_counts()

        # Should have 2 SFH (isolated buildings)
        assert type_counts.get('SFH', 0) == 2

        # Should have 3 TH (townhouse row)
        assert type_counts.get('TH', 0) == 3

        # Should have 2 MFH (medium cluster)
        assert type_counts.get('MFH', 0) == 2

        # Should have 3 AB (large cluster)
        assert type_counts.get('AB', 0) == 3

        # Verify specific buildings
        assert result.iloc[0]['building_type'] == 'SFH'  # First isolated
        assert result.iloc[1]['building_type'] == 'SFH'  # Second isolated
        assert result.iloc[2]['building_type'] == 'TH'   # Townhouse 1
        assert result.iloc[3]['building_type'] == 'TH'   # Townhouse 2
        assert result.iloc[4]['building_type'] == 'TH'   # Townhouse 3
        assert result.iloc[5]['building_type'] == 'MFH'  # MFH 1
        assert result.iloc[6]['building_type'] == 'MFH'  # MFH 2
        assert result.iloc[7]['building_type'] == 'AB'   # AB 1
        assert result.iloc[8]['building_type'] == 'AB'   # AB 2
        assert result.iloc[9]['building_type'] == 'AB'   # AB 3

    def test_empty_buildings_input(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test handling of empty buildings input."""
        empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor.classify_building_type(empty_buildings)

        assert len(result) == 0

    def test_missing_floor_area_column(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test that missing floor_area column raises an error."""
        buildings_data = {
            'id': [1],
            'geometry': [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        with pytest.raises(ValueError, match="Missing required 'floor_area' column"):
            building_processor.classify_building_type(buildings)

    def test_neighbor_detection_accuracy(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test that neighbor detection correctly identifies touching buildings."""
        # Create buildings where some touch and some don't
        buildings_data = {
            'id': [1, 2, 3, 4],
            'floor_area': [100, 100, 100, 100],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),      # Building 1
                Polygon([(10, 0), (10, 10), (20, 10), (20, 0)]),    # Building 2 (touches 1)
                # Building 3 (isolated - moved further away)
                Polygon([(35, 0), (35, 10), (45, 10), (45, 0)]),
                Polygon([(20, 5), (20, 15), (30, 15), (30, 5)])     # Building 4 (touches 2)
            ]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        neighbors_dict = building_processor._find_direct_neighbors(buildings)

        # Building 0 should have building 1 as neighbor
        assert neighbors_dict[0] == [1]

        # Building 1 should have buildings 0 and 3 as neighbors
        assert set(neighbors_dict[1]) == {0, 3}

        # Building 2 should have no neighbors (isolated)
        assert neighbors_dict[2] == []

        # Building 3 should have building 1 as neighbor
        assert neighbors_dict[3] == [1]

    def test_cluster_expansion(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test that cluster expansion finds all connected buildings."""
        # Create a chain of connected buildings
        neighbors_dict = {
            0: [1],        # Building 0 connects to 1
            1: [0, 2],     # Building 1 connects to 0 and 2
            2: [1, 3],     # Building 2 connects to 1 and 3
            3: [2],        # Building 3 connects to 2
            4: [],         # Building 4 is isolated
            5: [6],        # Building 5 connects to 6
            6: [5]         # Building 6 connects to 5
        }

        clusters_dict = building_processor._expand_to_clusters(neighbors_dict)

        # Buildings 0-3 should all be in the same cluster
        assert clusters_dict[0] == {0, 1, 2, 3}
        assert clusters_dict[1] == {0, 1, 2, 3}
        assert clusters_dict[2] == {0, 1, 2, 3}
        assert clusters_dict[3] == {0, 1, 2, 3}

        # Building 4 should be in its own cluster
        assert clusters_dict[4] == {4}

        # Buildings 5-6 should be in the same cluster
        assert clusters_dict[5] == {5, 6}
        assert clusters_dict[6] == {5, 6}


def test_building_id_assignment_fully_contained(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment for a building fully contained within one block."""
    # Building completely inside block 1
    building_data = {
        'id': [1],
        'geometry': [Polygon([(10, 10), (10, 30), (30, 30), (30, 10)])]
    }
    buildings = gpd.GeoDataFrame(building_data, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # Should have exactly one building with an ID
    assert len(result) == 1
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Should be assigned to block 1 (BLOCKCE20='6789')
    assert result.iloc[0]['census_block_id'] == '6789'
    assert result.iloc[0]['building_id'] == '1234567890001'  # STATEFP+COUNTYFP+BLOCKCE+0001


def test_building_id_assignment_85_percent_overlap(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment for a building with 85% overlap in one block."""
    # Building that spans boundary but 85% in block 1
    # Building: 20x20, positioned from (70,10) to (105,30)
    # 85.7% is in block 1 (x: 70-100 = 30 units), 14.3% is in block 2 (x: 100-105 = 5 units)
    # Total width = 35 units, so 30/35 = 85.7%
    building_data = {
        'id': [1],
        'geometry': [Polygon([(70, 10), (70, 30), (105, 30), (105, 10)])]
    }
    buildings = gpd.GeoDataFrame(building_data, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # Should have exactly one building with an ID (assigned to block with larger overlap)
    assert len(result) == 1
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Should be assigned to block 1 (BLOCKCE20='6789') since it has more overlap
    assert result.iloc[0]['census_block_id'] == '6789'
    assert result.iloc[0]['building_id'] == '1234567890001'


def test_building_id_assignment_50_50_split_assigned(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment for a building with 50/50 split - should be assigned to one."""
    # Building that spans boundary exactly 50/50
    # Building: 20x20, positioned from (90,10) to (110,30)
    # 50% in block 1 (x: 90-100), 50% in block 2 (x: 100-110)
    # Both overlaps are positive and equal. Should be assigned to the first one evaluated with max overlap.
    # Given the order in sample_census_blocks, block '6789' is likely first.
    building_data = {
        'id': [1],  # Original building identifier
        'geometry': [Polygon([(90, 10), (90, 30), (110, 30), (110, 10)])]
    }
    buildings = gpd.GeoDataFrame(building_data, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # Should have exactly one building (assigned because overlap is positive)
    assert len(result) == 1
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Expected to be assigned to block '6789' as it's the first one with max overlap
    assert result.iloc[0]['census_block_id'] == '6789'
    assert result.iloc[0]['building_id'] == '1234567890001'


def test_building_id_assignment_multiple_buildings_same_block(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test sequential ID assignment for multiple buildings in the same block."""
    # Two buildings fully contained in block 1
    buildings_data = {
        'id': [1, 2, 3, 4],
        'geometry': [
            Polygon([(10, 10), (10, 30), (30, 30), (30, 10)]),   # Building ID 1: Fully in block 1
            Polygon([(70, 10), (70, 30), (105, 30), (105, 10)]),  # Building ID 2: 85% in block 1
            Polygon([(90, 10), (90, 30), (110, 30), (110, 10)]),  # Building ID 3: 50/50 split
            # Building ID 4: Fully in block 2
            Polygon([(150, 10), (150, 30), (170, 30), (170, 10)])
        ]
    }
    buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")
    # Ensure buildings GeoDataFrame has a unique index for groupby(level=0) to work as expected
    buildings = buildings.reset_index(drop=True)

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # Should have 4 buildings (all assigned as they have positive overlap)
    assert len(result) == 4
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Check assignments by block
    # Building 1 (id:0) -> block '6789'
    # Building 2 (id:1) -> block '6789' (85% overlap)
    # Building 3 (id:2) -> block '6789' (50% overlap, assigned to first max)
    # Building 4 (id:3) -> block '6790'

    block1_buildings = result[result['census_block_id'] == '6789']
    block2_buildings = result[result['census_block_id'] == '6790']

    assert len(block1_buildings) == 3
    assert len(block2_buildings) == 1

    block1_ids = sorted(block1_buildings['building_id'].tolist())
    block2_ids = sorted(block2_buildings['building_id'].tolist())

    assert block1_ids == ['1234567890001', '1234567890002', '1234567890003']
    assert block2_ids == ['1234567900001']


def test_building_id_assignment_mixed_scenarios(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment with mixed scenarios - some assigned, some removed."""
    buildings_data = {
        'id': [1, 2, 3, 4],
        'geometry': [
            Polygon([(10, 10), (10, 30), (30, 30), (30, 10)]),   # Fully in block 1
            Polygon([(70, 10), (70, 30), (105, 30), (105, 10)]),  # 85% in block 1, 15% in block 2
            Polygon([(90, 10), (90, 30), (110, 30), (110, 10)]),  # 50/50 split - should be removed
            Polygon([(150, 10), (150, 30), (170, 30), (170, 10)])  # Fully in block 2
        ]
    }
    buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # Should have 3 buildings (one removed due to 50/50 split)
    assert len(result) == 3
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Check assignments by block
    block1_buildings = result[result['census_block_id'] == '6789']
    block2_buildings = result[result['census_block_id'] == '6790']

    assert len(block1_buildings) == 2  # Buildings 1 and 2
    assert len(block2_buildings) == 1  # Building 4

    # Check sequential numbering within each block
    block1_ids = sorted(block1_buildings['building_id'].tolist())
    block2_ids = sorted(block2_buildings['building_id'].tolist())

    assert block1_ids == ['1234567890001', '1234567890002']
    assert block2_ids == ['1234567900001']


def test_building_id_assignment_empty_input(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment with empty buildings input."""
    empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

    result = building_processor._assign_building_id(empty_buildings, sample_census_blocks)

    assert len(result) == 0
    # Should still have the expected columns
    expected_columns = {'building_id', 'census_block_id'}
    assert expected_columns.issubset(set(result.columns))


def test_building_id_assignment_no_census_blocks(
    building_processor: BuildingHeuristicsProcessor
) -> None:
    """Test building ID assignment with no census blocks."""
    building_data = {
        'id': [1],
        'geometry': [Polygon([(10, 10), (10, 30), (30, 30), (30, 10)])]
    }
    buildings = gpd.GeoDataFrame(building_data, crs="EPSG:5070")
    empty_blocks = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, empty_blocks)

    # Should return original buildings with null IDs
    assert len(result) == 1
    assert result['building_id'].isna().all()
    assert result['census_block_id'].isna().all()


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


class TestFreeWallsCalculation:
    """Test suite for free walls calculation functionality."""

    def test_calculate_free_walls_isolated_buildings(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_isolated_buildings: gpd.GeoDataFrame
    ) -> None:
        """Test free walls calculation for isolated buildings."""
        result = building_processor.calculate_free_walls(sample_isolated_buildings)

        # All isolated buildings should have 4 free walls
        assert all(result['free_walls'] == 4)

        # All buildings should have empty neighbors lists
        assert all(result['neighbors'].apply(len) == 0)

    def test_calculate_free_walls_townhouse_row(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_townhouse_cluster: gpd.GeoDataFrame
    ) -> None:
        """Test free walls calculation for a row of townhouses."""
        result = building_processor.calculate_free_walls(sample_townhouse_cluster)

        # Check free walls count
        free_walls = result['free_walls'].tolist()

        # End buildings should have 3 free walls (1 neighbor)
        assert free_walls[0] == 3  # First building
        assert free_walls[4] == 3  # Last building

        # Middle buildings should have 2 free walls (2 neighbors)
        assert free_walls[1] == 2  # Second building
        assert free_walls[2] == 2  # Third building
        assert free_walls[3] == 2  # Fourth building

        # Check neighbor counts
        neighbor_counts = result['neighbors'].apply(len).tolist()
        assert neighbor_counts == [1, 2, 2, 2, 1]

    def test_calculate_free_walls_complex_cluster(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_apartment_cluster: gpd.GeoDataFrame
    ) -> None:
        """Test free walls calculation for a complex building cluster."""
        result = building_processor.calculate_free_walls(sample_apartment_cluster)

        # All buildings should have some neighbors
        assert all(result['neighbors'].apply(len) > 0)

        # All buildings should have fewer than 4 free walls
        assert all(result['free_walls'] < 4)

        # Free walls should equal 4 minus neighbor count
        for idx, row in result.iterrows():
            expected_free_walls = max(0, 4 - len(row['neighbors']))
            assert row['free_walls'] == expected_free_walls

    def test_calculate_free_walls_empty_input(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test free walls calculation with empty input."""
        empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor.calculate_free_walls(empty_buildings)

        assert len(result) == 0


class TestFloorsCalculation:
    """Test suite for floors calculation functionality."""

    def test_calculate_floors_with_microsoft_data(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test floors calculation using Microsoft Buildings data."""
        # Create sample buildings
        buildings_data = {
            'id': [1, 2],
            'floor_area': [100, 200],
            'building_type': ['SFH', 'MFH'],
            'occupants': [3, 8],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
                Polygon([(20, 0), (20, 15), (35, 15), (35, 0)])
            ]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        # Create sample Microsoft Buildings data with height information
        ms_buildings_data = {
            'height': [9.0, 15.0],  # 3 floors and 5 floors respectively
            'confidence': [0.8, 0.9],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),  # Overlaps building 1
                Polygon([(20, 0), (20, 15), (35, 15), (35, 0)])  # Overlaps building 2
            ]
        }
        ms_buildings = gpd.GeoDataFrame(ms_buildings_data, crs="EPSG:5070")

        result = building_processor.calculate_floors(buildings, ms_buildings)

        # Should have floors and height columns
        assert 'floors' in result.columns
        assert 'height' in result.columns

        # Check that floors were calculated from Microsoft data
        assert result.iloc[0]['floors'] == 3  # 9m / 3m per floor
        assert result.iloc[1]['floors'] == 5  # 15m / 3m per floor

        # Check that height was assigned from Microsoft data
        assert result.iloc[0]['height'] == 9.0
        assert result.iloc[1]['height'] == 15.0

    def test_calculate_floors_without_microsoft_data(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test floors calculation when no Microsoft Buildings data is available."""
        buildings_data = {
            'id': [1, 2, 3],
            'floor_area': [80, 150, 400],
            'building_type': ['SFH', 'TH', 'MFH'],
            'occupants': [2, 4, 12],
            'geometry': [
                Polygon([(0, 0), (0, 8), (10, 8), (10, 0)]),
                Polygon([(20, 0), (20, 10), (35, 10), (35, 0)]),
                Polygon([(50, 0), (50, 20), (70, 20), (70, 0)])
            ]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        # Empty Microsoft Buildings data
        empty_ms_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor.calculate_floors(buildings, empty_ms_buildings)

        # Should have floors and height columns
        assert 'floors' in result.columns
        assert 'height' in result.columns

        # Check estimated floors based on building type and occupants
        # SFH with small area should have 1 floor
        assert result.iloc[0]['floors'] == 1

        # TH should have 2-3 floors
        assert result.iloc[1]['floors'] >= 2
        assert result.iloc[1]['floors'] <= 3

        # MFH should have 2-5 floors
        assert result.iloc[2]['floors'] >= 2
        assert result.iloc[2]['floors'] <= 5

        # Height should be estimated from floors (floors * 3.5)
        for idx, row in result.iterrows():
            expected_height = row['floors'] * 3.5
            assert row['height'] >= expected_height * 0.9  # Allow some tolerance
            assert row['height'] <= expected_height * 1.1

    def test_calculate_floors_empty_input(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test floors calculation with empty input."""
        empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")
        empty_ms_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor.calculate_floors(empty_buildings, empty_ms_buildings)

        assert len(result) == 0

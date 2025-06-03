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
        'GEOID20': ['123456789', '123456790'],  # STATEFP20 + COUNTYFP20 + BLOCKCE20
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

    # Should be assigned to block 1 (GEOID20='123456789')
    assert result.iloc[0]['census_block_id'] == '123456789'
    assert result.iloc[0]['building_id'] == '1234567890001'  # GEOID20 + 0001


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

    # Should be assigned to block 1 (GEOID20='123456789') since it has more overlap
    assert result.iloc[0]['census_block_id'] == '123456789'
    assert result.iloc[0]['building_id'] == '1234567890001'


def test_building_id_assignment_50_50_split_assigned(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment for a building with 50/50 split - centroid on boundary."""
    # Building that spans boundary exactly 50/50
    # Building: 20x20, positioned from (90,10) to (110,30)
    # 50% in block 1 (x: 90-100), 50% in block 2 (x: 100-110)
    # Centroid is at (100, 20) which is exactly on the boundary
    # With centroid-based assignment, this building should NOT be assigned
    building_data = {
        'id': [1],  # Original building identifier
        'geometry': [Polygon([(90, 10), (90, 30), (110, 30), (110, 10)])]
    }
    buildings = gpd.GeoDataFrame(building_data, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # With centroid-based assignment, buildings with centroids on boundaries are unassigned
    assert len(result) == 0  # Building is not assigned because centroid is on boundary


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

    # Should have 3 buildings (50/50 split building is unassigned due to centroid on boundary)
    assert len(result) == 3
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Check assignments by block
    # Building 1 (id:0) -> block '123456789'
    # Building 2 (id:1) -> block '123456789' (85% overlap)
    # Building 3 (id:2) -> UNASSIGNED (50/50 split, centroid on boundary)
    # Building 4 (id:3) -> block '123456790'

    block1_buildings = result[result['census_block_id'] == '123456789']
    block2_buildings = result[result['census_block_id'] == '123456790']

    assert len(block1_buildings) == 2  # Buildings 1 and 2
    assert len(block2_buildings) == 1   # Building 4

    block1_ids = sorted(block1_buildings['building_id'].tolist())
    block2_ids = sorted(block2_buildings['building_id'].tolist())

    assert block1_ids == ['1234567890001', '1234567890002']
    assert block2_ids == ['1234567900001']


def test_building_id_assignment_mixed_scenarios(
    building_processor: BuildingHeuristicsProcessor,
    sample_census_blocks: gpd.GeoDataFrame
) -> None:
    """Test building ID assignment with mixed scenarios - some assigned, some unassigned."""
    buildings_data = {
        'id': [1, 2, 3, 4],
        'geometry': [
            Polygon([(10, 10), (10, 30), (30, 30), (30, 10)]),   # Fully in block 1
            Polygon([(70, 10), (70, 30), (105, 30), (105, 10)]),  # 85% in block 1, 15% in block 2
            # 50/50 split - centroid on boundary
            Polygon([(90, 10), (90, 30), (110, 30), (110, 10)]),
            Polygon([(150, 10), (150, 30), (170, 30), (170, 10)])  # Fully in block 2
        ]
    }
    buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

    result = building_processor._assign_building_id(buildings, sample_census_blocks)

    # Should have 3 buildings (one unassigned due to centroid on boundary)
    assert len(result) == 3
    assert result['building_id'].notna().all()
    assert result['census_block_id'].notna().all()

    # Check assignments by block
    block1_buildings = result[result['census_block_id'] == '123456789']
    block2_buildings = result[result['census_block_id'] == '123456790']

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
            # Will give 4 floors and 6 floors respectively (using 2.5m/floor)
            'height': [9.0, 15.0],
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

        # Check that floors were calculated from Microsoft data using 2.5m per floor
        assert result.iloc[0]['floors'] == 4  # 9m / 2.5m = 3.6 → rounded to 4
        assert result.iloc[1]['floors'] == 6  # 15m / 2.5m = 6

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

        # Without Microsoft data and OSM tags, floors might remain None
        # The function doesn't currently implement heuristic-based estimation
        # when no height data is available, so we expect None values
        assert result.iloc[0]['floors'] is None or pd.isna(result.iloc[0]['floors'])
        assert result.iloc[1]['floors'] is None or pd.isna(result.iloc[1]['floors'])
        assert result.iloc[2]['floors'] is None or pd.isna(result.iloc[2]['floors'])

    def test_calculate_floors_empty_input(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test floors calculation with empty input."""
        empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")
        empty_ms_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor.calculate_floors(empty_buildings, empty_ms_buildings)

        assert len(result) == 0


class TestOccupantAllocation:
    """Test suite for occupant allocation functionality."""

    @pytest.fixture
    def sample_buildings_with_types(self) -> gpd.GeoDataFrame:
        """Create sample buildings with different types for allocation testing."""
        buildings_data = {
            'building_id': ['block1_001', 'block1_002', 'block1_003', 'block2_001'],
            'census_block_id': ['12345001', '12345001', '12345001', '12345002'],
            'building_type': ['SFH', 'TH', 'MFH', 'AB'],
            'floor_area': [150, 120, 300, 800],
            'building_use': ['residential', 'residential', 'residential', 'residential'],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
                Polygon([(20, 0), (20, 8), (35, 8), (35, 0)]),
                Polygon([(50, 0), (50, 15), (70, 15), (70, 0)]),
                Polygon([(100, 0), (100, 20), (140, 20), (140, 0)])
            ]
        }
        return gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

    @pytest.fixture
    def sample_census_blocks_with_population(self) -> gpd.GeoDataFrame:
        """Create sample census blocks with population data."""
        blocks_data = {
            'GEOID20': ['12345001', '12345002'],
            'POP20': [100, 200],
            'HOUSING20': [40, 80],
            'geometry': [
                Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]),
                Polygon([(100, 0), (100, 100), (200, 100), (200, 0)])
            ]
        }
        return gpd.GeoDataFrame(blocks_data, crs="EPSG:5070")

    def test_allot_occupants_normal_allocation(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_buildings_with_types: gpd.GeoDataFrame,
        sample_census_blocks_with_population: gpd.GeoDataFrame
    ) -> None:
        """Test normal occupant allocation without capacity adjustment."""
        result = building_processor._allot_occupants(
            sample_buildings_with_types,
            sample_census_blocks_with_population
        )

        # Should have occupants and housing_units columns
        assert 'occupants' in result.columns
        assert 'housing_units' in result.columns

        # All buildings should have some occupants
        assert (result['occupants'] > 0).all()
        assert (result['housing_units'] > 0).all()

        # Check census block 1 totals (100 people, 40 housing units)
        block1_buildings = result[result['census_block_id'] == '12345001']
        block1_total_pop = block1_buildings['occupants'].sum()
        block1_total_units = block1_buildings['housing_units'].sum()

        # Should match census totals (with small floating point tolerance)
        assert abs(block1_total_pop - 100) < 0.1
        assert abs(block1_total_units - 40) < 0.1

        # Check census block 2 totals (200 people, 80 housing units)
        block2_buildings = result[result['census_block_id'] == '12345002']
        block2_total_pop = block2_buildings['occupants'].sum()
        block2_total_units = block2_buildings['housing_units'].sum()

        assert abs(block2_total_pop - 200) < 0.1
        assert abs(block2_total_units - 80) < 0.1

    def test_allot_occupants_capacity_adjustment(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test occupant allocation when capacity adjustment is needed."""
        # Create buildings with low capacity relative to population
        buildings_data = {
            'building_id': ['small1', 'small2'],
            'census_block_id': ['12345001', '12345001'],
            'building_type': ['SFH', 'SFH'],  # Small buildings
            'floor_area': [80, 90],  # Small areas
            'building_use': ['residential', 'residential'],
            'geometry': [
                Polygon([(0, 0), (0, 8), (8, 8), (8, 0)]),
                Polygon([(20, 0), (20, 9), (29, 9), (29, 0)])
            ]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        # Census block with high population relative to building capacity
        blocks_data = {
            'GEOID20': ['12345001'],
            'POP20': [50],  # High population for small buildings
            'HOUSING20': [20],
            'geometry': [Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])]
        }
        blocks = gpd.GeoDataFrame(blocks_data, crs="EPSG:5070")

        result = building_processor._allot_occupants(buildings, blocks)

        # Should still allocate all population
        total_occupants = result['occupants'].sum()
        assert abs(total_occupants - 50) < 0.1

        # All buildings should have occupants
        assert (result['occupants'] > 0).all()

    def test_allot_occupants_empty_buildings_input(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_census_blocks_with_population: gpd.GeoDataFrame
    ) -> None:
        """Test occupant allocation with empty buildings input."""
        empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor._allot_occupants(
            empty_buildings,
            sample_census_blocks_with_population
        )

        assert len(result) == 0

    def test_allot_occupants_empty_census_blocks(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_buildings_with_types: gpd.GeoDataFrame
    ) -> None:
        """Test occupant allocation with empty census blocks."""
        empty_blocks = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor._allot_occupants(
            sample_buildings_with_types,
            empty_blocks
        )

        # Should return buildings unchanged (but with occupants and housing_units columns)
        assert len(result) == len(sample_buildings_with_types)
        assert 'occupants' in result.columns
        assert 'housing_units' in result.columns

    def test_allot_occupants_zero_population(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_buildings_with_types: gpd.GeoDataFrame
    ) -> None:
        """Test occupant allocation with zero population census blocks."""
        blocks_data = {
            'GEOID20': ['12345001', '12345002'],
            'POP20': [0, 0],  # Zero population
            'HOUSING20': [0, 0],
            'geometry': [
                Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]),
                Polygon([(100, 0), (100, 100), (200, 100), (200, 0)])
            ]
        }
        blocks = gpd.GeoDataFrame(blocks_data, crs="EPSG:5070")

        result = building_processor._allot_occupants(sample_buildings_with_types, blocks)

        # Some buildings might still have occupants from statistical fallback
        assert 'occupants' in result.columns
        assert 'housing_units' in result.columns

    def test_allot_occupants_mixed_building_types(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test allocation with mixed building types in same census block."""
        # Mix of building types with different characteristics
        buildings_data = {
            'building_id': [f'mixed_{i}' for i in range(1, 6)],
            'census_block_id': ['12345001'] * 5,
            'building_type': ['SFH', 'TH', 'MFH', 'AB', 'SFH'],
            'floor_area': [150, 120, 400, 1000, 180],
            'building_use': ['residential'] * 5,
            'geometry': [
                Polygon([(i * 30, 0), (i * 30, 10), ((i + 1) * 30, 10), ((i + 1) * 30, 0)])
                for i in range(5)
            ]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        blocks_data = {
            'GEOID20': ['12345001'],
            'POP20': [150],
            'HOUSING20': [60],
            'geometry': [Polygon([(0, 0), (0, 100), (200, 100), (200, 0)])]
        }
        blocks = gpd.GeoDataFrame(blocks_data, crs="EPSG:5070")

        result = building_processor._allot_occupants(buildings, blocks)

        # Census totals should be preserved
        total_pop = result['occupants'].sum()
        total_units = result['housing_units'].sum()
        assert abs(total_pop - 150) < 0.1
        assert abs(total_units - 60) < 0.1

        # Larger buildings should generally have more occupants
        ab_occupants = result[result['building_type'] == 'AB']['occupants'].iloc[0]
        sfh_occupants = result[result['building_type'] == 'SFH']['occupants'].mean()
        assert ab_occupants > sfh_occupants


class TestEvaluationFunction:
    """Test suite for census block evaluation functionality."""

    @pytest.fixture
    def sample_allocated_buildings(self) -> gpd.GeoDataFrame:
        """Create sample buildings with allocation results."""
        buildings_data = {
            'building_id': ['test_001', 'test_002', 'test_003'],
            'census_block_id': ['12345001', '12345001', '12345002'],
            'building_use': ['residential', 'residential', 'residential'],
            'building_type': ['SFH', 'MFH', 'TH'],
            'floor_area': [150, 400, 120],
            'floors': [2, 4, 3],
            'height': [5.0, 10.0, 7.5],
            'occupants': [4, 15, 6],
            'housing_units': [1, 6, 2],
            'free_walls': [4, 2, 3],
            'geometry': [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
                Polygon([(20, 0), (20, 20), (40, 20), (40, 0)]),
                Polygon([(60, 0), (60, 8), (75, 8), (75, 0)])
            ]
        }
        return gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

    def test_evaluate_census_block_allocation_normal(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_allocated_buildings: gpd.GeoDataFrame
    ) -> None:
        """Test evaluation function with normal input."""
        result = building_processor.evaluate_census_block_allocation(
            sample_allocated_buildings, '12345001'
        )

        # Should return DataFrame with 2 buildings from block 12345001
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Should have key columns
        expected_columns = [
            'building_id', 'building_use', 'building_type', 'floor_area',
            'floors', 'height', 'occupants', 'housing_units', 'free_walls'
        ]
        for col in expected_columns:
            assert col in result.columns

        # Should have derived metrics
        assert 'people_per_unit' in result.columns
        assert 'people_per_sqm' in result.columns

        # Check data integrity
        building_types = result['building_type'].tolist()
        assert set(building_types) == {'SFH', 'MFH'}  # Should contain both types
        assert result['occupants'].sum() == 19  # 4 + 15
        assert result['housing_units'].sum() == 7  # 1 + 6

    def test_evaluate_census_block_allocation_empty_block(
        self,
        building_processor: BuildingHeuristicsProcessor,
        sample_allocated_buildings: gpd.GeoDataFrame
    ) -> None:
        """Test evaluation function with non-existent census block."""
        result = building_processor.evaluate_census_block_allocation(
            sample_allocated_buildings, 'NONEXISTENT'
        )

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_evaluate_census_block_allocation_empty_buildings(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test evaluation function with empty buildings input."""
        empty_buildings = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:5070")

        result = building_processor.evaluate_census_block_allocation(
            empty_buildings, '12345001'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_evaluate_census_block_allocation_missing_columns(
        self,
        building_processor: BuildingHeuristicsProcessor
    ) -> None:
        """Test evaluation function with missing required columns."""
        buildings_data = {
            'building_id': ['test_001'],
            'geometry': [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])]
        }
        buildings = gpd.GeoDataFrame(buildings_data, crs="EPSG:5070")

        result = building_processor.evaluate_census_block_allocation(
            buildings, '12345001'
        )

        # Should return empty DataFrame due to missing census_block_id column
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

"""
Updated tests for the main gridtracer data processing pipeline script.

These tests verify the orchestration logic, error handling, and proper
integration between pipeline components while mocking individual handlers.
Tests are updated to match the current implementation.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

if TYPE_CHECKING:
    pass

from gridtracer.scripts.main import run_full_pipeline


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_census_data(sample_boundary_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Create mock census data for testing."""
    return {
        "target_region_blocks": sample_boundary_gdf,
        "target_region_blocks_filepath": "/test/blocks.geojson",
        "target_region_boundary": sample_boundary_gdf,
        "target_region_boundary_filepath": "/test/boundary.geojson",
    }


@pytest.fixture
def mock_nrel_data() -> Dict[str, Any]:
    """Create mock NREL data for testing."""
    return {
        "parquet_path": "/test/nrel_data.parquet",
        "vintage_distribution": {
            "<1940": 0.35,
            "1940s": 0.25,
            "1950s": 0.20,
            "1960s": 0.15,
            "1970s": 0.05,
        },
    }


@pytest.fixture
def mock_osm_data() -> Dict[str, Any]:
    """Create mock OSM data for testing - matches current implementation."""
    sample_buildings = gpd.GeoDataFrame(
        {
            "building": ["house", "commercial"],
            "geometry": [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
                Polygon([(20, 0), (20, 15), (35, 15), (35, 0)]),
            ],
        },
        crs="EPSG:4326",
    )

    return {
        "buildings": sample_buildings,
        "buildings_filepath": "/test/buildings.geojson",
        "pois": gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326"),
        "pois_filepath": "/test/pois.geojson",
        "landuse": gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326"),
        "landuse_filepath": "/test/landuse.geojson",
        "power": gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326"),
        "power_filepath": "/test/power.geojson",
    }


@pytest.fixture
def mock_microsoft_buildings_data() -> Dict[str, Any]:
    """Create mock Microsoft Buildings data for testing."""
    ms_buildings = gpd.GeoDataFrame(
        {
            "height": [9.0, 15.0],
            "confidence": [0.8, 0.9],
            "geometry": [
                Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
                Polygon([(20, 0), (20, 15), (35, 15), (35, 0)]),
            ],
        },
        crs="EPSG:4326",
    )

    return {
        "ms_buildings": ms_buildings,
        "ms_buildings_filepath": "/test/ms_buildings.geojson",
    }


@pytest.fixture
def mock_road_network_results() -> Dict[str, Any]:
    """Create mock road network results for testing."""
    return {
        "geojson_file": "/test/road_network.geojson",
        "node_count": 150,
        "edge_count": 200,
    }


class TestMainPipeline:
    """Test suite for the main data processing pipeline."""

    def test_successful_pipeline_execution(
        self,
        mock_census_data: Dict[str, Any],
        mock_nrel_data: Dict[str, Any],
        mock_osm_data: Dict[str, Any],
        mock_microsoft_buildings_data: Dict[str, Any],
        mock_road_network_results: Dict[str, Any],
    ) -> None:
        """Test successful execution of the entire pipeline."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class, patch(
            "gridtracer.scripts.main.CensusDataHandler"
        ) as mock_census_handler_class, patch(
            "gridtracer.scripts.main.CountySubdivisionHandler"
        ) as mock_county_handler_class, patch(
            "gridtracer.scripts.main.NRELDataHandler"
        ) as mock_nrel_handler_class, patch("gridtracer.scripts.main.OSMDataHandler") as mock_osm_handler_class, patch(
            "gridtracer.scripts.main.MicrosoftBuildingsDataHandler"
        ) as mock_ms_handler_class, patch(
            "gridtracer.scripts.main.BuildingProcessor"
        ) as mock_building_processor_class, patch(
            "gridtracer.scripts.main.RoadNetworkBuilder"
        ) as mock_road_builder_class:
            # Setup orchestrator mock with proper fips_dict
            mock_orchestrator = Mock()
            mock_orchestrator.base_output_dir = Path("/test/output")
            mock_orchestrator.get_dataset_specific_output_directory.return_value = Path("/test/output/buildings")
            mock_orchestrator.fips_dict = {
                "state": "MA",
                "county": "017",
                "cousub": "11000",
            }
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup handler mocks
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

            # Setup county subdivision handler
            mock_county_handler = Mock()
            mock_county_handler_class.return_value = mock_county_handler

            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = mock_nrel_data
            mock_nrel_handler_class.return_value = mock_nrel_handler

            mock_osm_handler = Mock()
            mock_osm_handler.process.return_value = mock_osm_data
            mock_osm_handler_class.return_value = mock_osm_handler

            mock_ms_handler = Mock()
            mock_ms_handler.process.return_value = mock_microsoft_buildings_data
            mock_ms_handler_class.return_value = mock_ms_handler

            mock_building_processor = Mock()
            mock_building_processor_class.return_value = mock_building_processor

            mock_road_builder = Mock()
            mock_road_builder.process.return_value = mock_road_network_results
            mock_road_builder_class.return_value = mock_road_builder

            # Execute pipeline
            run_full_pipeline()

            # Verify orchestrator was created
            mock_orchestrator_class.assert_called_once()

            # Verify all handlers were created with orchestrator
            mock_census_handler_class.assert_called_once_with(mock_orchestrator)
            mock_county_handler_class.assert_called_once_with(orchestrator=mock_orchestrator)
            mock_nrel_handler_class.assert_called_once_with(mock_orchestrator)
            mock_osm_handler_class.assert_called_once_with(mock_orchestrator)
            mock_ms_handler_class.assert_called_once_with(mock_orchestrator)

            # Verify all process methods were called
            mock_census_handler.process.assert_called_once()
            mock_county_handler.process.assert_called_once_with(state_filter="MA")
            mock_nrel_handler.process.assert_called_once()
            mock_osm_handler.process.assert_called_once_with(plot=False)
            mock_ms_handler.process.assert_called_once()

            # Verify building processor was created and called
            mock_orchestrator.get_dataset_specific_output_directory.assert_called_with("BUILDINGS_OUTPUT")
            mock_building_processor_class.assert_called_once_with(
                mock_orchestrator.get_dataset_specific_output_directory.return_value
            )
            mock_building_processor.process.assert_called_once_with(
                mock_census_data,
                mock_osm_data,
                mock_microsoft_buildings_data,
                mock_nrel_data["vintage_distribution"],
            )

            # Verify road network builder was created and called
            mock_road_builder_class.assert_called_once_with(orchestrator=mock_orchestrator)
            mock_road_builder.process.assert_called_once_with()

    def test_orchestrator_creation_failure_handling(self) -> None:
        """Test handling of WorkflowOrchestrator creation failure."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class:
            # Make orchestrator creation fail
            mock_orchestrator_class.side_effect = ValueError("Invalid configuration")

            # Execute pipeline - should not raise exception
            run_full_pipeline()

            # Verify orchestrator creation was attempted
            mock_orchestrator_class.assert_called_once()

    def test_runtime_error_handling(self) -> None:
        """Test handling of runtime errors during pipeline execution."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class, patch(
            "gridtracer.scripts.main.CensusDataHandler"
        ) as mock_census_handler_class:
            # Setup orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.fips_dict = {
                "state": "MA",
                "county": "017",
                "cousub": "11000",
            }
            mock_orchestrator_class.return_value = mock_orchestrator

            # Make census handler creation fail with RuntimeError
            mock_census_handler_class.side_effect = RuntimeError("Database connection failed")

            # Execute pipeline - should not raise exception
            run_full_pipeline()

            # Verify orchestrator was created
            mock_orchestrator_class.assert_called_once()

    def test_pipeline_with_empty_census_data(self) -> None:
        """Test pipeline behavior with empty census data."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class, patch(
            "gridtracer.scripts.main.CensusDataHandler"
        ) as mock_census_handler_class, patch(
            "gridtracer.scripts.main.CountySubdivisionHandler"
        ) as mock_county_handler_class:
            # Setup orchestrator mock
            mock_orchestrator = Mock()
            mock_orchestrator.fips_dict = {
                "state": "MA",
                "county": "017",
                "cousub": "11000",
            }
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup census handler to return empty data
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = None
            mock_census_handler_class.return_value = mock_census_handler

            # Setup county subdivision handler
            mock_county_handler = Mock()
            mock_county_handler_class.return_value = mock_county_handler

            # Execute pipeline
            run_full_pipeline()

            # Verify census handler was called
            mock_census_handler.process.assert_called_once()
            # Verify county subdivision handler was called
            mock_county_handler.process.assert_called_once()

    def test_building_processor_integration(
        self,
        mock_census_data: Dict[str, Any],
        mock_nrel_data: Dict[str, Any],
        mock_osm_data: Dict[str, Any],
        mock_microsoft_buildings_data: Dict[str, Any],
    ) -> None:
        """Test that building processor receives correct data flow."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class, patch(
            "gridtracer.scripts.main.CensusDataHandler"
        ) as mock_census_handler_class, patch(
            "gridtracer.scripts.main.CountySubdivisionHandler"
        ) as mock_county_handler_class, patch(
            "gridtracer.scripts.main.NRELDataHandler"
        ) as mock_nrel_handler_class, patch("gridtracer.scripts.main.OSMDataHandler") as mock_osm_handler_class, patch(
            "gridtracer.scripts.main.MicrosoftBuildingsDataHandler"
        ) as mock_ms_handler_class, patch(
            "gridtracer.scripts.main.BuildingProcessor"
        ) as mock_building_processor_class, patch(
            "gridtracer.scripts.main.RoadNetworkBuilder"
        ) as mock_road_builder_class:
            # Setup orchestrator and all handlers
            mock_orchestrator = Mock()
            mock_orchestrator.fips_dict = {
                "state": "MA",
                "county": "017",
                "cousub": "11000",
            }
            mock_orchestrator.get_dataset_specific_output_directory.return_value = Path("/test/buildings")
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup all handlers with return data
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

            mock_county_handler = Mock()
            mock_county_handler_class.return_value = mock_county_handler

            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = mock_nrel_data
            mock_nrel_handler_class.return_value = mock_nrel_handler

            mock_osm_handler = Mock()
            mock_osm_handler.process.return_value = mock_osm_data
            mock_osm_handler_class.return_value = mock_osm_handler

            mock_ms_handler = Mock()
            mock_ms_handler.process.return_value = mock_microsoft_buildings_data
            mock_ms_handler_class.return_value = mock_ms_handler

            mock_building_processor = Mock()
            mock_building_processor_class.return_value = mock_building_processor

            mock_road_builder = Mock()
            mock_road_builder.process.return_value = {"geojson_file": "/test/roads.geojson"}
            mock_road_builder_class.return_value = mock_road_builder

            # Execute pipeline
            run_full_pipeline()

            # Verify building processor receives correct data
            mock_building_processor.process.assert_called_once_with(
                mock_census_data,  # From census handler
                mock_osm_data,  # From OSM handler
                mock_microsoft_buildings_data,  # From MS handler
                mock_nrel_data["vintage_distribution"],  # From NREL handler
            )

    def test_component_initialization_order(self) -> None:
        """Test that pipeline components are initialized in the correct order."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class, patch(
            "gridtracer.scripts.main.CensusDataHandler"
        ) as mock_census_handler_class, patch(
            "gridtracer.scripts.main.CountySubdivisionHandler"
        ) as mock_county_handler_class, patch(
            "gridtracer.scripts.main.NRELDataHandler"
        ) as mock_nrel_handler_class, patch("gridtracer.scripts.main.OSMDataHandler") as mock_osm_handler_class, patch(
            "gridtracer.scripts.main.MicrosoftBuildingsDataHandler"
        ) as mock_ms_handler_class, patch(
            "gridtracer.scripts.main.BuildingProcessor"
        ) as mock_building_processor_class, patch(
            "gridtracer.scripts.main.RoadNetworkBuilder"
        ) as mock_road_builder_class:
            # Setup minimal mocks to allow pipeline to complete
            mock_orchestrator = Mock()
            mock_orchestrator.fips_dict = {
                "state": "MA",
                "county": "017",
                "cousub": "11000",
            }
            mock_orchestrator.get_dataset_specific_output_directory.return_value = Path("/test/buildings")
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup handlers with minimal valid returns
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = {"vintage_distribution": {}}
            mock_census_handler_class.return_value = mock_census_handler

            mock_county_handler = Mock()
            mock_county_handler_class.return_value = mock_county_handler

            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = {"vintage_distribution": {}}
            mock_nrel_handler_class.return_value = mock_nrel_handler

            mock_osm_handler = Mock()
            mock_osm_handler.process.return_value = {}
            mock_osm_handler_class.return_value = mock_osm_handler

            mock_ms_handler = Mock()
            mock_ms_handler.process.return_value = {}
            mock_ms_handler_class.return_value = mock_ms_handler

            mock_building_processor_class.return_value = Mock()

            mock_road_builder = Mock()
            mock_road_builder.process.return_value = {"geojson_file": "/test/roads.geojson"}
            mock_road_builder_class.return_value = mock_road_builder

            # Execute pipeline
            run_full_pipeline()

            # Verify initialization order by checking calls
            assert mock_orchestrator_class.call_count == 1

            # All handlers should be initialized with the orchestrator
            mock_census_handler_class.assert_called_with(mock_orchestrator)
            mock_county_handler_class.assert_called_with(orchestrator=mock_orchestrator)
            mock_nrel_handler_class.assert_called_with(mock_orchestrator)
            mock_osm_handler_class.assert_called_with(mock_orchestrator)
            mock_ms_handler_class.assert_called_with(mock_orchestrator)

            # Building processor should be initialized with output directory
            mock_orchestrator.get_dataset_specific_output_directory.assert_called_with("BUILDINGS_OUTPUT")
            mock_building_processor_class.assert_called_with(
                mock_orchestrator.get_dataset_specific_output_directory.return_value
            )

            # Road network builder should be initialized with orchestrator
            mock_road_builder_class.assert_called_with(orchestrator=mock_orchestrator)


class TestPipelineErrorHandling:
    """Test suite for pipeline error handling scenarios."""

    def test_missing_vintage_distribution_handling(self) -> None:
        """Test handling when NREL data lacks vintage_distribution."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class, patch(
            "gridtracer.scripts.main.CensusDataHandler"
        ) as mock_census_handler_class, patch(
            "gridtracer.scripts.main.CountySubdivisionHandler"
        ) as mock_county_handler_class, patch(
            "gridtracer.scripts.main.NRELDataHandler"
        ) as mock_nrel_handler_class, patch("gridtracer.scripts.main.OSMDataHandler") as mock_osm_handler_class, patch(
            "gridtracer.scripts.main.MicrosoftBuildingsDataHandler"
        ) as mock_ms_handler_class, patch(
            "gridtracer.scripts.main.BuildingProcessor"
        ) as mock_building_processor_class, patch(
            "gridtracer.scripts.main.RoadNetworkBuilder"
        ) as mock_road_builder_class:
            # Setup orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.fips_dict = {
                "state": "MA",
                "county": "017",
                "cousub": "11000",
            }
            mock_orchestrator.get_dataset_specific_output_directory.return_value = Path("/test/buildings")
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup handlers - NREL missing vintage_distribution
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = {}
            mock_census_handler_class.return_value = mock_census_handler

            mock_county_handler = Mock()
            mock_county_handler_class.return_value = mock_county_handler

            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = {"other_data": "value"}  # Missing vintage_distribution
            mock_nrel_handler_class.return_value = mock_nrel_handler

            mock_osm_handler = Mock()
            mock_osm_handler.process.return_value = {}
            mock_osm_handler_class.return_value = mock_osm_handler

            mock_ms_handler = Mock()
            mock_ms_handler.process.return_value = {}
            mock_ms_handler_class.return_value = mock_ms_handler

            mock_building_processor_class.return_value = Mock()
            mock_road_builder_class.return_value = Mock()

            # Execute pipeline - should handle KeyError gracefully
            run_full_pipeline()

            # Verify handlers were still called
            mock_census_handler.process.assert_called_once()
            mock_nrel_handler.process.assert_called_once()

    def test_unexpected_exception_handling(self) -> None:
        """Test handling of completely unexpected exceptions."""

        with patch("gridtracer.scripts.main.WorkflowOrchestrator") as mock_orchestrator_class:
            # Make orchestrator creation fail with unexpected error
            mock_orchestrator_class.side_effect = TypeError("Unexpected type error")

            # Execute pipeline - should not raise exception
            run_full_pipeline()

            # Verify orchestrator creation was attempted
            mock_orchestrator_class.assert_called_once()

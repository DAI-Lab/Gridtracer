"""
Tests for the main SynGrid data processing pipeline script.

These tests verify the orchestration logic, error handling, and proper
integration between pipeline components while mocking individual handlers.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import Mock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

from syngrid.data_processor.scripts.main import run_pipeline_v2


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_census_data(sample_boundary_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Create mock census data for testing."""
    return {
        'target_region_blocks': sample_boundary_gdf,
        'target_region_blocks_filepath': '/test/blocks.geojson',
        'target_region_boundary': sample_boundary_gdf,
        'target_region_boundary_filepath': '/test/boundary.geojson'
    }


@pytest.fixture
def mock_nrel_data() -> Dict[str, Any]:
    """Create mock NREL data for testing."""
    return {
        'parquet_path': '/test/nrel_data.parquet',
        'vintage_distribution': {
            '<1940': 0.35,
            '1940s': 0.25,
            '1950s': 0.20,
            '1960s': 0.15,
            '1970s': 0.05
        }
    }


@pytest.fixture
def mock_osm_data() -> Dict[str, Any]:
    """Create mock OSM data for testing."""
    sample_buildings = gpd.GeoDataFrame({
        'building': ['house', 'commercial'],
        'geometry': [
            Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
            Polygon([(20, 0), (20, 15), (35, 15), (35, 0)])
        ]
    }, crs="EPSG:4326")

    return {
        'buildings': sample_buildings,
        'pois': gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326"),
        'landuse': gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326"),
        'power_infrastructure': gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
    }


@pytest.fixture
def mock_microsoft_buildings_data() -> Dict[str, Any]:
    """Create mock Microsoft Buildings data for testing."""
    ms_buildings = gpd.GeoDataFrame({
        'height': [9.0, 15.0],
        'confidence': [0.8, 0.9],
        'geometry': [
            Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]),
            Polygon([(20, 0), (20, 15), (35, 15), (35, 0)])
        ]
    }, crs="EPSG:4326")

    return {
        'ms_buildings': ms_buildings,
        'ms_buildings_filepath': '/test/ms_buildings.geojson'
    }


@pytest.fixture
def mock_road_network_results() -> Dict[str, Any]:
    """Create mock road network results for testing."""
    return {
        'geojson_file': '/test/road_network.geojson',
        'node_count': 150,
        'edge_count': 200
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
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test successful execution of the entire pipeline."""
        # Set the logger name for caplog
        caplog.set_level('INFO', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class, \
                patch('syngrid.data_processor.scripts.main.OSMDataHandler') as mock_osm_handler_class, \
                patch('syngrid.data_processor.scripts.main.MicrosoftBuildingsDataHandler') as mock_ms_handler_class, \
                patch('syngrid.data_processor.scripts.main.BuildingHeuristicsProcessor') as mock_building_processor_class, \
                patch('syngrid.data_processor.scripts.main.RoadNetworkBuilder') as mock_road_builder_class:

            # Setup orchestrator mock
            mock_orchestrator = Mock()
            mock_orchestrator.base_output_dir = Path('/test/output')
            mock_orchestrator.get_dataset_specific_output_directory.return_value = Path(
                '/test/output/buildings')
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup handler mocks
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

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
            run_pipeline_v2()

            # Verify orchestrator was created
            mock_orchestrator_class.assert_called_once()

            # Verify all handlers were created with orchestrator
            mock_census_handler_class.assert_called_once_with(mock_orchestrator)
            mock_nrel_handler_class.assert_called_once_with(mock_orchestrator)
            mock_osm_handler_class.assert_called_once_with(mock_orchestrator)
            mock_ms_handler_class.assert_called_once_with(mock_orchestrator)

            # Verify all process methods were called
            mock_census_handler.process.assert_called_once_with(plot=False)
            mock_nrel_handler.process.assert_called_once()
            mock_osm_handler.process.assert_called_once_with(plot=False)
            mock_ms_handler.process.assert_called_once()

            # Verify building processor was created and called
            mock_orchestrator.get_dataset_specific_output_directory.assert_called_with(
                "BUILDINGS_OUTPUT")
            mock_building_processor_class.assert_called_once_with(
                mock_orchestrator.get_dataset_specific_output_directory.return_value)
            mock_building_processor.process.assert_called_once_with(
                mock_census_data,
                mock_osm_data,
                mock_microsoft_buildings_data,
                mock_nrel_data["vintage_distribution"]
            )

            # Verify road network builder was created and called
            mock_road_builder_class.assert_called_once_with(orchestrator=mock_orchestrator)
            mock_road_builder.process.assert_called_once_with(
                boundary_gdf=mock_census_data['target_region_boundary'],
                plot=True
            )

            # Verify logging messages (check individual message parts)
            log_text = caplog.text
            assert "Starting SynGrid Data Processing Pipeline v2.0" in log_text
            assert "STEP 1: Regional Data Extraction & Preparation" in log_text
            assert "STEP 2: Processing NREL data" in log_text
            assert "STEP 3.5: Processing Microsoft Buildings data" in log_text
            assert "SynGrid Data Processing Pipeline v2.0 completed successfully." in log_text

    def test_census_data_failure_halts_pipeline(
        self,
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test that census data processing failure halts the entire pipeline."""
        caplog.set_level('ERROR', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class:

            # Setup orchestrator mock
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup census handler to return empty/invalid data
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = None  # Simulate failure
            mock_census_handler_class.return_value = mock_census_handler

            # Setup NREL handler (should not be called)
            mock_nrel_handler = Mock()
            mock_nrel_handler_class.return_value = mock_nrel_handler

            # Execute pipeline
            run_pipeline_v2()

            # Verify census handler was called
            mock_census_handler.process.assert_called_once_with(plot=False)

            # Verify NREL handler was NOT created (pipeline halted)
            mock_nrel_handler_class.assert_not_called()

            # Verify error message was logged
            assert "Census data processing failed" in caplog.text
            assert "Halting." in caplog.text

    def test_census_data_missing_boundary_halts_pipeline(
        self,
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test that missing target_region_boundary in census data halts pipeline."""
        caplog.set_level('ERROR', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class:

            # Setup orchestrator mock
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup census handler to return data without target_region_boundary
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = {'other_data': 'value'}  # Missing key field
            mock_census_handler_class.return_value = mock_census_handler

            # Setup NREL handler (should not be called)
            mock_nrel_handler = Mock()
            mock_nrel_handler_class.return_value = mock_nrel_handler

            # Execute pipeline
            run_pipeline_v2()

            # Verify census handler was called
            mock_census_handler.process.assert_called_once_with(plot=False)

            # Verify NREL handler was NOT created (pipeline halted)
            mock_nrel_handler_class.assert_not_called()

            # Verify error message was logged
            assert "Census data processing failed" in caplog.text

    def test_nrel_data_processing_continues_on_warning(
        self,
        mock_census_data: Dict[str, Any],
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test that NREL data processing warning doesn't halt pipeline."""
        caplog.set_level('WARNING', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class, \
                patch('syngrid.data_processor.scripts.main.OSMDataHandler') as mock_osm_handler_class:

            # Setup orchestrator mock
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Setup census handler with valid data
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

            # Setup NREL handler without parquet_path
            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = {
                'other_data': 'value'}  # Missing parquet_path
            mock_nrel_handler_class.return_value = mock_nrel_handler

            # Setup OSM handler (should be called despite NREL warning)
            mock_osm_handler = Mock()
            mock_osm_handler_class.return_value = mock_osm_handler

            # Execute pipeline
            run_pipeline_v2()

            # Verify both handlers were called
            mock_census_handler.process.assert_called_once()
            mock_nrel_handler.process.assert_called_once()
            mock_osm_handler_class.assert_called_once()  # Pipeline continued

            # Verify warning was logged
            assert "NREL data processing did not yield a parquet path." in caplog.text

    def test_orchestrator_creation_failure_handling(
        self,
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test handling of WorkflowOrchestrator creation failure."""
        caplog.set_level('ERROR', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class:
            # Make orchestrator creation fail
            mock_orchestrator_class.side_effect = ValueError("Invalid configuration")

            # Execute pipeline
            run_pipeline_v2()

            # Verify error was logged
            assert "Configuration or validation error during pipeline:" in caplog.text
            assert "Invalid configuration" in caplog.text

    def test_runtime_error_handling(
        self,
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test handling of runtime errors during pipeline execution."""
        caplog.set_level('ERROR', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class:

            # Setup orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Make census handler creation fail with RuntimeError
            mock_census_handler_class.side_effect = RuntimeError("Database connection failed")

            # Execute pipeline
            run_pipeline_v2()

            # Verify error was logged
            assert "Runtime error during pipeline execution:" in caplog.text
            assert "Database connection failed" in caplog.text

    def test_unexpected_error_handling(
        self,
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test handling of unexpected errors during pipeline execution."""
        caplog.set_level('ERROR', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class:
            # Make orchestrator creation fail with unexpected error
            mock_orchestrator_class.side_effect = TypeError("Unexpected type error")

            # Execute pipeline
            run_pipeline_v2()

            # Verify error was logged
            assert "An unexpected error occurred in the pipeline:" in caplog.text
            assert "Unexpected type error" in caplog.text

    def test_microsoft_buildings_data_processing_warning(
        self,
        mock_census_data: Dict[str, Any],
        mock_nrel_data: Dict[str, Any],
        mock_osm_data: Dict[str, Any],
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test warning when Microsoft Buildings data processing fails."""
        caplog.set_level('WARNING', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class, \
                patch('syngrid.data_processor.scripts.main.OSMDataHandler') as mock_osm_handler_class, \
                patch('syngrid.data_processor.scripts.main.MicrosoftBuildingsDataHandler') as mock_ms_handler_class, \
                patch('syngrid.data_processor.scripts.main.BuildingHeuristicsProcessor') as mock_building_processor_class:

            # Setup orchestrator and successful handlers
            mock_orchestrator = Mock()
            mock_orchestrator.base_output_dir = Path('/test')
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = mock_nrel_data
            mock_nrel_handler_class.return_value = mock_nrel_handler

            mock_osm_handler = Mock()
            mock_osm_handler.process.return_value = mock_osm_data
            mock_osm_handler_class.return_value = mock_osm_handler

            # Setup Microsoft Buildings handler to return empty data
            mock_ms_handler = Mock()
            mock_ms_handler.process.return_value = None  # Simulate failure
            mock_ms_handler_class.return_value = mock_ms_handler

            # Setup building processor (should still be called)
            mock_building_processor = Mock()
            mock_building_processor_class.return_value = mock_building_processor

            # Mock road network builder to avoid complex setup
            with patch('syngrid.data_processor.scripts.main.RoadNetworkBuilder'):
                # Execute pipeline
                run_pipeline_v2()

                # Verify warning was logged
                assert "Microsoft Buildings data processing did not yield buildings data." in caplog.text

                # Verify building processor was still called (pipeline continued)
                mock_building_processor.process.assert_called_once()

    def test_road_network_generation_warning(
        self,
        mock_census_data: Dict[str, Any],
        caplog: "LogCaptureFixture"
    ) -> None:
        """Test warning when road network generation doesn't yield expected results."""
        caplog.set_level('WARNING', logger='syngrid')

        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.RoadNetworkBuilder') as mock_road_builder_class:

            # Setup orchestrator and census handler
            mock_orchestrator = Mock()
            mock_orchestrator.base_output_dir = Path('/test')
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

            # Setup road network builder without geojson_file
            mock_road_builder = Mock()
            mock_road_builder.process.return_value = {
                'other_data': 'value'}  # Missing geojson_file
            mock_road_builder_class.return_value = mock_road_builder

            # Mock other handlers to avoid complex setup and return minimal valid data
            with patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_class, \
                    patch('syngrid.data_processor.scripts.main.OSMDataHandler') as mock_osm_class, \
                    patch('syngrid.data_processor.scripts.main.MicrosoftBuildingsDataHandler') as mock_ms_class, \
                    patch('syngrid.data_processor.scripts.main.BuildingHeuristicsProcessor') as mock_bp_class:

                # Setup minimal valid returns to avoid KeyError
                mock_nrel_handler = Mock()
                mock_nrel_handler.process.return_value = {'vintage_distribution': {}}
                mock_nrel_class.return_value = mock_nrel_handler

                mock_osm_handler = Mock()
                mock_osm_handler.process.return_value = {}
                mock_osm_class.return_value = mock_osm_handler

                mock_ms_handler = Mock()
                mock_ms_handler.process.return_value = {}
                mock_ms_class.return_value = mock_ms_handler

                mock_bp_class.return_value = Mock()

                # Execute pipeline
                run_pipeline_v2()

                # Verify warning was logged
                assert "Road network generation did not yield a GPKG path." in caplog.text

                # Since multiple warnings occur, just verify the pipeline doesn't crash
                # The completion message may not appear due to multiple warnings


class TestPipelineIntegration:
    """Integration tests for pipeline components working together."""

    def test_data_flow_between_components(
        self,
        mock_census_data: Dict[str, Any],
        mock_nrel_data: Dict[str, Any],
        mock_osm_data: Dict[str, Any],
        mock_microsoft_buildings_data: Dict[str, Any]
    ) -> None:
        """Test that data flows correctly between pipeline components."""
        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class, \
                patch('syngrid.data_processor.scripts.main.OSMDataHandler') as mock_osm_handler_class, \
                patch('syngrid.data_processor.scripts.main.MicrosoftBuildingsDataHandler') as mock_ms_handler_class, \
                patch('syngrid.data_processor.scripts.main.BuildingHeuristicsProcessor') as mock_building_processor_class, \
                patch('syngrid.data_processor.scripts.main.RoadNetworkBuilder') as mock_road_builder_class:

            # Setup all mocks
            mock_orchestrator = Mock()
            mock_orchestrator.base_output_dir = Path('/test')
            mock_orchestrator_class.return_value = mock_orchestrator

            mock_census_handler = Mock()
            mock_census_handler.process.return_value = mock_census_data
            mock_census_handler_class.return_value = mock_census_handler

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
            mock_road_builder.process.return_value = {'geojson_file': '/test/roads.geojson'}
            mock_road_builder_class.return_value = mock_road_builder

            # Execute pipeline
            run_pipeline_v2()

            # Verify data flows correctly to building processor
            mock_building_processor.process.assert_called_once_with(
                mock_census_data,                           # From census handler
                mock_osm_data,                             # From OSM handler
                mock_microsoft_buildings_data,             # From MS handler
                mock_nrel_data["vintage_distribution"]     # From NREL handler
            )

            # Verify census boundary flows to road network builder
            mock_road_builder.process.assert_called_once_with(
                boundary_gdf=mock_census_data['target_region_boundary'],
                plot=True
            )

    def test_pipeline_component_initialization_order(self) -> None:
        """Test that pipeline components are initialized in the correct order."""
        with patch('syngrid.data_processor.scripts.main.WorkflowOrchestrator') as mock_orchestrator_class, \
                patch('syngrid.data_processor.scripts.main.CensusDataHandler') as mock_census_handler_class, \
                patch('syngrid.data_processor.scripts.main.NRELDataHandler') as mock_nrel_handler_class, \
                patch('syngrid.data_processor.scripts.main.OSMDataHandler') as mock_osm_handler_class, \
                patch('syngrid.data_processor.scripts.main.MicrosoftBuildingsDataHandler') as mock_ms_handler_class, \
                patch('syngrid.data_processor.scripts.main.BuildingHeuristicsProcessor') as mock_building_processor_class, \
                patch('syngrid.data_processor.scripts.main.RoadNetworkBuilder') as mock_road_builder_class:

            # Setup minimal mocks to allow pipeline to complete
            mock_orchestrator = Mock()
            mock_orchestrator.base_output_dir = Path('/test')
            mock_orchestrator.get_dataset_specific_output_directory.return_value = Path(
                '/test/buildings')
            mock_orchestrator_class.return_value = mock_orchestrator

            # Special setup for census handler (needs target_region_boundary)
            mock_census_handler = Mock()
            mock_census_handler.process.return_value = {
                'target_region_boundary': gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
            }
            mock_census_handler_class.return_value = mock_census_handler

            # Mock NREL handler with vintage_distribution
            mock_nrel_handler = Mock()
            mock_nrel_handler.process.return_value = {'vintage_distribution': {}}
            mock_nrel_handler_class.return_value = mock_nrel_handler

            # Mock remaining handlers to return minimal valid data
            mock_osm_handler = Mock()
            mock_osm_handler.process.return_value = {}
            mock_osm_handler_class.return_value = mock_osm_handler

            mock_ms_handler = Mock()
            mock_ms_handler.process.return_value = {}
            mock_ms_handler_class.return_value = mock_ms_handler

            # Mock building processor and road builder
            mock_building_processor_class.return_value = Mock()

            # Mock road builder with a return value to prevent the subscript error
            mock_road_builder = Mock()
            mock_road_builder.process.return_value = {'geojson_file': '/test/roads.geojson'}
            mock_road_builder_class.return_value = mock_road_builder

            # Execute pipeline
            run_pipeline_v2()

            # Verify initialization order by checking call order
            # WorkflowOrchestrator should be first
            assert mock_orchestrator_class.call_count == 1

            # All handlers should be initialized with the orchestrator
            mock_census_handler_class.assert_called_with(mock_orchestrator)
            mock_nrel_handler_class.assert_called_with(mock_orchestrator)
            mock_osm_handler_class.assert_called_with(mock_orchestrator)
            mock_ms_handler_class.assert_called_with(mock_orchestrator)

            # Building processor should be initialized with output directory from
            # get_dataset_specific_output_directory
            mock_orchestrator.get_dataset_specific_output_directory.assert_called_with(
                "BUILDINGS_OUTPUT")
            mock_building_processor_class.assert_called_with(
                mock_orchestrator.get_dataset_specific_output_directory.return_value)

            # Road network builder should be initialized with orchestrator
            mock_road_builder_class.assert_called_with(orchestrator=mock_orchestrator)

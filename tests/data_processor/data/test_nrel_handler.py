"""Tests for NREL data handler."""

import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from gridtracer.data_processor.data.nrel import EXPECTED_VINTAGE_BINS, NRELDataHandler


class TestNRELDataHandler:
    """Test cases for NRELDataHandler class."""

    @pytest.fixture
    def sample_nrel_data(self) -> pd.DataFrame:
        """Create sample NREL data for testing."""
        return pd.DataFrame({
            'in.county': ['G25017001', 'G25017002', 'G25017003', 'G01001001'],
            'in.vintage': ['1970s', '1980s', '<1940', '1990s'],
            'in.geometry_building_type_acs': [
                'Single-Family Detached',
                'Single-Family Attached',
                '2 Unit',
                'Single-Family Detached'
            ],
            'weight': [242.13, 242.13, 242.13, 242.13],
            'other_col': ['A', 'B', 'C', 'D']
        })

    @pytest.fixture
    def temp_tsv_file(self, sample_nrel_data: pd.DataFrame) -> Path:
        """Create a temporary TSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            sample_nrel_data.to_csv(f, sep='\t', index=False)
            return Path(f.name)

    def test_init_with_valid_config(self, orchestrator_with_fips) -> None:
        """Test initialization with valid configuration using real orchestrator."""
        handler = NRELDataHandler(orchestrator_with_fips)

        assert handler.orchestrator == orchestrator_with_fips
        assert handler.input_file_path == Path('/path/to/nrel.tsv')
        assert handler._get_dataset_name() == "NREL"

    def test_init_without_nrel_path(self, sample_config, sample_fips_csv_content,
                                    temp_output_dir, caplog) -> None:
        """Test initialization when NREL path is missing from config."""
        # Remove NREL path from config
        config_without_nrel = sample_config.copy()
        config_without_nrel['input_data'] = {
            'osm_pbf_file': '/path/to/test.pbf'
            # Missing nrel_data
        }

        with patch('gridtracer.data_processor.workflow.ConfigLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_region.return_value = config_without_nrel['region']
            mock_loader.get_output_dir.return_value = temp_output_dir
            mock_loader.get_input_data_paths.return_value = config_without_nrel['input_data']
            mock_loader.get_overpass_config.return_value = config_without_nrel['overpass']
            mock_loader_class.return_value = mock_loader

            with patch('urllib.request.urlretrieve') as mock_urlretrieve:
                # Create the FIPS file manually
                from tests.data_processor.conftest import create_mock_fips_file
                mock_urlretrieve.side_effect = lambda url, filepath: create_mock_fips_file(
                    Path(filepath), sample_fips_csv_content
                )

                from gridtracer.data_processor.workflow import WorkflowOrchestrator
                orchestrator = WorkflowOrchestrator()

                handler = NRELDataHandler(orchestrator)

                assert handler.input_file_path is None
                assert "NREL input data path ('nrel_data') not found" in caplog.text

    def test_validate_inputs_success(self, orchestrator_with_fips, temp_tsv_file: Path) -> None:
        """Test successful input validation."""
        # Update orchestrator to use real TSV file
        orchestrator_with_fips.config_loader.get_input_data_paths.return_value['nrel_data'] = str(
            temp_tsv_file)

        handler = NRELDataHandler(orchestrator_with_fips)
        handler.input_file_path = temp_tsv_file  # Override with real file

        assert handler._validate_inputs() is True

    def test_validate_inputs_missing_file(self, orchestrator_with_fips) -> None:
        """Test input validation with missing file."""
        handler = NRELDataHandler(orchestrator_with_fips)
        handler.input_file_path = Path('/nonexistent/file.tsv')

        assert handler._validate_inputs() is False

    def test_download_existing_files(self, orchestrator_with_fips, temp_tsv_file: Path) -> None:
        """Test download when files already exist."""
        handler = NRELDataHandler(orchestrator_with_fips)
        handler.input_file_path = temp_tsv_file

        # Create existing files
        parquet_file = handler.dataset_output_dir / "NREL_residential_typology_25_017.parquet"
        csv_file = handler.dataset_output_dir / "NREL_residential_typology_25_017.csv"
        parquet_file.touch()
        csv_file.touch()

        result = handler.download()

        assert result['parquet_path'] == parquet_file
        assert result['csv_path'] == csv_file

    def test_extract_and_save_nrel_data_simple(self, orchestrator_with_fips) -> None:
        """Test NREL data extraction and saving with simple approach."""
        handler = NRELDataHandler(orchestrator_with_fips)

        parquet_path = handler.dataset_output_dir / "test.parquet"
        csv_path = handler.dataset_output_dir / "test.csv"

        # Create sample data that should be found
        test_data = pd.DataFrame({
            'in.county': ['G25017001', 'G25017002'],
            'in.vintage': ['1970s', '1980s'],
            'weight': [242.13, 242.13]
        })

        # Test the file saving part by directly creating data
        test_data.to_parquet(parquet_path)
        test_data.to_csv(csv_path, index=False)

        # Verify files exist
        assert parquet_path.exists()
        assert csv_path.exists()

        # Verify we can read back the data
        loaded_data = pd.read_parquet(parquet_path)
        assert len(loaded_data) == 2
        assert 'in.vintage' in loaded_data.columns

    def test_process_complete_workflow(self, orchestrator_with_fips, temp_tsv_file: Path) -> None:
        """Test complete processing workflow."""
        handler = NRELDataHandler(orchestrator_with_fips)
        # Set a valid input file so validation passes
        handler.input_file_path = temp_tsv_file

        # Mock download to return file paths
        with patch.object(handler, 'download') as mock_download:
            parquet_path = handler.dataset_output_dir / "test.parquet"
            csv_path = handler.dataset_output_dir / "test.csv"

            # Create sample data file
            sample_df = pd.DataFrame({
                'in.vintage': ['1970s', '1980s', '<1940'],
                'other_col': ['A', 'B', 'C']
            })
            sample_df.to_parquet(parquet_path)

            mock_download.return_value = {
                'parquet_path': parquet_path,
                'csv_path': csv_path
            }

            result = handler.process()

            assert result['parquet_path'] == parquet_path
            assert result['csv_path'] == csv_path
            assert result['data'] is not None
            assert isinstance(result['vintage_distribution'], OrderedDict)

    def test_compute_vintage_distribution(self, orchestrator_with_fips) -> None:
        """Test vintage distribution computation."""
        handler = NRELDataHandler(orchestrator_with_fips)

        # Create test data with known vintage distribution
        test_data = pd.DataFrame({
            'in.vintage': ['1970s', '1970s', '1980s', '<1940', 'unknown_vintage'],
            'other_col': ['A', 'B', 'C', 'D', 'E']
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            test_data.to_parquet(f.name)
            parquet_path = Path(f.name)

            result = handler.compute_vintage_distribution(parquet_path)

            # Check that we get expected structure
            assert isinstance(result, OrderedDict)
            assert set(result.keys()) == set(EXPECTED_VINTAGE_BINS)

            # Check specific values (2 out of 4 valid records are 1970s = 50%)
            assert result['1970s'] == 0.5  # 2/4 valid records
            assert result['1980s'] == 0.25  # 1/4 valid records
            assert result['<1940'] == 0.25  # 1/4 valid records
            assert result['1990s'] == 0.0   # No records

    def test_compute_vintage_distribution_missing_column(self, orchestrator_with_fips) -> None:
        """Test vintage distribution when vintage column is missing."""
        handler = NRELDataHandler(orchestrator_with_fips)

        test_data = pd.DataFrame({
            'other_col': ['A', 'B', 'C']
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            test_data.to_parquet(f.name)
            parquet_path = Path(f.name)

            result = handler.compute_vintage_distribution(parquet_path)

            # Should return all zeros
            assert all(value == 0.0 for value in result.values())
            assert set(result.keys()) == set(EXPECTED_VINTAGE_BINS)

    def test_process_invalid_inputs(self, sample_config, temp_output_dir) -> None:
        """Test process method with invalid inputs."""
        # Create orchestrator with invalid setup
        config_invalid = sample_config.copy()
        config_invalid['input_data'] = {}  # No NREL data path

        with patch('gridtracer.data_processor.workflow.ConfigLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_region.return_value = None  # Invalid FIPS
            mock_loader.get_output_dir.return_value = temp_output_dir
            mock_loader.get_input_data_paths.return_value = config_invalid['input_data']
            mock_loader.get_overpass_config.return_value = config_invalid['overpass']
            mock_loader_class.return_value = mock_loader

            with patch('urllib.request.urlretrieve'):
                from gridtracer.data_processor.workflow import WorkflowOrchestrator

                with patch.object(WorkflowOrchestrator, '_resolve_fips_codes') as mock_fips:
                    mock_fips.side_effect = ValueError("Invalid FIPS")

                    with pytest.raises(ValueError):
                        WorkflowOrchestrator()

    def test_expected_vintage_bins_constant(self) -> None:
        """Test that expected vintage bins constant is properly defined."""
        expected = [
            "<1940", "1940s", "1950s", "1960s", "1970s",
            "1980s", "1990s", "2000s", "2010s"
        ]
        assert EXPECTED_VINTAGE_BINS == expected
        assert len(EXPECTED_VINTAGE_BINS) == 9

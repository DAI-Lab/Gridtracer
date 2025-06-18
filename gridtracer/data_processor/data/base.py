"""
Base class for data handlers in the gridtracer data processing pipeline.

This module provides the base DataHandler class which defines common functionality
for downloading, processing, and saving different types of data sources.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import geopandas as gpd

if TYPE_CHECKING:
    from gridtracer.data_processor.workflow import (
        WorkflowOrchestrator,)  # Forward reference for type hinting

# Set up logging
logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """
    Abstract base class for all data source handlers.

    This class defines the common interface and functionality that all data source
    handlers should implement. It relies on a WorkflowOrchestrator instance
    for accessing shared configuration, FIPS data, and pre-defined output paths.
    """

    def __init__(self, orchestrator: 'WorkflowOrchestrator'):
        """
        Initialize the DataHandler.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance,
                providing access to configuration, FIPS, and output paths.
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.dataset_name: str = self._get_dataset_name()
        # Get the pre-created dataset-specific output directory from the orchestrator
        self.dataset_output_dir: Path = self.orchestrator.get_dataset_specific_output_directory(
            self.dataset_name
        )
        self.logger.debug(
            f"DataHandler for '{self.dataset_name}' initialized. Output dir: {self.dataset_output_dir}")

    @abstractmethod
    def _get_dataset_name(self) -> str:  # Added type hint
        """
        Get the name of the dataset. This name must be one of the keys
        defined in `ALL_DATASETS` in the `WorkflowOrchestrator`.

        Returns:
            str: Dataset name (e.g., "Census", "OSM").
        """

    @abstractmethod
    def download(self):
        """
        Download the data for the region.
        Implementations should use `self.orchestrator` for region details and paths.

        Returns:
            dict: Dictionary containing the downloaded data or paths to it.
        """

    @abstractmethod
    def process(self, boundary_gdf: Optional[gpd.GeoDataFrame] = None):  # Added type hints
        """
        Process the data for the region.

        This method should implement the complete data processing workflow for
        the specific data source, including downloading, transforming, and saving.
        It should use `self.orchestrator` to get the region boundary if `boundary_gdf` is None.

        Args:
            boundary_gdf (Optional[GeoDataFrame]): Boundary to use for clipping.
                If None, the method should attempt to get it from `self.orchestrator.get_region_boundary()`.

        Returns:
            dict: Dictionary containing processed data and file paths.
        """

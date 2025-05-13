"""
OpenStreetMap data extraction module for SynGrid.

This module provides functions to extract buildings, roads, POIs, and power infrastructure
from OpenStreetMap data for a specified region.
"""

from .osm_extractor import osm_data_extraction

__all__ = ['osm_data_extraction']

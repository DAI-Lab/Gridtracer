"""
Data handlers for the SynGrid data processing pipeline.

This package provides classes to handle different data sources like Census, NREL,
NLCD, and OpenStreetMap.
"""

from syngrid.data_processor.data.base import DataHandler
from syngrid.data_processor.data.census import CensusDataHandler
from syngrid.data_processor.data.nrel import NRELDataHandler

# Try to import OSMDataHandler, but don't fail if it's not available yet
try:
    from syngrid.data_processor.data.osm.osm_data_handler import OSMDataHandler
    __all__ = [
        'DataHandler',
        'CensusDataHandler',
        'NRELDataHandler',
        'OSMDataHandler',
    ]
except ImportError:
    # OSM module might not be ready yet
    __all__ = [
        'DataHandler',
        'CensusDataHandler',
        'NRELDataHandler',
    ]

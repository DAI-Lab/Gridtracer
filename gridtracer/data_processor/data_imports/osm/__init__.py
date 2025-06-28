"""
OSM data processing module.

This module contains utilities for processing OpenStreetMap (OSM) data
for the gridtracer pipeline, including road network generation.
"""

from gridtracer.data_processor.data.osm.road_network_builder import RoadNetworkBuilder

__all__ = ['RoadNetworkBuilder']

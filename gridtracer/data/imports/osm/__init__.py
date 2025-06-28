"""
OSM data processing module.

This module contains utilities for processing OpenStreetMap (OSM) data
for the gridtracer pipeline, including road network generation.
"""

from gridtracer.data.imports.osm.road_network_builder import RoadNetworkBuilder

__all__ = ['RoadNetworkBuilder']

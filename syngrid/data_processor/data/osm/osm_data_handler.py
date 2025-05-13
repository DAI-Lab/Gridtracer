"""
OpenStreetMap data handler for SynGrid.

This module provides functionality to extract building, POI, and land use data 
from OpenStreetMap PBF files using PyROSM.
"""

import logging
from pathlib import Path

from pyrosm import OSM

from syngrid.data_processor.data.base import DataHandler

# Set up logging
logger = logging.getLogger(__name__)


class OSMDataHandler(DataHandler):
    """
    Handler for OpenStreetMap data.
    
    This class handles the extraction of buildings, POIs, and land use data 
    from OpenStreetMap PBF files using PyROSM.
    """
    
    def __init__(self, fips_dict, osm_pbf_file=None, output_dir=None):
        """
        Initialize the OSM data handler.
        
        Args:
            fips_dict (dict): Dictionary containing region information
            osm_pbf_file (str or Path, optional): Path to the OSM PBF file
            output_dir (str or Path, optional): Base output directory
        """
        super().__init__(fips_dict, output_dir)
        self.osm_pbf_file = Path(osm_pbf_file) if osm_pbf_file else None
        self.osm = None
    
    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.
        
        Returns:
            str: Dataset name
        """
        return "OSM"
    
    def initialize_osm(self):
        """
        Initialize the PyROSM OSM object with the PBF file.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.osm_pbf_file is None:
            logger.error("No OSM PBF file provided")
            return False
        
        try:
            logger.info(f"Initializing PyROSM with PBF file: {self.osm_pbf_file}")
            self.osm = OSM(str(self.osm_pbf_file))
            return True
        except Exception as e:
            logger.error(f"Error initializing PyROSM: {e}")
            return False
    
    def download(self):
        """
        Extract data from OSM PBF file.
        
        Unlike other data handlers, this doesn't download data from the web
        but rather processes a local OSM PBF file.
        
        Returns:
            dict: Dictionary containing extracted data:
                - buildings: GeoDataFrame of buildings
                - pois: GeoDataFrame of POIs
                - landuse: GeoDataFrame of land use polygons
        """
        results = {
            'buildings': None,
            'buildings_filepath': None,
            'pois': None,
            'pois_filepath': None,
            'landuse': None,
            'landuse_filepath': None
        }
        
        if not self.initialize_osm():
            return results
        
        # Extract buildings
        try:
            logger.info("Extracting buildings from OSM PBF file")
            buildings = self.osm.get_buildings()
            
            if buildings is not None and not buildings.empty:
                logger.info(f"Extracted {len(buildings)} buildings")
                buildings_filepath = self.dataset_output_dir / "buildings.geojson"
                buildings.to_file(buildings_filepath, driver="GeoJSON")
                logger.info(f"Saved buildings to {buildings_filepath}")
                
                results['buildings'] = buildings
                results['buildings_filepath'] = buildings_filepath
            else:
                logger.warning("No buildings found in OSM PBF file")
        except Exception as e:
            logger.error(f"Error extracting buildings: {e}")
        
        # Extract POIs
        try:
            logger.info("Extracting POIs from OSM PBF file")
            pois = self.osm.get_pois()
            
            if pois is not None and not pois.empty:
                logger.info(f"Extracted {len(pois)} POIs")
                pois_filepath = self.dataset_output_dir / "pois.geojson"
                pois.to_file(pois_filepath, driver="GeoJSON")
                logger.info(f"Saved POIs to {pois_filepath}")
                
                results['pois'] = pois
                results['pois_filepath'] = pois_filepath
            else:
                logger.warning("No POIs found in OSM PBF file")
        except Exception as e:
            logger.error(f"Error extracting POIs: {e}")
        
        # TODO: Extract landuse in future iterations
        
        return results
    
    def process(self, boundary_gdf=None):
        """
        Process OSM data for the region.
        
        Extract buildings, POIs, and land use from OSM PBF file and clip to region boundary.
        
        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to clip data to.
                If None, extracts all data in the PBF file.
                
        Returns:
            dict: Dictionary containing processed data and file paths.
        """
        logger.info(
            f"Processing OSM data for {self.fips_dict['state']} - {self.fips_dict['county']}"
        )
        
        # Extract all data from PBF file
        osm_data = self.download()
        
        # If we have a boundary, clip the data
        if boundary_gdf is not None and not boundary_gdf.empty:
            logger.info("Clipping OSM data to region boundary")
            
            # Clip buildings if we have them
            if osm_data['buildings'] is not None and not osm_data['buildings'].empty:
                try:
                    # Clip to boundary
                    clipped_buildings = self.clip_to_boundary(
                        osm_data['buildings'], 
                        boundary_gdf
                    )
                    
                    if clipped_buildings is not None and not clipped_buildings.empty:
                        # Save clipped buildings
                        clipped_filepath = self.dataset_output_dir / "buildings_clipped.geojson"
                        clipped_buildings.to_file(clipped_filepath, driver="GeoJSON")
                        logger.info(f"Saved clipped buildings to {clipped_filepath}")
                        
                        # Update results
                        osm_data['buildings'] = clipped_buildings
                        osm_data['buildings_clipped_filepath'] = clipped_filepath
                    else:
                        logger.warning("No buildings remain after clipping to boundary")
                except Exception as e:
                    logger.error(f"Error clipping buildings to boundary: {e}")
            
            # Clip POIs if we have them
            if osm_data['pois'] is not None and not osm_data['pois'].empty:
                try:
                    # Clip to boundary
                    clipped_pois = self.clip_to_boundary(
                        osm_data['pois'], 
                        boundary_gdf
                    )
                    
                    if clipped_pois is not None and not clipped_pois.empty:
                        # Save clipped POIs
                        clipped_filepath = self.dataset_output_dir / "pois_clipped.geojson"
                        clipped_pois.to_file(clipped_filepath, driver="GeoJSON")
                        logger.info(f"Saved clipped POIs to {clipped_filepath}")
                        
                        # Update results
                        osm_data['pois'] = clipped_pois
                        osm_data['pois_clipped_filepath'] = clipped_filepath
                    else:
                        logger.warning("No POIs remain after clipping to boundary")
                except Exception as e:
                    logger.error(f"Error clipping POIs to boundary: {e}")
        
        return osm_data
    
    def extract_by_bbox(self, boundary_gdf):
        """
        Extract OSM data using a bounding box from the boundary.
        
        This is an alternative to the regular process method when we want to 
        extract data directly using a bounding box rather than extracting all 
        and then clipping.
        
        Args:
            boundary_gdf (GeoDataFrame): Boundary to define the bounding box
                
        Returns:
            dict: Dictionary containing processed data and file paths.
        """
        if boundary_gdf is None or boundary_gdf.empty:
            logger.error("No boundary provided for bounding box extraction")
            return self.process()  # Fall back to regular processing
        
        # Initialize OSM
        if not self.initialize_osm():
            return {
                'buildings': None,
                'buildings_filepath': None,
                'pois': None,
                'pois_filepath': None,
                'landuse': None,
                'landuse_filepath': None
            }
        
        try:
            # Get total bounds from boundary
            minx, miny, maxx, maxy = boundary_gdf.total_bounds
            
            # Create results dictionary
            results = {
                'buildings': None,
                'buildings_filepath': None,
                'pois': None,
                'pois_filepath': None,
                'landuse': None,
                'landuse_filepath': None
            }
            
            # Extract buildings with bounding box filter
            bbox_str = f"{minx}, {miny}, {maxx}, {maxy}"
            logger.info(f"Extracting buildings using bounding box: {bbox_str}")
            buildings = self.osm.get_buildings(bounding_box=[minx, miny, maxx, maxy])
            
            if buildings is not None and not buildings.empty:
                logger.info(f"Extracted {len(buildings)} buildings within bounding box")
                buildings_filepath = self.dataset_output_dir / "buildings_bbox.geojson"
                buildings.to_file(buildings_filepath, driver="GeoJSON")
                logger.info(f"Saved buildings to {buildings_filepath}")
                
                # Further clip to exact boundary shape
                clipped_buildings = self.clip_to_boundary(buildings, boundary_gdf)
                
                if clipped_buildings is not None and not clipped_buildings.empty:
                    clipped_filepath = self.dataset_output_dir / "buildings_clipped.geojson"
                    clipped_buildings.to_file(clipped_filepath, driver="GeoJSON")
                    logger.info(f"Saved clipped buildings to {clipped_filepath}")
                    
                    results['buildings'] = clipped_buildings
                    results['buildings_filepath'] = clipped_filepath
                else:
                    results['buildings'] = buildings
                    results['buildings_filepath'] = buildings_filepath
            else:
                logger.warning("No buildings found within bounding box")
            
            # Extract POIs with bounding box filter
            logger.info(f"Extracting POIs using bounding box: {bbox_str}")
            pois = self.osm.get_pois(bounding_box=[minx, miny, maxx, maxy])
            
            if pois is not None and not pois.empty:
                logger.info(f"Extracted {len(pois)} POIs within bounding box")
                pois_filepath = self.dataset_output_dir / "pois_bbox.geojson"
                pois.to_file(pois_filepath, driver="GeoJSON")
                logger.info(f"Saved POIs to {pois_filepath}")
                
                # Further clip to exact boundary shape
                clipped_pois = self.clip_to_boundary(pois, boundary_gdf)
                
                if clipped_pois is not None and not clipped_pois.empty:
                    clipped_filepath = self.dataset_output_dir / "pois_clipped.geojson"
                    clipped_pois.to_file(clipped_filepath, driver="GeoJSON")
                    logger.info(f"Saved clipped POIs to {clipped_filepath}")
                    
                    results['pois'] = clipped_pois
                    results['pois_filepath'] = clipped_filepath
                else:
                    results['pois'] = pois
                    results['pois_filepath'] = pois_filepath
            else:
                logger.warning("No POIs found within bounding box")
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting data with bounding box: {e}")
            return self.process()  # Fall back to regular processing 
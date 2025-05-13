"""
Census data handler for SynGrid.

This module provides functionality to download and process US Census TIGER data
including boundaries (state, county, subdivision) and census blocks.
"""

import logging
from pathlib import Path

import geopandas as gpd

from syngrid.data_processor.data.base import DataHandler

# Set up logging
logger = logging.getLogger(__name__)


class CensusDataHandler(DataHandler):
    """
    Handler for Census TIGER data.

    This class handles downloading and processing US Census TIGER data,
    including boundaries (state, county, subdivision) and census blocks.
    """

    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name
        """
        return "Census"

    def download(self):
        """
        Download the Census data for the region specified in fips_dict.

        Returns:
            dict: Dictionary containing the downloaded data:
                - blocks: GeoDataFrame of census blocks
                - subdivision: GeoDataFrame of county subdivision (if applicable)
                - boundary: GeoDataFrame of the region boundary
        """
        logger.info(
            f"Downloading Census data for {self.fips_dict['state']} - {self.fips_dict['county']}")

        # Results dictionary
        results = {
            'blocks': None,
            'blocks_filepath': None,
            'subdivision': None,
            'subdivision_filepath': None,
            'boundary': None,
            'boundary_filepath': None
        }

        try:
            # Extract FIPS codes
            state_fips = self.fips_dict['state_fips']
            county_fips = self.fips_dict['county_fips']
            subdivision_fips = self.fips_dict.get('subdivision_fips')

            # 1. Download Subdivision boundaries first (if we have a subdivision)
            subdivision_present = subdivision_fips is not None
            subdivision_filepath = self.dataset_output_dir / \
                f"{state_fips}_{county_fips}_subdivisions.geojson"
            subdivision_gdf = None
            target_subdivision = None

            if not subdivision_filepath.exists():
                # Construct URL for county subdivisions
                subdivision_url = (
                    f"https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/"
                    f"tl_2020_{state_fips}_cousub.zip"
                )
                logger.info(f"Downloading county subdivisions from: {subdivision_url}")

                # Download and read the subdivision data
                subdivisions = gpd.read_file(subdivision_url)
                logger.debug(f"Subdivisions columns: {subdivisions.columns}")

                # Filter for specific county
                county_subdivisions = subdivisions[subdivisions['COUNTYFP'] == county_fips]
                logger.info(
                    f"Subdivisions in county {self.fips_dict['county']}: {len(county_subdivisions)}")

                # Save to file
                county_subdivisions.to_file(subdivision_filepath, driver='GeoJSON')
                logger.info(f"Saved subdivisions to: {subdivision_filepath}")
                subdivision_gdf = county_subdivisions
            else:
                logger.info(f"Loading subdivisions from existing file: {subdivision_filepath}")
                subdivision_gdf = gpd.read_file(subdivision_filepath)

            # If we have a specific subdivision, find it
            if subdivision_present and subdivision_fips:
                target_subdivision = subdivision_gdf[
                    subdivision_gdf['COUSUBFP'] == subdivision_fips
                ]

                if len(target_subdivision) == 0:
                    logger.warning(
                        f"Subdivision with FIPS {subdivision_fips} not found in county "
                        f"{self.fips_dict['county']}"
                    )
                else:
                    logger.info(
                        f"Found subdivision: {target_subdivision.iloc[0]['NAME']} "
                        f"(FIPS: {subdivision_fips})"
                    )

                    # Save specific subdivision to a separate file
                    target_subdiv_filepath = (
                        self.dataset_output_dir
                        / f"{state_fips}_{county_fips}_{subdivision_fips}_subdivision.geojson"
                    )

                    target_subdivision.to_file(target_subdiv_filepath, driver='GeoJSON')
                    logger.info(f"Saved target subdivision to: {target_subdiv_filepath}")

                    results['subdivision'] = target_subdivision
                    results['subdivision_filepath'] = target_subdiv_filepath

            # 2. Download Census Blocks
            blocks_filepath = self.dataset_output_dir / \
                f"{state_fips}_{county_fips}_blocks.geojson"

            if not blocks_filepath.exists():
                # Construct URL for census blocks
                blocks_url = (
                    f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
                    f"tl_2020_{state_fips}_tabblock20.zip"
                )
                logger.info(f"Downloading Census blocks from: {blocks_url}")

                # Download and read the blocks data
                blocks = gpd.read_file(blocks_url)
                logger.info(f"Total blocks downloaded: {len(blocks)}")

                # Filter for specific county
                blocks = blocks[blocks['COUNTYFP20'] == county_fips]
                logger.info(f"Blocks in county {self.fips_dict['county']}: {len(blocks)}")

                # Save all county blocks first
                blocks.to_file(blocks_filepath, driver='GeoJSON')
                logger.info(f"Saved all county blocks to: {blocks_filepath}")

                # If we have a target subdivision, filter blocks by spatial intersection
                if target_subdivision is not None and not target_subdivision.empty:
                    logger.info("Filtering blocks by intersection with subdivision")

                    # Ensure both GeoDataFrames have the same CRS
                    if target_subdivision.crs != blocks.crs:
                        target_subdivision = target_subdivision.to_crs(blocks.crs)

                    # Perform spatial join to find blocks that intersect with the subdivision
                    blocks_in_subdiv = gpd.sjoin(
                        blocks,
                        target_subdivision,
                        how='inner',
                        predicate='intersects'
                    )

                    # Format for log message
                    subdiv_name = self.fips_dict.get('subdivision')
                    block_count = len(blocks_in_subdiv)
                    logger.info(f"Found {block_count} blocks in subdivision {subdiv_name}")

                    # Save the filtered blocks
                    subdiv_blocks_filepath = (
                        self.dataset_output_dir
                        / f"{state_fips}_{county_fips}_{subdivision_fips}_blocks.geojson"
                    )
                    blocks_in_subdiv.to_file(subdiv_blocks_filepath, driver='GeoJSON')
                    logger.info(f"Saved subdivision blocks to: {subdiv_blocks_filepath}")

                    # Update results
                    results['blocks'] = blocks_in_subdiv
                    results['blocks_filepath'] = subdiv_blocks_filepath
                else:
                    # Use all county blocks if no subdivision specified
                    results['blocks'] = blocks
                    results['blocks_filepath'] = blocks_filepath
            else:
                logger.info(f"Loading blocks from existing file: {blocks_filepath}")

                # If we have a target subdivision and corresponding blocks file
                if target_subdivision is not None and not target_subdivision.empty:
                    subdiv_blocks_filepath = (
                        self.dataset_output_dir
                        / f"{state_fips}_{county_fips}_{subdivision_fips}_blocks.geojson"
                    )

                    if subdiv_blocks_filepath.exists():
                        logger.info(f"Loading subdivision blocks from: {subdiv_blocks_filepath}")
                        blocks = gpd.read_file(subdiv_blocks_filepath)
                        results['blocks'] = blocks
                        results['blocks_filepath'] = subdiv_blocks_filepath
                    else:
                        # We need to load county blocks and filter them
                        blocks = gpd.read_file(blocks_filepath)

                        # Ensure both GeoDataFrames have the same CRS
                        if target_subdivision.crs != blocks.crs:
                            target_subdivision = target_subdivision.to_crs(blocks.crs)

                        # Perform spatial join
                        blocks_in_subdiv = gpd.sjoin(
                            blocks,
                            target_subdivision,
                            how='inner',
                            predicate='intersects'
                        )

                        # Format for log message
                        subdiv_name = self.fips_dict.get('subdivision')
                        block_count = len(blocks_in_subdiv)
                        logger.info(f"Found {block_count} blocks in subdivision {subdiv_name}")

                        # Save the filtered blocks
                        blocks_in_subdiv.to_file(subdiv_blocks_filepath, driver='GeoJSON')
                        logger.info(f"Saved subdivision blocks to: {subdiv_blocks_filepath}")

                        # Update results
                        results['blocks'] = blocks_in_subdiv
                        results['blocks_filepath'] = subdiv_blocks_filepath
                else:
                    # Just load county blocks if no subdivision
                    blocks = gpd.read_file(blocks_filepath)
                    results['blocks'] = blocks
                    results['blocks_filepath'] = blocks_filepath

            # 3. Create Region Boundary
            if results['blocks'] is not None and not results['blocks'].empty:
                try:
                    # Generate the boundary filepath
                    if results['blocks_filepath']:
                        blocks_path = Path(results['blocks_filepath'])
                        boundary_path = blocks_path.parent / f"{blocks_path.stem}_boundary.geojson"

                        # Create a unified outer boundary by unary_union of all block geometries
                        unified_geometry = results['blocks'].geometry.unary_union

                        # Create a new GeoDataFrame with the boundary
                        blocks_crs = results['blocks'].crs
                        boundary_gdf = gpd.GeoDataFrame(
                            geometry=[unified_geometry], crs=blocks_crs)

                        # Save the boundary to file
                        boundary_gdf.to_file(boundary_path, driver='GeoJSON')
                        logger.info(f"Saved region boundary to: {boundary_path}")

                        # Update results
                        results['boundary'] = boundary_gdf
                        results['boundary_filepath'] = boundary_path
                except Exception as e:
                    logger.error(f"Error creating region boundary: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Error downloading census data: {str(e)}")

            # Try to load from local file if download fails
            if 'blocks_filepath' in locals() and blocks_filepath.exists():
                logger.info(f"Loading blocks from local file: {blocks_filepath}")
                blocks = gpd.read_file(blocks_filepath)
                results['blocks'] = blocks
                results['blocks_filepath'] = blocks_filepath

                # Try to create boundary even in exception case
                try:
                    boundary_path = blocks_filepath.parent / \
                        f"{blocks_filepath.stem}_boundary.geojson"
                    unified_geometry = blocks.geometry.unary_union
                    boundary_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=blocks.crs)
                    boundary_gdf.to_file(boundary_path, driver='GeoJSON')
                    logger.info(f"Saved region boundary to: {boundary_path}")
                    results['boundary'] = boundary_gdf
                    results['boundary_filepath'] = boundary_path
                except Exception as boundary_err:
                    logger.error(f"Error creating region boundary: {str(boundary_err)}")

            # We still return partial results if available
            if any(value is not None for value in results.values()):
                return results

            raise ValueError(f"Failed to download Census data: {str(e)}")

    def process(self, boundary_gdf=None):
        """
        Process the Census data for the region.

        Downloads Census data and returns GeoDataFrames for blocks, subdivisions,
        and boundaries.

        Args:
            boundary_gdf (GeoDataFrame, optional): Not used by this method
                Census data defines its own boundary based on blocks.

        Returns:
            dict: Dictionary containing:
                - blocks: GeoDataFrame of census blocks
                - subdivision: GeoDataFrame of county subdivision (if applicable)
                - boundary: GeoDataFrame of the region boundary
                - blocks_filepath: Path to the blocks GeoJSON file
                - subdivision_filepath: Path to the subdivision GeoJSON file
                - boundary_filepath: Path to the boundary GeoJSON file
        """
        logger.info(
            f"Processing Census data for {self.fips_dict['state']} - {self.fips_dict['county']}")

        # Download the data
        census_data = self.download()

        # Download already produces the final output, so we return it directly
        return census_data

import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd

from gridtracer.config import config
from gridtracer.data.imports.base import DataHandler


class CensusDataHandler(DataHandler):
    """
    Handler for US Census TIGER data.

    This class handles downloading and processing US Census TIGER data,
    including boundaries (state, county, subdivision) and census blocks,
    utilizing the WorkflowOrchestrator for configuration and context.
    It can also optionally visualize the processed census blocks.
    """

    def _get_dataset_name(self) -> str:
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name, matching an entry in orchestrator's ALL_DATASETS.
        """
        return "CENSUS"

    def _download_and_read_census_shp(self, shp_url: str, filename_prefix: str) -> Optional[gpd.GeoDataFrame]:
        """
        Helper to download, save, and read a Census shapefile (zipped).

        Args:
            shp_url (str): URL to the .zip shapefile.
            filename_prefix (str): Prefix for the output GeoJSON filename.

        Returns:
            Optional[gpd.GeoDataFrame]: GeoDataFrame if successful, else None.
        """
        output_geojson_path = self.dataset_output_dir / f"{filename_prefix}.geojson"

        if not output_geojson_path.exists():
            self.logger.debug(f"Downloading and processing from: {shp_url}")
            try:
                # Download file first to temporary location, then read it
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                    urllib.request.urlretrieve(shp_url, tmp_file.name)

                    # Read the downloaded file
                    gdf = gpd.read_file(tmp_file.name)

                    # Clean up temporary file
                    Path(tmp_file.name).unlink()

                gdf.to_file(output_geojson_path, driver="GeoJSON")
                return gdf
            except Exception as e:
                self.logger.error(f"Failed to download/process {shp_url}: {e}", exc_info=True)
                return None
        else:
            self.logger.debug("Loading from existing file")
            try:
                return gpd.read_file(output_geojson_path)
            except Exception as e:
                self.logger.error(f"Failed to load {output_geojson_path}: {e}", exc_info=True)
                return None

    def download_subdivisions(self, fips: Dict[str, str]) -> Optional[gpd.GeoDataFrame]:
        """
        Download and process county subdivision boundaries if needed.

        Args:
            fips: Dictionary containing FIPS codes

        Returns:
            Optional[gpd.GeoDataFrame]: Specific subdivision GDF if found, None otherwise
        """
        state_fips = fips["state_fips"]
        county_fips = fips["county_fips"]
        target_subdiv_fips = fips["subdivision_fips"]
        is_subdivision_run = self.orchestrator.is_subdivision_processing()

        if not (is_subdivision_run and target_subdiv_fips):
            return None

        census_urls = config.get_census_urls()
        base_url = census_urls.get("BASE_URL", "https://www2.census.gov/geo/tiger/TIGER2020")
        year = census_urls.get("YEAR", "2020")
        subdivision_url = f"{base_url}/COUSUB/tl_{year}_{state_fips}_cousub.zip"
        all_county_subdivisions_gdf = self._download_and_read_census_shp(
            subdivision_url,
            filename_prefix=f"{state_fips}_{county_fips}_all_subdivisions",
        )

        if all_county_subdivisions_gdf is None or all_county_subdivisions_gdf.empty:
            self.logger.warning("Could not load county subdivisions data (COUSUB).")
            return None

        county_filtered_subdivisions = all_county_subdivisions_gdf[
            all_county_subdivisions_gdf["COUNTYFP"] == county_fips
        ]

        if county_filtered_subdivisions.empty:
            self.logger.warning(f"No subdivisions for county FIPS {county_fips} in COUSUB file.")
            return None

        specific_subdivision_gdf = county_filtered_subdivisions[
            county_filtered_subdivisions["COUSUBFP"] == target_subdiv_fips
        ].copy()

        if not specific_subdivision_gdf.empty:
            self.logger.debug(f"Found target subdivision: {target_subdiv_fips}")
            return specific_subdivision_gdf
        else:
            self.logger.warning(f"Target subdivision FIPS {target_subdiv_fips} not found in county {county_fips}.")
            return None

    def download_blocks(self, fips: Dict[str, str]) -> Optional[gpd.GeoDataFrame]:
        """
        Download and filter census blocks for the target county.

        Args:
            fips: Dictionary containing FIPS codes

        Returns:
            Optional[gpd.GeoDataFrame]: Filtered county blocks if successful, None otherwise
        """
        state_fips = fips["state_fips"]
        county_fips = fips["county_fips"]

        census_urls = config.get_census_urls()
        base_url = census_urls.get("BASE_URL", "https://www2.census.gov/geo/tiger/TIGER2020")
        year = census_urls.get("YEAR", "2020")
        blocks_url = f"{base_url}/TABBLOCK20/tl_{year}_{state_fips}_tabblock20.zip"
        all_blocks_in_county_gdf = self._download_and_read_census_shp(
            blocks_url, filename_prefix=f"{state_fips}_{county_fips}_all_county_blocks"
        )

        if all_blocks_in_county_gdf is None or all_blocks_in_county_gdf.empty:
            self.logger.error(
                "Could not load county blocks data (TABBLOCK20). Cannot determine region boundary or blocks."
            )
            raise ValueError("County blocks (TABBLOCK20) could not be loaded.")

        # Filter for the specific county (as shapefiles can be state-wide)
        if "COUNTYFP20" in all_blocks_in_county_gdf.columns:
            county_blocks_gdf_filtered = all_blocks_in_county_gdf[
                all_blocks_in_county_gdf["COUNTYFP20"] == county_fips
            ].copy()
        elif "COUNTYFP" in all_blocks_in_county_gdf.columns:
            county_blocks_gdf_filtered = all_blocks_in_county_gdf[
                all_blocks_in_county_gdf["COUNTYFP"] == county_fips
            ].copy()
        else:
            self.logger.error(
                "Could not find a suitable county FIPS column in the blocks data. Checked 'COUNTYFP20', 'COUNTYFP'."
            )
            raise ValueError("County FIPS column not found in blocks data.")

        if county_blocks_gdf_filtered.empty:
            self.logger.error(f"No blocks found for county FIPS {county_fips} after filtering. Cannot proceed.")
            raise ValueError(f"No blocks found for county FIPS {county_fips}.")

        # County blocks loaded and filtered successfully
        self.logger.debug(
            f"Loaded {
                len(county_blocks_gdf_filtered)} blocks for county {county_fips}"
        )
        return county_blocks_gdf_filtered

    def download(self) -> Dict[str, Any]:
        """
        Downloads and processes Census data for the specified region.

        This method fetches county subdivision boundaries (if applicable) and
        census blocks, filters them to the specific target region (county or
        subdivision), and determines the authoritative boundary for that region.
        Intermediate boundaries (like full county extent or specific subdivision if
        different from final target) are saved to disk but not returned in the main dict.

        Returns:
            dict: Dictionary containing processed GeoDataFrames and their file paths:
                - 'target_region_blocks': GeoDataFrame of census blocks for the target region.
                - 'target_region_blocks_filepath': Path to the saved blocks GeoJSON.
                - 'target_region_boundary': GeoDataFrame representing the final authoritative
                                            boundary for the processing run.
                - 'target_region_boundary_filepath': Path to the final region boundary GeoJSON.
        """
        fips = self.orchestrator.fips_dict
        self.logger.info(f"Processing Census data for {fips['state']} - {fips['county']} - {fips['subdivision']}")

        # Download subdivision and blocks data
        specific_subdivision_gdf = self.download_subdivisions(fips)
        county_blocks_gdf_filtered = self.download_blocks(fips)

        # Store intermediate data for processing step
        return {
            "subdivision_gdf": specific_subdivision_gdf,
            "county_blocks_gdf": county_blocks_gdf_filtered,
            "fips": fips,
        }

    def clip_and_filter_data(
        self,
        county_blocks_gdf: gpd.GeoDataFrame,
        subdivision_gdf: Optional[gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        """
        Clip blocks to subdivision boundary and filter geometries.

        Args:
            county_blocks_gdf: County-level census blocks
            subdivision_gdf: Subdivision boundary for clipping (if available)
            fips: FIPS codes dictionary

        Returns:
            gpd.GeoDataFrame: Processed and filtered blocks
        """
        is_subdivision_run = self.orchestrator.is_subdivision_processing()

        clipping_boundary_for_blocks: Optional[gpd.GeoDataFrame] = None
        if is_subdivision_run and subdivision_gdf is not None and not subdivision_gdf.empty:
            clipping_boundary_for_blocks = subdivision_gdf

        if clipping_boundary_for_blocks is not None:
            if county_blocks_gdf.crs != clipping_boundary_for_blocks.crs:
                county_blocks_gdf = county_blocks_gdf.to_crs(clipping_boundary_for_blocks.crs)

            processed_target_blocks_gdf = gpd.clip(county_blocks_gdf, clipping_boundary_for_blocks)
        else:
            processed_target_blocks_gdf = county_blocks_gdf

        # Filter out non-polygon geometries before saving
        if processed_target_blocks_gdf is not None and not processed_target_blocks_gdf.empty:
            # Keep only Polygon geometries
            processed_target_blocks_gdf = processed_target_blocks_gdf[
                processed_target_blocks_gdf.geometry.geom_type.isin(["Polygon"])
            ].copy()

        return processed_target_blocks_gdf

    def process_boundaries(
        self,
        processed_blocks_gdf: gpd.GeoDataFrame,
        subdivision_gdf: Optional[gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        """
        Create the final authoritative region boundary.

        Args:
            processed_blocks_gdf: Processed census blocks
            subdivision_gdf: Subdivision boundary (if available)
            fips: FIPS codes dictionary

        Returns:
            gpd.GeoDataFrame: Authoritative boundary
        """
        is_subdivision_run = self.orchestrator.is_subdivision_processing()

        if is_subdivision_run and subdivision_gdf is not None and not subdivision_gdf.empty:
            authoritative_boundary_gdf = subdivision_gdf
        elif processed_blocks_gdf is not None and not processed_blocks_gdf.empty:
            try:
                if processed_blocks_gdf.crs is None:
                    self.logger.warning(
                        "Processed target blocks GDF has no CRS. Cannot reliably create authoritative boundary."
                    )
                    raise ValueError("CRS missing from processed blocks, cannot create boundary.")
                unified_geometry = processed_blocks_gdf.geometry.unary_union
                authoritative_boundary_gdf = gpd.GeoDataFrame(geometry=[unified_geometry], crs=processed_blocks_gdf.crs)
            except Exception as e:
                self.logger.error(
                    f"Error creating authoritative target region boundary from blocks: {e}",
                    exc_info=True,
                )
                raise
        else:
            self.logger.error(
                "Cannot determine authoritative target region boundary: No specific subdivision and no processed blocks."
            )
            raise ValueError("Failed to determine an authoritative target region boundary for the run.")

        return authoritative_boundary_gdf

    def process(self, boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
        """
        Processes Census data: downloads, filters, sets the region boundary,
        and optionally visualizes the census blocks.

        Args:
            boundary_gdf (Optional[gpd.GeoDataFrame]): Boundary to use for clipping.
                If None, the method should attempt to get it from `self.orchestrator.get_region_boundary()`.

        Returns:
            Dict[str, Any]: A dictionary containing processed GeoDataFrames for
            blocks, subdivision (if applicable), the final region boundary,
            and paths to their saved files.
        """
        try:
            # Download raw data
            download_data = self.download()

            # Extract downloaded components
            subdivision_gdf = download_data["subdivision_gdf"]
            county_blocks_gdf = download_data["county_blocks_gdf"]

            # Process and filter data
            processed_target_blocks_gdf = self.clip_and_filter_data(county_blocks_gdf, subdivision_gdf)

            # Create authoritative boundary
            authoritative_boundary_gdf = self.process_boundaries(processed_target_blocks_gdf, subdivision_gdf)

            # Prepare results
            results: Dict[str, Any] = {
                "target_region_blocks": None,
                "target_region_blocks_filepath": None,
                "target_region_boundary": None,
                "target_region_boundary_filepath": None,
            }

            # Save blocks
            if processed_target_blocks_gdf is not None and not processed_target_blocks_gdf.empty:
                results["target_region_blocks"] = processed_target_blocks_gdf
                results["target_region_blocks_filepath"] = self.dataset_output_dir / "target_region_blocks.geojson"
                processed_target_blocks_gdf.to_file(results["target_region_blocks_filepath"], driver="GeoJSON")
            else:
                self.logger.warning(
                    "No target region blocks resulted after processing/clipping. "
                    "This might indicate an issue if blocks were expected."
                )

            # Save boundary
            if authoritative_boundary_gdf is not None and not authoritative_boundary_gdf.empty:
                results["target_region_boundary"] = authoritative_boundary_gdf
                results["target_region_boundary_filepath"] = self.dataset_output_dir / "target_region_boundary.geojson"
                authoritative_boundary_gdf.to_file(results["target_region_boundary_filepath"], driver="GeoJSON")

                self.orchestrator.set_region_boundary(authoritative_boundary_gdf)
            else:
                self.logger.error(
                    "Authoritative target region boundary GDF is empty or None. " "Cannot set in orchestrator."
                )
                raise ValueError("Authoritative target region boundary could not be established.")
            self.logger.info("Finished processing census data for the target_region")

            return results
        except Exception as e:
            self.logger.error(f"Census data processing failed: {e}", exc_info=True)
            raise

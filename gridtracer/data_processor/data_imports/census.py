from typing import Dict, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd  # For timestamp in plotting

from gridtracer.data_processor.data_imports.base import DataHandler


class CensusDataHandler(DataHandler):
    """
    Handler for Census TIGER data.

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

    def _download_and_read_census_shp(
            self, shp_url: str, filename_prefix: str) -> Optional[gpd.GeoDataFrame]:
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
            self.logger.info(f"Downloading and processing from: {shp_url}")
            try:
                gdf = gpd.read_file(shp_url)
                gdf.to_file(output_geojson_path, driver='GeoJSON')
                self.logger.info(f"Saved to: {output_geojson_path}")
                return gdf
            except Exception as e:
                self.logger.error(f"Failed to download/process {shp_url}: {e}", exc_info=True)
                return None
        else:
            self.logger.info(f"Loading from existing file: {output_geojson_path}")
            try:
                return gpd.read_file(output_geojson_path)
            except Exception as e:
                self.logger.error(f"Failed to load {output_geojson_path}: {e}", exc_info=True)
                return None

    def download_and_process_data(self) -> Dict[str, any]:
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
        fips = self.orchestrator.get_fips_dict()
        if not fips:
            self.logger.error("FIPS dictionary not available from orchestrator.")
            raise ValueError("FIPS dictionary is missing for Census processing.")

        state_fips = fips['state_fips']
        county_fips = fips['county_fips']
        target_subdiv_fips = fips.get('subdivision_fips')
        is_subdivision_run = self.orchestrator.is_subdivision_processing()

        self.logger.info(
            f"Processing Census data for State FIPS: {state_fips}, County FIPS: {county_fips}"
            f"{f', Target Subdivision FIPS: {target_subdiv_fips}' if is_subdivision_run and target_subdiv_fips else ''}"
        )

        results: Dict[str, any] = {
            'target_region_blocks': None, 'target_region_blocks_filepath': None,
            'target_region_boundary': None, 'target_region_boundary_filepath': None
        }

        specific_subdivision_gdf: Optional[gpd.GeoDataFrame] = None

        # 1. Process County Subdivisions (COUSUB) - if it's a subdivision run
        if is_subdivision_run and target_subdiv_fips:
            subdivision_url = f"https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/tl_2020_{state_fips}_cousub.zip"
            all_county_subdivisions_gdf = self._download_and_read_census_shp(
                subdivision_url,
                # Suffix for clarity
                filename_prefix=f"{state_fips}_{county_fips}_all_subdivisions_ref"
            )

            if all_county_subdivisions_gdf is not None and not all_county_subdivisions_gdf.empty:
                county_filtered_subdivisions = all_county_subdivisions_gdf[
                    all_county_subdivisions_gdf['COUNTYFP'] == county_fips
                ]
                if not county_filtered_subdivisions.empty:
                    specific_subdivision_gdf = county_filtered_subdivisions[
                        county_filtered_subdivisions['COUSUBFP'] == target_subdiv_fips
                    ].copy()

                    if not specific_subdivision_gdf.empty:
                        # Save this specific subdivision boundary for reference
                        subdiv_boundary_ref_path = self.dataset_output_dir / \
                            f"{state_fips}_{county_fips}_{target_subdiv_fips}_subdivision_boundary_ref.geojson"
                        specific_subdivision_gdf.to_file(
                            subdiv_boundary_ref_path, driver='GeoJSON')
                        self.logger.info(
                            f"Reference file for target subdivision boundary saved: {subdiv_boundary_ref_path}")
                    else:
                        self.logger.warning(
                            f"Target subdivision FIPS {target_subdiv_fips} not found in county {county_fips}.")
                else:
                    self.logger.warning(
                        f"No subdivisions for county FIPS {county_fips} in COUSUB file.")
            else:
                self.logger.warning("Could not load county subdivisions data (COUSUB).")

        # 2. Process Census Blocks (TABBLOCK20)
        blocks_url = f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{state_fips}_tabblock20.zip"
        # This GDF will store all blocks for the county initially
        all_blocks_in_county_gdf = self._download_and_read_census_shp(
            blocks_url,
            # Suffix for clarity
            filename_prefix=f"{state_fips}_{county_fips}_all_county_blocks_ref"
        )

        processed_target_blocks_gdf: Optional[gpd.GeoDataFrame] = None
        if all_blocks_in_county_gdf is not None and not all_blocks_in_county_gdf.empty:
            # Filter for the specific county (as shapefiles can be state-wide)
            if 'COUNTYFP20' in all_blocks_in_county_gdf.columns:
                county_blocks_gdf_filtered = all_blocks_in_county_gdf[all_blocks_in_county_gdf['COUNTYFP20'] == county_fips].copy(
                )
            elif 'COUNTYFP' in all_blocks_in_county_gdf.columns:  # Fallback or alternative common name
                county_blocks_gdf_filtered = all_blocks_in_county_gdf[all_blocks_in_county_gdf['COUNTYFP'] == county_fips].copy(
                )
            else:
                self.logger.error(
                    f"Could not find a suitable county FIPS column in the blocks data. Checked 'COUNTYFP20', 'COUNTYFP'.")
                raise ValueError("County FIPS column not found in blocks data.")

            if county_blocks_gdf_filtered.empty:
                self.logger.error(
                    f"No blocks found for county FIPS {county_fips} after filtering. Cannot proceed.")
                raise ValueError(f"No blocks found for county FIPS {county_fips}.")

            # Save the boundary of ALL blocks in the county (full county extent) for reference
            try:
                if county_blocks_gdf_filtered.crs is None:
                    self.logger.warning(
                        "Full county blocks GDF has no CRS. Cannot reliably create its boundary reference.")
                else:
                    full_county_extent_geom = county_blocks_gdf_filtered.geometry.unary_union
                    full_county_extent_gdf = gpd.GeoDataFrame(
                        geometry=[full_county_extent_geom], crs=county_blocks_gdf_filtered.crs)
                    full_county_extent_ref_path = self.dataset_output_dir / \
                        f"{state_fips}_{county_fips}_full_county_extent_ref.geojson"
                    full_county_extent_gdf.to_file(full_county_extent_ref_path, driver='GeoJSON')
                    self.logger.info(
                        f"Reference file for full county extent (from all blocks) saved: {full_county_extent_ref_path}")
            except Exception as e:
                self.logger.warning(
                    f"Could not save full county extent reference boundary: {e}",
                    exc_info=True)

            clipping_boundary_for_blocks: Optional[gpd.GeoDataFrame] = None
            if is_subdivision_run and specific_subdivision_gdf is not None and not specific_subdivision_gdf.empty:
                clipping_boundary_for_blocks = specific_subdivision_gdf
                self.logger.info("Targeting blocks within the specific county subdivision.")
            else:
                self.logger.info("Targeting all blocks within the county.")

            if clipping_boundary_for_blocks is not None:
                self.logger.info(f"Clipping county blocks to the target subdivision boundary.")
                if county_blocks_gdf_filtered.crs != clipping_boundary_for_blocks.crs:
                    self.logger.info(
                        f"Aligning CRS for block clipping: Blocks {county_blocks_gdf_filtered.crs} to Boundary {clipping_boundary_for_blocks.crs}")
                    county_blocks_gdf_filtered = county_blocks_gdf_filtered.to_crs(
                        clipping_boundary_for_blocks.crs)

                processed_target_blocks_gdf = gpd.clip(
                    county_blocks_gdf_filtered, clipping_boundary_for_blocks)
                blocks_filename_suffix = f"_blocks" if target_subdiv_fips else "_blocks"
            else:
                processed_target_blocks_gdf = county_blocks_gdf_filtered
                blocks_filename_suffix = "_blocks"

            # Filter out non-polygon geometries before saving
            if processed_target_blocks_gdf is not None and not processed_target_blocks_gdf.empty:
                # Filter out non-polygon geometries before saving
                original_count = len(processed_target_blocks_gdf)

                # Keep only Polygon and MultiPolygon geometries
                processed_target_blocks_gdf = processed_target_blocks_gdf[
                    processed_target_blocks_gdf.geometry.geom_type.isin(['Polygon'])
                ].copy()

                final_count = len(processed_target_blocks_gdf)
                dropped_count = original_count - final_count

                if dropped_count > 0:
                    self.logger.info(
                        f"Filtered out {dropped_count} non-polygon blocks from clipping artifacts")
                    self.logger.info(
                        f"Retained {final_count}/{original_count} valid polygon blocks")

            if processed_target_blocks_gdf is not None and not processed_target_blocks_gdf.empty:
                results['target_region_blocks'] = processed_target_blocks_gdf
                results['target_region_blocks_filepath'] = self.dataset_output_dir / \
                    f"target_region{blocks_filename_suffix}.geojson"
                processed_target_blocks_gdf.to_file(
                    results['target_region_blocks_filepath'], driver='GeoJSON')
                self.logger.info(
                    f"Saved target region blocks: {results['target_region_blocks_filepath']}")
            else:
                self.logger.warning(
                    "No target region blocks resulted after processing/clipping. This might indicate an issue if blocks were expected.")
        else:
            self.logger.error(
                "Could not load county blocks data (TABBLOCK20). Cannot determine region boundary or blocks.")
            raise ValueError("County blocks (TABBLOCK20) could not be loaded.")

        # 3. Define and Set the Final Authoritative Region Boundary for the Orchestrator
        authoritative_boundary_gdf: Optional[gpd.GeoDataFrame] = None
        boundary_filename_suffix_for_file: str

        if is_subdivision_run and specific_subdivision_gdf is not None and not specific_subdivision_gdf.empty:
            authoritative_boundary_gdf = specific_subdivision_gdf
            boundary_filename_suffix_for_file = f"_boundary"
            self.logger.info(
                "Using specific subdivision as the authoritative target region boundary.")
        elif processed_target_blocks_gdf is not None and not processed_target_blocks_gdf.empty:
            self.logger.info(
                "Defining authoritative target region boundary from the extent of processed blocks.")
            try:
                if processed_target_blocks_gdf.crs is None:
                    self.logger.warning(
                        "Processed target blocks GDF has no CRS. Cannot reliably create authoritative boundary.")
                    raise ValueError("CRS missing from processed blocks, cannot create boundary.")
                unified_geometry = processed_target_blocks_gdf.geometry.unary_union
                authoritative_boundary_gdf = gpd.GeoDataFrame(
                    geometry=[unified_geometry], crs=processed_target_blocks_gdf.crs)
                boundary_filename_suffix_for_file = "_boundary"
            except Exception as e:
                self.logger.error(
                    f"Error creating authoritative target region boundary from blocks: {e}",
                    exc_info=True)
                raise
        else:
            self.logger.error(
                "Cannot determine authoritative target region boundary: No specific subdivision and no processed blocks.")
            raise ValueError(
                "Failed to determine an authoritative target region boundary for the run.")

        if authoritative_boundary_gdf is not None and not authoritative_boundary_gdf.empty:
            results['target_region_boundary'] = authoritative_boundary_gdf
            results['target_region_boundary_filepath'] = self.dataset_output_dir / \
                f"target_region{boundary_filename_suffix_for_file}.geojson"
            authoritative_boundary_gdf.to_file(
                results['target_region_boundary_filepath'], driver='GeoJSON')
            self.logger.info(
                f"Saved authoritative target region boundary: {results['target_region_boundary_filepath']}")

            self.orchestrator.set_region_boundary(authoritative_boundary_gdf)
            self.logger.info(
                "Authoritative target region boundary successfully set in WorkflowOrchestrator.")
        else:
            self.logger.error(
                "Authoritative target region boundary GDF is empty or None. Cannot set in orchestrator.")
            raise ValueError("Authoritative target region boundary could not be established.")

        return results

    def _visualize_census_data(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        boundary_to_plot_gdf: Optional[gpd.GeoDataFrame] = None,
        plot_title_override: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize census blocks on a map with their individual boundaries.
        If boundary_to_plot_gdf is provided, its boundary will be overlaid.

        Args:
            blocks_gdf (GeoDataFrame): GeoDataFrame containing census blocks to plot.
                                     These should be the blocks relevant to the current scope
                                     (e.g., subdivision blocks or all county blocks).
            boundary_to_plot_gdf (Optional[gpd.GeoDataFrame]): GeoDataFrame whose boundary
                                                             will be drawn on the plot. This could
                                                             be a specific subdivision or the whole county boundary.
            plot_title_override (Optional[str]): Optional title for the plot. If None,
                                                 a title is generated based on FIPS info.
        Returns:
            Optional[str]: Path to the saved plot file, or None if plotting failed.
        """
        if blocks_gdf is None or blocks_gdf.empty:
            self.logger.error("No blocks provided to _visualize_census_data.")
            return None

        self.logger.info(f"Visualizing {len(blocks_gdf)} census blocks.")

        fips = self.orchestrator.get_fips_dict()
        if not fips:  # Should not happen if orchestrator is correctly initialized
            self.logger.error("FIPS dictionary not available from orchestrator for plotting.")
            return None

        plot_output_dir = self.orchestrator.get_dataset_specific_output_directory("PLOTS")

        # Determine plot title
        title = plot_title_override
        if title is None:
            # Use the name from the boundary_to_plot_gdf if available and it's a subdivision
            # Otherwise, use the FIPS info from the orchestrator.
            if self.orchestrator.is_subdivision_processing() and \
               boundary_to_plot_gdf is not None and not boundary_to_plot_gdf.empty and \
               'NAME' in boundary_to_plot_gdf.columns:
                title = f"Census Blocks in {boundary_to_plot_gdf.iloc[0]['NAME']}"
            elif self.orchestrator.is_subdivision_processing() and fips.get('subdivision'):
                title = f"Census Blocks in {fips['subdivision']}"
            else:
                title = f"Census Blocks in {fips['county']}, {fips['state']}"

        # Create figure and axis
        _, ax = plt.subplots(figsize=(15, 15))

        # Convert to Web Mercator for basemap compatibility
        blocks_mercator = blocks_gdf.to_crs(epsg=3857)
        # Define initial bounds from blocks
        current_plot_bounds = list(blocks_mercator.total_bounds)

        # Plot blocks
        blocks_mercator.plot(ax=ax, alpha=0.2, edgecolor='red', facecolor='skyblue', linewidth=1.0)

        # Plot the provided boundary if it exists
        if boundary_to_plot_gdf is not None and not boundary_to_plot_gdf.empty:
            boundary_mercator = boundary_to_plot_gdf.to_crs(epsg=3857)
            boundary_mercator.plot(
                ax=ax,
                facecolor='none',
                edgecolor='green',
                linewidth=2.0,
                linestyle='--')

            # Update overall plot bounds to ensure the boundary is visible
            specific_boundary_bounds = list(boundary_mercator.total_bounds)
            current_plot_bounds = [
                min(current_plot_bounds[0], specific_boundary_bounds[0]),
                min(current_plot_bounds[1], specific_boundary_bounds[1]),
                max(current_plot_bounds[2], specific_boundary_bounds[2]),
                max(current_plot_bounds[3], specific_boundary_bounds[3])
            ]

        # Add basemap
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                zoom='auto',
                crs="EPSG:3857",
                attribution_size=8)
        except Exception as e:
            self.logger.warning(f"Could not add basemap for census blocks plot: {e}")

        ax.set_xlim(current_plot_bounds[0], current_plot_bounds[2])
        ax.set_ylim(current_plot_bounds[1], current_plot_bounds[3])
        plt.title(title, pad=20, fontsize=16)
        ax.set_axis_off()

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        safe_title = title.replace(" ", "_").replace(",", "").lower()
        output_file = plot_output_dir / f"{safe_title}_{timestamp}.png"

        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved census blocks visualization to: {output_file}")
            return str(output_file)
        except Exception as e:
            self.logger.error(f"Failed to save census blocks plot: {e}", exc_info=True)
            return None
        finally:
            plt.close()

    def download(self) -> Dict[str, any]:
        """
        Satisfies the abstract 'download' method from DataHandler.
        For CensusDataHandler, the primary data acquisition and initial processing
        happens in download_and_process_data. This method can be a wrapper
        or delegate if a distinct "raw download" step isn't strictly separate.

        Returns:
            Dict[str, any]: A dictionary containing references to downloaded/processed
                            data, typically file paths or preliminary GeoDataFrames.
        """
        self.logger.info(
            "Executing 'download' method for CensusDataHandler, which defers to internal processing.")
        # In this specific case, download_and_process_data handles what would
        # conceptually be "downloading" and some initial structuring.
        # We return a subset of its results or a status.
        # For simplicity here, we'll just call it and let it populate files.
        # The 'process' method will then ensure the orchestrator is updated.

        # Option A: If download_and_process_data is considered the "download" step.
        # However, download_and_process_data also sets the boundary in the orchestrator,
        # which might be more of a "process" step.
        # For now, let's assume download_and_process_data is too broad for just "download".

        # Option B: Minimal implementation if 'download_and_process_data' is called by 'process'
        # This indicates that the main work is done in 'process' which calls
        # 'download_and_process_data'.
        self.logger.debug(
            "CensusDataHandler.download(): No separate raw download step; main logic in process().")
        return {"status": "Download logic is integrated into process() via download_and_process_data()"}

    def process(self, plot: bool = False) -> Dict[str, any]:
        """
        Processes Census data: downloads, filters, sets the region boundary,
        and optionally visualizes the census blocks.

        Args:
            plot (bool): If True, visualizes the processed census blocks. Defaults to False.

        Returns:
            Dict[str, any]: A dictionary containing processed GeoDataFrames for
            blocks, subdivision (if applicable), the final region boundary,
            and paths to their saved files.
        """
        self.logger.info(
            f"Processing Census data via orchestrator context. Plotting: {plot}")  # Changed plot_blocks to plot

        try:
            # The main work, including what would be "downloading", happens here.
            processed_data = self.download_and_process_data()

            if plot:  # Changed plot_blocks to plot
                blocks_for_plot = processed_data.get('target_region_blocks')

                boundary_to_draw = processed_data.get('target_region_boundary')

                if blocks_for_plot is not None and not blocks_for_plot.empty:
                    self.logger.info("Proceeding with census blocks visualization.")
                    self._visualize_census_data(
                        blocks_gdf=blocks_for_plot,
                        boundary_to_plot_gdf=boundary_to_draw
                    )
                else:
                    self.logger.warning(
                        "Skipping census blocks visualization as no blocks data is available.")

            return processed_data
        except Exception as e:
            self.logger.error(f"Census data processing failed: {e}", exc_info=True)
            raise

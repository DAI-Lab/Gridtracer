"""
Road network builder module for creating routable road networks from OSM data.

This module provides functionality to extract and process OpenStreetMap road
data to create a PostgreSQL/PostGIS compatible SQL file for pgRouting.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import geopandas as gpd
import pandas as pd
import yaml
from shapely.wkb import dumps as wkb_dumps
from tqdm import tqdm

from syngrid.data_processor.data.base import DataHandler
from syngrid.data_processor.utils.utils import visualize_road_network

if TYPE_CHECKING:
    from syngrid.data_processor.workflow import WorkflowOrchestrator

# Initialize logger
logger = logging.getLogger(__name__)

# --- SQL Templates ---
HEADER_SQL = """SET client_encoding = 'UTF8';

DROP TABLE IF EXISTS public_2po_4pgr;

CREATE TABLE public_2po_4pgr (
    id SERIAL PRIMARY KEY,
    osm_id bigint UNIQUE, -- Added UNIQUE constraint for potential merging
    osm_name character varying,
    osm_meta character varying,
    osm_source_id bigint,
    osm_target_id bigint,
    clazz integer,
    flags integer,
    source integer,
    target integer,
    km double precision,
    kmh integer,
    cost double precision,
    reverse_cost double precision,
    x1 double precision,
    y1 double precision,
    x2 double precision,
    y2 double precision,
    geom_way geometry(LineString, 4326)
);
"""

INDEX_SQL = """
-- Build spatial index after data load
CREATE INDEX IF NOT EXISTS osm2po_routing_geom_idx
    ON public_2po_4pgr USING GIST (geom_way);

-- Optional: Indexes on source/target for routing queries
CREATE INDEX IF NOT EXISTS osm2po_routing_source_idx
    ON public_2po_4pgr (source);
CREATE INDEX IF NOT EXISTS osm2po_routing_target_idx
    ON public_2po_4pgr (target);

-- Analyze table after index creation and data loading
ANALYZE public_2po_4pgr;
"""


class RoadNetworkBuilder(DataHandler):
    """
    Class for building a routable road network from OSM data.

    This class processes OpenStreetMap data to create a PostgreSQL/PostGIS
    compatible SQL file that can be imported for routing purposes.
    """

    def __init__(
            self,
            orchestrator: 'WorkflowOrchestrator',
            config_file: Optional[str] = None):
        """
        Initialize the road network builder.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance.
            config_file (str, optional): Path to the YAML configuration file
        """
        super().__init__(orchestrator)
        self.orchestrator = orchestrator  # Store orchestrator if needed, or ensure DataHandler does
        self.config_file = config_file or 'syngrid/data_processor/data/osm/osm2po_config.yaml'
        self.config = self._load_config()

    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name
        """
        return "STREET_NETWORK"

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Returns:
            Dict containing the configuration settings.
        """
        default_config = {
            'way_tag_resolver': {
                'tags': {
                    "motorway": {"clazz": 11, "maxspeed": 120, "flags": ["car"]},
                },
                'final_mask': None,  # Default: no mask
                'flag_list': ["car", "bike", "foot"]  # Default flags for bitmask
            },
            'tileSize': 'x',  # Default: no tiling
            'network_type': 'driving',
            'osm_pbf_file': None,  # Must be provided separately
            'output_dir': 'sql_output_chunks',
            'total_bounds': None  # e.g., [min_lon, min_lat, max_lon, max_lat]
        }

        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                default_config.update(config)
                return default_config
        except FileNotFoundError:
            logger.warning(
                f"Configuration file '{self.config_file}' not found. "
                f"Using default values."
            )
            return default_config

    def _flags_to_int(self, flags_set: Set[str]) -> int:
        """
        Convert a set of flags to a bitmask integer.

        Args:
            flags_set: Set of flag strings to convert.

        Returns:
            Integer representation of the flags.
        """
        flag_list = self.config.get('way_tag_resolver', {}).get(
            'flag_list', ["car", "bike", "foot"]
        )
        flag_bitmask = {flag: 1 << i for i, flag in enumerate(flag_list)}

        mask = 0
        if flags_set:
            for flag in flags_set:
                mask |= flag_bitmask.get(flag, 0)
        return mask

    def _resolve_way_tags(self, tags_dict: Dict[str, Any]) -> tuple:
        """
        Resolves OSM way tags to internal clazz, maxspeed, and flags
        based on the loaded configuration.

        Args:
            tags_dict: Dictionary of OSM tags.

        Returns:
            Tuple of (clazz, maxspeed, flags_set).
        """
        way_tag_config = self.config.get('way_tag_resolver', {}).get('tags', {})
        highway_type = tags_dict.get("highway")
        if highway_type and highway_type in way_tag_config:
            cfg = way_tag_config[highway_type]
            flags = set(cfg.get('flags', []))
            return cfg.get("clazz", 0), cfg.get("maxspeed", 0), flags
        return 0, 0, set()

    def _is_way_allowed_by_final_mask(self, flags_set: Set[str]) -> bool:
        """
        Checks if a way's resolved flags intersect with the final_mask_flags.

        Args:
            flags_set: Set of flag strings to check.

        Returns:
            Boolean indicating if the way is allowed.
        """
        final_mask_str = self.config.get('way_tag_resolver', {}).get('final_mask')
        final_mask_flags = set(final_mask_str.split(',')) if final_mask_str else None

        if final_mask_flags is None:  # No mask defined, allow all
            return True
        if not flags_set:  # No flags resolved, disallow if mask exists
            return False
        return not final_mask_flags.isdisjoint(flags_set)

    def _process_and_write_edges(
        self, edges_gdf: gpd.GeoDataFrame, tile_identifier: str = ""
    ) -> List[str]:
        """
        Process edge data for SQL generation.

        Args:
            edges_gdf: GeoDataFrame containing network edges.
            tile_identifier: String or number to identify the tile (for logging).

        Returns:
            List of SQL value tuple strings.
        """
        if edges_gdf.empty:
            logger.info(f"[{tile_identifier}] No edges to process.")
            return []  # Return empty list

        logger.info(
            f"[{tile_identifier}] Processing {len(edges_gdf)} edges: "
            f"Resolving tags, calculating costs, applying filters..."
        )
        processed_edges_list = []
        original_crs = edges_gdf.crs  # Preserve CRS

        for idx, row in tqdm(edges_gdf.iterrows()):
            # Adapt tag extraction based on actual pyrosm output format
            tags = {"highway": row.get("highway")}  # How does pyrosm store tags?
            if 'tags' in row and isinstance(row['tags'], dict):
                tags.update(row['tags'])

            # Extract way ID for logging
            way_id = row.get("osm_id", row.get("id", idx))

            clazz, maxspeed, flags = self._resolve_way_tags(tags)

            if not self._is_way_allowed_by_final_mask(flags):
                logger.debug(
                    "Skipping way %s due to final mask. Flags: %s",
                    way_id, flags
                )
                continue  # Skip this edge

            if clazz == 0:
                logger.debug("Skipping way %s due to clazz 0. Tags: %s", way_id, tags)
                continue  # Skip ways that didn't resolve

            processed_edge = row.to_dict()
            processed_edge['clazz'] = clazz
            processed_edge['kmh'] = maxspeed
            processed_edge['flags_set'] = flags  # Store resolved flags

            length_m = processed_edge.get('length', 0)
            if length_m == 0 and processed_edge.get('geometry'):
                try:
                    length_m = processed_edge['geometry'].length
                except Exception:  # Catch specific exceptions if possible
                    length_m = 0  # Ignore errors

            processed_edge['km'] = length_m / 1000.0
            if maxspeed > 0:
                processed_edge['cost'] = processed_edge['km'] / maxspeed  # Hours
            else:
                processed_edge['cost'] = processed_edge['km'] * 10  # High penalty

            is_oneway = str(row.get('oneway', 'no')).lower() in [
                'yes', 'true', '1', '-1'
            ]
            if is_oneway:
                if str(row.get('oneway', 'no')) == '-1':
                    processed_edge['reverse_cost'] = processed_edge['cost']
                else:
                    processed_edge['reverse_cost'] = float('inf')
            else:
                processed_edge['reverse_cost'] = processed_edge['cost']

            processed_edges_list.append(processed_edge)

        if not processed_edges_list:
            logger.info(f"[{tile_identifier}] No edges remained after processing.")
            return []  # Return empty list

        edges = gpd.GeoDataFrame(
            processed_edges_list, geometry='geometry', crs=original_crs
        )
        logger.info(f"[{tile_identifier}] Processing complete. {len(edges)} edges remaining.")

        insert_value_tuples = []
        if 'osm_id' not in edges.columns:
            edges['osm_id'] = edges.index

        # Iterate again, this time on processed geodataframe
        for idx, row in edges.iterrows():
            osm_id = row.get('osm_id', idx)
            osm_name_val = row.get('name')
            ref_val = row.get('ref')  # Get the 'ref' tag value

            final_name_for_sql = "NULL"  # Default to NULL

            # Prioritize osm_name_val if it exists and is not empty
            if pd.notna(osm_name_val) and str(osm_name_val).strip():
                name_str = str(osm_name_val)
                escaped_name = name_str.replace("'", "''")
                final_name_for_sql = f"'{escaped_name}'"
            # Else, if ref_val exists and is not empty, use it
            elif pd.notna(ref_val) and str(ref_val).strip():
                ref_str = str(ref_val)
                escaped_ref = ref_str.replace("'", "''")
                final_name_for_sql = f"'{escaped_ref}'"

            # Assign the determined name to osm_name for the SQL query
            osm_name = final_name_for_sql

            osm_meta = "NULL"  # Placeholder
            osm_source_id = row.get("u", -1)
            osm_target_id = row.get("v", -1)
            clazz = row.get("clazz", 0)
            flags_int = self._flags_to_int(row.get("flags_set", set()))
            source = osm_source_id
            target = osm_target_id
            km = row.get("km", 0.0)
            kmh = row.get("kmh", 0)
            cost = row.get("cost", float('inf'))
            reverse_cost = row.get("reverse_cost", float('inf'))

            try:
                coords = list(row.geometry.coords)
                x1, y1 = coords[0]
                x2, y2 = coords[-1]
                geom_hex_ewkb = wkb_dumps(row.geometry, hex=True, srid=4326)
            except Exception as e:
                logger.warning(
                    f"[{tile_identifier}] Skipping edge {osm_id} "
                    f"due to geometry error: {e}"
                )
                continue

            # Shorter f-string parts for readability
            part1 = f"({osm_id}, {osm_name}, {osm_meta}, "
            part2 = f"{osm_source_id}, {osm_target_id}, {clazz}, {flags_int}, "
            part3 = f"{source}, {target}, {km:.7f}, {kmh}, {cost:.7f}, "
            part4 = f"{reverse_cost:.7f}, {x1:.7f}, {y1:.7f}, {x2:.7f}, {y2:.7f}, "
            part5 = f"'{geom_hex_ewkb}')"
            insert_tuple_str = part1 + part2 + part3 + part4 + part5
            insert_value_tuples.append(insert_tuple_str)

        return insert_value_tuples

    def build_network(
        self,
        network_type: str = 'driving',
        boundary_gdf: Optional[gpd.GeoDataFrame] = None,
        plot: bool = False
    ) -> Dict[str, Any]:
        """
        Build a routable road network from OSM data.

        Args:
            network_type: Type of network to extract (e.g., 'driving', 'walking', 'cycling').
            boundary_gdf: GeoDataFrame containing the boundary polygon for clipping.

        Returns:
            Dictionary containing results of the network building process:
                - edges: GeoDataFrame of edges
                - sql_file: Path to the generated SQL file
                - geojson_file: Path to the network GeoJSON file
                - visualization_file: Path to the network visualization
        """
        results = {
            'nodes': None,
            'edges': None,
            'sql_file': None,
            'geojson_file': None,
            'visualization_file': None
        }

        osm = self.orchestrator.get_osm_parser()
        if osm is None:
            logger.error(
                "OSM parser not available from orchestrator. Cannot build road network."
            )
            return results

        try:
            logger.info(
                f"Extracting network using pre-initialized OSM parser. Network type: {network_type}")
            nodes, edges_gdf = osm.get_network(network_type=network_type, nodes=True)
            logger.info(f"Loaded {len(nodes)} nodes and {len(edges_gdf)} total edges.")

            # Export road network to a single GeoJSON file
            geojson_path = self.dataset_output_dir / "road_network.geojson"
            try:
                # Save to GeoJSON
                edges_gdf.to_file(geojson_path, driver='GeoJSON')
                logger.info(f"Saved road network to GeoJSON: {geojson_path}")
                results['geojson_file'] = geojson_path
            except Exception as e:
                logger.error(f"Error exporting network to GeoJSON: {e}")

            # Process the edges for SQL
            insert_value_tuples = self._process_and_write_edges(edges_gdf, "ALL_DATA")

            # Generate SQL file
            full_sql_content = []
            full_sql_content.append(HEADER_SQL)

            if insert_value_tuples:
                # Constructing the full INSERT INTO block
                insert_block_parts = [
                    "INSERT INTO public_2po_4pgr (osm_id, osm_name, osm_meta, ",
                    "osm_source_id, osm_target_id, clazz, flags, source, target, km, ",
                    "kmh, cost, reverse_cost, x1, y1, x2, y2, geom_way) VALUES"
                ]
                insert_block = "".join(insert_block_parts)
                insert_block += "\n" + ",\n".join(insert_value_tuples) + ";\n"
                full_sql_content.append(insert_block)
                logger.info(f"Generated {len(insert_value_tuples)} insert statements.")
            else:
                logger.warning("No insert statements generated.")

            full_sql_content.append(INDEX_SQL)

            output_sql_file = self.dataset_output_dir / "osm2po_routing_network.sql"
            try:
                with open(output_sql_file, "w", encoding="utf-8") as f:
                    # Add some spacing between main SQL sections
                    f.write("\n\n".join(full_sql_content))
                logger.info(f"All SQL commands written to {output_sql_file}")
                results['sql_file'] = output_sql_file
            except IOError as e:
                logger.error(f"Error writing full SQL file {output_sql_file}: {e}")

            # Store the edges in results
            results['edges'] = edges_gdf

            # Create a simple visualization of the road network using the exported GeoJSON
            if plot:
                try:
                    plot_output_dir = self.orchestrator.get_dataset_specific_output_directory(
                        "PLOTS")
                    viz_file = visualize_road_network(
                        network_data=geojson_path,
                        boundary_gdf=boundary_gdf,
                        output_dir=plot_output_dir,
                        title=f"Road Network - {network_type}"
                    )

                    if viz_file:
                        logger.info(f"Road network visualization created: {viz_file}")
                        results['visualization_file'] = viz_file
                except Exception as e:
                    logger.error(f"Error creating road network visualization: {e}")

                logger.info(f"Processing finished. Output generated in: {self.dataset_output_dir}")

        except Exception as e:
            logger.error(f"Error building road network: {e}")

        return results

    def download(self):
        """
        Required method for DataHandler - not implemented.
        This class does not directly download data.
        """
        raise NotImplementedError("Data downloading not implemented for this class")

    def process(self, boundary_gdf=None, plot=False):
        """
        Process the data for the region.

        This method implements the complete data processing workflow
        for the road network, essentially a wrapper around build_network.

        Args:
            boundary_gdf (GeoDataFrame, optional): Boundary to use for clipping
                If provided, the network will be clipped to this boundary

        Returns:
            dict: Dictionary containing processed data and file paths:
                - edges: GeoDataFrame of network edges
                - sql_file: Path to the generated SQL file
                - geojson_file: Path to the network GeoJSON file
                - visualization_file: Path to the network visualization
        """
        network_type = self.config.get('network_type', 'driving')

        # Build the network with boundary clipping during initial loading
        results = self.build_network(
            network_type=network_type,
            boundary_gdf=boundary_gdf,
            plot=plot
        )

        return results

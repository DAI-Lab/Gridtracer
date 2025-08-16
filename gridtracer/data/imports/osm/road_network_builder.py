"""
Road network builder module for creating routable road networks from OSM data.

This module provides functionality to extract and process OpenStreetMap road
data to create a PostgreSQL/PostGIS compatible SQL file for pgRouting.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
import yaml
from shapely.wkb import dumps as wkb_dumps

from gridtracer.data.imports.base import DataHandler

if TYPE_CHECKING:
    from gridtracer.data.workflow import WorkflowOrchestrator

# Constants
METERS_TO_KM = 1000.0
MIN_SPEED_KMH = 1  # Minimum speed to avoid division by zero

# --- SQL Templates ---
HEADER_SQL = """SET client_encoding = 'UTF8';

DROP TABLE IF EXISTS road_network;

CREATE TABLE road_network (
    id integer,
    osm_id bigint,
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
    y2 double precision
);
SELECT AddGeometryColumn('road_network', 'geom_way', 4326, 'LINESTRING', 2);

"""

INDEX_SQL = """
-- Build spatial index after data load
CREATE INDEX IF NOT EXISTS routing_geom_idx
    ON road_network USING GIST (geom_way);

-- Optional: Indexes on source/target for routing queries
CREATE INDEX IF NOT EXISTS routing_source_idx
    ON road_network (source);
CREATE INDEX IF NOT EXISTS routing_target_idx
    ON road_network (target);

-- Analyze table after index creation and data loading
ANALYZE road_network;
"""


class RoadNetworkBuilder(DataHandler):
    """
    Class for building a routable road network from OSM data.

    This class processes OpenStreetMap data to create a PostgreSQL/PostGIS
    compatible SQL file that can be imported for routing purposes.
    """

    def __init__(
        self,
        orchestrator: "WorkflowOrchestrator",
        road_extraction_config_file: Optional[str] = None,
    ):
        """
        Initialize the road network builder.

        Args:
            orchestrator (WorkflowOrchestrator): The workflow orchestrator instance.
            config_file (str, optional): Path to the YAML configuration file
        """
        super().__init__(orchestrator)

        self.road_extraction_config_file = (
            road_extraction_config_file or Path(__file__).parent / "road_network_config.yaml"
        )
        self.config = self._load_road_extraction_config()

    def _get_dataset_name(self):
        """
        Get the name of the dataset for directory naming.

        Returns:
            str: Dataset name
        """
        return "ROAD_NETWORK"

    def _load_road_extraction_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Returns:
            Dict containing the configuration settings.
        """
        default_config = {
            "way_tag_resolver": {
                "tags": {
                    "motorway": {"clazz": 11, "maxspeed": 120, "flags": ["car"]},
                },
                # Default flags for bitmask
                "flag_list": ["car", "bike", "foot"],
            },
            "osm_pbf_file": None,  # Must be provided separately
            "output_dir": "sql_output_chunks",
        }

        try:
            with open(self.road_extraction_config_file) as f:
                config = yaml.safe_load(f)
                default_config.update(config)
                return default_config
        except FileNotFoundError:
            self.logger.warning(
                f"Configuration file '{self.road_extraction_config_file}' not found. " f"Using default values."
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
        flag_list = self.config.get("way_tag_resolver", {}).get("flag_list", ["car", "bike", "foot"])
        flag_bitmask = {flag: 1 << i for i, flag in enumerate(flag_list)}

        mask = 0
        if flags_set:
            for flag in flags_set:
                mask |= flag_bitmask.get(flag, 0)
        return mask

    def _normalize_list_value(self, value: Any) -> Any:
        """
        Normalize a value that might be a list to a single value.

        Args:
            value: The value to normalize (could be list, string, or None).

        Returns:
            Single value (first element if list, original value otherwise).
        """
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _resolve_way_tags(self, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Way tag resolution for the entire DataFrame.

        Args:
            edges_gdf: GeoDataFrame containing edges to process.

        Returns:
            GeoDataFrame with added clazz, kmh, and flags_set columns.
        """
        # Create lookup dictionaries from config
        way_tag_config = self.config.get("way_tag_resolver", {}).get("tags", {})

        clazz_map = {k: v.get("clazz", 0) for k, v in way_tag_config.items()}
        maxspeed_map = {k: v.get("maxspeed", 0) for k, v in way_tag_config.items()}
        flags_map = {k: set(v.get("flags", [])) for k, v in way_tag_config.items()}
        # Normalize highway values (handle lists)
        edges_gdf["highway_normalized"] = edges_gdf["highway"].apply(self._normalize_list_value)

        # Map highway types to clazz, maxspeed, and flags
        edges_gdf["clazz"] = edges_gdf["highway_normalized"].map(clazz_map).fillna(0).astype(int)
        edges_gdf["kmh"] = edges_gdf["highway_normalized"].map(maxspeed_map).fillna(0).astype(int)
        edges_gdf["flags_set"] = edges_gdf["highway_normalized"].map(flags_map)
        edges_gdf["flags_set"] = edges_gdf["flags_set"].apply(lambda x: x if pd.notna(x) else set())

        return edges_gdf

    def _extract_coordinates(self, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract start/end coordinates from geometry column.

        Args:
            edges: GeoDataFrame with geometry column.

        Returns:
            GeoDataFrame with added x1, y1, x2, y2 columns.
        """

        def get_coords(geom):
            try:
                coords = list(geom.coords)
                return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]
            except Exception:
                return None, None, None, None

        coord_data = edges["geometry"].apply(get_coords)
        edges[["x1", "y1", "x2", "y2"]] = pd.DataFrame(coord_data.tolist(), index=edges.index)
        return edges

    def _process_names(self, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Name/ref processing with SQL escaping.

        Args:
            edges: GeoDataFrame with name and ref columns.

        Returns:
            GeoDataFrame with added final_name column for SQL.
        """
        # Normalize name and ref columns if they exist
        if "name" in edges.columns:
            edges["name_norm"] = edges["name"].apply(self._normalize_list_value)
        else:
            edges["name_norm"] = None

        if "ref" in edges.columns:
            edges["ref_norm"] = edges["ref"].apply(self._normalize_list_value)
        else:
            edges["ref_norm"] = None

        # Create boolean masks for valid values
        name_valid = edges["name_norm"].notna() & (edges["name_norm"].astype(str).str.strip() != "")
        ref_valid = edges["ref_norm"].notna() & (edges["ref_norm"].astype(str).str.strip() != "")

        # Initialize with NULL
        edges["final_name"] = "NULL"

        # Prioritize name over ref, escape SQL quotes
        if name_valid.any():
            edges.loc[name_valid, "final_name"] = (
                "'" + edges.loc[name_valid, "name_norm"].astype(str).str.replace("'", "''", regex=False) + "'"
            )

        if ref_valid.any():
            edges.loc[~name_valid & ref_valid, "final_name"] = (
                "'"
                + edges.loc[~name_valid & ref_valid, "ref_norm"].astype(str).str.replace("'", "''", regex=False)
                + "'"
            )

        return edges

    def _normalize_osm_id(self, edges_gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Extract and normalize OSM ID from various possible columns.

        Args:
            edges_gdf: GeoDataFrame containing edges.

        Returns:
            Series of normalized OSM IDs.
        """
        if "osmid" in edges_gdf.columns:
            osmid_norm = edges_gdf["osmid"].apply(self._normalize_list_value)
            return osmid_norm.combine_first(edges_gdf.get("osm_id", edges_gdf.get("id", pd.Series(edges_gdf.index))))
        return edges_gdf.get("osm_id", edges_gdf.get("id", pd.Series(edges_gdf.index)))

    def _calculate_edge_metrics(self, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate length, distance and cost metrics for edges.

        Args:
            edges_gdf: GeoDataFrame containing edges.

        Returns:
            GeoDataFrame with calculated metrics.
        """
        # Calculate length in meters
        edges_gdf["length_m"] = edges_gdf["length"].fillna(0)

        # For rows where length is 0 or missing, calculate from geometry
        zero_length_mask = edges_gdf["length_m"] == 0
        if zero_length_mask.any():
            try:
                geom_lengths = edges_gdf.loc[zero_length_mask, "geometry"].length
                edges_gdf.loc[zero_length_mask, "length_m"] = geom_lengths.fillna(0)
            except Exception:
                edges_gdf.loc[zero_length_mask, "length_m"] = 0

        # Calculate derived metrics with division by zero protection
        edges_gdf["km"] = edges_gdf["length_m"] / METERS_TO_KM
        edges_gdf["kmh_safe"] = edges_gdf["kmh"].replace(0, MIN_SPEED_KMH)  # Avoid division by zero
        edges_gdf["cost"] = edges_gdf["km"] / edges_gdf["kmh_safe"]  # Hours
        edges_gdf["reverse_cost"] = edges_gdf["cost"]  # Always set to cost

        return edges_gdf

    def _prepare_sql_columns(self, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepare all columns needed for SQL generation.

        Args:
            edges_gdf: GeoDataFrame containing edges.

        Returns:
            GeoDataFrame with all SQL columns prepared.
        """
        # Process names and coordinates
        edges_gdf = self._process_names(edges_gdf)
        edges_gdf = self._extract_coordinates(edges_gdf)

        # Convert flags to integers
        edges_gdf["flags_int"] = edges_gdf["flags_set"].apply(self._flags_to_int)

        # Generate WKB geometry hex strings
        def safe_wkb_dumps(geom):
            try:
                return wkb_dumps(geom, hex=True, srid=4326)
            except Exception:
                return None

        edges_gdf["geom_hex"] = edges_gdf["geometry"].apply(safe_wkb_dumps)

        # Filter out rows with geometry errors
        valid_geom_mask = edges_gdf["geom_hex"].notna() & edges_gdf["x1"].notna()
        if not valid_geom_mask.all():
            edges_gdf = edges_gdf[valid_geom_mask].copy()

        if edges_gdf.empty:
            return edges_gdf

        # Fill missing values with defaults - batch operation
        int_columns = ["u", "v", "source", "target"]
        for col in int_columns:
            if col in edges_gdf.columns:
                edges_gdf[col] = edges_gdf[col].fillna(-1).astype(int)
            else:
                edges_gdf[col] = -1

        float_columns = ["cost", "reverse_cost"]
        for col in float_columns:
            edges_gdf[col] = edges_gdf[col].fillna(float("inf"))

        return edges_gdf

    def _generate_sql_tuples_vectorized(self, edges_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Generate SQL tuples using vectorized string operations for better performance.

        Args:
            edges_gdf: GeoDataFrame with all required columns.

        Returns:
            List of SQL value tuple strings.
        """
        # Pre-format all numeric columns that need specific precision
        formatted_data = {
            "km": edges_gdf["km"].map("{:.7f}".format),
            "cost": edges_gdf["cost"].map("{:.7f}".format),
            "reverse_cost": edges_gdf["reverse_cost"].map("{:.7f}".format),
            "x1": edges_gdf["x1"].map("{:.7f}".format),
            "y1": edges_gdf["y1"].map("{:.7f}".format),
            "x2": edges_gdf["x2"].map("{:.7f}".format),
            "y2": edges_gdf["y2"].map("{:.7f}".format),
        }

        # Build SQL strings using vectorized concatenation
        sql_tuples = (
            "("
            + edges_gdf.index.astype(str)
            + ", "
            + edges_gdf["osm_id"].astype(str)
            + ", "
            + edges_gdf["final_name"]
            + ", NULL, "
            + edges_gdf["u"].astype(str)
            + ", "
            + edges_gdf["v"].astype(str)
            + ", "
            + edges_gdf["clazz"].astype(str)
            + ", "
            + edges_gdf["flags_int"].astype(str)
            + ", "
            + edges_gdf["source"].astype(str)
            + ", "
            + edges_gdf["target"].astype(str)
            + ", "
            + formatted_data["km"]
            + ", "
            + edges_gdf["kmh"].astype(str)
            + ", "
            + formatted_data["cost"]
            + ", "
            + formatted_data["reverse_cost"]
            + ", "
            + formatted_data["x1"]
            + ", "
            + formatted_data["y1"]
            + ", "
            + formatted_data["x2"]
            + ", "
            + formatted_data["y2"]
            + ", '"
            + edges_gdf["geom_hex"]
            + "')"
        )

        return sql_tuples.tolist()

    def _build_sql_file(self, insert_value_tuples: List[str]) -> List[str]:
        """
        Build the SQL file.

        Args:
            insert_value_tuples: List of SQL value tuple strings.

        Returns:
            List of SQL strings.
        """
        full_sql_content = []
        full_sql_content.append(HEADER_SQL)

        if insert_value_tuples:
            # Chunk the insert statements into groups of 1000
            chunk_size = 1000

            # Insert statement template
            insert_prefix_parts = ["INSERT INTO road_network VALUES"]
            insert_prefix = "".join(insert_prefix_parts)

            # Generate chunked INSERT statements
            for i in range(0, len(insert_value_tuples), chunk_size):
                chunk = insert_value_tuples[i : i + chunk_size]
                chunk_insert = insert_prefix + "\n" + ",\n".join(chunk) + ";\n"
                full_sql_content.append(chunk_insert)

        else:
            self.logger.warning("No insert statements generated.")

        full_sql_content.append(INDEX_SQL)

        return full_sql_content

    def _process_and_write_edges(self, edges_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Process edge data for SQL generation.

        Args:
            edges_gdf: GeoDataFrame containing network edges.

        Returns:
            List of SQL value tuple strings.
        """
        if edges_gdf.empty:
            return []  # Return empty list

        # Resolve way tags
        edges_gdf = self._resolve_way_tags(edges_gdf)
        edges_gdf = edges_gdf[edges_gdf["clazz"] != 0]

        # Handle osm_id normalization
        edges_gdf["osm_id"] = self._normalize_osm_id(edges_gdf)

        # Calculate edge metrics
        edges_gdf = self._calculate_edge_metrics(edges_gdf)

        # Prepare SQL columns
        edges_gdf = self._prepare_sql_columns(edges_gdf)

        # Generate SQL tuples
        insert_value_tuples = self._generate_sql_tuples_vectorized(edges_gdf)

        # Build sql file
        full_sql_content = self._build_sql_file(insert_value_tuples)

        return full_sql_content

    def build_network(self) -> Dict[str, Any]:
        """
        Build a routable road network from OSM data.

        Returns:
            Dictionary containing results of the network building process:
                - edges: GeoDataFrame of edges
                - sql_file: Path to the generated SQL file
                - geojson_file: Path to the network GeoJSON file
        """
        results: Dict[str, Union[Path, None]] = {
            "nodes": None,
            "edges": None,
            "sql_file": None,
            "geojson_file": None,
        }

        osm = self.orchestrator.get_osm_parser()
        if osm is None:
            self.logger.error("OSM parser not available from orchestrator. Cannot build road network.")
            return results

        try:
            self.logger.info("Extracting network using pre-initialized OSM parser")
            nodes, edges_gdf = osm.get_network(nodes=True, network_type="driving")

            G = osm.to_graph(nodes, edges_gdf, graph_type="networkx")
            G_simplified = ox.simplification.simplify_graph(G)

            # Convert back to GeoDataFrames
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_simplified)

            # Optional: if needed in WGS84
            nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
            edges_gdf = edges_gdf.to_crs("EPSG:4326")

            edges_gdf = edges_gdf.reset_index()

            self.logger.info(
                f"Loaded {
                    len(nodes)} nodes and {
                    len(edges_gdf)} total edges."
            )

            unique_node_ids = pd.Series(pd.concat([edges_gdf["u"], edges_gdf["v"]]).unique())

            node_id_map = dict(zip(unique_node_ids, range(len(unique_node_ids))))

            # Step 3: Apply the mapping to your edge table
            edges_gdf["source"] = edges_gdf["u"].map(node_id_map)
            edges_gdf["target"] = edges_gdf["v"].map(node_id_map)

            # Export road network to a single GeoJSON file
            geojson_path = self.dataset_output_dir / "road_network.geojson"
            try:
                # Save to GeoJSON
                edges_gdf.to_file(geojson_path, driver="GeoJSON")
                self.logger.info(f"Saved road network to GeoJSON: {geojson_path}")
                results["geojson_file"] = geojson_path
            except Exception as e:
                self.logger.error(f"Error exporting network to GeoJSON: {e}")

            # Process the edges for SQL
            full_sql_content = self._process_and_write_edges(edges_gdf)

            output_sql_file = self.dataset_output_dir / "road_network.sql"
            try:
                with open(output_sql_file, "w", encoding="utf-8") as f:
                    # Add some spacing between main SQL sections
                    f.write("\n\n".join(full_sql_content))
                self.logger.info(f"All SQL commands written to {output_sql_file}")
                results["sql_file"] = output_sql_file
            except OSError as e:
                self.logger.error(f"Error writing full SQL file {output_sql_file}: {e}")

            # Store the edges in results
            results["edges"] = edges_gdf

        except Exception as e:
            self.logger.error(f"Error building road network: {e}")

        return results

    def download(self):
        """
        Required method for DataHandler - not implemented.
        This class does not directly download data.
        """
        raise NotImplementedError("Data downloading not implemented for this class")

    def process(self):
        """
        Process the data for the region.

        This method implements the complete data processing workflow
        for the road network, essentially a wrapper around build_network.

        Returns:
            dict: Dictionary containing processed data and file paths:
                - edges: GeoDataFrame of network edges
                - sql_file: Path to the generated SQL file
                - geojson_file: Path to the network GeoJSON file
        """
        # Define expected output file paths
        geojson_path = self.dataset_output_dir / "road_network.geojson"
        sql_path = self.dataset_output_dir / "road_network.sql"

        # Check if output files already exist
        if geojson_path.exists() and sql_path.exists():
            self.logger.info("Road network files already exist!")

            # Load existing edges data if available
            try:
                edges_gdf = gpd.read_file(geojson_path)
                self.logger.info(
                    f"Loaded existing road network with {
                        len(edges_gdf)} edges"
                )
            except Exception as e:
                self.logger.warning(f"Could not load existing GeoJSON file: {e}")
                edges_gdf = None

            return {
                "nodes": None,
                "edges": edges_gdf,
                "sql_file": sql_path,
                "geojson_file": geojson_path,
            }
        else:
            results = self.build_network()

            return results

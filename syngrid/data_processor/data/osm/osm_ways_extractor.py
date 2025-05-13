# Notes: this is an attempt to read the OSM data from a geofabrik .pbf
# file. And create a routable network from it.


# import re # Unused import
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
# from shapely.geometry import box # Unused import
import yaml
from pyrosm import OSM
from shapely.wkb import dumps as wkb_dumps
from tqdm import tqdm

from syngrid.data_processor.utils.utils import create_region_path

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

# --- Configuration Loading ---
CONFIG_FILE = 'osm_pbf_parser/osm2po_config.yaml'
DEFAULT_CONFIG = {
    'way_tag_resolver': {
        'tags': {
            "motorway": {"clazz": 11, "maxspeed": 120, "flags": ["car"]},
        },
        'final_mask': None,  # Default: no mask
        'flag_list': ["car", "bike", "foot"]  # Default flags for bitmask
    },
    'tileSize': 'x',  # Default: no tiling
    'network_type': 'driving',
    # MUST be provided
    'osm_pbf_file': '/Users/magic-rabbit/Downloads/osm2po/pbf_input/Andorra.pbf',
    'output_dir': 'sql_output_chunks',
    'total_bounds': None  # e.g., [min_lon, min_lat, max_lon, max_lat]
}


def load_config():
    """Load configuration from file or use defaults."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            DEFAULT_CONFIG.update(config)
            return DEFAULT_CONFIG
    except FileNotFoundError:
        logger.warning(
            f"Configuration file '{CONFIG_FILE}' not found. "
            f"Using default values."
        )
        return DEFAULT_CONFIG


def extract_osm_ways(osm_pbf_file=None, output_dir=None, fips_dict=None, network_type='driving'):
    """
    Extract OSM ways from a PBF file and process them for routing.

    Args:
        osm_pbf_file (str): Path to the OSM PBF file
        output_dir (str or Path): Directory to save output files
        fips_dict (dict): Dictionary containing FIPS codes for standardized path creation
        network_type (str): Type of network to extract (e.g., 'driving', 'walking', 'cycling')

    Returns:
        dict: Dictionary containing results of the extraction process
            - nodes: GeoDataFrame of nodes
            - edges: GeoDataFrame of edges
            - sql_file: Path to the generated SQL file
            - raw_edges_file: Path to the raw edges CSV file
    """
    # Load configuration
    config = load_config()

    # Override config with parameters
    if osm_pbf_file:
        config['osm_pbf_file'] = osm_pbf_file
    if network_type:
        config['network_type'] = network_type

    # Extract config values
    way_tag_config = config.get('way_tag_resolver', {}).get('tags', {})
    final_mask_str = config.get('way_tag_resolver', {}).get('final_mask')
    final_mask_flags = set(final_mask_str.split(',')) if final_mask_str else None
    flag_list = config.get('way_tag_resolver', {}).get('flag_list', ["car", "bike", "foot"])
    osm_pbf_file = config.get('osm_pbf_file')

    # Set up output directory
    if output_dir is None:
        if fips_dict is not None:
            # Use the standardized directory structure if FIPS dict is provided
            output_dir = create_region_path(fips_dict, "OSM_NETWORK")
        else:
            # Fall back to config output directory if no FIPS info
            output_dir = Path(config.get('output_dir', 'sql_output_chunks'))
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Results dictionary
    results = {
        'nodes': None,
        'edges': None,
        'sql_file': None,
        'raw_edges_file': None
    }

    if not osm_pbf_file:
        logger.error("No OSM PBF file provided")
        return results

    # Helper to create flag bitmask mapping
    flag_bitmask = {flag: 1 << i for i, flag in enumerate(flag_list)}

    def flags_to_int(flags_set):
        mask = 0
        if flags_set:
            for flag in flags_set:
                mask |= flag_bitmask.get(flag, 0)
        return mask

    def resolve_way_tags(tags_dict):
        """
        Resolves OSM way tags to internal clazz, maxspeed, and flags
        based on the loaded configuration.
        """
        highway_type = tags_dict.get("highway")
        if highway_type and highway_type in way_tag_config:
            cfg = way_tag_config[highway_type]
            flags = set(cfg.get('flags', []))
            return cfg.get("clazz", 0), cfg.get("maxspeed", 0), flags
        return 0, 0, set()

    def is_way_allowed_by_final_mask(flags_set):
        """Checks if a way's resolved flags intersect with the final_mask_flags."""
        if final_mask_flags is None:  # No mask defined, allow all
            return True
        if not flags_set:  # No flags resolved, disallow if mask exists
            return False
        return not final_mask_flags.isdisjoint(flags_set)

    def process_and_write_edges(edges_gdf, tile_identifier=""):
        """
        Applies tag resolution, filtering, cost calculation, and generates SQL value tuples.
        Returns a list of SQL VALUE tuple strings.

        Args:
            edges_gdf: GeoDataFrame containing network edges.
            tile_identifier: String or number to identify the tile (for logging).
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

            clazz, maxspeed, flags = resolve_way_tags(tags)

            if final_mask_flags and not is_way_allowed_by_final_mask(flags):
                logger.debug(
                    "Skipping way %s due to final mask. Flags: %s, Final Mask: %s",
                    way_id, flags, final_mask_flags
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
            flags_int = flags_to_int(row.get("flags_set", set()))
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

    # Main execution logic
    logger.info(f"Loading OSM data from: {osm_pbf_file}")
    try:
        osm = OSM(osm_pbf_file)
        nodes, edges_gdf = osm.get_network(network_type=network_type, nodes=True)
        logger.info(f"Loaded {len(nodes)} nodes and {len(edges_gdf)} total edges.")

        # Save raw edges for reference
        raw_edges_csv_path = output_dir / "raw_osm_network_edges.csv"
        try:
            edges_gdf.to_csv(raw_edges_csv_path, index=False)
            logger.info(f"Saved raw network edges to {raw_edges_csv_path}")
            results['raw_edges_file'] = raw_edges_csv_path
        except Exception as e:
            logger.error(f"Error saving raw network edges to CSV: {e}")

        # Process the edges
        insert_value_tuples = process_and_write_edges(edges_gdf, "ALL_DATA")

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

        output_sql_file = output_dir / "osm2po_full_setup.sql"
        try:
            with open(output_sql_file, "w", encoding="utf-8") as f:
                # Add some spacing between main SQL sections
                f.write("\n\n".join(full_sql_content))
            logger.info(f"All SQL commands written to {output_sql_file}")
            results['sql_file'] = output_sql_file
        except IOError as e:
            logger.error(f"Error writing full SQL file {output_sql_file}: {e}")

        # Store the nodes and edges in results
        results['nodes'] = nodes
        results['edges'] = edges_gdf

        logger.info(f"Processing finished. Output generated in: {output_dir}")

    except Exception as e:
        logger.error(f"Error extracting OSM ways: {e}")

    return results


if __name__ == "__main__":
    # Example usage
    # Example FIPS region dict
    fips_dict = {
        'state': 'Massachusetts',
        'state_fips': '25',
        'county': 'Suffolk',
        'county_fips': '025',
        'subdivision': 'Boston',
        'subdivision_fips': '07000'
    }

    # Default PBF file from config (customize as needed)
    config = load_config()
    osm_pbf_file = config.get('osm_pbf_file')

    # Extract OSM ways
    results = extract_osm_ways(
        osm_pbf_file=osm_pbf_file,
        fips_dict=fips_dict,
        network_type='driving'
    )

    # Print results - TODO: remove
    if results['nodes'] is not None:
        print(f"Nodes: {len(results['nodes'])} features")
    if results['edges'] is not None:
        print(f"Edges: {len(results['edges'])} features")
    if results['sql_file'] is not None:
        print(f"SQL file: {results['sql_file']}")
    if results['raw_edges_file'] is not None:
        print(f"Raw edges file: {results['raw_edges_file']}")

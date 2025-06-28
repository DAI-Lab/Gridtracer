#!/usr/bin/env python3
"""
Census Tract Analysis and Visualization Tool

This script provides a comprehensive tool for analyzing and visualizing data for a
specific US Census Tract from a GeoJSON file of census blocks.

It can perform the following functions:
1.  Aggregate census block data (population, housing) for a given tract.
2.  Calculate area, population/housing density, and generate geometry hashes.
3.  Generate an interactive HTML map using Folium.
4.  Generate a static PNG map using Matplotlib.

How to Run:
The script requires a GeoJSON file containing census blocks and a specific
TRACTCE20 code to analyze. The output paths for the maps are optional.

The GeoJSON file for a given region is typically found in the project's output
directory after running other processing steps.

Example Command:
    python -m gridtracer.analysis.tract_analysis \\
        gridtracer/data_processor/output/MA/Middlesex_County/Cambridge_city/CENSUS/target_region_blocks.geojson \\
        354200 \\
        --output-html tract_354200_interactive.html \\
        --output-png tract_354200_static.png
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union


def calculate_area_km2(gdf: gpd.GeoDataFrame) -> float:
    """Calculate area in square kilometers."""
    if gdf.crs and gdf.crs.is_geographic:
        gdf_projected = gdf.to_crs('EPSG:5070')
    else:
        gdf_projected = gdf
    area_m2 = gdf_projected.geometry.area.sum()
    return area_m2 / 1_000_000


def generate_geometry_hash(geometry) -> str:
    """Generate a SHA-256 hash of the geometry's WKT representation."""
    wkt_str = str(geometry.wkt).encode('utf-8')
    return hashlib.sha256(wkt_str).hexdigest()


def generate_wkb_hex(geometry, source_crs: str, target_crs: str = 'EPSG:5070') -> str:
    """Generate WKB hex representation of the geometry in the target CRS."""
    projected_gds = gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs)
    return projected_gds.iloc[0].wkb_hex


def aggregate_tract_data(geojson_path: str, tract_code: str) -> Dict[str, Any]:
    """Aggregate census block data for a specific tract."""
    try:
        print(f"Reading GeoJSON file: {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError("GeoJSON file is empty.")

        print(f"Loaded {len(gdf)} total blocks.")
        tract_blocks = gdf[gdf['TRACTCE20'] == tract_code].copy()
        if tract_blocks.empty:
            raise ValueError(f"No blocks found for tract code: {tract_code}")

        print(f"Found {len(tract_blocks)} blocks for tract {tract_code}")

        total_population = tract_blocks['POP20'].sum()
        total_housing = tract_blocks['HOUSING20'].sum(
        ) if 'HOUSING20' in tract_blocks.columns else 0

        union_geometry = unary_union(tract_blocks.geometry)
        area_km2 = calculate_area_km2(tract_blocks)
        source_crs = str(gdf.crs) if gdf.crs else 'EPSG:4269'

        return {
            'tract_code': tract_code,
            'total_population': int(total_population),
            'total_housing_units': int(total_housing),
            'area_km2': round(area_km2, 4),
            'num_blocks': len(tract_blocks),
            'geometry_hash': generate_geometry_hash(union_geometry),
            'geometry_wkb_hex': generate_wkb_hex(union_geometry, source_crs),
            'union_geometry': union_geometry,
            'centroid': union_geometry.centroid,
            'blocks_data': tract_blocks,
            'crs': source_crs,
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {geojson_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing data: {e}")


def print_summary(result: Dict[str, Any]) -> None:
    """Print a formatted summary of the aggregated tract data."""
    summary = f"""
============================================================
CENSUS TRACT {result['tract_code']} SUMMARY
============================================================
Total Population:     {result['total_population']:,}
Total Housing Units:  {result['total_housing_units']:,}
Area:                 {result['area_km2']} km²
Number of Blocks:     {result['num_blocks']}
Coordinate System:    {result['crs']}
Geometry Hash:        {result['geometry_hash']}
WKB Hex (EPSG:5070):  {result['geometry_wkb_hex']}
============================================================
"""
    print(summary)
    if result['area_km2'] > 0:
        pop_density = result['total_population'] / result['area_km2']
        print(f"Population Density:   {pop_density:.1f} people/km²")
        if result['total_housing_units'] > 0:
            housing_density = result['total_housing_units'] / result['area_km2']
            print(f"Housing Density:      {housing_density:.1f} units/km²")
    print("=" * 60)


def create_interactive_map(result: Dict[str, Any], output_path: str) -> None:
    """Create an interactive Folium map of the tract."""
    print(f"Generating interactive map for tract {result['tract_code']}...")
    centroid = result['centroid']
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles='OpenStreetMap')
    folium.TileLayer('CartoDB positron', name="CartoDB Positron").add_to(m)
    folium.TileLayer('CartoDB dark_matter', name="CartoDB Dark Matter").add_to(m)

    # Add Tract Boundary
    union_gdf = gpd.GeoDataFrame([1], geometry=[result['union_geometry']], crs=result['crs'])
    popup_html = f"""
    <b>Census Tract {result['tract_code']}</b><br>
    Population: {result['total_population']:,}<br>
    Housing Units: {result['total_housing_units']:,}<br>
    Area: {result['area_km2']} km²
    """
    folium.GeoJson(
        union_gdf.to_json(),
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 3,
            'fillOpacity': 0.3},
        popup=folium.Popup(popup_html, max_width=300),
        name=f"Tract {result['tract_code']} Outline",
        tooltip=f"Tract {result['tract_code']}"
    ).add_to(m)

    # Add Individual Blocks
    blocks_gdf = result['blocks_data']
    blocks_fg = folium.FeatureGroup(name='Individual Census Blocks', show=True)
    for _, block in blocks_gdf.iterrows():
        block_popup_html = f"""
        <b>Block {block['BLOCKCE20']}</b><br>
        Population: {block['POP20']}<br>
        Housing: {block.get('HOUSING20', 0)}<br>
        Area (land): {block.get('ALAND20', 0):,} m²
        """
        folium.GeoJson(
            gpd.GeoDataFrame([block.geometry], columns=['geometry'], crs=result['crs']).to_json(),
            style_function=lambda x: {
                'fillColor': 'blue', 'color': 'navy', 'weight': 1, 'fillOpacity': 0.2, 'opacity': 0.6
            },
            popup=folium.Popup(block_popup_html, max_width=250),
            tooltip=f"Block {block['BLOCKCE20']}"
        ).add_to(blocks_fg)
    blocks_fg.add_to(m)

    folium.Marker(
        [centroid.y, centroid.x],
        popup=f"Tract {result['tract_code']} Centroid",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    folium.LayerControl().add_to(m)

    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")


def create_static_map(result: Dict[str, Any], output_path: str) -> None:
    """Create a static Matplotlib plot of the tract."""
    print(f"Generating static map for tract {result['tract_code']}...")
    union_gdf = gpd.GeoDataFrame([1], geometry=[result['union_geometry']], crs=result['crs'])
    tract_mercator = union_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 12))
    tract_mercator.plot(ax=ax, edgecolor='red', facecolor='red', alpha=0.4, linewidth=2.5)

    label_text = (
        f"Tract: {result['tract_code']}\\n"
        f"Population: {result['total_population']:,}\\n"
        f"Housing: {result['total_housing_units']:,}"
    )
    centroid = tract_mercator.geometry.iloc[0].centroid
    ax.text(
        centroid.x, centroid.y, label_text,
        ha='center', va='center', fontsize=12, color='black', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(f"Census Tract Outline: {result['tract_code']}", fontsize=16)
    ax.set_axis_off()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static map saved to: {output_path}")


def main() -> None:
    """Main function to handle command-line arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize a specific US Census Tract.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__  # Use the module docstring as the epilog
    )
    parser.add_argument('geojson_path', help='Path to the GeoJSON file with census blocks')
    parser.add_argument('tract_code', help='TRACTCE20 code to analyze (e.g., 354200)')
    parser.add_argument(
        '--output-html',
        help='Output path for the interactive HTML map (optional)')
    parser.add_argument('--output-png', help='Output path for the static PNG map (optional)')

    args = parser.parse_args()

    if not Path(args.geojson_path).exists():
        print(f"Error: GeoJSON file not found at {args.geojson_path}")
        sys.exit(1)

    try:
        result = aggregate_tract_data(args.geojson_path, args.tract_code)
        print_summary(result)

        if args.output_html:
            create_interactive_map(result, args.output_html)
        if args.output_png:
            create_static_map(result, args.output_png)

        if not args.output_html and not args.output_png:
            print("\\nWarning: No output paths provided. Only summary was printed.")

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

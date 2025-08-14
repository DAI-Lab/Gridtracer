#!/usr/bin/env python3
"""
Census Tract Analysis and Visualization Tool

This script provides a comprehensive tool for analyzing and visualizing data for a
specific US Census Tract, a single Census Block, or ALL tracts from a GeoJSON file.

It can perform the following functions:
1.  Aggregate census block data (population, housing) for a given tract.
2.  Filter data for a single census block if a block code is provided.
3.  Analyze ALL distinct tracts in a file if no tract code is specified.
4.  Calculate area, population/housing density, and generate geometry hashes.
5.  Generate an interactive HTML map using Folium.
6.  Generate a static PNG map using Matplotlib.

How to Run:
The script requires a GeoJSON file containing census blocks. You can either analyze
a specific tract, a single block within a tract, or all tracts in the file.

Example Command (All Tracts):
    python -m gridtracer.analysis.tract_analysis \\
        gridtracer/output/MA/Middlesex_County/Cambridge_city/CENSUS/target_region_blocks.geojson

Example Command (Specific Tract):
    python -m gridtracer.analysis.tract_analysis \\
        gridtracer/output/MA/Middlesex_County/Cambridge_city/CENSUS/target_region_blocks.geojson \\
        --tract-code 354200

Example Command (Specific Block):
    python -m gridtracer.analysis.tract_analysis \\
        gridtracer/output/MA/Middlesex_County/Cambridge_city/CENSUS/target_region_blocks.geojson \\
        --tract-code 354200 \\
        --block-code 2024
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def calculate_area_km2(gdf: gpd.GeoDataFrame) -> float:
    """Calculate area in square kilometers.

    Args:
        gdf: GeoDataFrame to calculate area for

    Returns:
        Area in square kilometers
    """
    if gdf.crs and gdf.crs.is_geographic:
        gdf_projected = gdf.to_crs('EPSG:5070')
    else:
        gdf_projected = gdf
    area_m2 = gdf_projected.geometry.area.sum()
    return area_m2 / 1_000_000


def generate_geometry_hash(geometry: BaseGeometry) -> str:
    """Generate a SHA-256 hash of the geometry's WKT representation.

    Args:
        geometry: Shapely geometry object

    Returns:
        SHA-256 hash as hexadecimal string
    """
    wkt_str = str(geometry.wkt).encode('utf-8')
    return hashlib.sha256(wkt_str).hexdigest()


def generate_wkb_hex(
    geometry: BaseGeometry, source_crs: str, target_crs: str = 'EPSG:5070'
) -> str:
    """Generate WKB hex representation of the geometry in the target CRS.

    Args:
        geometry: Shapely geometry object
        source_crs: Source coordinate reference system
        target_crs: Target coordinate reference system

    Returns:
        WKB hex representation of the geometry
    """
    projected_gds = gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs)
    return projected_gds.iloc[0].wkb_hex


def analyze_all_tracts(geojson_path: str) -> Dict[str, Any]:
    """Analyze all distinct tracts in the GeoJSON file.

    Args:
        geojson_path: Path to the GeoJSON file

    Returns:
        Dictionary containing analysis results for all tracts
    """
    try:
        print(f"Reading GeoJSON file: {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError("GeoJSON file is empty.")

        print(f"Loaded {len(gdf)} total blocks.")

        # Get all unique tract codes
        unique_tracts = gdf['TRACTCE20'].unique()
        print(f"Found {len(unique_tracts)} distinct tracts: {sorted(unique_tracts)}")

        tract_summaries = []
        all_geometries = []
        source_crs = str(gdf.crs) if gdf.crs else 'EPSG:4269'

        for tract_code in sorted(unique_tracts):
            tract_blocks = gdf[gdf['TRACTCE20'] == tract_code].copy()

            total_population = tract_blocks['POP20'].sum()
            total_housing = (
                tract_blocks['HOUSING20'].sum() if 'HOUSING20' in tract_blocks.columns else 0
            )

            union_geometry = unary_union(tract_blocks.geometry)
            area_km2 = calculate_area_km2(tract_blocks)

            tract_summary = {
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
            }
            tract_summaries.append(tract_summary)
            all_geometries.append(union_geometry)

        # Calculate overall statistics
        total_pop = sum(t['total_population'] for t in tract_summaries)
        total_housing = sum(t['total_housing_units'] for t in tract_summaries)
        total_area = sum(t['area_km2'] for t in tract_summaries)

        # Create combined geometry
        combined_geometry = unary_union(all_geometries)

        return {
            'analysis_type': 'all_tracts',
            'total_tracts': len(unique_tracts),
            'total_blocks': len(gdf),
            'total_population': total_pop,
            'total_housing_units': total_housing,
            'total_area_km2': round(total_area, 4),
            'tract_summaries': tract_summaries,
            'combined_geometry': combined_geometry,
            'combined_centroid': combined_geometry.centroid,
            'crs': source_crs,
            'geometry_hash': generate_geometry_hash(combined_geometry),
            'geometry_wkb_hex': generate_wkb_hex(combined_geometry, source_crs),
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {geojson_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing data: {e}")


def aggregate_tract_data(
    geojson_path: str, tract_code: str, block_code: Optional[str] = None
) -> Dict[str, Any]:
    """Aggregate census block data for a specific tract or a single block within it.

    Args:
        geojson_path: Path to the GeoJSON file
        tract_code: Census tract code to analyze
        block_code: Optional census block code within the tract

    Returns:
        Dictionary containing aggregated data for the tract or block
    """
    try:
        print(f"Reading GeoJSON file: {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError("GeoJSON file is empty.")

        print(f"Loaded {len(gdf)} total blocks.")
        tract_blocks = gdf[gdf['TRACTCE20'] == tract_code].copy()
        if tract_blocks.empty:
            raise ValueError(f"No blocks found for tract code: {tract_code}")

        if block_code:
            level_label = f"Block {block_code}"
            print(f"Filtering for {level_label} in tract {tract_code}")
            target_data = tract_blocks[tract_blocks['BLOCKCE20'] == block_code].copy()
            if target_data.empty:
                raise ValueError(f"No block found with code {block_code} in tract {tract_code}")
        else:
            level_label = f"Tract {tract_code}"
            target_data = tract_blocks
            print(f"Found {len(target_data)} blocks for {level_label}")

        total_population = target_data['POP20'].sum()
        total_housing = (
            target_data['HOUSING20'].sum() if 'HOUSING20' in target_data.columns else 0
        )

        union_geometry = unary_union(target_data.geometry)
        area_km2 = calculate_area_km2(target_data)
        source_crs = str(gdf.crs) if gdf.crs else 'EPSG:4269'

        return {
            'analysis_type': 'single_tract_or_block',
            'level_label': level_label,
            'tract_code': tract_code,
            'block_code': block_code,
            'total_population': int(total_population),
            'total_housing_units': int(total_housing),
            'area_km2': round(area_km2, 4),
            'num_blocks': len(target_data),
            'geometry_hash': generate_geometry_hash(union_geometry),
            'geometry_wkb_hex': generate_wkb_hex(union_geometry, source_crs),
            'union_geometry': union_geometry,
            'centroid': union_geometry.centroid,
            'blocks_data': target_data,  # This will be all blocks for a tract, or one for a block
            'crs': source_crs,
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {geojson_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing data: {e}")


def print_all_tracts_summary(result: Dict[str, Any]) -> None:
    """Print a formatted summary for all tracts analysis.

    Args:
        result: Dictionary containing analysis results for all tracts
    """
    print(f"""
============================================================
ALL CENSUS TRACTS SUMMARY
============================================================
Total Tracts:         {result['total_tracts']}
Total Blocks:         {result['total_blocks']:,}
Total Population:     {result['total_population']:,}
Total Housing Units:  {result['total_housing_units']:,}
Total Area:           {result['total_area_km2']} km²
Coordinate System:    {result['crs']}
Combined Geometry Hash: {result['geometry_hash']}
WKB Hex (EPSG:5070):  {result['geometry_wkb_hex']}
============================================================
""")

    if result['total_area_km2'] > 0:
        pop_density = result['total_population'] / result['total_area_km2']
        print(f"Overall Pop Density:  {pop_density:.1f} people/km²")
        if result['total_housing_units'] > 0:
            housing_density = result['total_housing_units'] / result['total_area_km2']
            print(f"Overall Housing Density: {housing_density:.1f} units/km²")

    print("\nINDIVIDUAL TRACT DETAILS:")
    print("-" * 60)
    for tract in result['tract_summaries']:
        pop_density = tract['total_population'] / tract['area_km2'] if tract['area_km2'] > 0 else 0
        print(f"Tract {tract['tract_code']:>6} | Pop: {tract['total_population']:>6,} | "
              f"Housing: {tract['total_housing_units']:>6,} | Area: {tract['area_km2']:>8} km² | "
              f"Blocks: {tract['num_blocks']:>3} | Density: {pop_density:>6.1f} people/km²")
    print("=" * 60)


def print_summary(result: Dict[str, Any]) -> None:
    """Print a formatted summary of the aggregated tract or block data.

    Args:
        result: Dictionary containing analysis results
    """
    title = f"TRACT {result['tract_code']}"
    if result['block_code']:
        title = f"BLOCK {result['block_code']} (in Tract {result['tract_code']})"

    summary = f"""
============================================================
CENSUS {title} SUMMARY
============================================================
Total Population:     {result['total_population']:,}
Total Housing Units:  {result['total_housing_units']:,}
Area:                 {result['area_km2']} km²
"""
    if not result['block_code']:
        summary += f"Number of Blocks:     {result['num_blocks']}\n"

    summary += f"""Coordinate System:    {result['crs']}
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


def create_all_tracts_interactive_map(result: Dict[str, Any], output_path: str) -> None:
    """Create an interactive Folium map showing all tracts.

    Args:
        result: Dictionary containing analysis results for all tracts
        output_path: Path where to save the HTML map
    """
    print(f"Generating interactive map for {result['total_tracts']} tracts...")

    centroid = result['combined_centroid']
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='OpenStreetMap')
    folium.TileLayer('CartoDB positron', name="CartoDB Positron").add_to(m)
    folium.TileLayer('CartoDB dark_matter', name="CartoDB Dark Matter").add_to(m)

    # Color palette for different tracts
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    # Add each tract as a separate layer
    for i, tract in enumerate(result['tract_summaries']):
        color = colors[i % len(colors)]
        tract_gdf = gpd.GeoDataFrame([1], geometry=[tract['union_geometry']], crs=result['crs'])

        popup_html = f"""
        <b>Census Tract {tract['tract_code']}</b><br>
        Population: {tract['total_population']:,}<br>
        Housing Units: {tract['total_housing_units']:,}<br>
        Area: {tract['area_km2']} km²<br>
        Blocks: {tract['num_blocks']}<br>
                 Pop Density: {tract['total_population'] / tract['area_km2']:.1f} people/km²
        """

        folium.GeoJson(
            tract_gdf.to_json(),
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.4
            },
            popup=folium.Popup(popup_html, max_width=300),
            name=f"Tract {tract['tract_code']}",
            tooltip=f"Tract {tract['tract_code']}"
        ).add_to(m)

        # Add centroid marker
        folium.Marker(
            [tract['centroid'].y, tract['centroid'].x],
            popup=f"Tract {tract['tract_code']} Center",
            icon=folium.Icon(color=color, icon='info-sign', icon_size=(8, 8))
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")


def create_interactive_map(result: Dict[str, Any], output_path: str) -> None:
    """Create an interactive Folium map of the tract or block.

    Args:
        result: Dictionary containing analysis results
        output_path: Path where to save the HTML map
    """
    level_type = "Block" if result['block_code'] else "Tract"
    code = result['block_code'] if result['block_code'] else result['tract_code']
    print(f"Generating interactive map for {level_type} {code}...")

    centroid = result['centroid']
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=15, tiles='OpenStreetMap')
    folium.TileLayer('CartoDB positron', name="CartoDB Positron").add_to(m)
    folium.TileLayer('CartoDB dark_matter', name="CartoDB Dark Matter").add_to(m)

    # Add Tract/Block Boundary
    union_gdf = gpd.GeoDataFrame([1], geometry=[result['union_geometry']], crs=result['crs'])
    popup_html = f"""
    <b>Census {level_type} {code}</b><br>
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
        name=f"{level_type} {code} Outline",
        tooltip=f"{level_type} {code}"
    ).add_to(m)

    # Add Individual Blocks layer only for tract-level analysis
    if not result['block_code']:
        blocks_gdf = result['blocks_data']
        blocks_fg = folium.FeatureGroup(name='Individual Census Blocks', show=True)
        for _, block in blocks_gdf.iterrows():
            block_popup_html = f"""
            <b>Block {block['BLOCKCE20']}</b><br>
            Population: {block['POP20']}<br>
            Housing: {block.get('HOUSING20', 0)}<br>
            Area (land): {block.get('ALAND20', 0):,} m²
            """
            block_gdf = gpd.GeoDataFrame(
                [block.geometry], columns=['geometry'], crs=result['crs']
            )
            folium.GeoJson(
                block_gdf.to_json(),
                style_function=lambda x: {
                    'fillColor': 'blue', 'color': 'navy', 'weight': 1,
                    'fillOpacity': 0.2, 'opacity': 0.6
                },
                popup=folium.Popup(block_popup_html, max_width=250),
                tooltip=f"Block {block['BLOCKCE20']}"
            ).add_to(blocks_fg)
        blocks_fg.add_to(m)

    folium.Marker(
        [centroid.y, centroid.x],
        popup=f"{level_type} {code} Centroid",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    folium.LayerControl().add_to(m)

    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")


def create_all_tracts_static_map(result: Dict[str, Any], output_path: str) -> None:
    """Create a static Matplotlib plot showing all tracts.

    Args:
        result: Dictionary containing analysis results for all tracts
        output_path: Path where to save the PNG map
    """
    print(f"Generating static map for {result['total_tracts']} tracts...")

    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot each tract with different colors
    colors = plt.cm.Set3(range(len(result['tract_summaries'])))
    for i, (tract, color) in enumerate(zip(result['tract_summaries'], colors)):
        tract_gdf = gpd.GeoDataFrame([1], geometry=[tract['union_geometry']], crs=result['crs'])
        tract_mercator = tract_gdf.to_crs(epsg=3857)
        tract_mercator.plot(ax=ax, edgecolor='black', facecolor=color, alpha=0.6, linewidth=1.5)

        # Add tract labels at centroids
        centroid_series = gpd.GeoSeries([tract['centroid']], crs=result['crs'])
        centroid_mercator = centroid_series.to_crs(epsg=3857).iloc[0]
        ax.annotate(
            f"T{tract['tract_code']}",
            (centroid_mercator.x, centroid_mercator.y),
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3')
        )

    title = f"All Census Tracts Overview ({result['total_tracts']} tracts)"
    label_text = (
        f"Total Tracts: {result['total_tracts']}\n"
        f"Total Population: {result['total_population']:,}\n"
        f"Total Housing: {result['total_housing_units']:,}\n"
        f"Total Area: {result['total_area_km2']} km²"
    )

    # Place text at the top-center of the plot
    ax.text(
        0.5, 0.98, label_text,
        transform=ax.transAxes,
        ha='center', va='top', fontsize=14, color='black', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_axis_off()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static map saved to: {output_path}")


def create_static_map(result: Dict[str, Any], output_path: str) -> None:
    """Create a static Matplotlib plot of the tract or block.

    Args:
        result: Dictionary containing analysis results
        output_path: Path where to save the PNG map
    """
    level_type = "Block" if result['block_code'] else "Tract"
    code = result['block_code'] if result['block_code'] else result['tract_code']
    print(f"Generating static map for {level_type} {code}...")

    union_gdf = gpd.GeoDataFrame([1], geometry=[result['union_geometry']], crs=result['crs'])
    tract_mercator = union_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 12))
    tract_mercator.plot(ax=ax, edgecolor='red', facecolor='red', alpha=0.4, linewidth=2.5)

    if result['block_code']:
        title = f"Census Block Outline: {result['block_code']} (Tract: {result['tract_code']})"
        label_text = (
            f"Block: {result['block_code']}\n"
            f"Population: {result['total_population']:,}\n"
            f"Housing: {result['total_housing_units']:,}"
        )
    else:
        title = f"Census Tract Outline: {result['tract_code']}"
        label_text = (
            f"Tract: {result['tract_code']}\n"
            f"Population: {result['total_population']:,}\n"
            f"Housing: {result['total_housing_units']:,}"
        )

    # Place text at the top-center of the plot
    ax.text(
        0.5, 0.98, label_text,
        transform=ax.transAxes,
        ha='center', va='top', fontsize=12, color='black', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static map saved to: {output_path}")


def main() -> None:
    """Main function to handle command-line arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize US Census Tract(s) or Blocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__  # Use the module docstring as the epilog
    )
    parser.add_argument('geojson_path', help='Path to the GeoJSON file with census blocks')
    parser.add_argument(
        '--tract-code',
        help='TRACTCE20 code to analyze (e.g., 354200). If not provided, all tracts are analyzed.'
    )
    parser.add_argument(
        '--block-code',
        type=str,
        help='BLOCKCE20 code to analyze a single block within the tract (requires --tract-code)',
    )
    parser.add_argument(
        '--output-html',
        help='Output path for the interactive HTML map (optional)')
    parser.add_argument('--output-png', help='Output path for the static PNG map (optional)')

    args = parser.parse_args()

    if not Path(args.geojson_path).exists():
        print(f"Error: GeoJSON file not found at {args.geojson_path}")
        sys.exit(1)

    if args.block_code and not args.tract_code:
        print("Error: --block-code requires --tract-code to be specified")
        sys.exit(1)

    try:
        if args.tract_code:
            # Analyze specific tract or block
            result = aggregate_tract_data(
                args.geojson_path, args.tract_code, args.block_code
            )
            print_summary(result)

            if args.output_html:
                create_interactive_map(result, args.output_html)
            if args.output_png:
                create_static_map(result, args.output_png)
        else:
            # Analyze all tracts
            result = analyze_all_tracts(args.geojson_path)
            print_all_tracts_summary(result)

            if args.output_html:
                create_all_tracts_interactive_map(result, args.output_html)
            if args.output_png:
                create_all_tracts_static_map(result, args.output_png)

        if not args.output_html and not args.output_png:
            print("\\nWarning: No output paths provided. Only summary was printed.")

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

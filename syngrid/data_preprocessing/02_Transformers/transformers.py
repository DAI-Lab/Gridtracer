import csv
import json
import os

import click
import pyproj
import requests
import us
from shapely.geometry import Point
from shapely.ops import transform

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Template for Overpass query
QUERY_TEMPLATE = """
[out:json][timeout:500];
area["name"="{state_name}"]->.searchArea;
(
  node["power"="transformer"]["abandoned"!~"yes"]["abandoned:substation"!~"y"](area.searchArea);
  way["power"="transformer"]["abandoned:building"!~"transformer"](area.searchArea);
  relation["power"="transformer"](area.searchArea);
  node["power"="substation"]["abandoned"!~"yes"]["abandoned:substation"!~"y"](area.searchArea);
  way["power"="substation"]["abandoned:building"!~"transformer"](area.searchArea);
  relation["power"="substation"](area.searchArea);
  node["power"="pole"]["transformer"="distribution"](area.searchArea);
);
out body;
>;
out skel qt;
"""


def fetch_osm_data(state_abbr, state_name):
    print(f"Fetching data for {state_name} ({state_abbr})...")
    query = QUERY_TEMPLATE.format(state_name=state_name)
    response = requests.post(OVERPASS_URL, data={"data": query})
    response.raise_for_status()
    return response.json()


def deduplicate_nearby_features(geojson_data, distance_threshold_meters=10):
    """
    Deduplicate features that are within distance_threshold_meters of each other.
    Prioritizes ways over nodes, and takes the first node if both are nodes.
    Args:
        geojson_data (dict): The GeoJSON data
        distance_threshold_meters (float): Distance threshold in meters
    Returns:
        dict: Deduplicated GeoJSON data
    """
    # Create a projection for accurate distance calculation
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:32633')  # UTM zone 33N, adjust based on your location
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    # Convert features to Shapely Points and store their indices
    points = []
    for idx, feature in enumerate(geojson_data['features']):
        if feature['geometry']['type'] == 'Point':
            coords = feature['geometry']['coordinates']
            point = Point(coords)
            # Transform to UTM for accurate distance calculation
            point_utm = transform(project, point)
            points.append({
                'index': idx,
                'point': point_utm,
                'type': feature['properties']['type'],
                'feature': feature
            })

    # Find duplicates
    to_remove = set()
    for i, point1 in enumerate(points):
        if i in to_remove:
            continue

        for j, point2 in enumerate(points[i + 1:], i + 1):
            if j in to_remove:
                continue

            # Calculate distance in meters
            distance = point1['point'].distance(point2['point'])

            if distance <= distance_threshold_meters:
                # Determine which feature to keep
                if point1['type'] == 'Way' and point2['type'] == 'Node':
                    to_remove.add(j)
                elif point1['type'] == 'Node' and point2['type'] == 'Way':
                    to_remove.add(i)
                    break  # Break inner loop as point1 is being removed
                else:
                    # If both are the same type, keep the first one
                    to_remove.add(j)

    # Create new features list excluding duplicates
    new_features = [
        feature for idx, feature in enumerate(geojson_data['features'])
        if idx not in to_remove
    ]

    # Create new GeoJSON with deduplicated features
    deduplicated_geojson = {
        'type': geojson_data['type'],
        'name': geojson_data['name'],
        'crs': geojson_data['crs'],
        'features': new_features
    }

    print(f"Removed {len(to_remove)} duplicate features")
    return deduplicated_geojson


def save_files(data, state_abbr, state_fips):
    # Create output directory structure
    base_dir = os.path.join("syngrid", "data_preprocessing", "data_output")
    output_dir = os.path.join(base_dir, "transformers", f"US-{state_abbr}")
    os.makedirs(output_dir, exist_ok=True)

    # Save raw response
    raw_path = os.path.join(output_dir, f"US-{state_abbr}_{state_fips}_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw response to {raw_path}")

    # Transform data into GeoJSON format
    geojson_data = {
        "type": "FeatureCollection",
        "name": f"US-{state_abbr}",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": []
    }

    # Create a lookup dictionary for nodes
    node_lookup = {
        element["id"]: element
        for element in data.get("elements", [])
        if element.get("type") == "node"
    }

    def get_unique_coords(coords_list):
        """Remove duplicate coordinates while preserving order."""
        seen = set()
        unique_coords = []
        for coord in coords_list:
            # Round coordinates to 6 decimal places to handle floating point differences
            coord_tuple = (round(coord[0], 6), round(coord[1], 6))
            if coord_tuple not in seen:
                seen.add(coord_tuple)
                unique_coords.append(coord)
        return unique_coords

    # Process each element and convert to GeoJSON Feature
    for element in data.get("elements", []):
        if element.get("type") in ["node", "way", "relation"]:
            if not element.get("tags"):
                continue

            # Get coordinates based on geometry type
            if element.get("type") == "node":
                coords = [element.get("lon"), element.get("lat")]
                geom_type = "Point"
            else:
                # For ways and relations, we'll calculate the centroid
                if "nodes" in element and element["nodes"]:
                    # Get all node coordinates
                    node_coords = []
                    for node_id in element["nodes"]:
                        if node_id in node_lookup:
                            node = node_lookup[node_id]
                            node_coords.append([node.get("lon"), node.get("lat")])

                    # Remove duplicate coordinates
                    node_coords = get_unique_coords(node_coords)

                    if node_coords:
                        # Calculate centroid
                        lons = [coord[0] for coord in node_coords]
                        lats = [coord[1] for coord in node_coords]
                        coords = [sum(lons) / len(lons), sum(lats) / len(lats)]
                        geom_type = "Point"  # Changed to Point since we're using centroid
                    else:
                        coords = []
                        geom_type = "Point" if element.get("type") == "node" else (
                            "LineString" if element.get("type") == "way" else "Polygon"
                        )
                elif element.get("type") == "relation" and "members" in element:
                    # For relations, collect coordinates from all member nodes
                    member_coords = []
                    for member in element.get("members", []):
                        if member.get("type") == "node" and member.get("ref") in node_lookup:
                            node = node_lookup[member["ref"]]
                            member_coords.append([
                                node.get("lon"),
                                node.get("lat")
                            ])
                        elif member.get("type") == "way":
                            # Find the way in elements
                            for way in data.get("elements", []):
                                if way.get("type") == "way" and way.get("id") == member.get("ref"):
                                    # Get coordinates for all nodes in this way
                                    for node_id in way.get("nodes", []):
                                        if node_id in node_lookup:
                                            node = node_lookup[node_id]
                                            member_coords.append([
                                                node.get("lon"),
                                                node.get("lat")
                                            ])

                    # Remove duplicate coordinates
                    member_coords = get_unique_coords(member_coords)

                    if member_coords:
                        # Calculate centroid from all member coordinates
                        lons = [coord[0] for coord in member_coords]
                        lats = [coord[1] for coord in member_coords]
                        coords = [sum(lons) / len(lons), sum(lats) / len(lats)]
                        geom_type = "Point"  # Changed to Point since we're using centroid
                    else:
                        coords = []
                        geom_type = "Point" if element.get("type") == "node" else (
                            "LineString" if element.get("type") == "way" else "Polygon"
                        )
                else:
                    coords = []
                    geom_type = "Point" if element.get("type") == "node" else (
                        "LineString" if element.get("type") == "way" else "Polygon"
                    )

            feature = {
                "type": "Feature",
                "properties": {
                    "id": element.get("id"),
                    "type": element.get("type").capitalize(),
                    "tags": element.get("tags", {}),
                    "name": element.get("tags", {}).get("name", ""),
                    "operator": element.get("tags", {}).get("operator", ""),
                    "power": element.get("tags", {}).get("power", "")
                },
                "geometry": {
                    "type": geom_type,
                    "coordinates": coords
                }
            }
            geojson_data["features"].append(feature)

    # Deduplicate nearby features
    geojson_data = deduplicate_nearby_features(geojson_data)

    geojson_path = os.path.join(output_dir, f"US-{state_abbr}_{state_fips}.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        f.write('{\n')
        f.write('  "type": "FeatureCollection",\n')
        f.write(f'  "name": "US-{state_abbr}",\n')
        f.write('  "crs": {\n')
        f.write('    "type": "name",\n')
        f.write('    "properties": {\n')
        f.write('      "name": "urn:ogc:def:crs:OGC:1.3:CRS84"\n')
        f.write('    }\n')
        f.write('  },\n')
        f.write('  "features": [\n')

        features = []
        for feature in geojson_data["features"]:
            feature_str = json.dumps(feature, separators=(',', ':'))
            features.append(f"    {feature_str}")

        f.write(',\n'.join(features))
        f.write('\n  ]\n}')

    print(f"Saved GeoJSON to {geojson_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"US-{state_abbr}_{state_fips}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["id", "type", "tags", "name", "operator", "power"])

        # Write data rows
        for element in data.get("elements", []):
            if element.get("type") in ["node", "way", "relation"]:
                writer.writerow([
                    element.get("id"),
                    element.get("type"),
                    str(element.get("tags", {})),
                    element.get("tags", {}).get("name", ""),
                    element.get("tags", {}).get("operator", ""),
                    element.get("tags", {}).get("power", "")
                ])
    print(f"Saved CSV to {csv_path}")


@click.command()
@click.option('--state', default="all", help="Specify a US state abbreviation (e.g., CA) or 'all'")
def main(state):
    if state.lower() == "all":
        for state_obj in us.states.STATES:
            state_abbr = state_obj.abbr
            state_fips = state_obj.fips
            state_name = state_obj.name
            data = fetch_osm_data(state_abbr, state_name)
            save_files(data, state_abbr, state_fips)
    else:
        state_obj = us.states.lookup(state)
        if state_obj is None:
            print(f"Unknown state abbreviation: {state}")
            return
        state_abbr = state_obj.abbr
        state_name = state_obj.name
        data = fetch_osm_data(state_abbr, state_name)
        save_files(data, state_abbr)


if __name__ == "__main__":
    main()

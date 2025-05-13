import logging
from pathlib import Path

import osmnx as ox
from shapely.geometry import Polygon

from syngrid.data_processor.utils.utils import create_region_path

# Initialize logger
logger = logging.getLogger(__name__)


def extract_osm_features(polygon, fips_region=None, output_dir=None, save_files=True):
    """
    Extract OSM features (buildings, POIs, and power infrastructure) from a given polygon boundary,
    with optional metadata from a FIPS region dictionary.

    Args:
        polygon (shapely.geometry.Polygon): A Shapely polygon defining the area of interest
        fips_region (dict, optional): Dictionary containing region information with keys:
            - state: State name
            - state_fips: State FIPS code
            - county: County name
            - county_fips: County FIPS code
            - subdivision: Subdivision name (optional)
            - subdivision_fips: Subdivision FIPS code (optional)
        output_dir (str or Path, optional): Directory to save output files
        save_files (bool): Whether to save the extracted features as shapefiles

    Returns:
        dict: Dictionary containing the extracted features:
            - buildings: GeoDataFrame of buildings
            - pois: GeoDataFrame of points of interest
            - power: GeoDataFrame of power infrastructure (or None if not found)
            - filepaths: Dict of file paths if files were saved
    """
    # Set default options for osmnx
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # Prepare results dictionary
    results = {
        'buildings': None,
        'pois': None,
        'power': None,
        'filepaths': {}
    }

    # Configure output directory
    if save_files:
        if output_dir is None and fips_region is not None:
            # Create output directory based on FIPS region if provided
            output_dir = create_region_path(fips_region, "OSM_BUILDINGS")

        # If still None, use a default directory
        if output_dir is None:
            output_dir = Path("syngrid/data_processor/output/osm/extracted_features")
            output_dir.mkdir(parents=True, exist_ok=True)
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

    # 1. Extract Buildings
    try:
        logger.info("Extracting buildings from OSM")
        buildings = ox.features_from_polygon(polygon, tags={"building": True})
        results['buildings'] = buildings
        logger.info(f"Buildings found: {len(buildings)}")

        if save_files and not buildings.empty:
            buildings_file = output_dir / "buildings.shp"
            buildings.to_file(buildings_file)
            results['filepaths']['buildings'] = buildings_file
            logger.info(f"Saved buildings to {buildings_file}")
    except Exception as e:
        logger.error(f"Error extracting buildings: {str(e)}")
        results['buildings'] = None

    # 2. Extract POIs (like amenities, shops, etc.)
    try:
        logger.info("Extracting POIs from OSM")
        pois = ox.features_from_polygon(polygon, tags={
            "amenity": True,
            "shop": True,
            "tourism": True,
            "leisure": True,
            "office": True
        })

        if not pois.empty:
            results['pois'] = pois
            logger.info(f"POIs found: {len(pois)}")

            if save_files:
                # Split POIs by geometry type to avoid shapefile errors
                geometry_types = pois.geometry.type.unique()
                logger.info(f"POI geometry types: {', '.join(geometry_types)}")

                for geom_type in geometry_types:
                    type_suffix = geom_type.lower()
                    type_pois = pois[pois.geometry.type == geom_type]

                    if not type_pois.empty:
                        pois_file = output_dir / f"pois_{type_suffix}.shp"
                        try:
                            type_pois.to_file(pois_file)
                            results['filepaths'][f'pois_{type_suffix}'] = pois_file
                            logger.info(f"Saved {len(type_pois)} {geom_type} POIs to {pois_file}")
                        except Exception as type_error:
                            logger.error(f"Error saving {geom_type} POIs: {str(type_error)}")
        else:
            logger.info("No POIs found in the specified area")
    except Exception as e:
        logger.error(f"Error extracting POIs: {str(e)}")
        results['pois'] = None

    # 3. Extract Power Infrastructure (transformers, substations, poles)
    try:
        logger.info("Extracting power infrastructure from OSM")

        # Create a comprehensive power infrastructure tag dictionary based on the Overpass query
        power_tags = {
            "power": [
                "transformer",         # Transformer nodes/ways/relations
                "substation",          # Substation nodes/ways/relations
                "pole"                 # Poles that may have transformers
            ]
        }

        # This will extract all features matching any of the power types
        power_features = ox.features_from_polygon(polygon, tags=power_tags)

        # Further filter to match the Overpass query criteria
        # - Exclude abandoned transformers/substations
        # - Include distribution transformer poles
        if not power_features.empty:
            logger.info(f"Found {len(power_features)} power features before filtering")

            # Filter out abandoned infrastructure
            try:
                # Filter out abandoned infrastructure
                abandoned_mask = (
                    (power_features.get('abandoned', '') != 'yes')
                    & (power_features.get('abandoned:substation', '') != 'yes')
                    & (power_features.get('abandoned:building', '') != 'transformer')
                )

                # For poles, keep only those with distribution transformers
                poles_mask = (
                    (power_features['power'] == 'pole')
                    & (power_features.get('transformer', '') == 'distribution')
                )

                # Combine masks: keep non-abandoned infrastructure and poles with transformers
                if 'power' in power_features.columns:
                    transformers_mask = power_features['power'].isin(['transformer', 'substation'])
                    final_mask = (transformers_mask & abandoned_mask) | poles_mask
                    power_features = power_features[final_mask]

                logger.info(f"Power features after filtering: {len(power_features)}")

                results['power'] = power_features

                # Split by geometry type like we did for POIs
                if save_files and not power_features.empty:
                    geometry_types = power_features.geometry.type.unique()
                    logger.info(f"Power geometry types: {', '.join(geometry_types)}")

                    for geom_type in geometry_types:
                        type_suffix = geom_type.lower()
                        type_power = power_features[power_features.geometry.type == geom_type]

                        if not type_power.empty:
                            power_file = output_dir / f"power_{type_suffix}.shp"
                            try:
                                type_power.to_file(power_file)
                                results['filepaths'][f'power_{type_suffix}'] = power_file
                                logger.info(
                                    f"Saved {len(type_power)} {geom_type} power features "
                                    f"to {power_file}"
                                )
                            except Exception as type_error:
                                logger.error(
                                    f"Error saving {geom_type} power features: {str(type_error)}"
                                )
            except Exception as filter_error:
                logger.error(
                    f"Error filtering power features: {str(filter_error)}"
                )
        else:
            logger.info(
                "No power infrastructure features found in the specified area"
            )
            results['power'] = None

    except Exception as e:
        logger.error(f"Error extracting power infrastructure: {str(e)}")
        results['power'] = None

    if save_files:
        logger.info(f"All features saved to {output_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    # Example polygon (Boston area)
    polygon = Polygon([
        (-71.0589, 42.3601),
        (-71.0570, 42.3601),
        (-71.0570, 42.3615),
        (-71.0589, 42.3615),
        (-71.0589, 42.3601),
    ])

    # Example FIPS region dict
    fips_dict = {
        'state': 'Massachusetts',
        'state_fips': '25',
        'county': 'Suffolk',
        'county_fips': '025',
        'subdivision': 'Boston',
        'subdivision_fips': '07000'
    }

    # Extract features
    results = extract_osm_features(polygon, fips_region=fips_dict)

    # Print results - TODO: remove
    buildings_count = len(results['buildings']) if results['buildings'] is not None else 0
    pois_count = len(results['pois']) if results['pois'] is not None else 0

    print(f"Buildings: {buildings_count} features")
    print(f"POIs: {pois_count} features")
    if results['power'] is not None and not results['power'].empty:
        print(f"Power: {len(results['power'])} features")
    else:
        print("No power infrastructure found")

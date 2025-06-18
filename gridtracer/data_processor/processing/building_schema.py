import dataclasses
from typing import List, Optional

import geopandas as gpd


@dataclasses.dataclass
class ResidentialBuildingOutput:
    """
    Defines the output schema for residential buildings.

    This class represents the standardized output format for residential building data
    processed through the building classification pipeline. Field names use the
    original processing column names as default.
    """

    # Required fields (no default values) must come first - using original column names
    id: str  # Unique identifier
    floor_area: float  # Total floor area in m²
    building_use: str  # Primary use - should be 'residential'
    free_walls: int  # Number of free walls
    building_type: str  # Building typology (SFH, TH, MFH, AB)
    occupants: int  # Number of occupants
    floors: int  # Number of floors/stories
    construction_year: str  # Construction period
    geometry: object  # Building footprint geometry

    # Optional fields (with default values) come after required fields
    comment: Optional[str] = None  # Optional comment field
    refurb_walls: Optional[str] = None  # Wall refurbishment status
    refurb_roof: Optional[str] = None  # Roof refurbishment status
    refurb_basement: Optional[str] = None  # Basement refurbishment status
    refurb_windows: Optional[str] = None  # Window refurbishment status

    @classmethod
    def get_schema_fields(cls) -> List[str]:
        """
        Returns a list of all field names in the schema.

        Returns:
        --------
        List[str] : List of field names in original column order
        """
        return [field.name for field in dataclasses.fields(cls)]

    @classmethod
    def prepare_default_output(cls, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepares residential buildings data for default output format.

        This method filters columns according to the schema and adds missing optional columns,
        but preserves original column names.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Source buildings data with original column names

        Returns:
        --------
        GeoDataFrame : Buildings data with schema-compliant columns in original naming
        """
        # Create a copy to avoid modifying original
        output_buildings = buildings.copy()

        # Get schema fields (original column names)
        schema_fields = cls.get_schema_fields()

        # Filter to only include columns that exist in the schema
        available_schema_columns = [
            col for col in schema_fields if col in output_buildings.columns
        ]
        output_buildings = output_buildings[available_schema_columns]

        # Add missing optional columns with None values
        for field in schema_fields:
            if field not in output_buildings.columns and field != 'geometry':
                output_buildings[field] = None

        # Reorder columns according to schema field order
        output_buildings = output_buildings[schema_fields]

        return output_buildings


@dataclasses.dataclass
class NonResidentialBuildingOutput:
    """
    Defines the output schema for non-residential buildings.

    This class represents the standardized output format for non-residential building data
    processed through the building classification pipeline. Field names use the
    original processing column names as default.
    """

    # Required fields (no default values) must come first - using original column names
    id: str  # Unique identifier
    floor_area: float  # Total floor area in m²
    building_use: str  # Primary use (commercial, industrial, public)
    free_walls: int  # Number of free walls
    geometry: object  # Building footprint geometry

    # Optional fields (with default values) come after required fields
    comment: Optional[str] = None  # Optional comment field

    @classmethod
    def get_schema_fields(cls) -> List[str]:
        """
        Returns a list of all field names in the schema.

        Returns:
        --------
        List[str] : List of field names
        """
        return [field.name for field in dataclasses.fields(cls)]

    @classmethod
    def prepare_default_output(cls, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepares non-residential buildings data for default output format.

        This method filters columns according to the schema and adds missing optional columns,
        but preserves original column names.

        Parameters:
        -----------
        buildings : GeoDataFrame
            Source buildings data with original column names

        Returns:
        --------
        GeoDataFrame : Buildings data with schema-compliant columns in original naming
        """
        # Create a copy to avoid modifying original
        output_buildings = buildings.copy()

        # Get schema fields (original column names)
        schema_fields = cls.get_schema_fields()

        # Filter to only include columns that exist in the schema
        available_schema_columns = [
            col for col in schema_fields if col in output_buildings.columns
        ]
        output_buildings = output_buildings[available_schema_columns]

        # Add missing optional columns with None values
        for field in schema_fields:
            if field not in output_buildings.columns and field != 'geometry':
                output_buildings[field] = None

        # Reorder columns according to schema field order
        output_buildings = output_buildings[schema_fields]

        return output_buildings

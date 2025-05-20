import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class BuildingOutputSchema:
    """
    Defines the output schema for classified buildings.

    This class represents the standardized output format for building data
    processed through the building classification pipeline. It includes all
    necessary attributes for energy demand modeling and grid planning.
    """

    building_id: str  # Unique identifier
    geometry: object  # Building footprint geometry
    floor_area: Optional[float] = None  # Total floor area (m²)
    building_use: str  # Primary use (residential, commercial, industrial, public)
    building_type: Optional[str] = None  # Building typology (NREL profile)
    floor_number: Optional[int] = None  # Number of floors/stories
    occupants: Optional[int] = None  # Number of occupants
    households: Optional[int] = None  # Number of household units
    construction_year: Optional[str] = None  # Construction period
    height: Optional[float] = None  # Building height (m)

    @classmethod
    def get_schema_fields(cls) -> List[str]:
        """
        Returns a list of all field names in the schema.

        Returns:
        --------
        List[str] : List of field names
        """
        return [field.name for field in dataclasses.fields(cls)]

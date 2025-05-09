import csv
import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def lookup_fips_codes(region):
    state = region.get('state')
    county = region.get('county')
    subdivision = region.get('county_subdivision')
    lookup_url = region.get('lookup_url')

    if not state or not county or not lookup_url:
        logger.error("Missing required parameters: state, county, and lookup_url are required")
        return None

    # Create output directory
    output_dir = Path("syngrid/data_processor/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Local file path
    filename = os.path.basename(lookup_url)
    local_file_path = output_dir / filename

    # Download file if it doesn't exist
    if not local_file_path.exists():
        logger.info(f"Downloading Census data file from {lookup_url}")
        urllib.request.urlretrieve(lookup_url, local_file_path)
        logger.debug(f"Census data file saved to {local_file_path}")
    else:
        logger.debug(f"Using existing Census data file: {local_file_path}")

    try:
        # Read and clean the data
        with open(local_file_path, 'r', encoding='latin-1') as infile:
            reader = csv.reader(infile)
            processed_rows = []
            merged_count = 0

            for i, row in enumerate(reader):
                # Skip header
                if i == 0 or (row and row[0] == 'STATE'):
                    continue
                # Process rows based on column count
                if len(row) == 7:
                    processed_rows.append(row)
                elif len(row) == 8:
                    # Merge third-last and second-last columns (5 and 6 in zero-based indexing)
                    merged_row = row[:5]  # First 5 columns
                    merged_row.append(row[5] + ' ' + row[6])  # Merge columns 5 and 6
                    merged_row.append(row[7])  # Last column
                    processed_rows.append(merged_row)
                    merged_count += 1

        logger.debug(f"Merged {merged_count} rows with 8 columns into 7 columns")

        # Create DataFrame from clean data
        column_names = [
            'state',
            'state_fips',
            'county_fips',
            'county',
            'subdivision_fips',
            'subdivision',
            'funcstat']
        df = pd.DataFrame(processed_rows, columns=column_names)

        # Now look up the specific data we need
        state_df = df[df['state'] == state]
        if state_df.empty:
            raise ValueError(f"State '{state}' not found in Census data")

        # Filter by exact county match
        county_matches = state_df[state_df['county'] == county]
        if county_matches.empty:
            raise ValueError(f"County '{county}' not found in state '{state}'")

        # Get the county data
        county_data = county_matches.iloc[0]
        state_fips = county_data['state_fips']
        county_fips = county_data['county_fips']

        # Initialize result with consistent structure
        result = {
            'state': state,
            'state_fips': state_fips,
            'county': county,
            'county_fips': county_fips,
            'subdivision': None,
            'subdivision_fips': None,
            'funcstat': None
        }

        # If subdivision provided, get exact match
        if subdivision:
            subdiv_match = county_matches[county_matches['subdivision'] == subdivision]

            if subdiv_match.empty:
                raise ValueError(
                    f"Subdivision '{subdivision}' not found in county '{county}', state '{state}'")
            subdiv_data = subdiv_match.iloc[0]
            result['subdivision'] = subdivision
            result['subdivision_fips'] = subdiv_data['subdivision_fips']
            result['funcstat'] = subdiv_data['funcstat']

        return result

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        logger.error(f"Error processing Census data: {str(e)}")
        raise ValueError(f"Failed to lookup FIPS codes: {str(e)}")

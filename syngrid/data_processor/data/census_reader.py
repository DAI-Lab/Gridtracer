import logging
import sys
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import Polygon
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('census_exploration.log')
    ]
)
logger = logging.getLogger(__name__)


class CensusReader:
    def __init__(self, state_fips, county_fips, plot_output_dir=None):
        """Initialize the Census Reader"""
        self.state_fips = state_fips
        self.county_fips = county_fips
        self.blocks_gdf = None
        self.household_data = None
        self.census_data_dir = Path("census_Data_exploration/census_data")
        self.census_data_dir.mkdir(exist_ok=True)

        # Set up plot output directory
        if plot_output_dir is None:
            self.plot_output_dir = Path("plots_output_dir")
        else:
            self.plot_output_dir = Path(plot_output_dir)
        self.plot_output_dir.mkdir(exist_ok=True)

        logger.info("Created census_data directory at %s", self.census_data_dir)
        logger.info("Created plot output directory at %s", self.plot_output_dir)

    def get_census_blocks(self):
        """Download and load Census blocks for Suffolk County"""
        # Use the c orrect URL for Massachusetts Census blocks
        url = (
            "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
            "tl_2020_{self.state_fips}_tabblock20.zip"
        )
        logger.inf("Downloading Census blocks from: %s", url)

        try:
            # Download and load the data
            blocks = gpd.read_file(url)
            logger.info("Total blocks downloaded: %d", len(blocks))

            # Filter for Suffolk County
            blocks = blocks[blocks['COUNTYFP20'] == self.county_fips]
            logger.info("Blocks in Suffolk County: %d", len(blocks))

            # Save to local file
            output_file = self.census_data_dir / "suffolk_blocks.geojson"
            blocks.to_file(output_file, driver='GeoJSON')
            logger.info("Saved blocks to: %s", output_file)

            # Set the blocks_gdf attribute
            self.blocks_gdf = blocks
            return blocks

        except Exception as e:
            logger.error("Error downloading blocks: %s", e, exc_info=True)
            # Try to load from local file if download fails
            local_file = self.census_data_dir / "suffolk_blocks.geojson"
            if local_file.exists():
                logger.info("Loading blocks from local file: %s", local_file)
                blocks = gpd.read_file(local_file)
                self.blocks_gdf = blocks
                return blocks
            else:
                logger.error("Could not download or load Census blocks")
                raise Exception("Could not download or load Census blocks")

    def get_household_data(self, force_download=False):
        """Get H9 household data for all blocks with batching for API limits"""
        logger.info("Fetching household data from Census API...")

        if self.blocks_gdf is None:
            logger.error("Census blocks not loaded")
            raise ValueError(
                "Census blocks not loaded. Call get_census_blocks() first."
            )

        # Define cache file path
        cache_file = self.census_data_dir / "household_data.csv"

        # Check if cached data exists and load it if not forcing download
        if not force_download and cache_file.exists():
            logger.info("Loading household data from cache: %s", cache_file)
            self.household_data = pd.read_csv(cache_file)
            return self.household_data

        api_url = "https://api.census.gov/data/2020/dec/dhc"
        all_block_ids = self.blocks_gdf['GEOID20'].tolist()
        batch_size = 50
        all_data = []

        logger.info(
            "Processing %d blocks in batches of %d",
            len(all_block_ids),
            batch_size
        )

        # Process blocks in batches
        for i in tqdm(range(0, len(all_block_ids), batch_size)):
            batch_ids = all_block_ids[i:i + batch_size]
            logger.info(
                "Processing batch %d/%d (%d blocks)",
                (i // batch_size) + 1,
                (len(all_block_ids) + batch_size - 1) // batch_size,
                len(batch_ids)
            )

            # Format block IDs for API
            formatted_ids = [f"1000000US{bid}" for bid in batch_ids]

            params = {
                "get": "group(H9)",
                "ucgid": ",".join(formatted_ids),
                "key": CENSUS_API_KEY
            }

            try:
                response = requests.get(api_url, params=params)
                response.raise_for_status()

                # Check for API key related errors
                if response.status_code == 403:
                    logger.error("API key invalid or expired")
                    raise ValueError("Invalid or expired Census API key")

                data = response.json()

                # Skip header row after first batch
                if i == 0:
                    all_data.extend(data)
                else:
                    all_data.extend(data[1:])

            except requests.exceptions.RequestException as e:
                logger.error(
                    "API request failed for batch %d: %s",
                    (i // batch_size) + 1,
                    e
                )
                raise

        # Convert to DataFrame
        self.household_data = pd.DataFrame(
            all_data[1:],  # Skip header row
            columns=all_data[0]  # Use header row for column names
        )

        # Save to cache
        self.household_data.to_csv(cache_file, index=False)
        logger.info(
            "Saved %d blocks of household data to: %s",
            len(self.household_data),
            cache_file
        )

        return self.household_data

    @staticmethod
    def get_column_description(column):
        """Get description for Census column"""
        descriptions = {
            'H9_001N': 'Total Occupied Housing Units',
            'H9_002N': 'Owner-occupied Housing Units',
            'H9_003N': 'Renter-occupied Housing Units',
            'H9_004N': 'Average Household Size (Owner-occupied)',
            'H9_005N': 'Average Household Size (Renter-occupied)',
            'H9_008N': 'Average Household Size (All Units)'
        }
        return descriptions.get(column, column)

    def plot_blocks_with_data(self, column='H9_001N', title=None):
        """Plot Census blocks colored by specified data column"""
        logger.info("Creating visualization for column: %s", column)

        if self.blocks_gdf is None:
            logger.error("Census blocks not loaded")
            raise ValueError(
                "Census blocks not loaded. Call get_census_blocks() first."
            )
        if self.household_data is None:
            logger.error("Household data not loaded")
            raise ValueError(
                "Household data not loaded. Call get_household_data() first."
            )

        # Clean the GEO_ID by removing the '1000000US' prefix
        self.household_data['GEO_ID'] = (
            self.household_data['GEO_ID'].str.replace('1000000US', '')
        )
        logger.info(
            "Cleaned GEO_IDs in household data to match Census block format"
        )

        # Merge Census data with geographic blocks
        merged_gdf = self.blocks_gdf.merge(
            self.household_data,
            left_on='GEOID20',
            right_on='GEO_ID',
            how='inner'
        )
        logger.info("Merged data contains %d blocks", len(merged_gdf))

        # Convert column to numeric
        merged_gdf[column] = pd.to_numeric(merged_gdf[column])

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot blocks with choropleth
        merged_gdf.to_crs(epsg=3857).plot(
            column=column,
            ax=ax,
            legend=True,
            legend_kwds={
                'label': self.get_column_description(column),
                'orientation': 'vertical',
                'shrink': 0.8
            },
            missing_kwds={
                'color': 'lightgrey'
            }
        )

        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

        # Set title
        if title is None:
            title = (
                f"Boston Census Blocks - "
                f"{self.get_column_description(column)}"
            )
        plt.title(title, pad=20, fontsize=16)

        # Remove axis
        ax.set_axis_off()

        # Save plot
        output_file = self.plot_output_dir / f"census_blocks_{column}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info("Saved plot to: %s", output_file)
        plt.close()

        # Print summary statistics
        logger.info("\nSummary Statistics:")
        stats = merged_gdf[column].describe()
        logger.info("\n%s", stats)

    def plot_blocks(self, blocks, title="Census Blocks for BackBay TestData"):
        """Plot Census blocks with basemap"""
        if blocks.empty:
            logger.error("No blocks to plot")
            return

        logger.info("Plotting %d blocks", len(blocks))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        # Convert to Web Mercator for basemap compatibility
        blocks_mercator = blocks.to_crs(epsg=3857)

        # Plot blocks with improved styling
        blocks_mercator.plot(
            ax=ax,
            alpha=0.2,  # More transparent fill
            edgecolor='red',  # Red boundaries like in plot_blocks_with_data
            facecolor='skyblue',  # Light fill color
            linewidth=1.0  # Slightly thicker lines
        )

        # Add block IDs at centroids
        for idx, block in blocks_mercator.iterrows():
            # Get centroid coordinates
            centroid = block.geometry.centroid
            # Extract last 6 digits of GEOID20 for cleaner display
            block_id = block['GEOID20'][-6:]
            # Add text with white background for better visibility
            ax.annotate(
                block_id,
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    pad=1
                )
            )

        # Get the bounds of the plotted area
        xmin, ymin, xmax, ymax = blocks_mercator.total_bounds

        # Add basemap with explicit zoom level
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom=15,  # Explicit zoom level for detailed street view
            crs=blocks_mercator.crs,
            attribution_size=8
        )

        # Set the axis limits to match the blocks
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Set title and remove axes
        plt.title(title, pad=20, fontsize=16)  # Match title style
        ax.set_axis_off()

        # Save plot
        output_file = self.plot_output_dir / "census_blocks.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info("Saved plot to: %s", output_file)
        plt.close()

    def load_existing_data(self, blocks_file=None, household_file=None):
        """Load existing census block and household data from files"""
        logger.info("Loading existing census data from files")

        # Load blocks data if file provided
        if blocks_file:
            try:
                logger.info("Loading blocks from: %s", blocks_file)
                self.blocks_gdf = gpd.read_file(blocks_file)
                logger.info("Loaded %d blocks", len(self.blocks_gdf))
            except Exception as e:
                logger.error("Error loading blocks file: %s", e, exc_info=True)
                raise

        # Load household data if file provided
        if household_file:
            try:
                logger.info("Loading household data from: %s", household_file)
                self.household_data = pd.read_csv(household_file)
                logger.info("Loaded household data for %d blocks", len(self.household_data))
            except Exception as e:
                logger.error("Error loading household file: %s", e, exc_info=True)
                raise

        return self.blocks_gdf, self.household_data

    def plot_blocks_with_boundary(self, blocks, boundary_coords,
                                  title="Census Blocks with Back Bay Boundary"):
        """Plot Census blocks with Back Bay boundary overlay"""
        if blocks.empty:
            logger.error("No blocks to plot")
            return

        logger.info("Plotting %d blocks with boundary", len(blocks))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        # Convert to Web Mercator for basemap compatibility
        blocks_mercator = blocks.to_crs(epsg=3857)

        # Create and transform boundary polygon
        boundary_polygon = gpd.GeoDataFrame(
            geometry=[Polygon(boundary_coords)],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Plot blocks with improved styling
        blocks_mercator.plot(
            ax=ax,
            alpha=0.2,
            edgecolor='red',
            facecolor='skyblue',
            linewidth=1.0
        )

        # Plot boundary
        boundary_polygon.plot(
            ax=ax,
            facecolor='none',
            edgecolor='green',
            linewidth=2.0,
            linestyle='--'
        )

        # Add block IDs at centroids
        for idx, block in blocks_mercator.iterrows():
            centroid = block.geometry.centroid
            block_id = block['GEOID20'][-6:]
            ax.annotate(
                block_id,
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    pad=1
                )
            )

        # Get the bounds of both blocks and boundary
        blocks_bounds = blocks_mercator.total_bounds
        boundary_bounds = boundary_polygon.total_bounds
        xmin = min(blocks_bounds[0], boundary_bounds[0])
        ymin = min(blocks_bounds[1], boundary_bounds[1])
        xmax = max(blocks_bounds[2], boundary_bounds[2])
        ymax = max(blocks_bounds[3], boundary_bounds[3])

        # Add basemap with explicit zoom level
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom=15,
            crs=blocks_mercator.crs,
            attribution_size=8
        )

        # Set the axis limits to show both blocks and boundary
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Set title and remove axes
        plt.title(title, pad=20, fontsize=16)
        ax.set_axis_off()

        # Save plot
        output_file = self.plot_output_dir / "census_blocks_with_boundary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info("Saved plot to: %s", output_file)
        plt.close()


def main():
    logger.info("Starting Census data exploration")

    # Initialize reader with custom plot output directory
    reader = CensusReader(
        state_fips="25",
        county_fips="025",
        plot_output_dir="census_data_exploration/test_data/plots"
    )

    try:
        # Load existing data
        reader.load_existing_data(
            blocks_file=Path("census_data_exploration/test_data/test_blocks_10.geojson"),
            household_file=Path("census_data_exploration/test_data/test_buildings_10.csv")
        )

        # Define Back Bay boundary coordinates
        backbay_coords = [
            (-71.08729044718838, 42.352058162996514),  # NW corner
            (-71.08530408364521, 42.34792938709825),  # SW corner
            (-71.07510623509738, 42.35065572032843),  # SE corner
            (-71.07716354019527, 42.35477121092043),  # NE corner
            (-71.08729044718838, 42.352058162996514)  # Close polygon
        ]

        # Plot blocks with boundary
        reader.plot_blocks_with_boundary(reader.blocks_gdf, backbay_coords)

        # Generate other visualizations
        logger.info("\nGenerating other visualizations...")
        reader.plot_blocks_with_data('H9_008N', 'Average Household Size')
        reader.plot_blocks(reader.blocks_gdf)

        logger.info("Census data exploration completed successfully")

    except Exception as e:
        logger.error("Error during execution: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()

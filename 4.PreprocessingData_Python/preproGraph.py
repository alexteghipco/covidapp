''' ------------------------------------------------------------------ '''
''' Create a matlab friendly file of the abstract embeddings for later '''
''' ------------------------------------------------------------------ '''
import numpy as np
data = np.load("D:\\Science\\covid\\abstract_embeddings.npy")
import h5py
with h5py.File("D:\\Science\\covid\\abstract_embeddings.h5", "w") as hf:
    hf.create_dataset("data", data=data)

''' ------------------------------- '''
''' Function for concatenating data '''
''' ------------------------------- '''
def load_and_process_sharded_csv(folder_path, file_pattern, output_combined_path, output_yearly_path):
    # Collect and sort all file paths
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        print(f"No files found in {folder_path}. Please check the folder path and file naming convention.")
        return None

    # Load and concatenate files
    combined_df = pd.concat((pd.read_csv(file, encoding='utf-8') for file in file_paths), ignore_index=True)
    print(f"Concatenation complete for {folder_path}. Here is a preview of the combined DataFrame:")
    print(combined_df.head())

    # Remove missing data if needed
    combined_df = combined_df.dropna(subset=['node1', 'node2'])

    # Save the combined DataFrame
    combined_df.to_csv(output_combined_path, index=False, encoding='utf-8')

    # Write out each year into a separate matrix
    os.makedirs(output_yearly_path, exist_ok=True)
    for year, data in combined_df.groupby('year'):
        file_path = os.path.join(output_yearly_path,
                                 f"{os.path.basename(output_combined_path).split('_combined')[0]}_{int(year)}.csv")
        data.to_csv(file_path, index=False, encoding='utf-8')

    print(f"Data for each year saved successfully in {output_yearly_path}.")


''' ------------------------------------------------------------------------------------- '''
''' Concatenate author, country, higher-level organization, and within-organization data  '''
''' ------------------------------------------------------------------------------------- '''
import os
import glob
# Author-level data (sharded CSV files)
author_folder = r"D:\Science\covid\matrices\authors"
author_file_pattern = os.path.join(author_folder, "author_collab_matrix_*.csv")
author_output_combined = r"D:\Science\covid\matrices\authors_combined.csv"
author_output_yearly = r"D:\Science\covid\matrices\authors_by_year"
load_and_process_sharded_csv(author_folder, author_file_pattern, author_output_combined, author_output_yearly)

# Country-level data (single CSV file, still split by year)
country_folder = r"D:\Science\covid\matrices\countries"
country_file = os.path.join(country_folder, "*.csv")
country_output_combined = r"D:\Science\covid\matrices\countries_combined.csv"
country_output_yearly = r"D:\Science\covid\matrices\countries_by_year"
load_and_process_sharded_csv(country_folder, country_file, country_output_combined, country_output_yearly)

# High-level organization data (sharded CSV files)
high_level_folder = r"D:\Science\covid\matrices\highLevel"
high_level_file_pattern = os.path.join(high_level_folder, "high_Level_collab_matrix_*.csv")
high_level_output_combined = r"D:\Science\covid\matrices\high_Level_combined.csv"
high_level_output_yearly = r"D:\Science\covid\matrices\high_Level_by_year"
load_and_process_sharded_csv(high_level_folder, high_level_file_pattern, high_level_output_combined, high_level_output_yearly)

# Low-level organization data (single CSV file, still split by year)
low_level_folder = r"D:\Science\covid\matrices\lowLevel"
low_level_file = os.path.join(low_level_folder, "*.csv")
low_level_output_combined = r"D:\Science\covid\matrices\low_Level_combined.csv"
low_level_output_yearly = r"D:\Science\covid\matrices\low_Level_by_year"
load_and_process_sharded_csv(low_level_folder, low_level_file, low_level_output_combined, low_level_output_yearly)

''' ------------------------------------------------------------------------ '''
''' Process yearly country data to get network positions, normalized weights '''
''' ------------------------------------------------------------------------ '''
import os
import pandas as pd
import networkx as nx

# Paths to input and output directories
input_dir_countries = r"D:\Science\covid\matrices\countries_by_year"
output_dir_countries = r"D:\Science\covid\matrices\processed_countries"
os.makedirs(output_dir_countries, exist_ok=True)

# Gather all unique country pairs across all years
all_data_countries = []
years_countries = sorted([
    int(f.split('_')[-1].split('.')[0])
    for f in os.listdir(input_dir_countries)
    if f.startswith("countries_")
])

for year in years_countries:
    file_path = os.path.join(input_dir_countries, f"countries_{year}.csv")
    df = pd.read_csv(file_path, encoding='utf-8')
    all_data_countries.append(df[['node1', 'node2']])

# Combine all pairs to get full set of unique country pairs
combined_pairs_countries = pd.concat(all_data_countries).drop_duplicates().reset_index(drop=True)

# Create a static layout using all unique country pairs
G_countries = nx.from_pandas_edgelist(combined_pairs_countries, "node1", "node2")
static_pos_countries = nx.spring_layout(G_countries, seed=42)  # Consistent layout across all years

# Process and save each year's data
for year in years_countries:
    file_path = os.path.join(input_dir_countries, f"countries_{year}.csv")
    df = pd.read_csv(file_path, encoding='utf-8')

    # Merge with all unique pairs to include all pairs each year
    df_full = combined_pairs_countries.merge(df, on=["node1", "node2"], how="left").fillna({"weight": 0})

    # Normalize weights
    max_weight = df_full["weight"].max()
    df_full["normalized_weight"] = df_full["weight"] / max_weight if max_weight > 0 else df_full["weight"]

    # Add static positions
    df_full["x0"] = df_full["node1"].map(lambda node: static_pos_countries[node][0])
    df_full["y0"] = df_full["node1"].map(lambda node: static_pos_countries[node][1])
    df_full["x1"] = df_full["node2"].map(lambda node: static_pos_countries[node][0])
    df_full["y1"] = df_full["node2"].map(lambda node: static_pos_countries[node][1])

    # Save processed data
    output_file = os.path.join(output_dir_countries, f"processed_country_{year}.csv")
    df_full.to_csv(output_file, index=False, encoding='utf-8')

''' -------------------------------- '''
''' Process yearly organization data '''
''' -------------------------------- '''
# Paths to input and output directories
input_dir_orgs = r"D:\Science\covid\matrices\high_Level_by_year"
output_dir_orgs = r"D:\Science\covid\matrices\high_level_processed"
os.makedirs(output_dir_orgs, exist_ok=True)

# Gather all unique organization pairs across all years
all_data_orgs = []
years_orgs = sorted([
    int(f.split('_')[-1].split('.')[0])
    for f in os.listdir(input_dir_orgs)
    if f.startswith("high_Level")
])

for year in years_orgs:
    file_path = os.path.join(input_dir_orgs, f"high_Level_{year}.csv")
    df = pd.read_csv(file_path, encoding='utf-8')
    all_data_orgs.append(df[['node1', 'node2']])

# Combine all pairs to get full set of unique organization pairs
combined_pairs_orgs = pd.concat(all_data_orgs).drop_duplicates().reset_index(drop=True)

# Create a static layout using all unique organization pairs
G_orgs = nx.from_pandas_edgelist(combined_pairs_orgs, "node1", "node2")
static_pos_orgs = nx.spring_layout(G_orgs, seed=42)  # Consistent layout across all years

# Process and save each year's data
for year in years_orgs:
    file_path = os.path.join(input_dir_orgs, f"high_Level_{year}.csv")
    df = pd.read_csv(file_path, encoding='utf-8')

    # Merge with all unique pairs to include all pairs each year
    df_full = combined_pairs_orgs.merge(df, on=["node1", "node2"], how="left").fillna({"weight": 0})

    # Normalize weights
    max_weight = df_full["weight"].max()
    df_full["normalized_weight"] = df_full["weight"] / max_weight if max_weight > 0 else df_full["weight"]

    # Add static positions
    df_full["x0"] = df_full["node1"].map(lambda node: static_pos_orgs[node][0])
    df_full["y0"] = df_full["node1"].map(lambda node: static_pos_orgs[node][1])
    df_full["x1"] = df_full["node2"].map(lambda node: static_pos_orgs[node][0])
    df_full["y1"] = df_full["node2"].map(lambda node: static_pos_orgs[node][1])

    # Save processed data
    output_file = os.path.join(output_dir_orgs, f"processed_high_level{year}.csv")
    df_full.to_csv(output_file, index=False, encoding='utf-8')

''' ------------------------------------------------------------------------------------ '''
''' Get mapping hierarchy between countries, organizations and lower-level organizations '''
''' ------------------------------------------------------------------------------------ '''
import os
import pandas as pd

# Path to the hierarchy data
hierarchy_path = r"D:\Science\covid\matrices\hierarchy"
output_dir = r"D:\Science\covid\matrices\processed_hierarchy"
os.makedirs(output_dir, exist_ok=True)

# Define data types for columns
dtype_spec = {
    'publication_id': str,
    'publication_year': 'Int64',  # Nullable integer
    'author_id': str,
    'author_name': str,
    'org_grid_id': str,
    'country_name': str,
    'latitude': float,
    'longitude': float,
    'organization_name': str,
    'high_level_org_id': str,
    'high_level_org_name': str,
    'low_level_org_id': str,
    'low_level_org_name': str
}

# Load all hierarchy data with specified data types
hierarchy_files = [os.path.join(hierarchy_path, f) for f in os.listdir(hierarchy_path) if f.endswith('.csv')]
hierarchy_df = pd.concat(
    (pd.read_csv(f, encoding='utf-8', dtype=dtype_spec) for f in hierarchy_files),
    ignore_index=True
)

# Create a mapping between countries and high-level organizations
country_org_mapping = hierarchy_df[['country_name', 'high_level_org_name']].drop_duplicates()

# Save country-to-organization mapping as a CSV file
country_org_mapping.to_csv(os.path.join(output_dir, "country_org_mapping.csv"), index=False)

# Create a mapping between high-level and low-level organizations
org_hierarchy_mapping = hierarchy_df[['high_level_org_name', 'low_level_org_name']].drop_duplicates()

# Save organization hierarchy mapping as a CSV file
org_hierarchy_mapping.to_csv(os.path.join(output_dir, "org_hierarchy_mapping.csv"), index=False)

print("Mappings saved successfully!")

''' ------------------------------------------------------------------------------------------------------------ '''
''' Split by org for within country data for speed...This was unused as there were too few organizations to plot '''
''' ------------------------------------------------------------------------------------------------------------ '''
import os
import pandas as pd

# Paths
country_org_mapping_path = r"D:\Science\covid\matrices\processed_hierarchy\country_org_mapping.csv"
high_level_dir = r"D:\Science\covid\matrices\high_Level_by_year"
output_dir = r"D:\Science\covid\matrices\processed_within_country"
os.makedirs(output_dir, exist_ok=True)

# Load the country-to-organization mapping
country_org_mapping = pd.read_csv(country_org_mapping_path, encoding='utf-8')

# Get unique countries
countries = country_org_mapping['country_name'].unique()

# Process each country
for country in countries:
    # Get the organizations within this country
    country_orgs = country_org_mapping[country_org_mapping['country_name'] == country]['high_level_org_name'].unique()

    # Process each year in the `high_level_by_year` directory
    for yearly_file in os.listdir(high_level_dir):
        if yearly_file.startswith("high_Level_") and yearly_file.endswith(".csv"):
            # Extract the year from the filename
            year = yearly_file.split('_')[-1].split('.')[0]
            file_path = os.path.join(high_level_dir, yearly_file)

            # Load the high-level organization data for this year
            df = pd.read_csv(file_path, encoding='utf-8')

            # Filter rows where both `node1` and `node2` are within the list of country organizations
            filtered_df = df[(df['node1'].isin(country_orgs)) & (df['node2'].isin(country_orgs))]

            # If there is data to save, store it in the output directory
            if not filtered_df.empty:
                # Define the output file path
                output_file = os.path.join(output_dir, f"{country}_within_country_{year}.csv")

                # Save the filtered data to a CSV
                filtered_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"Saved: {output_file}")
            else:
                print(f"No data for {country} in year {year}.")

''' ------------------------------------- '''
''' Get within-country mappings for speed '''
''' ------------------------------------- '''
import os
import pandas as pd
import networkx as nx

# Paths to input and output directories
country_org_mapping_path = r"D:\Science\covid\matrices\processed_hierarchy\country_org_mapping.csv"
input_dir_orgs = r"D:\Science\covid\matrices\high_Level_by_year"
output_dir = r"D:\Science\covid\matrices\processed_within_country"
os.makedirs(output_dir, exist_ok=True)

# Load the country-to-organization mapping
country_org_mapping = pd.read_csv(country_org_mapping_path, encoding='utf-8')

# Get unique countries
countries = country_org_mapping['country_name'].unique()

# Gather all unique organization pairs across all years for each country
for country in countries:
    # Get the organizations within this country
    country_orgs = country_org_mapping[country_org_mapping['country_name'] == country]['high_level_org_name'].unique()

    # Collect all data for this country's organization pairs across all years
    all_data_country = []
    years_orgs = sorted([
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(input_dir_orgs)
        if f.startswith("high_Level")
    ])

    for year in years_orgs:
        file_path = os.path.join(input_dir_orgs, f"high_Level_{year}.csv")
        df = pd.read_csv(file_path, encoding='utf-8')
        # Filter only pairs where both nodes are in the specified country's organizations
        df_country = df[(df['node1'].isin(country_orgs)) & (df['node2'].isin(country_orgs))]
        all_data_country.append(df_country[['node1', 'node2']])

    # Combine all pairs to get the full set of unique organization pairs for the country
    combined_pairs_country = pd.concat(all_data_country).drop_duplicates().reset_index(drop=True)

    # Create a static layout for consistent visualization across years
    G_country = nx.from_pandas_edgelist(combined_pairs_country, "node1", "node2")
    static_pos_country = nx.spring_layout(G_country, seed=42)  # Consistent layout across all years

    # Process and save each year's data with static positions and normalized weights
    for year in years_orgs:
        file_path = os.path.join(input_dir_orgs, f"high_Level_{year}.csv")
        df = pd.read_csv(file_path, encoding='utf-8')

        # Filter only pairs where both nodes are in the specified country's organizations
        df_country = df[(df['node1'].isin(country_orgs)) & (df['node2'].isin(country_orgs))]

        # Merge with all unique pairs to ensure each pair exists each year
        df_full = combined_pairs_country.merge(df_country, on=["node1", "node2"], how="left").fillna({"weight": 0})

        # Normalize weights
        max_weight = df_full["weight"].max()
        df_full["normalized_weight"] = df_full["weight"] / max_weight if max_weight > 0 else df_full["weight"]

        # Add static positions for nodes
        df_full["x0"] = df_full["node1"].map(lambda node: static_pos_country[node][0])
        df_full["y0"] = df_full["node1"].map(lambda node: static_pos_country[node][1])
        df_full["x1"] = df_full["node2"].map(lambda node: static_pos_country[node][0])
        df_full["y1"] = df_full["node2"].map(lambda node: static_pos_country[node][1])

        # Define the output file path
        output_file = os.path.join(output_dir, f"{country}_within_country_{year}.csv")

        # Save processed data
        df_full.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved: {output_file}")

'''-------------------------------------------------------------------------'''
''' Let's pregenerate organization-level data as well for rendering speed...'''
'''-------------------------------------------------------------------------'''
import runpy
runpy.run_path('preproOrgs.py')

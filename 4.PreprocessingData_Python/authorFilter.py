import pandas as pd
import gcsfs

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\alext\PycharmProjects\covid\COVIDApp_Teghipco\dsapp-440110-ef871808f4ac.json"

# Initialize GCS filesystem
fs = gcsfs.GCSFileSystem(token=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

# Paths to collaboration data files for the years you need
collaboration_data_paths = [
    'gs://covid-dash-app/processed_authors_parquet/author_collab_matrix_2020.parquet',
    'gs://covid-dash-app/processed_authors_parquet/author_collab_matrix_2021.parquet',
    'gs://covid-dash-app/processed_authors_parquet/author_collab_matrix_2022.parquet',
    'gs://covid-dash-app/processed_authors_parquet/author_collab_matrix_2023.parquet',
    'gs://covid-dash-app/processed_authors_parquet/author_collab_matrix_2024.parquet'
]

# Path to the original author mapping CSV file
author_mapping_path = 'gs://covid-dash-app/authorToOrg/authorToOrgCountry.csv'

# Output path for the filtered Parquet file
output_path = 'gs://covid-dash-app/authorToOrg/authorToOrgCountry_filtered.parquet'

unique_authors = set()
for path in collaboration_data_paths:
    with fs.open(path, 'rb') as f:
        df_collab = pd.read_parquet(f)
        unique_authors.update(df_collab['node1'].unique())
        unique_authors.update(df_collab['node2'].unique())


print(f"Total unique authors in collaboration data: {len(unique_authors)}")

# Load the author mapping CSV
with fs.open(author_mapping_path, 'rb') as f:
    df_mapping = pd.read_csv(f)

print(f"Total authors in mapping: {len(df_mapping)}")

# Filter the mapping to include only relevant authors
df_filtered = df_mapping[df_mapping['author_name'].isin(unique_authors)]
df_filtered = df_filtered.drop_duplicates(subset='author_name', keep='first').reset_index(drop=True)

print(f"Filtered mapping has {len(df_filtered)} records")

# Save the filtered mapping as a Parquet file
with fs.open(output_path, 'wb') as f:
    df_filtered.to_parquet(f)
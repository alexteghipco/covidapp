import pandas as pd
final_merged_data = pd.read_parquet(r"D:\Science\covid\pubs_all_with_altmetric.parquet")
authorList = pd.read_parquet(r"C:\Users\alext\Downloads\authorToOrg_authorToOrgCountry_filtered.parquet")

# Convert each numpy.ndarray in `authors` to a list
import numpy as np
final_merged_data['authors'] = final_merged_data['authors'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
print(f"All entries are now lists: {final_merged_data['authors'].apply(lambda x: isinstance(x, list)).all()}")

# Remove duplicate authors within each publication by name
def remove_duplicate_authors(authors):
    unique_authors = {}
    for author in authors:
        name = f"{author['first_name']} {author['last_name']}"
        if name not in unique_authors:  # Only add unique names
            unique_authors[name] = author
    return list(unique_authors.values())

# Apply the function to clean duplicates
final_merged_data['authors'] = final_merged_data['authors'].apply(
    lambda authors: remove_duplicate_authors(authors) if isinstance(authors, list) else authors
)
final_merged_data_exploded = final_merged_data.explode('authors').reset_index(drop=True)

final_merged_data_exploded['concatenated_name'] = final_merged_data_exploded['authors'].apply(
    lambda author: f"{author['first_name']} {author['last_name']}" if isinstance(author, dict) else None
)
final_merged_data_exploded = final_merged_data_exploded.dropna(subset=['concatenated_name']).reset_index(drop=True)
# Verify if there are still duplicates in publication_id and concatenated_name
duplicate_author_entries = final_merged_data_exploded.duplicated(subset=['publication_id', 'concatenated_name'], keep=False)
duplicates = final_merged_data_exploded[duplicate_author_entries]

print("Duplicate author-publication pairs after deduplication:")
print(duplicates[['publication_id', 'concatenated_name']].value_counts())

author_level_data = final_merged_data_exploded.merge(
    authorList[['author_name', 'highest_level_organization_name', 'country_name']],
    left_on='concatenated_name',
    right_on='author_name',
    how='left'
)
author_level_data.drop(columns=['author_name'], inplace=True)
# Fill missing values by assigning directly without inplace=True
author_level_data['highest_level_organization_name'] = author_level_data['highest_level_organization_name'].fillna("Unknown Organization")
author_level_data['country_name'] = author_level_data['country_name'].fillna("Unknown Country")

import pandas as pd

# Components for impact score calculation
components = [
    'field_citation_ratio', 'altmetric_score', 'Diff',
    'patent_count_x', 'dataset_count_x', 'grant_count_x', 'num_unique_orgs'
]

# Step 1: Replace negative values in 'Diff' with 0
author_level_data['Diff'] = author_level_data['Diff'].apply(lambda x: max(x, 0))

# Step 2: Calculate the number of unique publications per author (using `concatenated_name`)
author_level_data['num_publications'] = author_level_data.groupby('concatenated_name')['publication_id'].transform('nunique')
components.append('num_publications')

# Ensure all components are numeric, coercing errors to NaN, and then fill NaNs with 0
for component in components:
    author_level_data[component] = pd.to_numeric(author_level_data[component], errors='coerce').fillna(0)

# Proceed with normalization
for component in components:
    min_value = author_level_data[component].min()
    max_value = author_level_data[component].max()
    range_value = max_value - min_value if max_value != min_value else 1  # Prevent division by zero
    author_level_data[f'normalized_{component}'] = (
        author_level_data[component] - min_value
    ) / range_value

# Step 4: Define weights and calculate impact score
# Define weights for each normalized component
weights = {
    'normalized_field_citation_ratio': 0.2,
    'normalized_altmetric_score': 0.2,
    'normalized_Diff': 0.2,
    'normalized_num_publications': 0.2,
    'normalized_patent_count_x': 0.05,
    'normalized_dataset_count_x': 0.05,
    'normalized_grant_count_x': 0.05,
    'normalized_num_unique_orgs': 0.05
}

# Calculate impact score by multiplying each normalized component by its weight and summing them
author_level_data['impact_score'] = author_level_data[
    list(weights.keys())
].mul(list(weights.values())).sum(axis=1)

# Flatten authors to merge with organization info
flattened_authors = author_level_data[['publication_id', 'concatenated_name']]

# Merge flattened authors with organization information from `authorList`
merged_authors = pd.merge(
    flattened_authors,
    authorList[['author_name', 'highest_level_organization_name']],
    left_on='concatenated_name',
    right_on='author_name',
    how='left'
)

# Merge impact scores with each author
author_impact = pd.merge(
    merged_authors,
    author_level_data[['publication_id', 'impact_score']],
    on='publication_id',
    how='left'
)

# Check for duplicates in `concatenated_name` and `author_name`
print("Duplicate entries in concatenated_name:", flattened_authors['concatenated_name'].duplicated().sum())
print("Duplicate entries in author_name:", authorList['author_name'].duplicated().sum())

flattened_authors = flattened_authors.drop_duplicates(subset=['publication_id', 'concatenated_name']).reset_index(drop=True)

# Merge `flattened_authors` with organization information from `authorList`
merged_authors = pd.merge(
    flattened_authors,
    authorList[['author_name', 'highest_level_organization_name']],
    left_on='concatenated_name',
    right_on='author_name',
    how='left'
)

# Check the result after merging to confirm
print("Shape of merged_authors after merging with authorList:", merged_authors.shape)
print(merged_authors.head())

# Remove duplicate author entries in `author_level_data` for unique `publication_id` and `concatenated_name`
author_level_data = author_level_data.drop_duplicates(subset=['publication_id', 'concatenated_name']).reset_index(drop=True)

# Check shape to confirm duplicates are removed
print("Shape of author_level_data after removing duplicates:", author_level_data.shape)

# Remove duplicate author entries in `flattened_authors` for unique `publication_id` and `concatenated_name`
flattened_authors = flattened_authors.drop_duplicates(subset=['publication_id', 'concatenated_name']).reset_index(drop=True)

# Check shape to confirm duplicates are removed
print("Shape of flattened_authors after removing duplicates:", flattened_authors.shape)

# Merge `flattened_authors` with organization information from `authorList`
merged_authors = pd.merge(
    flattened_authors,
    authorList[['author_name', 'highest_level_organization_name']],
    left_on='concatenated_name',
    right_on='author_name',
    how='left'
)

# Check result after the first merge
print("Shape of merged_authors after merging with authorList:", merged_authors.shape)
print(merged_authors.head())

# Merge `impact_score` from `author_level_data`
author_impact = pd.merge(
    merged_authors,
    author_level_data[['publication_id', 'impact_score']],
    on='publication_id',
    how='left'
)

# Check the result of the second merge
print("Shape of author_impact after merging with impact_score:", author_impact.shape)
print(author_impact.head())

# Keep `publication_id` in the aggregation
author_metrics = merged_authors.groupby(['publication_id', 'concatenated_name']).agg({
    'highest_level_organization_name': 'first'  # Adjust as needed; this assumes one organization per author-publication pair
}).reset_index()

# Confirm the shape and columns
print("Shape of author_metrics after aggregation with publication_id:", author_metrics.shape)
print(author_metrics.head())

def chunked_merge(left_df, right_df, merge_key, chunk_size=100000):
    merged_chunks = []  # To store each chunk's result

    # Process the left DataFrame in chunks
    for start in range(0, len(left_df), chunk_size):
        end = start + chunk_size
        left_chunk = left_df.iloc[start:end]

        # Perform the merge for the current chunk
        merged_chunk = pd.merge(left_chunk, right_df, on=merge_key, how='left')
        merged_chunks.append(merged_chunk)

        # Optional: print progress
        print(f"Merged chunk {start} to {end} of {len(left_df)}")

    # Concatenate all merged chunks
    return pd.concat(merged_chunks, ignore_index=True)

# Merge in chunks
try:
    author_impact = chunked_merge(
        left_df=author_metrics,
        right_df=author_level_data[['publication_id', 'impact_score']],
        merge_key='publication_id',
        chunk_size=100000  # Adjust the chunk size as needed based on available memory
    )

    print("Shape of author_impact after chunked merge:", author_impact.shape)
    print(author_impact.head())

except Exception as e:
    print("Error during chunked merge:", e)

# Check the result of the second merge
print("Shape of author_impact after merging with impact_score:", author_impact.shape)
print(author_impact.head())

# Aggregate total and median impact scores by author
author_scores = author_impact.groupby('concatenated_name')['impact_score'].agg(
    total_impact_score='sum',
    median_impact_score='median'
).reset_index()

# Aggregate total impact scores by organization
organization_scores = author_impact.groupby('highest_level_organization_name')['impact_score'].agg(
    total_impact_score='sum'
).reset_index()

# Count unique authors per organization
authors_per_org = author_impact.groupby('highest_level_organization_name')['concatenated_name'].nunique().reset_index()
authors_per_org = authors_per_org.rename(columns={'concatenated_name': 'num_authors'})

# Merge to calculate average impact score per author in each organization
organization_scores = pd.merge(organization_scores, authors_per_org, on='highest_level_organization_name', how='left')
organization_scores['average_impact_per_author'] = (
    organization_scores['total_impact_score'] / organization_scores['num_authors']
)

# Save author-level and organization-level impact scores to CSV
author_scores.to_csv('author_scores.csv', index=False)
organization_scores.to_csv('organization_scores.csv', index=False)

# List of normalized components
normalized_components = [
    'normalized_field_citation_ratio', 'normalized_altmetric_score', 'normalized_Diff',
    'normalized_patent_count_x', 'normalized_dataset_count_x', 'normalized_grant_count_x', 'normalized_num_unique_orgs'
]

# Aggregate normalized components by averaging per author and organization
author_component_scores = author_impact.groupby('concatenated_name')[normalized_components].mean().reset_index()
organization_component_scores = author_impact.groupby('highest_level_organization_name')[normalized_components].mean().reset_index()

# Top 30 authors by total impact score
top_authors_by_total_score = author_scores.sort_values(by='total_impact_score', ascending=False).head(30)
top_authors_by_total_score.to_csv('top_30_authors_by_total_score.csv', index=False)

# Top 30 organizations by total impact score
top_orgs_by_total_score = organization_scores.sort_values(by='total_impact_score', ascending=False).head(30)
top_orgs_by_total_score.to_csv('top_30_organizations_by_total_score.csv', index=False)
#
# # List of normalized components
# normalized_components = [
#     'normalized_field_citation_ratio', 'normalized_altmetric_score', 'normalized_Diff',
#     'normalized_patent_count_x', 'normalized_dataset_count_x', 'normalized_grant_count_x', 'normalized_num_unique_orgs'
# ]
#
# # Calculate average normalized component scores for each author
# author_component_scores = author_impact.groupby('concatenated_name')[normalized_components].mean().reset_index()
#
# # Calculate average normalized component scores for each organization
# organization_component_scores = author_impact.groupby('highest_level_organization_name')[normalized_components].mean().reset_index()
#
# # Loop over each normalized component to get the top 30 authors and organizations
# for component in normalized_components:
#     # Top 30 authors by each component
#     top_authors_by_component = author_component_scores.sort_values(by=component, ascending=False).head(30)
#     top_authors_by_component.to_csv(f'top_authors_by_{component}.csv', index=False)
#
#     # Top 30 organizations by each component
#     top_orgs_by_component = organization_component_scores.sort_values(by=component, ascending=False).head(30)
#     top_orgs_by_component.to_csv(f'top_organizations_by_{component}.csv', index=False)
#
# # List of normalized components for individual ranking
# normalized_components = [
#     'normalized_field_citation_ratio', 'normalized_altmetric_score', 'normalized_Diff',
#     'normalized_patent_count_x', 'normalized_dataset_count_x', 'normalized_grant_count_x', 'normalized_num_unique_orgs'
# ]
#
# # Loop over each component to find top 30 authors and organizations
# for component in normalized_components:
#     # Top 30 authors by each component
#     top_authors_by_component = author_component_scores.sort_values(by=component, ascending=False).head(30)
#     top_authors_by_component.to_csv(f'top_authors_by_{component}.csv', index=False)
#
#     # Top 30 organizations by each component
#     top_orgs_by_component = organization_component_scores.sort_values(by=component, ascending=False).head(30)
#     top_orgs_by_component.to_csv(f'top_organizations_by_{component}.csv', index=False)

import pandas as pd
authorList = pd.read_parquet(r"C:\Users\alext\Downloads\authorToOrg_authorToOrgCountry_filtered.parquet")
auth_df = pd.read_csv("C:/Users/alext/PycharmProjects/covid/top_30_authors_by_total_score.csv")
merged_auth_df = auth_df.merge(authorList, left_on='concatenated_name', right_on='author_name', how='left')
merged_auth_df = merged_auth_df[['concatenated_name', 'total_impact_score', 'median_impact_score', 'country_name', 'highest_level_organization_name']]
print("Top 30 Authors with Country and Organization:")
print(merged_auth_df.head())






## GET AUTHOR-LEVEL DATA
# Function to get organization information for a list of researcher IDs
from google.cloud import bigquery

# Initialize the BigQuery client
import os
from google.cloud import bigquery

# Path to your JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/alext/PycharmProjects/covid/COVIDApp_Teghipco/dsapp-440110-ef871808f4ac.json"

# Initialize the BigQuery client
client = bigquery.Client()

def get_author_organizations(researcher_ids):
    # Convert the researcher IDs list to a format compatible with BigQuery
    researcher_ids_str = ', '.join([f"'{rid}'" for rid in researcher_ids])

    # Define the query with parameters
    query = f"""
        WITH AuthorAffiliations AS (
            -- Step 1: Extract grid_id for each researcher from affiliations
            SELECT
                a.researcher_id,
                p.publication_id,
                af.grid_id
            FROM
                `covid-19-dimensions-ai.data.publications` AS p,
                UNNEST(p.authors) AS a,  -- Extract each author
                UNNEST(a.affiliations_address) AS af  -- Extract affiliations
            WHERE
                a.researcher_id IN ({researcher_ids_str})  -- Filter by researcher IDs
        ), OrganizationInfo AS (
            -- Step 2: Find organization names and relationships based on grid_id
            SELECT
                g.id AS grid_id,
                g.name AS organization_name,
                r.label AS relationship_label,
                r.type AS relationship_type
            FROM
                `covid-19-dimensions-ai.data.grid` AS g
            LEFT JOIN
                UNNEST(g.relationships) AS r  -- Explode relationships to check for "Parent"
            WHERE
                r.type = "Parent"  -- Only include parent relationships
        )
        -- Join author affiliations with organization info
        SELECT
            aa.researcher_id,
            aa.publication_id,
            oi.organization_name,
            oi.relationship_label,
            oi.relationship_type
        FROM
            AuthorAffiliations AS aa
        LEFT JOIN
            OrganizationInfo AS oi
        ON
            aa.grid_id = oi.grid_id
    """

    # Run the query
    query_job = client.query(query)  # Make an API request
    results = query_job.result()  # Wait for the job to complete

    # Convert results to a list of dictionaries for easy access
    rows = [dict(row) for row in results]
    return rows


# Example usage
researcher_ids = ['r12345', 'r67890']  # Replace with actual researcher IDs
organizations_data = get_author_organizations(researcher_ids)

# Display results
for entry in organizations_data:
    print(entry)

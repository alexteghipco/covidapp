# config.py
# Paths
import os
#PROCESSED_DIR_COUNTRIES = r"D:\Science\covid\matrices\processed_countries"
#PROCESSED_DIR_ORGS_BY_ORG = r"D:\Science\covid\matrices\high_level_processed\processed_by_org"
#PROCESSED_DIR_WITHIN_COUNTRY = r"D:\Science\covid\matrices\processed_within_country"
#PROCESSED_HIERARCHY_DIR = r"D:\Science\covid\matrices\processed_hierarchy"
#COUNTRY_ORG_MAPPING_PATH = os.path.join(PROCESSED_HIERARCHY_DIR, "country_org_mapping.csv")
# Google Cloud Storage bucket paths
PROCESSED_DIR_COUNTRIES = 'gs://covid-dash-app/processed_countries_parquet'
PROCESSED_DIR_ORGS_BY_ORG = 'gs://covid-dash-app/processed_by_org_parquet'
PROCESSED_DIR_WITHIN_COUNTRY = 'gs://covid-dash-app/processed_within_country_parquet'
PROCESSED_HIERARCHY_DIR = 'gs://covid-dash-app/processed_hierarchy_parquet'
COUNTRY_ORG_MAPPING_PATH = f"{PROCESSED_HIERARCHY_DIR}/country_org_mapping.parquet"
PROCESSED_DIR_AUTHORS = 'covid-dash-app/processed_authors_parquet'
AUTHOR_ORG_COUNTRY_MAPPING_PATH = 'covid-dash-app/authorToOrg/authorToOrgCountry_filtered.parquet'

# Constants
AVAILABLE_METRICS = {
    'Degree': 'degree',
    'Betweenness Centrality': 'betweenness',
    'Closeness Centrality': 'closeness',
    'Eigenvector Centrality': 'eigenvector',
    'Collaboration Intensity Index (CII)': 'cii',
}

LAYOUT_OPTIONS = [
    {'label': 'Spring Layout', 'value': 'spring_layout'},
    {'label': 'Kamada-Kawai Layout', 'value': 'kamada_kawai_layout'},
    {'label': 'Circular Layout', 'value': 'circular_layout'},
    {'label': 'Shell Layout', 'value': 'shell_layout'},
    {'label': 'Spectral Layout', 'value': 'spectral_layout'}
]
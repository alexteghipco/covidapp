import os
import pandas as pd

def convert_csv_to_parquet(source_folder, target_folder_suffix="_parquet"):
    # Create target folder path
    target_folder = f"{source_folder}{target_folder_suffix}"
    os.makedirs(target_folder, exist_ok=True)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".csv"):
            # Construct full file path
            csv_file_path = os.path.join(source_folder, filename)

            # Read CSV file into DataFrame
            df = pd.read_csv(csv_file_path)

            # Construct the output Parquet file path
            parquet_file_name = f"{os.path.splitext(filename)[0]}.parquet"
            parquet_file_path = os.path.join(target_folder, parquet_file_name)

            # Convert to Parquet and save
            df.to_parquet(parquet_file_path, index=False)
            print(f"Converted {filename} to {parquet_file_name} and saved in {target_folder}")

PROCESSED_DIR_COUNTRIES = r"D:\Science\covid\matrices\processed_countries"
PROCESSED_DIR_ORGS_BY_ORG = r"D:\Science\covid\matrices\high_level_processed\processed_by_org"
PROCESSED_DIR_WITHIN_COUNTRY = r"D:\Science\covid\matrices\processed_within_country"
PROCESSED_HIERARCHY_DIR = r"D:\Science\covid\matrices\processed_hierarchy"
COUNTRY_ORG_MAPPING_PATH = os.path.join(PROCESSED_HIERARCHY_DIR, "country_org_mapping.csv")

convert_csv_to_parquet(PROCESSED_DIR_COUNTRIES)
convert_csv_to_parquet(PROCESSED_DIR_ORGS_BY_ORG)
convert_csv_to_parquet(PROCESSED_DIR_WITHIN_COUNTRY)
convert_csv_to_parquet(PROCESSED_HIERARCHY_DIR)

import os
import pandas as pd

# List of columns to remove
columns_to_remove = ["normalized", "x0", "y0", "x1", "y1"]

# Function to loop through all Parquet files in a directory, remove specific columns, and save back
def clean_parquet_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            file_path = os.path.join(directory, filename)

            # Load Parquet file into DataFrame
            df = pd.read_parquet(file_path)

            # Drop columns if they exist in the DataFrame
            df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

            # Save the modified DataFrame back to Parquet
            df.to_parquet(file_path, index=False)
            print(f"Processed and saved {filename} without specified columns.")


# Run the function for each of your directories
clean_parquet_files(r'D:\Science\covid\matrices\processed_countries_parquet')
clean_parquet_files(r'D:\Science\covid\matrices\processed_within_country_parquet')
clean_parquet_files(r'D:\Science\covid\matrices\high_level_processed\processed_by_org_parquet')

import pandas as pd

def combine_columns_dropping_duplicates(parquet_file1, parquet_file2, output_file_path):
    # Load both Parquet files into DataFrames
    df1 = pd.read_parquet(parquet_file1)
    df2 = pd.read_parquet(parquet_file2)

    # Drop duplicate columns from df2
    duplicate_columns = df1.columns.intersection(df2.columns)
    df2 = df2.drop(columns=duplicate_columns)

    # Concatenate along columns (axis=1)
    combined_df = pd.concat([df1, df2], axis=1)

    # Save the combined DataFrame as a new Parquet file
    combined_df.to_parquet(output_file_path, index=False)
    print(f"Combined file saved as {output_file_path}")

# Specify file paths
parquet_file1 = r"D:\Science\covid\pub_data_concat.parquet"
parquet_file2 = r"D:\Science\covid\pub_data_additional_concat.parquet"
output_file_path = r"D:\Science\covid\pubs_all.parquet"

# Run the function
combine_columns_dropping_duplicates(parquet_file1, parquet_file2, output_file_path)


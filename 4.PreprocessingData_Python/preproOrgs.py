import os
import pandas as pd
import re
from multiprocessing import Pool

# Paths to input and output directories
input_dir_orgs = r"D:\Science\covid\matrices\high_level_processed"
output_dir_orgs_by_org = os.path.join(input_dir_orgs, "processed_by_org")
os.makedirs(output_dir_orgs_by_org, exist_ok=True)

def sanitize_filename(name):
    """Sanitize the organization name to make it suitable as a file name."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def process_year_file(file):
    print(f"Processing file: {file}")  # Print each file being processed

    # Extract the year from the filename
    year = int(file.replace("processed_high_level", "").replace(".csv", ""))

    # Load the high-level organization data for the current year
    file_path = os.path.join(input_dir_orgs, file)
    df = pd.read_csv(file_path, encoding='utf-8')

    print(f"Loaded data for year {year}, file: {file_path}")  # Confirm data load

    # Dictionary to store rows for each organization
    org_data = {}

    # Process each row and add it to the corresponding organization in org_data
    for _, row in df.iterrows():
        for org in [row['node1'], row['node2']]:
            if org not in org_data:
                org_data[org] = []
            org_data[org].append(row)

    # Write out each organization’s data as a separate CSV file
    for org, rows in org_data.items():
        sanitized_org = sanitize_filename(org)
        output_file = os.path.join(output_dir_orgs_by_org, f"{sanitized_org}_{year}.csv")

        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping: {output_file}")
            continue

        # Save the organization data for the year
        org_df = pd.DataFrame(rows)
        org_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    # Get list of all files to process
    year_files = [
        f for f in os.listdir(input_dir_orgs) if f.startswith("processed_high_level") and f.endswith(".csv")
    ]

    print("Year files to process:", year_files)  # Check if files are found

    # Use multiprocessing to process each year's file in parallel
    if year_files:
        with Pool() as pool:
            pool.map(process_year_file, year_files)
    else:
        print("No files found to process in the input directory.")

    print("Organization-specific preprocessing completed.")


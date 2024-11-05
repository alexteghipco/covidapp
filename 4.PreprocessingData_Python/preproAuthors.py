import os
import pandas as pd

def combine_csv_files(input_folder, output_file_path):
    # List all CSV files in the input folder and sort them
    csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])

    # Initialize an empty list to store each DataFrame
    data_frames = []

    # Read each CSV file, sort, and append to the list
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        print(f"Processing file: {file}")
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Save the combined DataFrame to a Parquet file
    combined_df.to_parquet(output_file_path, index=False)
    print(f"Combined file saved as {output_file_path}")

# Specify the folder containing the CSV files and the output file path
input_folder = r"D:\Science\covid\pubsAdd"
output_file_path = r"D:\Science\covid\pub_data_additional_concat.parquet"

# Run the function
combine_csv_files(input_folder, output_file_path)

# Now let's make these csv files a lot smaller...
import pandas as pd
import os

# Define folder and threshold
folder_path = r"D:\Science\covid\matrices\authors_by_year"
threshold = 3

# List to store edges remaining for each year after applying threshold
edges_remaining = {}

# Iterate through files in the folder
for file_name in os.listdir(folder_path):
    if file_name.startswith("author_collab_matrix_") and file_name.endswith(".csv"):
        # Extract year from filename
        year = file_name.split("_")[-1].split(".")[0]

        # Load the matrix
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # Apply threshold and count edges
        filtered_edges = df[df["weight"] >= threshold]
        edges_remaining[year] = len(filtered_edges)

# Display edges remaining by year
print(edges_remaining)

# Above was for testing good threshold, now apply
import pandas as pd
import os

# Define paths and threshold
folder_path = r"D:\Science\covid\matrices\authors_by_year"
processed_folder = os.path.join(folder_path, "processed_authors")
threshold = 3

# Create processed folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

# Loop through each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        # Load the CSV file
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # Filter rows based on threshold
        filtered_df = df[df["weight"] >= threshold]

        # Define the output path with Parquet format
        output_file_path = os.path.join(processed_folder, file_name.replace(".csv", ".parquet"))

        # Save the DataFrame in Parquet format
        filtered_df.to_parquet(output_file_path, index=False)

        print(f"Processed and saved: {output_file_path}")

print("All files processed and saved in Parquet format.")

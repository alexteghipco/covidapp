import pandas as pd
from scipy.io import loadmat

mat_files = [r'D:\Science\covid\tsne_v2.mat',
             r'D:\Science\covid\final_consensus_cluster_labels.mat',
             r'D:\Science\covid\bc_dip.mat']

combined_data = {}
for file in mat_files:
    mat_data = loadmat(file)
    for var_name, var_value in mat_data.items():
        if not var_name.startswith('__'):
            combined_data[var_name] = var_value.flatten() if var_value.ndim > 1 else var_value


df = pd.DataFrame({key: pd.Series(value) for key, value in combined_data.items()})
df.to_parquet(r'D:\Science\covid\topicClusters.parquet', engine='pyarrow')

''' Now we need to get word clouds'''
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the `topicClusters.parquet` DataFrame
topic_clusters_df = pd.read_parquet(r'D:\Science\covid\topicClusters.parquet', engine='pyarrow')
print("Initial NaN values in finalClustersAll:", topic_clusters_df['finalClustersAll'].isna().sum())

# Step 1: Check if scaling is necessary
min_value = topic_clusters_df['finalClustersAll'].min()
max_value = topic_clusters_df['finalClustersAll'].max()
print(f"Initial range of finalClustersAll: {min_value} to {max_value}")

# Step 2: Normalize `finalClustersAll` values to range from 1 to 20
if 'finalClustersAll' in topic_clusters_df.columns:
    scaler = MinMaxScaler(feature_range=(1, 20))
    topic_clusters_df['finalClustersAll'] = scaler.fit_transform(topic_clusters_df[['finalClustersAll']]).round().astype(int)

topic_clusters_df = pd.read_parquet(r'D:\Science\covid\topicClusters.parquet', engine='pyarrow')
print("Initial NaN values in finalClustersAll:", topic_clusters_df['finalClustersAll'].isna().sum())

# Step 3: Load the `pub_data_concat.parquet` DataFrame
pub_data_df = pd.read_parquet(r'D:\Science\covid\pub_data_concat.parquet', engine='pyarrow')

# Step 4: Merge the two DataFrames on an identifier if needed, or just use them separately
df_combined = pd.DataFrame({
    'abstract_preferred': pub_data_df['abstract_preferred'],
    'finalClustersAll': topic_clusters_df['finalClustersAll']
})
del(pub_data_df)

# Below did not look great--indexing error and we forgot about other languages
# def process_text(text):
#     words = re.findall(r'\b\w+\b', text.lower())  # Tokenize and lowercase
#     return [word for word in words if word not in stop_words]
#
# cluster_word_counts = defaultdict(Counter)
# for cluster, group in df_combined.groupby('finalClustersAll'):
#     # Process all abstracts in the cluster
#     all_words = []
#     for text in group['abstract_preferred'].dropna():
#         all_words.extend(process_text(text))
#     cluster_word_counts[cluster] = Counter(all_words)
#
# global_word_count = Counter()
# for word_counts in cluster_word_counts.values():
#     global_word_count.update(word_counts)
#
# distinctive_words_per_cluster = {}
# for cluster, word_counts in cluster_word_counts.items():
#     distinctive_words = {}
#     for word, count in word_counts.items():
#         if global_word_count[word] > 0:  # Avoid division by zero
#             # Calculate distinctiveness as the ratio of cluster frequency to global frequency
#             distinctiveness_score = count / global_word_count[word]
#             distinctive_words[word] = distinctiveness_score
#     # Sort words by distinctiveness score and take the top 50
#     distinctive_words_per_cluster[cluster] = sorted(distinctive_words.items(), key=lambda x: x[1], reverse=True)[:50]
#
# for cluster, words in distinctive_words_per_cluster.items():
#     print(f"Cluster {cluster}:")
#     for word, score in words:
#         print(f"  {word}: {score:.2f}")
#     print("\n")

##############################################
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Step 1: Detect language and filter by English
def detect_language(text):
    try:
        return detect(text) == 'en'
    except:
        return False  # Return False if language detection fails

# Filter to keep only English abstracts
english_mask = df_combined['abstract_preferred'].apply(detect_language)
non_english_indices = df_combined[~english_mask].index.tolist()
df_combined = df_combined[english_mask]

# Stop words
def process_text(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize and lowercase
    return ' '.join(word for word in words if word not in stop_words)

# Apply text processing to clean abstracts
df_combined['processed_abstract'] = df_combined['abstract_preferred'].dropna().apply(process_text)
print("Initial NaN values in finalClustersAll:", df_combined['finalClustersAll'].isna().sum())

df_combined.to_parquet(r'D:\Science\covid\abstracts_clusters_cleaned.parquet', engine='pyarrow')
english_mask.to_parquet(r'D:\Science\covid\english_mask.parquet', engine='pyarrow')
non_english_indices.to_parquet(r'D:\Science\covid\english_mask.parquet', engine='pyarrow')

print("Initial NaN values in finalClustersAll:", df_combined['finalClustersAll'].isna().sum())

###############################################################################################################################################
#### Let's fix the clusters here it looks like indexing went wonky...we didn't join before removing nans from abstracts like embeddings.npy ###
###############################################################################################################################################
df_combined_orig = pd.read_parquet(r'D:\Science\covid\abstracts_clusters_cleaned.parquet', engine='pyarrow')
df_combined = pd.read_parquet(r'D:\Science\covid\abstracts_clusters_cleaned.parquet', engine='pyarrow')
topic_clusters_df = pd.read_parquet(r'D:\Science\covid\topicClusters.parquet', engine='pyarrow')
english_mask = pd.read_parquet(r'D:\Science\covid\english_mask.parquet')  # Load the saved `english_mask`
if isinstance(english_mask, pd.DataFrame):
    english_mask = english_mask['abstract_preferred']
english_mask = english_mask.astype(bool)

aligned_english_mask = english_mask.reindex(pub_data_df.index, fill_value=False)
pub_data_df_filtered = pub_data_df[aligned_english_mask]
topic_clusters_filtered = topic_clusters_df.iloc[:pub_data_df_filtered.shape[0]]

df_combined = pd.DataFrame({
    'abstract_preferred': pub_data_df_filtered['abstract_preferred'].values,  # Get abstracts
    'finalClustersAll': topic_clusters_filtered['finalClustersAll'].values    # Get corresponding clusters
})

print("Shape of df_combined:", df_combined.shape)
print("NaN values in finalClustersAll:", df_combined['finalClustersAll'].isna().sum())

df_combined.to_parquet(r'D:\Science\covid\abstracts_clusters_cleaned_with_clusters.parquet', engine='pyarrow')

#####################################
### Now get words in each cluster ###
#####################################

def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stop_words]
df_combined['processed_abstract'] = df_combined['abstract_preferred'].dropna().apply(preprocess_text)

# Step 2: Group abstracts by clusters and calculate word frequencies
cluster_word_counts = defaultdict(Counter)
for cluster, group in df_combined.groupby('finalClustersAll'):
    # Flatten all words in the cluster into a single list
    words = [word for abstract in group['processed_abstract'].dropna() for word in abstract]
    # Count word frequencies for the cluster
    cluster_word_counts[cluster] = Counter(words)

# Step 3: Identify the most unique words per cluster
unique_words_per_cluster = {}

# Calculate global word frequencies to compare
global_word_count = Counter(word for words in df_combined['processed_abstract'].dropna() for word in words)
for cluster, word_counts in cluster_word_counts.items():
    # Calculate a distinctiveness score for each word in the cluster
    distinctive_words = {
        word: (count / global_word_count[word]) if global_word_count[word] > 0 else 0
        for word, count in word_counts.items()
    }
    # Sort words by their distinctiveness score and keep the top 50
    unique_words_per_cluster[cluster] = sorted(distinctive_words.items(), key=lambda x: x[1], reverse=True)[:50]

for cluster, words in unique_words_per_cluster.items():
    print(f"Cluster {cluster}:")
    for word, score in words:
        print(f"  {word}: {score:.2f}")
    print("\n")

distinctiveness_dict = defaultdict(dict)
for cluster, words in unique_words_per_cluster.items():
    for word, score in words:
        distinctiveness_dict[word][f"Cluster_{cluster}"] = score
distinctiveness_df = pd.DataFrame(distinctiveness_dict).T.fillna(0)  # Transpose to have words as rows
csv_path = 'D:\Science\covid\word_cluster_distinctiveness.csv'
distinctiveness_df.to_csv(csv_path, index_label="Word")

######################################################
# Distinctiveness is not that great-- GET TF-IDF
######################################################
word_cluster_matrix = pd.DataFrame(cluster_word_counts).fillna(0)
total_counts_per_cluster = word_cluster_matrix.sum(axis=0)
normalized_word_cluster_matrix = word_cluster_matrix.div(total_counts_per_cluster, axis=1).fillna(0)
normalized_word_cluster_matrix.to_csv(r'D:\Science\covid\normalized_word_cluster_matrix.csv')

import numpy as np
total_counts_per_cluster = word_cluster_matrix.sum(axis=0)
tf_matrix = word_cluster_matrix.div(total_counts_per_cluster, axis=1).fillna(0)
num_clusters = word_cluster_matrix.shape[1]
word_in_clusters = (word_cluster_matrix > 0).sum(axis=1)
idf = np.log((num_clusters / word_in_clusters).replace(0, np.nan))  # Avoid division by zero with np.nan
tf_idf_matrix = tf_matrix.multiply(idf, axis=0).fillna(0)
tf_idf_matrix.to_csv(r'D:\Science\covid\tf_idf_word_cluster_matrix.csv')

###############################################################
# ABSTRACTS ARE FASCINATING BUT TOO LABORIOUS--SWITCH TO TITLES
###############################################################
pub_data_df_filtered = pub_data_df[aligned_english_mask]
topic_clusters_filtered = topic_clusters_df.iloc[:pub_data_df_filtered.shape[0]]
# Use .loc to assign the processed title to the filtered DataFrame to avoid SettingWithCopyWarning
pub_data_df_filtered = pub_data_df_filtered.copy()  # Create a copy to ensure we're not working on a view
def process_text(text):
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize and lowercase
    return ' '.join(word for word in words if word not in stop_words)
pub_data_df_filtered.loc[:, 'processed_title'] = pub_data_df_filtered['title_preferred'].apply(process_text)

df_combined_filtered = pub_data_df_filtered.copy()
df_combined_filtered['finalClustersAll'] = topic_clusters_filtered['finalClustersAll'].values

# Dictionary to store TF-IDF scores for each cluster
tfidf_results = {}
from sklearn.feature_extraction.text import TfidfVectorizer
# Loop through each cluster and calculate TF-IDF for titles
for cluster, group in df_combined_filtered.groupby('finalClustersAll'):
    titles = group['processed_title'].tolist()

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles)

    # Get feature names (words) and TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1  # Sum scores across all documents in the cluster

    # Create a dictionary of word: score pairs for the current cluster
    tfidf_scores = dict(zip(feature_names, scores))

    # Store the scores in tfidf_results with words as keys and scores for the current cluster
    for word, score in tfidf_scores.items():
        if word not in tfidf_results:
            tfidf_results[word] = {}
        tfidf_results[word][cluster] = score  # Assign score to the specific cluster

# Convert tfidf_results to a DataFrame
tfidf_df = pd.DataFrame(tfidf_results).T  # Transpose so rows are words and columns are clusters
tfidf_df = tfidf_df.fillna(0)  # Fill NaN with 0 for words that don't appear in some clusters

# Save the TF-IDF matrix to CSV
tfidf_df.to_csv(r'D:\Science\covid\tfidf_matrix_titles.csv')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

cluster_texts = df_combined_filtered.groupby('finalClustersAll')['processed_title'].apply(lambda x: ' '.join(x))

# now with clusters as "documents"
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cluster_texts)
feature_names = vectorizer.get_feature_names_out()
tf_icf_df = pd.DataFrame(tfidf_matrix.toarray(), index=cluster_texts.index, columns=feature_names)
tf_icf_df.to_csv(r'D:\Science\covid\tf_icf_matrix_titles.csv')

tf_icf_df = tf_icf_df.T
tf_icf_df.columns = [f"Cluster_{col}" for col in tf_icf_df.columns]
tf_icf_df.to_csv(r'D:\Science\covid\tf_icf_matrix_titles.csv')

############################## ENHANCE DISTINCTIVENESS BY TREATING CLUSTERS AS DOCS
# Concatenate all titles in each cluster into a single "document"
cluster_texts = df_combined_filtered.groupby('finalClustersAll')['processed_title'].apply(lambda x: ' '.join(x))

# Calculate TF-IDF across all clusters (treating each cluster as a document)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cluster_texts)
feature_names = vectorizer.get_feature_names_out()
tf_icf_df = pd.DataFrame(tfidf_matrix.toarray(), index=cluster_texts.index, columns=feature_names)

# Normalize TF-IDF scores within each cluster by subtracting the mean score for each word across clusters
mean_tfidf = tf_icf_df.mean(axis=0)
tf_icf_normalized_df = tf_icf_df - mean_tfidf

tf_icf_normalized_df = tf_icf_normalized_df.T
tf_icf_normalized_df.columns = [f"Cluster_{col}" for col in tf_icf_normalized_df.columns]
tf_icf_normalized_df.to_csv(r'D:\Science\covid\tf_icf_normalized_matrix.csv')

top_words_per_cluster = set()
for cluster in tf_icf_normalized_df.columns:
    # Get the top 500 words for the current cluster based on normalized TF-IDF score
    top_words = tf_icf_normalized_df[cluster].nlargest(500).index
    top_words_per_cluster.update(top_words)  # Add to the set of unique top words across clusters
top_words_per_cluster = list(top_words_per_cluster)
filtered_tf_icf_df = tf_icf_normalized_df.loc[top_words_per_cluster]
filtered_tf_icf_df.to_csv(r'D:\Science\covid\tf_icf_normalized_matrix_top500.csv')


# CHANGES OVER TIME
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load data and mask
pub_data_df = pd.read_parquet(r'D:\Science\covid\pub_data_concat.parquet', engine='pyarrow')
topic_clusters_df = pd.read_parquet(r'D:\Science\covid\topicClusters.parquet', engine='pyarrow')
english_mask = pd.read_parquet(r'D:\Science\covid\english_mask.parquet')

# Convert english_mask to boolean Series if necessary
if isinstance(english_mask, pd.DataFrame):
    english_mask = english_mask['abstract_preferred']
english_mask = english_mask.astype(bool)

# Step 2: Align the English mask with pub_data_df
aligned_english_mask = english_mask.reindex(pub_data_df.index, fill_value=False)
pub_data_df_filtered = pub_data_df[aligned_english_mask]

# Step 3: Align topic_clusters_df with pub_data_df_filtered
topic_clusters_filtered = topic_clusters_df.iloc[:pub_data_df_filtered.shape[0]]

# Step 4: Combine filtered data into a single DataFrame
df_combined = pd.DataFrame({
    'abstract_preferred': pub_data_df_filtered['abstract_preferred'].values,  # Get abstracts
    'year': pub_data_df_filtered['year'].values,                              # Get year information
    'finalClustersAll': topic_clusters_filtered['finalClustersAll'].values    # Get corresponding clusters
})

# Check alignment and NaN values
print("Shape of df_combined:", df_combined.shape)
print("NaN values in finalClustersAll:", df_combined['finalClustersAll'].isna().sum())

# Step 5: Group by year and cluster, then count occurrences
cluster_counts_by_year = df_combined.groupby(['year', 'finalClustersAll']).size().reset_index(name='count')

# Step 6: Plot the number of items in each cluster over time
plt.figure(figsize=(12, 8))
for cluster in cluster_counts_by_year['finalClustersAll'].unique():
    cluster_data = cluster_counts_by_year[cluster_counts_by_year['finalClustersAll'] == cluster]
    plt.plot(cluster_data['year'], cluster_data['count'], marker='o', label=f'Cluster {cluster}')

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Number of Items')
plt.title('Cluster Distribution Over Time')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Calculate year-over-year growth rate for each cluster
cluster_counts_by_year['growth_rate'] = cluster_counts_by_year.groupby('finalClustersAll')['count'].pct_change() * 100

# Plot growth rates instead of counts
plt.figure(figsize=(12, 8))
for cluster in cluster_counts_by_year['finalClustersAll'].unique():
    cluster_data = cluster_counts_by_year[cluster_counts_by_year['finalClustersAll'] == cluster]
    plt.plot(cluster_data['year'], cluster_data['growth_rate'], marker='o', label=f'Cluster {cluster}')

plt.xlabel('Year')
plt.ylabel('Year-over-Year Growth Rate (%)')
plt.title('Cluster Growth Rate Over Time')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Calculate total publications per year and relative share as before
cluster_counts_by_year['total_count'] = cluster_counts_by_year.groupby('year')['count'].transform('sum')
cluster_counts_by_year['relative_share'] = (cluster_counts_by_year['count'] / cluster_counts_by_year['total_count']) * 100

# Step 2: Calculate the slope of the relative share for each cluster
slopes = []
for cluster in cluster_counts_by_year['finalClustersAll'].unique():
    cluster_data = cluster_counts_by_year[cluster_counts_by_year['finalClustersAll'] == cluster]
    X = cluster_data['year'].values.reshape(-1, 1)  # Reshape for sklearn
    y = cluster_data['relative_share'].values

    # Fit a linear regression model to get the slope
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]  # Slope of the relative share over time
    slopes.append((cluster, slope))

# Create a DataFrame of slopes for plotting
slopes_df = pd.DataFrame(slopes, columns=['Cluster', 'Slope']).sort_values(by='Slope', ascending=False)

# Step 3: Plot the slope of the relative share as a bar chart
plt.figure(figsize=(10, 6))
plt.barh(slopes_df['Cluster'], slopes_df['Slope'], color='skyblue')
plt.xlabel('Slope of Relative Share (%)')
plt.ylabel('Cluster')
plt.title('Slope of Relative Share of Each Cluster Over Time')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming `cluster_counts_by_year` is available with 'year', 'finalClustersAll', and 'count'

# Step 1: Calculate the total number of publications for each year
cluster_counts_by_year['total_count'] = cluster_counts_by_year.groupby('year')['count'].transform('sum')

# Step 2: Calculate the relative share of each cluster for each year
cluster_counts_by_year['relative_share'] = (cluster_counts_by_year['count'] / cluster_counts_by_year['total_count']) * 100

# Step 3: Plot the relative share over time for each cluster
plt.figure(figsize=(12, 8))
for cluster in cluster_counts_by_year['finalClustersAll'].unique():
    cluster_data = cluster_counts_by_year[cluster_counts_by_year['finalClustersAll'] == cluster]
    plt.plot(cluster_data['year'], cluster_data['relative_share'], marker='o', label=f'Cluster {cluster}')

plt.xlabel('Year')
plt.ylabel('Relative Share (%)')
plt.title('Relative Share of Each Cluster in Proportion to Total Publications Over Time')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

unique_clusters = cluster_counts_by_year['finalClustersAll'].unique()
cluster_mapping = {old: new for old, new in zip(unique_clusters, range(1, 21))}
cluster_counts_by_year['finalClustersAll'] = cluster_counts_by_year['finalClustersAll'].map(cluster_mapping)

import gcsfs
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\alext\PycharmProjects\covid\COVIDApp_Teghipco\dsapp-440110-ef871808f4ac.json"
fs = gcsfs.GCSFileSystem(token=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
output_path = 'gs://covid-dash-app/wordClouds/topicTrends.parquet'
# Save the filtered mapping as a Parquet file
with fs.open(output_path, 'wb') as f:
    cluster_counts_by_year.to_parquet(f)

# need to fix Y2
import pandas as pd
import scipy.io

# Load the .mat file
mat_path = "D:\\Science\\covid\\tsne_v3.mat"
mat_data = scipy.io.loadmat(mat_path)

# Extract Y2 and finalClustersAll
Y2_matrix = mat_data.get('Y2')
final_clusters_all = mat_data.get('finalClustersAll')

# Check if both Y2 and finalClustersAll were successfully loaded
if Y2_matrix is None or final_clusters_all is None:
    raise KeyError("One or both of the keys 'Y2' and 'finalClustersAll' are missing from the .mat file.")

# Ensure Y2 is in matrix form (2D list of lists)
if Y2_matrix.ndim != 2:
    raise ValueError("Y2 is not a matrix. It should be a 2D array.")

# Convert to DataFrame
df = pd.DataFrame({
    'Y2': Y2_matrix.tolist(),  # This keeps Y2 as a matrix (list of lists)
    'finalClustersAll': final_clusters_all.flatten()  # Flatten if necessary
})

# Save to Parquet
parquet_path = "D:\\Science\\covid\\tsne.parquet"
df.to_parquet(parquet_path, index=False)
output_path = 'gs://covid-dash-app/wordClouds/tsne.parquet'
with fs.open(output_path, 'wb') as f:
    df.to_parquet(f)

import pandas as pd
import scipy.io

# Load the .mat file
mat_path = "D:\\Science\\covid\\bc_dip.mat"
mat_data = scipy.io.loadmat(mat_path)

# Convert the loaded .mat data to a DataFrame
# We exclude metadata fields (starting with '__') typically included in .mat files
data = {key: mat_data[key].flatten() for key in mat_data if not key.startswith('__')}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to Parquet
parquet_path = "D:\\Science\\covid\\clusterEval.parquet"
df.to_parquet(parquet_path, index=False)
print(f"Data successfully saved to {parquet_path}")
output_path = 'gs://covid-dash-app/wordClouds/clusterEval.parquet'
with fs.open(output_path, 'wb') as f:
    df.to_parquet(f)

df = pd.read_csv(r"D:\Science\covid\altmetric2\bayesian_optimization_predictions_checkpoint_altmetric.csv")
output_path = 'gs://covid-dash-app/ML/preds.parquet'
# Save the filtered mapping as a Parquet file
with fs.open(output_path, 'wb') as f:
    df.to_parquet(f)

''' Let's get metric '''
# First, add altmetric residual
#Publication ID	Actual Altmetric Score	Predicted Altmetric Score
import pandas as pd
alt = pd.read_csv(r"D:\Science\covid\altmetric2\bayesian_optimization_predictions_checkpoint_altmetric.csv")
parquet_data = pd.read_parquet(r"D:\Science\covid\pub_data_additional_concat.parquet")
alt['Diff'] = alt['Actual Altmetric Score'] - alt['Predicted Altmetric Score']
merged_data = parquet_data.merge(alt[['Publication ID', 'Diff']],
                                 left_on='publication_id',
                                 right_on='Publication ID',
                                 how='left')
merged_data.drop(columns=['Publication ID'], inplace=True)
parquet_data2 = pd.read_parquet(r"D:\Science\covid\pubs_all.parquet")
final_merged_data = merged_data.merge(parquet_data2, on='publication_id', how='left')
print(final_merged_data.head())
final_merged_data.to_parquet(r"D:\Science\covid\pubs_all_with_altmetric.parquet", index=False)

def concatenate_names(authors):
    return [f"{author['first_name']} {author['last_name']}" for author in authors]

# Create a new column 'concatenated_name' in final_merged_data with concatenated names from 'authors'
final_merged_data['concatenated_name'] = final_merged_data['authors'].apply(concatenate_names)

# Step 2: Explode 'concatenated_name' so each name has its own row for easier matching
final_merged_data_exploded = final_merged_data.explode('concatenated_name').reset_index(drop=True)

# Perform the merge to add organization and country information
authorList = pd.read_parquet(r"C:\Users\alext\Downloads\authorToOrg_authorToOrgCountry_filtered.parquet")
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

# We will manually get reasercher id...
author_level_data.to_parquet(r"D:\Science\covid\author_pubs.parquet", index=False)

# switched to procTop.py...different computer
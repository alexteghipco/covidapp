# ./pages/themes.py

from dash import html, dcc, callback, Output, Input
import plotly.graph_objs as go
import pandas as pd
import gcsfs
import numpy as np
from dash.exceptions import PreventUpdate
from cache import cache
#from . import config  # Adjust the import path based on your project structure
import ast  # For safer evaluation of string representations
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
import io

# Define a list of 20 distinct colors (using Matplotlib's tab20 palette)
DISTINCT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# Define a mapping from cluster numbers (1-20) to cluster IDs and names
CLUSTER_MAPPING = {
    1: {"id": 20, "name": "Social Impacts and Public Health"},
    2: {"id": 174, "name": "Medical and Patient-Focused Research"},
    3: {"id": 263, "name": "Nutritional and Prevention Research"},
    4: {"id": 312, "name": "Policy and Governance"},
    5: {"id": 328, "name": "Clinical Trials and Patient Outcomes"},
    6: {"id": 409, "name": "Educational Changes"},
    7: {"id": 413, "name": "Digital and Remote Learning"},
    8: {"id": 429, "name": "Social Media and Information Dissemination"},
    9: {"id": 505, "name": "Disease Spread and Impact"},
    10: {"id": 585, "name": "Health Risks and Vulnerabilities"},
    11: {"id": 643, "name": "Pandemic Response and Learning Systems"},
    12: {"id": 668, "name": "COVID and Sociocultural Dynamics"},
    13: {"id": 687, "name": "Environmental Impact and Behavior"},
    14: {"id": 759, "name": "Psychological and Social Support"},
    15: {"id": 763, "name": "Public Health and Healthcare Systems"},
    16: {"id": 886, "name": "Artificial Intelligence and Data Analysis"},
    17: {"id": 896, "name": "Social Distancing and Community Response"},
    18: {"id": 930, "name": "Digital Health and Telemedicine"},
    19: {"id": 967, "name": "Clinical Evidence and Treatment"},
    20: {"id": 971, "name": "Global and Policy Impact"}
}

# Create a reverse mapping from cluster ID to cluster number
CLUSTER_ID_TO_NUMBER = {v['id']: k for k, v in CLUSTER_MAPPING.items()}

def create_empty_figure(title="No Data Available"):
    """
    Creates an empty Plotly figure with a specified title.

    Parameters:
        title (str): The title to display on the empty figure.

    Returns:
        go.Figure: An empty Plotly figure with annotations.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": "No data available",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 20}
        }],
        plot_bgcolor='white'
    )
    return fig

def load_tsne_data():
    """
    Loads the tsne.parquet data from Google Cloud Storage.

    Returns:
        pd.DataFrame: DataFrame containing t-SNE embeddings and cluster assignments.
    """
    @cache.memoize(timeout=600)  # Cache for 10 minutes
    def _load():
        try:
            fs = gcsfs.GCSFileSystem()
            file_path = 'gs://covid-dash-app/wordClouds/tsne.parquet'
            with fs.open(file_path, 'rb') as f:
                df_tsne = pd.read_parquet(f)
            print(f"Loaded tsne.parquet with shape: {df_tsne.shape}")
            print("Columns in tsne.parquet:", df_tsne.columns.tolist())
            print("Sample 'Y2' entries before conversion:", df_tsne['Y2'].head(5).tolist())
            return df_tsne
        except Exception as e:
            print(f"Error loading tsne.parquet: {e}")
            return pd.DataFrame()

    return _load()

def load_cluster_eval_data():
    """
    Loads the clusterEval.parquet data from Google Cloud Storage.

    Returns:
        pd.DataFrame: DataFrame containing cluster evaluation metrics.
    """
    @cache.memoize(timeout=600)  # Cache for 10 minutes
    def _load():
        try:
            fs = gcsfs.GCSFileSystem()
            file_path = 'gs://covid-dash-app/wordClouds/clusterEval.parquet'
            with fs.open(file_path, 'rb') as f:
                df_eval = pd.read_parquet(f)
            print(f"Loaded clusterEval.parquet with shape: {df_eval.shape}")
            print("Columns in clusterEval.parquet:", df_eval.columns.tolist())
            return df_eval
        except Exception as e:
            print(f"Error loading clusterEval.parquet: {e}")
            return pd.DataFrame()

    return _load()

def load_wordcloud_data():
    """
    Loads the tf_icf_normalized_matrix_top500.csv from Google Cloud Storage.

    Returns:
        pd.DataFrame: DataFrame containing words and their tf-idf scores per cluster.
    """
    @cache.memoize(timeout=600)  # Cache for 10 minutes
    def _load():
        try:
            fs = gcsfs.GCSFileSystem()
            file_path = 'gs://covid-dash-app/wordClouds/tf_icf_normalized_matrix_top500.csv'
            with fs.open(file_path, 'r', encoding='utf-8') as f:
                df_wc = pd.read_csv(f, encoding='utf-8', index_col=0)
            print(f"Loaded tf_icf_normalized_matrix_top500.csv with shape: {df_wc.shape}")
            print("Columns in tf_icf_normalized_matrix_top500.csv:", df_wc.columns.tolist())
            return df_wc
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
            try:
                with fs.open(file_path, 'r', encoding='ISO-8859-1') as f:
                    df_wc = pd.read_csv(f, encoding='ISO-8859-1', index_col=0)
                print(f"Loaded tf_icf_normalized_matrix_top500.csv with 'ISO-8859-1' encoding and shape: {df_wc.shape}")
                print("Columns in tf_icf_normalized_matrix_top500.csv:", df_wc.columns.tolist())
                return df_wc
            except Exception as e_inner:
                print(f"Error loading tf_icf_normalized_matrix_top500.csv with 'ISO-8859-1' encoding: {e_inner}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading tf_icf_normalized_matrix_top500.csv: {e}")
            return pd.DataFrame()

    return _load()

def generate_wordcloud_image(words, scores, color):
    """
    Generates a word cloud image from words and their corresponding scores.

    Parameters:
        words (list): List of words.
        scores (list): Corresponding tf-idf scores for the words.
        color (str): Hex color code for the word cloud.

    Returns:
        str: Base64 encoded image string.
    """
    try:
        # Create a dictionary of words and their scores
        word_freq = dict(zip(words, scores))

        # Initialize WordCloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            prefer_horizontal=1.0,
            repeat=False
        )

        # Generate the word cloud
        wc.generate_from_frequencies(word_freq)

        # Recolor words based on the cluster color
        def single_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return color

        wc.recolor(color_func=single_color_func)

        # Save the word cloud to a bytes buffer
        buffer = io.BytesIO()
        wc.to_image().save(buffer, format='PNG')
        buffer.seek(0)

        # Encode the image to base64
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return img_base64
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

def create_tsne_plot(df_tsne, sample_fraction=0.7):
    try:
        if df_tsne.empty:
            print("tsne.parquet is empty. Returning empty figure.")
            return create_empty_figure("t-SNE Embedding - No Data Available")

        if sample_fraction < 1.0:
            df_tsne = df_tsne.sample(frac=sample_fraction, random_state=42)
            print(f"Sampled tsne.parquet to {sample_fraction * 100}% of data.")

        # Check if required columns exist
        required_columns = {'Y2', 'finalClustersAll'}
        if not required_columns.issubset(df_tsne.columns):
            missing = required_columns - set(df_tsne.columns)
            print(f"Missing required columns in tsne.parquet: {missing}")
            return create_empty_figure("t-SNE Embedding - Missing Data")

        # Attempt to parse 'Y2'
        if df_tsne['Y2'].dtype == object:
            print("Converting 'Y2' from string to list.")
            def safe_parse(x):
                try:
                    return ast.literal_eval(x) if isinstance(x, str) else x
                except Exception as e:
                    print(f"Failed to parse 'Y2': {x}, error: {e}")
                    return None

            df_tsne['Y2_parsed'] = df_tsne['Y2'].apply(safe_parse)
            print("Sample 'Y2_parsed' entries:", df_tsne['Y2_parsed'].head(5).tolist())
        else:
            df_tsne['Y2_parsed'] = df_tsne['Y2']

        # Filter valid 'Y2' entries
        valid_Y2 = df_tsne['Y2_parsed'].apply(
            lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2 and all(isinstance(i, (int, float)) for i in x)
        )
        print(f"Number of valid 'Y2' entries: {valid_Y2.sum()} out of {len(valid_Y2)}")
        df_tsne = df_tsne[valid_Y2]

        if df_tsne.empty:
            print("All 'Y2' entries were invalid after filtering. Returning empty figure.")
            return create_empty_figure("t-SNE Embedding - No Valid Data")

        # Now use 'Y2_parsed'
        df_tsne['X_tsne'] = df_tsne['Y2_parsed'].apply(lambda x: x[0])
        df_tsne['Y_tsne'] = df_tsne['Y2_parsed'].apply(lambda x: x[1])

        # **Debugging Information**
        unique_clusters = df_tsne['finalClustersAll'].unique()
        unique_clusters_sorted = sorted(unique_clusters)
        cluster_types = df_tsne['finalClustersAll'].apply(type).unique()
        print(f"Unique clusters in 'finalClustersAll': {unique_clusters_sorted}")
        print(f"Data types in 'finalClustersAll': {cluster_types}")

        # **Map cluster IDs to cluster numbers using CLUSTER_ID_TO_NUMBER**
        df_tsne['ClusterNumber'] = df_tsne['finalClustersAll'].map(CLUSTER_ID_TO_NUMBER)
        print(f"Sample 'ClusterNumber' entries:", df_tsne['ClusterNumber'].head(5).tolist())

        # **Handle clusters without a mapping**
        unmapped_clusters = df_tsne[df_tsne['ClusterNumber'].isna()]['finalClustersAll'].unique()
        if len(unmapped_clusters) > 0:
            print(f"Clusters with no mapping: {unmapped_clusters}")
            # Optionally, you can decide to drop these or assign a default value
            # Here, we'll drop them
            df_tsne = df_tsne.dropna(subset=['ClusterNumber'])
            print(f"Dropped clusters with no mapping. New shape: {df_tsne.shape}")

        # **Convert 'ClusterNumber' to integer**
        df_tsne['ClusterNumber'] = df_tsne['ClusterNumber'].astype(int)

        # **Map ClusterNumber to colors**
        color_mapping = {cluster_num: DISTINCT_COLORS[cluster_num - 1] for cluster_num in df_tsne['ClusterNumber'].unique()}
        print(f"Color mapping: {color_mapping}")

        # **Map ClusterNumber to Cluster Names**
        df_tsne['ClusterName'] = df_tsne['ClusterNumber'].apply(lambda x: CLUSTER_MAPPING[x]['name'] if x in CLUSTER_MAPPING else 'Unknown Cluster')

        # **Additional Debug Statement to Verify Mapping**
        unmapped_clusters_after = df_tsne[df_tsne['ClusterName'] == 'Unknown Cluster']['finalClustersAll'].unique()
        if len(unmapped_clusters_after) > 0:
            print(f"Clusters still with no mapping after processing: {unmapped_clusters_after}")

        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=df_tsne['X_tsne'],
            y=df_tsne['Y_tsne'],
            mode='markers',
            marker=dict(
                size=5,
                color=df_tsne['ClusterNumber'].map(color_mapping),
                showscale=False,  # Hide the colorscale as we're using discrete colors
                opacity=0.6  # Reduce opacity for overlapping points
            ),
            hovertext=df_tsne['ClusterName'],  # Use hovertext for tooltip
            hoverinfo='text',  # Specify that only text should be shown on hover
        ))

        fig.update_layout(
            title='t-SNE of LLM Abstract Embeddings with Consensus Cluster Solution',
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            hovermode='closest',
            dragmode='zoom',  # Allow zooming
            plot_bgcolor='white',  # Set plot background to white
            paper_bgcolor='white'  # Set paper background to white
        )

        print("t-SNE plot created successfully.")
        return fig
    except Exception as e:
        print(f"Error in create_tsne_plot: {e}")
        import traceback
        print(traceback.format_exc())
        return create_empty_figure("t-SNE Plotting Error")

def create_line_plot(df_eval):
    try:
        fig = go.Figure()

        # Add 'BC' line on the primary y-axis with specific color
        fig.add_trace(go.Scatter(
            x=df_eval['k'],
            y=df_eval['bc'],
            mode='lines+markers',
            name='BC',
            line=dict(color='#7209b7'),  # Updated color
            marker=dict(size=6),
            yaxis='y1'
        ))

        # Add 'Dip' line on the secondary y-axis with specific color
        fig.add_trace(go.Scatter(
            x=df_eval['k'],
            y=df_eval['dip'],
            mode='lines+markers',
            name='Dip',
            line=dict(color='#ff006e'),  # Specific color for 'dip'
            marker=dict(size=6),
            yaxis='y2'
        ))

        # Highlight the BC value at k=20 with a larger dot
        target_k = 20
        target_points = df_eval[df_eval['k'] == target_k]

        if not target_points.empty:
            for _, target_point in target_points.iterrows():
                fig.add_trace(go.Scatter(
                    x=[target_point['k']],
                    y=[target_point['bc']],
                    mode='markers',
                    name=f'Selected model k={target_k}',
                    marker=dict(color='#7209b7', size=12, symbol='circle'),  # Larger dot, same color as 'BC'
                    hoverinfo='text',
                    text=[f"Selected model k={target_k}: {target_point['bc']}"],
                    yaxis='y1'
                ))
        else:
            print(f"No data point found at k={target_k} to highlight.")

        # Identify continuous ranges where 'dipP' > 0.05
        dipP_mask = df_eval['dipP'] > 0.05
        dipP_indices = df_eval.index[dipP_mask].tolist()
        continuous_ranges = find_continuous_ranges(dipP_indices)

        # Add rectangle shapes for each continuous range
        for start_idx, end_idx in continuous_ranges:
            start_k = df_eval.loc[start_idx, 'k']
            end_k = df_eval.loc[end_idx, 'k']
            fig.add_shape(
                type="rect",
                x0=start_k,
                y0=0,
                x1=end_k,
                y1=1,
                yref="paper",
                fillcolor="#CBDCEB",
                opacity=0.5,
                layer="below",
                line=dict(width=0)
            )

        # Add a dummy trace for the blue box legend entry
        fig.add_trace(go.Scatter(
            x=[None],  # Invisible placeholder point
            y=[None],
            mode='lines',
            line=dict(color='#CBDCEB', width=0),  # Match blue box color, invisible line
            showlegend=True,
            name="Dip p > 0.05"  # Label for the blue box
        ))

        # Update layout for dual y-axes and white background
        fig.update_layout(
            title='Consensus of LLM Abstract Embedding Clustering',
            xaxis=dict(
                title='Clustering Solution (K)',
                showgrid=False,
                zeroline=False,
                showline=True,
                ticks='outside',  # Show ticks on the x-axis
            ),
            yaxis=dict(
                title='Bimodality Coefficient',
                side='left',
                showgrid=False,
                zeroline=False,
                showline=True
            ),
            yaxis2=dict(
                title='Dip Statistic',
                side='right',
                overlaying='y',
                showgrid=False,
                zeroline=False,
                showline=True
            ),
            legend=dict(
                title="Metrics",
                x=1.05,  # Position legend to the far right
                y=1,
                traceorder='normal',
                font=dict(
                    size=12,
                ),
                bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                bordercolor='rgba(0, 0, 0, 0)'  # No border
            ),
            hovermode='closest',
            dragmode='zoom',
            plot_bgcolor='white',  # Set plot background to white
            paper_bgcolor='white'  # Set paper background to white
        )

        return fig
    except Exception as e:
        print(f"Error in create_line_plot: {e}")
        import traceback
        print(traceback.format_exc())
        return create_empty_figure("Line Plotting Error")


@callback(
    Output('themes-line-plot', 'figure'),
    Input('url', 'pathname')
)
def update_line_plot(pathname):
    print(f"Callback Triggered: update_line_plot with pathname={pathname}")
    if pathname != '/themes':
        print("Not the Themes page. Preventing update.")
        raise PreventUpdate

    df_eval = load_cluster_eval_data()

    if df_eval.empty:
        print("clusterEval.parquet is empty. Returning empty figure.")
        return go.Figure(
            data=[],
            layout=go.Layout(
                title='BC and Dip over K - No Data Available',
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[{
                    'text': "No data available for the line plot.",
                    'xref': "paper",
                    'yref': "paper",
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
        )

    print("Creating line plot figure.")
    fig = create_line_plot(df_eval)

    print("Line plot figure created successfully.")
    return fig


@callback(
    Output('themes-tsne-plot', 'figure'),
    Input('url', 'pathname')
)
def update_tsne_plot(pathname):
    print(f"Callback Triggered: update_tsne_plot with pathname={pathname}")
    # if pathname != '/themes':
    #     print("Not the Themes page. Preventing update.")
    #     raise PreventUpdate
    if pathname != '/themes':
        print("Not the Themes page. Preventing update.")
        raise PreventUpdate

    df_tsne = load_tsne_data()

    if df_tsne.empty:
        print("tsne.parquet is empty. Returning empty figure.")
        return go.Figure(
            data=[],
            layout=go.Layout(
                title='t-SNE Embedding - No Data Available',
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[{
                    'text': "No data available for the t-SNE plot.",
                    'xref': "paper",
                    'yref': "paper",
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
        )

    # Ensure 'Y2' is a list or array of two elements
    if df_tsne['Y2'].dtype == object:
        print("Converting 'Y2' from string to list.")
        df_tsne['Y2'] = df_tsne['Y2'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    print("Creating t-SNE plot figure.")
    fig = create_tsne_plot(df_tsne, sample_fraction=0.3)

    print("t-SNE plot figure created successfully.")
    return fig


@callback(
    Output('themes-wordclouds', 'children'),
    Input('url', 'pathname')
)
def update_wordclouds(pathname):
    print(f"Callback Triggered: update_wordclouds with pathname={pathname}")
    if pathname != '/themes':
        print("Not the Themes page. Preventing update.")
        raise PreventUpdate

    df_wc = load_wordcloud_data()

    if df_wc.empty:
        print("tf_icf_normalized_matrix_top500.csv is empty. Returning no word clouds.")
        return html.Div([
            html.H3("Word Clouds - No Data Available", style={'textAlign': 'center'})
        ])

    wordclouds = []
    for cluster_num in range(1, 21):  # Cluster numbers 1 to 20
        mapping = CLUSTER_MAPPING.get(cluster_num)
        if not mapping:
            print(f"No mapping found for cluster number {cluster_num}. Skipping.")
            continue

        cluster_id = mapping['id']
        cluster_name = mapping['name']
        color = DISTINCT_COLORS[cluster_num - 1 % len(DISTINCT_COLORS)]
        cluster_column = f"Cluster_{cluster_num}"

        if cluster_column not in df_wc.columns:
            print(f"Cluster column '{cluster_column}' not found in CSV. Skipping cluster {cluster_num}.")
            continue

        # Extract words and their tf-idf scores
        cluster_series = df_wc[cluster_column].dropna()
        words = cluster_series.index.tolist()
        scores = cluster_series.values.tolist()

        if not words or not scores:
            print(f"No words or scores found for cluster {cluster_num}. Skipping.")
            continue

        # Generate word cloud image
        img_base64 = generate_wordcloud_image(words, scores, color)
        if img_base64 is None:
            print(f"Failed to generate word cloud for cluster {cluster_num}. Skipping.")
            continue

        # Create HTML components for the word cloud
        wordcloud_component = html.Div([
            html.H4(f"{cluster_name}", style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{img_base64}',
                style={'width': '100%', 'height': 'auto', 'border': f'2px solid {color}', 'borderRadius': '10px'}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'margin': '2.5%', 'verticalAlign': 'top'})

        wordclouds.append(wordcloud_component)

    # Check if any word clouds were generated
    if not wordclouds:
        print("No word clouds were generated. Returning 'No data available' message.")
        return html.Div([
            html.H3("Word Clouds - No Data Available", style={'textAlign': 'center'})
        ])

    # Arrange word clouds in a responsive grid
    return html.Div([
        html.H2("Word Clouds for Each Cluster", style={'textAlign': 'center'}),
        html.Div(wordclouds, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ])


def create_themes_layout():
    """
    Creates the layout for the Themes page.

    Returns:
        dash_html_components.Div: The layout for the Themes page.
    """
    return html.Div([
        # Header
        html.H2("Themes Analysis", style={'textAlign': 'center'}),

        # Description Text Area
        dcc.Textarea(
            id='themes-description',
            value="""
            To capture more context in grouping together publications relative to traditional methods such as Latent Dirichlet Allocation (LDA), publications were grouped into themes based on the similarity of their 765-dimensional embeddings within an LLM trained on PubMed and PMC articles (https://academic.oup.com/bioinformatics/article/36/4/1234/5566506).

            Publications (or embeddings) were grouped together with consensus clustering (k-means over 10k subsamples). Hartigan's dip statistic of bimodality was used to exclude consensus matrices that were unimodal (i.e., sample pairs weren't always being assigned to the same cluster or to different clusters consistently over subsamples). The solution with the highest bimodality coefficient beyond just 2-3 clusters was selected. Affinity propagation extracted the final solution from the consensus matrix.

            Similarity between publication embeddings is shown by t-SNE, with samples color coded by the independent clustering solution.

            Word clouds were generated for each cluster with normalized tf-idf (each cluster treated as a document and normalized tf-idf within each cluster).
            """,
        style={
                'width': '100%',
                'height': 100,
                'padding': '10px',
                'font-family': 'Arial',
                'font-size': '16px'
            },
            disabled=True  # Make it read-only
        ),

        html.Br(),

        # Line Plot
        html.Div([
            dcc.Graph(id='themes-line-plot')
        ], style={'width': '100%', 'display': 'inline-block'}),

        html.Br(),

        # t-SNE Embedding Plot
        html.Div([
            dcc.Graph(id='themes-tsne-plot')
        ], style={'width': '100%', 'display': 'inline-block'}),

        html.Br(),

        # Word Clouds Section
        html.Div([
            dcc.Loading(
                id="loading-wordclouds",
                type="default",
                children=html.Div(id='themes-wordclouds')
            )
        ], style={'width': '100%', 'display': 'inline-block'}),

    ], style={'padding': '20px', 'font-family': 'Arial'})

def find_continuous_ranges(indices):
    """
    Finds continuous ranges in a list of indices.

    Parameters:
        indices (list): List of integer indices.

    Returns:
        list of tuples: Each tuple contains the start and end index of a continuous range.
    """
    ranges = []
    if not indices:
        return ranges

    start = indices[0]
    end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append((start, end))
            start = indices[i]
            end = indices[i]
    ranges.append((start, end))
    return ranges

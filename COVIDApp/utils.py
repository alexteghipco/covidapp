# utils.py

from cache import cache  # Import the shared Cache instance
import re
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
#from COVIDApp_Teghipco import config
import config
import gcsfs  # Import gcsfs to work with GCS paths
import os
#import config  # Make sure you import your config module

import os

def get_available_years(bucket_dir, file_prefix):
    """
    Fetches the available years from filenames in the specified GCS bucket directory.

    Parameters:
    - bucket_dir (str): The GCS bucket directory path.
    - file_prefix (str): The prefix of the files to look for (e.g., "processed_countries_").

    Returns:
    - List[int]: A sorted list of unique years found in the filenames.
    """
    fs = gcsfs.GCSFileSystem()
    try:
        files = fs.ls(bucket_dir)
    except Exception as e:
        print(f"Error accessing GCS bucket: {e}")
        return []

    years = []
    for f in files:
        if f.startswith(file_prefix):
            # Extract year from filename, assuming format: processed_countries_2020.csv
            basename = os.path.basename(f)
            parts = basename.replace('.csv', '').split('_')
            try:
                year = int(parts[-1])
                years.append(year)
            except ValueError:
                continue  # Skip files where year extraction fails

    return sorted(set(years))


@cache.memoize(timeout=300)  # Cache for 5 minutes
def load_country_org_mapping():
    """
    Loads the country-organization mapping from a Parquet file on Google Cloud Storage.

    Returns:
        pd.DataFrame: DataFrame containing the country-organization mapping.
    """
    try:
        # Define the GCS path for the Parquet file
        file_path = config.COUNTRY_ORG_MAPPING_PATH

        # Initialize GCS filesystem
        fs = gcsfs.GCSFileSystem()

        # Load Parquet file from GCS using pandas
        with fs.open(file_path, 'rb') as f:
            mapping = pd.read_parquet(f)

        print(f"Loaded country_org_mapping with shape: {mapping.shape}")
        return mapping
    except Exception as e:
        print(f"Error loading country_org_mapping: {e}")
        return pd.DataFrame()


# Modify the load_processed_data function to handle 'author' graph_type
@cache.memoize(timeout=300)  # Cache for 5 minutes
def load_processed_data(graph_type, year, entity=None):
    """
    Loads processed data from Google Cloud Storage for a given graph type and year.

    Parameters:
        graph_type (str): Type of graph ('country', 'organization', 'within_country', 'author').
        year (int): The year for which to load the data.
        entity (str, optional): Specific entity to filter data for (e.g., organization name or country name).

    Returns:
        pd.DataFrame: DataFrame containing the processed data, or empty DataFrame on failure.
    """
    # Construct the file path based on graph_type
    if graph_type == 'country':
        file_path = f"{config.PROCESSED_DIR_COUNTRIES}/processed_countries_{year}.parquet"
    elif graph_type == 'organization' and entity:
        sanitized_entity = re.sub(r'[<>:"/\\|?*]', '_', entity)
        file_path = f"{config.PROCESSED_DIR_ORGS_BY_ORG}/{sanitized_entity}_{year}.parquet"
    elif graph_type == 'within_country' and entity:
        sanitized_entity = re.sub(r'[\s\/\"\'()]', '_', str(entity)).title()
        file_path = f"{config.PROCESSED_DIR_WITHIN_COUNTRY}/{sanitized_entity}_within_country_{year}.parquet"
    elif graph_type == 'author':
        file_path = f"{config.PROCESSED_DIR_AUTHORS}/author_collab_matrix_{year}.parquet"
    else:
        print(f"Invalid parameters: graph_type={graph_type}, year={year}, entity={entity}")
        return pd.DataFrame()

    # Ensure that the file_path has the correct format (i.e., starts with 'gs://')
    if not file_path.startswith('gs://'):
        file_path = f"gs://{file_path}"

    try:
        # Initialize GCS filesystem
        fs = gcsfs.GCSFileSystem()

        print(f"Attempting to open file at path: {file_path}")

        # Check if the file exists
        if not fs.exists(file_path):
            print(f"File does not exist: {file_path}")
            return pd.DataFrame()

        # Load Parquet file from GCS using pandas
        with fs.open(file_path, 'rb') as f:
            df = pd.read_parquet(f)

        required_columns = {'node1', 'node2', 'weight'}
        if required_columns.issubset(df.columns):
            print(f"DataFrame loaded successfully with shape: {df.shape}")
            return df
        else:
            print(f"Error: Missing required columns in {file_path}. Required columns: {required_columns}")
            print(f"DataFrame columns are: {df.columns}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file from {file_path}: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

from cache import cache  # Import the shared Cache instance
import pandas as pd
import gcsfs  # Import gcsfs to work with GCS paths
import config  # Make sure you import your config module

# utils.py

@cache.memoize(timeout=300)
def load_author_org_country_mapping():
    """
    Loads the filtered author to organization and country mapping from a Parquet file on Google Cloud Storage.

    Returns:
        pd.DataFrame: DataFrame containing the author to organization and country mapping.
    """
    try:
        file_path = f"gs://{config.AUTHOR_ORG_COUNTRY_MAPPING_PATH}"
        print(f"Attempting to open file at path: {file_path}")

        fs = gcsfs.GCSFileSystem()

        if not fs.exists(file_path):
            print(f"File does not exist: {file_path}")
            return pd.DataFrame()

        with fs.open(file_path, 'rb') as f:
            mapping = pd.read_parquet(f)

        print(f"Loaded author_org_country_mapping with shape: {mapping.shape}")

        expected_columns = {'author_name', 'country_name', 'highest_level_organization_name'}
        if not expected_columns.issubset(mapping.columns):
            print(f"Error: Missing expected columns: {expected_columns}")
            print(f"Available columns: {mapping.columns.tolist()}")
            return pd.DataFrame()

        # Set 'author_name' as the index for faster lookups
        mapping.set_index('author_name', inplace=True)

        return mapping
    except Exception as e:
        print(f"Error loading author_org_country_mapping: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

@cache.memoize(timeout=300)  # Cache for 5 minutes
def calculate_proportional_weights(df):
    """
    Calculates proportional weights for each edge based on the total collaborations of each node.
    The proportion is computed as the edge weight divided by the total weight of the node.
    Since the graph is undirected, we take the average proportion between the two nodes.

    Parameters:
        df (pd.DataFrame): DataFrame with 'node1', 'node2', and 'weight' columns.

    Returns:
        pd.DataFrame: DataFrame with an added 'proportional_weight' column.
    """
    # Get total collaborations per node
    nodes = pd.concat([
        df[['node1', 'weight']].rename(columns={'node1': 'node'}),
        df[['node2', 'weight']].rename(columns={'node2': 'node'})
    ], ignore_index=True)
    node_totals = nodes.groupby('node')['weight'].sum().reset_index()
    node_totals.columns = ['node', 'total_collaborations']

    # Merge total collaborations back to the dataframe for node1 and node2
    df = df.merge(node_totals, left_on='node1', right_on='node', how='left')
    df.rename(columns={'total_collaborations': 'total_collaborations_node1'}, inplace=True)
    df.drop('node', axis=1, inplace=True)

    df = df.merge(node_totals, left_on='node2', right_on='node', how='left')
    df.rename(columns={'total_collaborations': 'total_collaborations_node2'}, inplace=True)
    df.drop('node', axis=1, inplace=True)

    # Compute the proportions
    df['proportion_node1'] = df['weight'] / df['total_collaborations_node1']
    df['proportion_node2'] = df['weight'] / df['total_collaborations_node2']

    # Compute the average proportion
    df['proportional_weight'] = 0.5 * (df['proportion_node1'] + df['proportion_node2'])

    # Handle potential division by zero
    df['proportional_weight'] = df['proportional_weight'].fillna(0)

    return df


@cache.memoize(timeout=300)  # Cache for 5 minutes
def calculate_cii(df):
    """
    Calculates the Collaboration Intensity Index (CII) for each edge.

    Parameters:
        df (pd.DataFrame): DataFrame with 'node1', 'node2', and 'weight' columns.

    Returns:
        pd.DataFrame: DataFrame with an added 'cii' column representing the CII.
    """
    # Total collaborations in the network
    total_collaborations_network = df['weight'].sum()

    # Get total collaborations per node
    nodes = pd.concat([
        df[['node1', 'weight']].rename(columns={'node1': 'node'}),
        df[['node2', 'weight']].rename(columns={'node2': 'node'})
    ], ignore_index=True)
    node_totals = nodes.groupby('node')['weight'].sum().reset_index()
    node_totals.columns = ['node', 'total_collaborations']

    # Merge total collaborations back to the dataframe for node1 and node2
    df = df.merge(node_totals, left_on='node1', right_on='node', how='left')
    df.rename(columns={'total_collaborations': 'total_collaborations_node1'}, inplace=True)
    df.drop('node', axis=1, inplace=True)

    df = df.merge(node_totals, left_on='node2', right_on='node', how='left')
    df.rename(columns={'total_collaborations': 'total_collaborations_node2'}, inplace=True)
    df.drop('node', axis=1, inplace=True)

    # Calculate expected collaborations
    df['expected_weight'] = (df['total_collaborations_node1'] * df[
        'total_collaborations_node2']) / total_collaborations_network

    # Avoid division by zero
    df['expected_weight'] = df['expected_weight'].replace(0, np.nan)

    # Calculate CII
    df['cii'] = df['weight'] / df['expected_weight']

    # Replace infinities and NaNs with zeros
    df['cii'] = df['cii'].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


@cache.memoize(timeout=300)  # Cache for 5 minutes
def build_graph(df, weight_column='weight'):
    """
    Builds a NetworkX graph from a DataFrame containing edges and weights.

    Parameters:
        df (pd.DataFrame): DataFrame with 'node1', 'node2', and weight columns.
        weight_column (str): The column to use as edge weights.

    Returns:
        nx.Graph: NetworkX graph constructed from the DataFrame.
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            row['node1'],
            row['node2'],
            weight=row[weight_column],
            raw_weight=row['weight'] if 'weight' in row else 1
        )
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def compute_centrality(G, metric):
    """
    Computes centrality measures for the nodes in the graph.

    Parameters:
        G (nx.Graph): The NetworkX graph.
        metric (str): The centrality metric to compute ('degree', 'betweenness', 'closeness', 'eigenvector', 'cii').

    Returns:
        dict: Dictionary mapping nodes to their centrality values.
    """
    if metric == 'degree':
        centrality = nx.degree_centrality(G)
    elif metric == 'betweenness':
        centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
    elif metric == 'closeness':
        centrality = nx.closeness_centrality(G, distance='weight')
    elif metric == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        except nx.PowerIterationFailedConvergence:
            print("Eigenvector centrality did not converge. Assigning zero to all nodes.")
            centrality = {node: 0 for node in G.nodes()}
    elif metric == 'cii':
        # For 'cii', sum the CII values of edges connected to the node
        centrality = {}
        for node in G.nodes():
            centrality[node] = sum([edata['weight'] for _, _, edata in G.edges(node, data=True)])
    else:
        centrality = nx.degree_centrality(G)
    return centrality


def compute_layout(G, layout_type):
    """
    Computes the layout of the graph based on the selected layout algorithm.

    Parameters:
        G (nx.Graph): The NetworkX graph.
        layout_type (str): The layout algorithm to use.

    Returns:
        dict: Dictionary mapping nodes to their positions.
    """
    if len(G) == 0:
        return {}
    elif len(G) == 1:
        return {list(G.nodes())[0]: (0, 0)}
    else:
        try:
            if layout_type == 'spring_layout':
                pos = nx.spring_layout(G, k=0.15, iterations=20, weight='weight')  # Reduced iterations
            elif layout_type == 'kamada_kawai_layout':
                pos = nx.kamada_kawai_layout(G, weight='weight')
            elif layout_type == 'circular_layout':
                pos = nx.circular_layout(G)
            elif layout_type == 'shell_layout':
                pos = nx.shell_layout(G)
            elif layout_type == 'spectral_layout':
                pos = nx.spectral_layout(G)
            else:
                pos = nx.spring_layout(G, k=0.15, iterations=20, weight='weight')  # Default to spring layout
        except Exception as e:
            print(f"Error computing layout: {e}")
            pos = nx.spring_layout(G, k=0.15, iterations=20, weight='weight')  # Default to spring layout
        return pos


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


def create_networkx_figure(
    G,
    pos,
    node_sizes,
    node_colors,
    show_text=True,
    weight_label='Weight',
    org_country_mapping=None,
    highlighted_nodes=None,
    highlight_color='pink'
):
    """
    Creates a Plotly figure from a NetworkX graph with variable edge widths.
    Highlights specified nodes with a colored outline.
    Adds tooltips displaying country information for organizations.

    Parameters:
        G (nx.Graph): The NetworkX graph.
        pos (dict): Positions of the nodes.
        node_sizes (List[int]): Sizes for each node.
        node_colors (List[float]): Colors for each node (normalized to [0,1]).
        show_text (bool): Whether to display node labels.
        weight_label (str): Label to use for edge weights ('Weight', 'Proportion', 'CII').
        org_country_mapping (dict, optional): Mapping from organization names to country names.
        highlighted_nodes (List[str], optional): List of nodes to highlight with colored outlines.
        highlight_color (str): Color to use for highlighted node outlines.

    Returns:
        go.Figure: The Plotly figure representing the graph.
    """
    # Initialize edge trace list
    edge_trace_list = []

    # Define edge colors and widths
    min_width = 1.0  # Minimum edge width
    max_width = 5.0  # Maximum edge width

    # Get edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    min_weight = min(edge_weights.values()) if edge_weights else 1
    max_weight = max(edge_weights.values()) if edge_weights else 1

    # Normalize edge weights for width scaling
    def normalize_weight(w):
        return (w - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 1

    # Iterate over edges to create individual traces for hover functionality
    for edge in G.edges(data=True):
        u, v, data = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_weight = data.get('weight', 1)
        raw_weight = data.get('raw_weight', 1)
        normalized_weight = normalize_weight(edge_weight)

        edge_width = min_width + (max_width - min_width) * normalized_weight

        # Set all edges to Medium Grey with 50% opacity
        line_color = 'rgba(169, 169, 169, 0.5)'  # Medium grey

        # Edge trace
        if weight_label == 'Weight':
            weight_display = str(int(raw_weight))
        else:
            weight_display = f"{edge_weight:.4f}"
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=edge_width, color=line_color),
            hoverinfo='text',
            text=f"{weight_label}: {weight_display}",
            mode='lines'
        )
        edge_trace_list.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    customdata = []
    hover_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        customdata.append(node)
        if org_country_mapping:
            country = org_country_mapping.get(node, 'Unknown')
            hover_text.append(f"Organization: {node}<br>Country: {country}")
        else:
            hover_text.append(f"Organization: {node}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text' if show_text else 'markers',
        text=node_text if show_text else None,
        hoverinfo='text',
        hovertext=hover_text,  # Custom hover text with country info
        customdata=customdata,  # For clickData and further interactions
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Thermal',  # Set colormap to 'Thermal'
            showscale=True,
            colorbar=dict(title='Betweenness Centrality'),
            opacity=0.87  # Slightly reduce node opacity
        )
    )

    # Highlight specific nodes if provided
    if highlighted_nodes:
        # Create a list indicating whether each node is highlighted
        node_trace.marker.line = dict(
            width=[4 if node in highlighted_nodes else 0 for node in G.nodes()],
            color=[highlight_color if node in highlighted_nodes else 'rgba(0,0,0,0)' for node in G.nodes()]
        )
    else:
        node_trace.marker.line = dict(width=0, color='rgba(0,0,0,0)')

    # Combine all traces
    data = edge_trace_list + [node_trace]

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        title='',
                        font=dict(family='Arial'),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'  # Set background color to white
                    )
                    )
    return fig

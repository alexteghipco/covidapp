# author_network.py

from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import networkx as nx
import os
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go  # Make sure this is imported
print("Starting author_network.py initialization...")

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Set up Google Cloud credentials
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
# print(f"Set credentials path to: {credentials_path}")

from cache import cache
from utils import (
    load_processed_data,
    calculate_proportional_weights,
    calculate_cii,
    build_graph,
    compute_centrality,
    compute_layout,
    create_empty_figure,
    create_networkx_figure,
    load_author_org_country_mapping  # Import the new function
)

# Load author to organization and country mapping
author_org_country_mapping = load_author_org_country_mapping()

# if author_org_country_mapping.empty:
#     print("Author to organization mapping is empty. Please check the file and path.")
# else:
#     print(f"Author org country mapping loaded with shape: {author_org_country_mapping.shape}")
#     print(f"Columns in author_org_country_mapping: {author_org_country_mapping.columns.tolist()}")
#
#     # Proceed to set the index
#     author_org_country_mapping.set_index('author_name', inplace=True)
# Check for duplicates in the index

@cache.memoize(timeout=300)
def load_author_data(year):
    """Load author collaboration data using load_processed_data from utils."""
    print(f"\nAttempting to load author data for year {year}")
    try:
        df = load_processed_data('author', year)
        if df.empty:
            print("No data loaded for author network.")
        else:
            print(f"Author data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Exception occurred while loading author data: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

def create_author_network_layout():
    """Creates the layout for the author network page."""
    return html.Div([
        html.H2("Author-Level Collaboration Network",
                style={'font-family': 'Arial', 'textAlign': 'center'}),

        # Controls arranged horizontally
        html.Div([
            html.Button('Toggle Text', id='toggle-text-button-author', n_clicks=0,
                        style={'margin-right': '20px'}),

            html.Label("Select Layout Algorithm:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='layout-dropdown-author',
                options=[
                    {'label': 'Spring Layout', 'value': 'spring_layout'},
                    {'label': 'Kamada-Kawai Layout', 'value': 'kamada_kawai_layout'},
                    {'label': 'Circular Layout', 'value': 'circular_layout'},
                    {'label': 'Shell Layout', 'value': 'shell_layout'},
                    {'label': 'Spectral Layout', 'value': 'spectral_layout'}
                ],
                value='spring_layout',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block',
                       'verticalAlign': 'middle', 'margin-right': '20px'}
            ),

            html.Label("Select Node Size Metric:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='node-size-metric-dropdown-author',
                options=[
                    {'label': 'Degree Centrality', 'value': 'degree'},
                    {'label': 'Betweenness Centrality', 'value': 'betweenness'},
                    {'label': 'Closeness Centrality', 'value': 'closeness'},
                    {'label': 'Eigenvector Centrality', 'value': 'eigenvector'},
                    {'label': 'Collaboration Intensity Index (CII)', 'value': 'cii'},
                ],
                value='cii',
                clearable=False,
                style={'width': '300px', 'display': 'inline-block',
                       'verticalAlign': 'middle', 'margin-right': '20px'}
            ),

            html.Label("Select Weight Metric:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='weight-metric-dropdown-author',
                options=[
                    {'label': 'Raw Number of Collaborations', 'value': 'raw'},
                    {'label': 'Proportion of Collaborations', 'value': 'proportion'},
                    {'label': 'Collaboration Intensity Index (CII)', 'value': 'cii'}
                ],
                value='proportion',
                clearable=False,
                style={'width': '300px', 'display': 'inline-block',
                       'verticalAlign': 'middle', 'margin-right': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center',
                  'justify-content': 'center', 'padding': '10px',
                  'font-family': 'Arial'}),

        # Sliders Section
        html.Div([
            # Number of Top Collaborators Slider
            html.Div([
                html.Label("Number of Top Collaborators to Display:",
                           style={'font-family': 'Arial'}),
                dcc.Slider(
                    id='top-collaborators-slider-author',
                    min=5,
                    max=500,
                    step=5,
                    value=50,
                    marks={i: str(i) for i in [5, 100, 200, 300, 400, 500]},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'padding': '20px'}),

            # Weight Threshold Slider
            html.Div([
                html.Label("Minimum Collaboration Weight:",
                           style={'font-family': 'Arial'}),
                html.Div(id='threshold-display-author',
                         style={'font-family': 'Arial', 'textAlign': 'center'}),
                dcc.Slider(
                    id='weight-threshold-slider-author',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.01,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'padding': '20px'}),

            # Year Slider
            html.Div([
                html.Label("Select Year:", style={'font-family': 'Arial'}),
                dcc.Slider(
                    id='year-slider-author',
                    min=2020,
                    max=2024,
                    value=2020,
                    marks={year: str(year) for year in range(2020, 2025)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': '20px'}),
        ]),

        # Network Graph
        dcc.Loading(
            dcc.Graph(id='author-graph'),
            type='default'
        ),

        # Edge Weights Display
        html.Div([
            html.H3("Connected Edge Weights", style={'font-family': 'Arial'}),
            html.Div(id='edge-weights-display-author',
                     style={'font-family': 'Arial'})
        ], style={'padding': '20px'}),

        # Store components
        dcc.Store(id='text-toggle-author', data=True),

    ], style={'padding': '20px', 'border': '3px solid #ccc', 'margin': '20px'})

# Update the IDs in all callbacks accordingly

@callback(
    Output('text-toggle-author', 'data'),
    Input('toggle-text-button-author', 'n_clicks'),
    State('text-toggle-author', 'data')
)
def toggle_text(n_clicks, show_text):
    """Toggle the visibility of node labels."""
    if n_clicks is None:
        raise PreventUpdate
    return not show_text

@callback(
    Output('threshold-display-author', 'children'),
    [
        Input('weight-threshold-slider-author', 'value'),
        Input('weight-metric-dropdown-author', 'value'),
    ]
)
def display_threshold(value, selected_weight_metric):
    """Display the current threshold value with appropriate label."""
    weight_label_mapping = {
        'raw': 'Raw Number of Collaborations',
        'proportion': 'Proportion of Collaborations',
        'cii': 'Collaboration Intensity Index (CII)'
    }
    weight_label = weight_label_mapping.get(selected_weight_metric, 'Weight')
    return f"Threshold ({weight_label} ≥ {value}):"

@callback(
    Output('author-graph', 'figure'),
    [
        Input('year-slider-author', 'value'),
        Input('layout-dropdown-author', 'value'),
        Input('weight-threshold-slider-author', 'value'),
        Input('node-size-metric-dropdown-author', 'value'),
        Input('text-toggle-author', 'data'),
        Input('weight-metric-dropdown-author', 'value'),
        Input('top-collaborators-slider-author', 'value')
    ]
)
def update_graph(year, layout_type, weight_threshold, node_size_metric,
                 show_text, selected_weight_metric, n_top):
    print("\n=== UPDATE_GRAPH CALLBACK TRIGGERED ===")
    print("Loading initial data...")

    df = load_author_data(year)

    # Check loaded data
    if df.empty:
        print("ERROR: No data loaded!")
        return create_empty_figure("No author data available - see console for details")

    print(f"Initial data loaded successfully with shape: {df.shape}")
    print("Getting top collaborators...")
    df = get_top_collaborators(df, n_top)
    print(f"After filtering for top {n_top} collaborators, shape: {df.shape}")

    if df.empty:
        return create_empty_figure("Author Collaboration Network - No Data After Filtering")

    # Process weights
    print("Processing weights...")
    if selected_weight_metric == 'raw':
        df['computed_weight'] = df['weight']
    elif selected_weight_metric == 'proportion':
        df = calculate_proportional_weights(df)
        df['computed_weight'] = df['proportional_weight']
    elif selected_weight_metric == 'cii':
        df = calculate_cii(df)
        df['computed_weight'] = df['cii']
    else:
        df['computed_weight'] = df['weight']

    print(f"After weight processing, shape: {df.shape}")

    # Thresholding
    print(f"Applying threshold {weight_threshold}...")
    df = df[df['computed_weight'] >= weight_threshold]
    print(f"After thresholding, shape: {df.shape}")

    if df.empty:
        return create_empty_figure("Author Collaboration Network - No Data After Thresholding")

    # Build graph
    G = build_graph(df, weight_column='computed_weight')
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        return create_empty_figure("Author Collaboration Network - No Nodes to Display")

    # Compute centralities on thresholded graph
    if node_size_metric == 'cii':
        node_cii = {}
        for node in G.nodes():
            cii_sum = sum([edata['weight'] for _, _, edata in G.edges(node, data=True)])
            node_cii[node] = cii_sum
        centrality = node_cii
    else:
        centrality = compute_centrality(G, node_size_metric)

    # Node sizes based on selected metric
    if centrality:
        min_cent = min(centrality.values())
        max_cent = max(centrality.values())
        if max_cent - min_cent != 0:
            node_sizes = [
                10 + 30 * (centrality[node] - min_cent) / (max_cent - min_cent)
                for node in G.nodes()
            ]
        else:
            node_sizes = [20 for _ in G.nodes()]
    else:
        node_sizes = [20 for _ in G.nodes()]

    # Node colors based on betweenness centrality computed on thresholded graph
    betweenness_centrality = compute_centrality(G, 'betweenness')
    if betweenness_centrality:
        min_bet = min(betweenness_centrality.values())
        max_bet = max(betweenness_centrality.values())
        if max_bet - min_bet != 0:
            node_colors = [
                (betweenness_centrality[node] - min_bet) / (max_bet - min_bet)
                for node in G.nodes()
            ]
        else:
            node_colors = [0.5 for _ in G.nodes()]
    else:
        node_colors = [0.5 for _ in G.nodes()]

    # Compute layout
    pos = compute_layout(G, layout_type)

    # Prepare data for the nodes
    node_x = []
    node_y = []
    node_text = []
    customdata = []
    hover_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node if show_text else '')
        customdata.append(node)

        # Get organization and country from the mapping
        if node in author_org_country_mapping.index:
            org = author_org_country_mapping.loc[node]['highest_level_organization_name']
            country = author_org_country_mapping.loc[node]['country_name']
        else:
            org = 'Unknown'
            country = 'Unknown'

        hover_text.append(f"Author: {node}<br>Organization: {org}<br>Country: {country}")

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text' if show_text else 'markers',
        text=node_text,
        hoverinfo='text',
        hovertext=hover_text,
        customdata=customdata,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Thermal',
            showscale=True,
            colorbar=dict(title='Betweenness Centrality'),
            opacity=0.87
        )
    )

    # Edge traces
    edge_trace_list = []

    # Define edge colors and widths
    min_width = 0.5  # Minimum edge width
    max_width = 3  # Maximum edge width

    # Get edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    min_weight = min(edge_weights.values()) if edge_weights else 1
    max_weight = max(edge_weights.values()) if edge_weights else 1

    # Normalize edge weights for width scaling
    def normalize_weight(w):
        return (w - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 1

    for edge in G.edges(data=True):
        u, v, data = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_weight = data.get('weight', 1)
        normalized_weight = normalize_weight(edge_weight)
        edge_width = min_width + (max_width - min_width) * normalized_weight
        line_color = 'rgba(169, 169, 169, 0.5)'  # Medium grey

        if selected_weight_metric == 'raw':
            weight_display = str(int(data.get('raw_weight', edge_weight)))
        else:
            weight_display = f"{edge_weight:.4f}"

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=edge_width, color=line_color),
            hoverinfo='text',
            text=f"Edge between {u} and {v}<br>Weight: {weight_display}",
            mode='lines'
        )
        edge_trace_list.append(edge_trace)

    # Create figure with edge traces and node trace
    fig = go.Figure(data=edge_trace_list + [node_trace],
                    layout=go.Layout(
                        title='',
                        font=dict(family='Arial'),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'  # Set background color to white
                    ))
    return fig

@cache.memoize(timeout=300)
def get_top_collaborators(df, n_top):
    """Get the top n collaborators based on total collaboration weight."""
    if df.empty:
        return pd.DataFrame()

    # Calculate total collaborations for each author
    author1_weights = df.groupby('node1')['weight'].sum()
    author2_weights = df.groupby('node2')['weight'].sum()

    # Combine weights from both columns
    total_weights = author1_weights.add(author2_weights, fill_value=0)

    # Get top n authors
    top_authors = total_weights.nlargest(n_top).index.tolist()

    # Filter the DataFrame to only include connections between top authors
    filtered_df = df[
        (df['node1'].isin(top_authors)) &
        (df['node2'].isin(top_authors))
    ]

    return filtered_df

@callback(
    [
        Output('weight-threshold-slider-author', 'min'),
        Output('weight-threshold-slider-author', 'max'),
        Output('weight-threshold-slider-author', 'marks'),
        Output('weight-threshold-slider-author', 'value'),
        Output('weight-threshold-slider-author', 'step'),
    ],
    [
        Input('year-slider-author', 'value'),
        Input('weight-metric-dropdown-author', 'value'),
        Input('top-collaborators-slider-author', 'value'),
    ],
    [State('weight-threshold-slider-author', 'value')]
)
def update_author_threshold_slider(year, selected_weight_metric, n_top, current_value):
    df = load_author_data(year)
    if df.empty:
        # Return default values
        return 0, 1, {0: '0', 1: '1'}, 0.01, 0.01

    # Get top collaborators
    df = get_top_collaborators(df, n_top)
    if df.empty:
        return 0, 1, {0: '0', 1: '1'}, 0.01, 0.01

    # Process weights
    if selected_weight_metric == 'raw':
        df['computed_weight'] = df['weight']
    elif selected_weight_metric == 'proportion':
        df = calculate_proportional_weights(df)
        df['computed_weight'] = df['proportional_weight']
    elif selected_weight_metric == 'cii':
        df = calculate_cii(df)
        df['computed_weight'] = df['cii']
    else:
        df['computed_weight'] = df['weight']

    weights = df['computed_weight']
    min_weight = weights.min()
    max_weight = weights.max()

    if selected_weight_metric == 'raw':
        step = 1
        marks = {int(i): str(int(i)) for i in np.linspace(min_weight, max_weight, num=5)}
        default_value = max(int(min_weight), 1)
    else:
        step = round((max_weight - min_weight) / 50, 5) if max_weight != min_weight else 0.01
        marks = {round(i, 4): str(round(i, 4)) for i in np.linspace(min_weight, max_weight, num=5)}
        default_value = round(min_weight, 4)

    return min_weight, max_weight, marks, default_value, step

@callback(
    Output('edge-weights-display-author', 'children'),
    [
        Input('author-graph', 'clickData'),
        Input('year-slider-author', 'value'),
        Input('weight-threshold-slider-author', 'value'),
        Input('weight-metric-dropdown-author', 'value'),
        Input('node-size-metric-dropdown-author', 'value'),
        Input('top-collaborators-slider-author', 'value')
    ]
)
def display_edge_weights(click_data, year, weight_threshold,
                         selected_weight_metric, node_size_metric, n_top):
    if not click_data:
        return "Click on a node in the graph to see its connected edge weights."

    clicked_node = click_data['points'][0].get('customdata', None)
    if not clicked_node:
        return "No node data available."

    df = load_author_data(year)
    if df.empty:
        return "No data available."

    df = get_top_collaborators(df, n_top)
    if df.empty:
        return "No data available."

    # Process weights
    if selected_weight_metric == 'raw':
        df['computed_weight'] = df['weight']
    elif selected_weight_metric == 'proportion':
        df = calculate_proportional_weights(df)
        df['computed_weight'] = df['proportional_weight']
    elif selected_weight_metric == 'cii':
        df = calculate_cii(df)
        df['computed_weight'] = df['cii']
    else:
        df['computed_weight'] = df['weight']

    # Thresholding
    df = df[df['computed_weight'] >= weight_threshold]
    if df.empty:
        return "No data available after thresholding."

    # Build graph
    G = build_graph(df, weight_column='computed_weight')

    # Compute centrality for node size metric on thresholded graph
    if node_size_metric == 'cii':
        node_cii = {}
        for node in G.nodes():
            cii_sum = sum([edata['weight'] for _, _, edata in G.edges(node, data=True)])
            node_cii[node] = cii_sum
        centrality = node_cii
    else:
        centrality = compute_centrality(G, node_size_metric)

    # Find edges connected to the clicked node
    connected_edges = df[
        (df['node1'] == clicked_node) | (df['node2'] == clicked_node)
    ]

    if connected_edges.empty:
        return "No connected edges found for the selected node."

    # Sort edges by computed weight descending
    connected_edges = connected_edges.sort_values(by='computed_weight', ascending=False)

    # Create a table to display the edges
    weight_label_mapping = {
        'raw': 'Weight',
        'proportion': 'Proportion',
        'cii': 'CII'
    }
    weight_label = weight_label_mapping.get(selected_weight_metric, 'Weight')

    node_size_label_mapping = {
        'degree': 'Degree Centrality',
        'betweenness': 'Betweenness Centrality',
        'closeness': 'Closeness Centrality',
        'eigenvector': 'Eigenvector Centrality',
        'cii': 'Node CII'
    }
    node_size_label = node_size_label_mapping.get(node_size_metric, 'Node Metric')

    table_header = [
        html.Thead(html.Tr([
            html.Th("Connected Node"),
            html.Th("Organization"),
            html.Th("Country"),
            html.Th(weight_label),
            html.Th(node_size_label)
        ]))
    ]

    table_rows = []
    for _, row in connected_edges.iterrows():
        if row['node1'] == row['node2']:
            continue  # Skip self-loops
        other_node = row['node2'] if row['node1'] == clicked_node else row['node1']

        weight_value = row['computed_weight']
        if selected_weight_metric == 'raw':
            weight_display = str(int(weight_value))
        else:
            weight_display = f"{weight_value:.4f}"

        # Get the node size metric value for the connected node
        node_metric_value = centrality.get(other_node, 0)
        node_metric_display = f"{node_metric_value:.4f}"

        # Get organization and country
        if other_node in author_org_country_mapping.index:
            org = author_org_country_mapping.loc[other_node]['highest_level_organization_name']
            country = author_org_country_mapping.loc[other_node]['country_name']
        else:
            org = 'Unknown'
            country = 'Unknown'

        table_rows.append(html.Tr([
            html.Td(other_node),
            html.Td(org),
            html.Td(country),
            html.Td(weight_display),
            html.Td(node_metric_display)
        ]))

    table_body = [html.Tbody(table_rows)]

    table = html.Table(table_header + table_body, style={'width': '100%', 'textAlign': 'left'})

    return html.Div([
        html.H4(f"Edge Weights and Node Metrics for {clicked_node}", style={'font-family': 'Arial'}),
        table
    ])

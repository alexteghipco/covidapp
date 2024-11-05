# app.py
import os

import pandas as pd
import numpy as np
import logging
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Import layout modules
from pages.top_scores import create_top_scores_layout
from pages.themeTrends import create_theme_trends_layout
from pages.altmetricPred import create_altmetric_predictions_layout

import config  # Import the config module
from cache import cache  # Import the shared Cache instance

# Initialize the Dash application with Bootstrap for better styling
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "COVID publication trends"
server = app.server

# Initialize Flask-Caching with the Dash app
cache.init_app(app.server)

# Define the navbar with custom color and updated link text
navbar = dbc.NavbarSimple(
    brand="COVID publication and research patterns",
    color="dark",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Entity-Level Networks", href="/")),
        dbc.NavItem(dbc.NavLink("Author-Level Networks", href="/author-level")),
        dbc.NavItem(dbc.NavLink("Themes", href="/themes")),
        dbc.NavItem(dbc.NavLink("Theme Trends", href="/theme-trends")),
        dbc.NavItem(dbc.NavLink("Altmetric Predictions", href="/altmetric-predictions")),
        dbc.NavItem(dbc.NavLink("Top Impact", href="/top-scores")),
    ],
    style={
        "background-color": "#9b2226",
    },
)

# Define the layout with a Location component and a container for page content
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', style={'padding': '20px'})
], style={'font-family': 'Arial'})

# Import pages' layouts and callbacks
from pages.main_page import create_main_page_layout
from pages.author_network import create_author_network_layout
from pages.themes import create_themes_layout

# Update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/' or pathname == '/main-page':
        return create_main_page_layout()
    elif pathname == '/author-level':
        return create_author_network_layout()
    elif pathname == '/themes':
        return create_themes_layout()
    elif pathname == '/theme-trends':
        return create_theme_trends_layout()
    elif pathname == '/altmetric-predictions':
        return create_altmetric_predictions_layout()
    elif pathname == '/top-scores':
        return create_top_scores_layout()
    else:
        return html.H3("404: Page not found", style={'textAlign': 'center', 'color': 'red'})

# Import utility functions
from utils import (
    load_country_org_mapping,
    load_processed_data,
    calculate_proportional_weights,
    calculate_cii,
    build_graph,
    compute_centrality,
    compute_layout,
    create_empty_figure,
    create_networkx_figure
)

import dash  # Import Dash for callback_context

# Load country-org mapping once
country_org_mapping = load_country_org_mapping()


# ------------------------ Callbacks for Main Page ------------------------ #

# Toggle Text Visibility
@app.callback(
    Output('text-toggle', 'data'),
    Input('toggle-text-button', 'n_clicks'),
    State('text-toggle', 'data')
)
def toggle_text(n_clicks, show_text):
    """
    Toggles the visibility of node labels on the graph.
    """
    if n_clicks is None:
        raise PreventUpdate
    return not show_text


# Update selected node and selected country based on any graph click
@app.callback(
    [
        Output('selected-node', 'data'),
        Output('selected-country', 'data')
    ],
    [
        Input('country-graph', 'clickData'),
        Input('organization-graph', 'clickData'),
        Input('within-country-graph', 'clickData')
    ],
    [State('selected-node', 'data'),
     State('selected-country', 'data')]
)
def update_selected_node(country_click, org_click, wc_click, current_node, current_country):
    """
    Updates the selected node and selected country when a node in any graph is clicked.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'country-graph' and country_click:
        clicked_node = country_click['points'][0].get('customdata', None)
        if clicked_node:
            return {'node': clicked_node, 'type': 'country'}, clicked_node

    elif triggered_id == 'organization-graph' and org_click:
        clicked_node = org_click['points'][0].get('customdata', None)
        if clicked_node:
            return {'node': clicked_node, 'type': 'organization'}, current_country

    elif triggered_id == 'within-country-graph' and wc_click:
        clicked_node = wc_click['points'][0].get('customdata', None)
        if clicked_node:
            return {'node': clicked_node, 'type': 'within_country'}, current_country

    # If no valid clickData, prevent update
    raise PreventUpdate


# Helper function to update threshold sliders
def update_threshold_slider(graph_type, year, current_value, selected_weight_metric, selected_country=None):
    """
    Generic function to update threshold sliders based on the graph type, year, and selected weight metric.
    """
    if graph_type == 'country':
        df = load_processed_data('country', year)
    elif graph_type == 'organization' and selected_country:
        relevant_orgs = country_org_mapping[country_org_mapping['country_name'] == selected_country][
            'high_level_org_name'].unique()
        if len(relevant_orgs) == 0:
            return 1, 100, {1: '1', 100: '100'}, 1, 1
        org_dfs = [
            load_processed_data('organization', year, entity=org)
            for org in relevant_orgs
        ]
        org_dfs = [df for df in org_dfs if not df.empty]
        if not org_dfs:
            return 1, 100, {1: '1', 100: '100'}, 1, 1
        df = pd.concat(org_dfs, ignore_index=True)
    elif graph_type == 'within_country' and selected_country:
        df = load_processed_data('within_country', year, entity=selected_country)
    else:
        df = pd.DataFrame()

    if df.empty:
        # Return default values
        return 0, 1, {0: '0', 1: '1'}, 0.04, 0.01  # Adjusted defaults for 'proportion' and 'cii'

    if selected_weight_metric == 'raw':
        weights = df['weight']
        min_weight = weights.min()
        max_weight = weights.max()
        step = 1
        marks = {int(i): str(int(i)) for i in np.linspace(min_weight, max_weight, num=5, dtype=int)}
        default_value = max(1, current_value)
    elif selected_weight_metric == 'proportion':
        df = calculate_proportional_weights(df)
        weights = df['proportional_weight']
        min_weight = weights.min()
        max_weight = weights.max()
        step = round((max_weight - min_weight) / 50, 5) if max_weight != min_weight else 0.01
        marks = {round(i, 4): str(round(i, 4)) for i in np.linspace(min_weight, max_weight, num=5)}
        default_value = max(round(min_weight, 3), 0.04)  # Avoid default_value being too high
    elif selected_weight_metric == 'cii':
        df = calculate_cii(df)
        weights = df['cii']
        min_weight = weights.min()
        max_weight = weights.max()
        step = round((max_weight - min_weight) / 50, 5) if max_weight != min_weight else 0.01
        marks = {round(i, 3): str(round(i, 3)) for i in np.linspace(min_weight, max_weight, num=5)}
        default_value = round(min_weight, 3)  # Set to min_weight to include some edges
    else:
        # Default to raw
        weights = df['weight']
        min_weight = weights.min()
        max_weight = weights.max()
        step = 1
        marks = {int(i): str(int(i)) for i in np.linspace(min_weight, max_weight, num=5, dtype=int)}
        default_value = max(1, current_value)

    return min_weight, max_weight, marks, default_value, step


# ------------------------ Callbacks for Country-Level Network ------------------------ #

# Update threshold slider for Country-Level graph
@app.callback(
    [
        Output('weight-threshold-slider-country', 'min'),
        Output('weight-threshold-slider-country', 'max'),
        Output('weight-threshold-slider-country', 'marks'),
        Output('weight-threshold-slider-country', 'value'),
        Output('weight-threshold-slider-country', 'step'),
    ],
    [
        Input('year-slider-country', 'value'),
        Input('weight-metric-dropdown', 'value'),
    ],
    [State('weight-threshold-slider-country', 'value')]
)
def update_country_threshold_slider(year, selected_weight_metric, current_value):
    min_weight, max_weight, marks, default_value, step = update_threshold_slider(
        'country', year, current_value, selected_weight_metric)
    return min_weight, max_weight, marks, default_value, step


# Update threshold display for Country-Level graph
@app.callback(
    Output('threshold-display-country', 'children'),
    [
        Input('weight-threshold-slider-country', 'value'),
        Input('weight-metric-dropdown', 'value'),
    ]
)
def display_country_threshold(value, selected_weight_metric):
    weight_label_mapping = {
        'raw': 'Raw Number of Collaborations',
        'proportion': 'Proportion of Collaborations',
        'cii': 'Collaboration Intensity Index (CII)'
    }
    weight_label = weight_label_mapping.get(selected_weight_metric, 'Weight')
    return f"Threshold ({weight_label} ≥ {value}):"


# Update Country-Level graph
@app.callback(
    Output('country-graph', 'figure'),
    [
        Input('year-slider-country', 'value'),
        Input('layout-dropdown', 'value'),
        Input('weight-threshold-slider-country', 'value'),
        Input('node-size-metric-dropdown', 'value'),
        Input('text-toggle', 'data'),  # Corrected to 'data'
        Input('weight-metric-dropdown', 'value'),
    ]
)
def update_country_graph(year, layout_type, weight_threshold, node_size_metric, show_text, selected_weight_metric):
    """
    Updates the Country-Level Collaboration Network graph based on user inputs.
    """
    # Load data
    country_df = load_processed_data('country', year)
    if country_df.empty:
        return create_empty_figure("Country Collaboration Network - No Data")

    # Compute weights based on selected metric
    if selected_weight_metric == 'raw':
        country_df['computed_weight'] = country_df['weight']
    elif selected_weight_metric == 'proportion':
        country_df = calculate_proportional_weights(country_df)
        country_df['computed_weight'] = country_df['proportional_weight']
    elif selected_weight_metric == 'cii':
        country_df = calculate_cii(country_df)
        # 'cii' is now calculated directly in the DataFrame
    else:
        country_df['computed_weight'] = country_df['weight']

    # Thresholding
    if selected_weight_metric == 'cii':
        country_df = country_df[country_df['cii'] >= weight_threshold].copy()
    else:
        country_df = country_df[country_df['computed_weight'] >= weight_threshold].copy()

    if country_df.empty:
        return create_empty_figure("Country Collaboration Network - No Data")

    # Build graph
    if selected_weight_metric == 'cii':
        G = build_graph(country_df, weight_column='cii')
    else:
        G = build_graph(country_df, weight_column='computed_weight')

    if G.number_of_nodes() == 0:
        return create_empty_figure("Country Collaboration Network - No Data")

    # Compute centralities on thresholded graph
    if node_size_metric == 'cii':
        # For node CII, compute sum of edge CII values connected to the node
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

    # Map weight labels
    weight_label_mapping = {
        'raw': 'Weight',
        'proportion': 'Proportion',
        'cii': 'CII'
    }
    weight_label = weight_label_mapping.get(selected_weight_metric, 'Weight')

    # Create network figure
    fig = create_networkx_figure(
        G,
        pos,
        node_sizes,
        node_colors,
        show_text=bool(show_text),  # Convert list to bool
        weight_label=weight_label,
        org_country_mapping=None,  # No country mapping needed at country-level
        highlighted_nodes=None  # No highlighting needed at country-level
    )

    return fig


# ------------------------ Callbacks for Organization-Level Network ------------------------ #

# Update threshold slider for Organization-Level graph
@app.callback(
    [
        Output('weight-threshold-slider-organization', 'min'),
        Output('weight-threshold-slider-organization', 'max'),
        Output('weight-threshold-slider-organization', 'marks'),
        Output('weight-threshold-slider-organization', 'value'),
        Output('weight-threshold-slider-organization', 'step'),
    ],
    [
        Input('year-slider-organization', 'value'),
        Input('selected-country', 'data'),
        Input('weight-metric-dropdown', 'value'),
    ],
    [State('weight-threshold-slider-organization', 'value')]
)
def update_organization_threshold_slider(year, selected_country, selected_weight_metric, current_value):
    min_weight, max_weight, marks, default_value, step = update_threshold_slider(
        'organization', year, current_value, selected_weight_metric, selected_country)
    return min_weight, max_weight, marks, default_value, step


# Update threshold display for Organization-Level graph
@app.callback(
    Output('threshold-display-organization', 'children'),
    [
        Input('weight-threshold-slider-organization', 'value'),
        Input('weight-metric-dropdown', 'value'),
    ]
)
def display_organization_threshold(value, selected_weight_metric):
    weight_label_mapping = {
        'raw': 'Raw Number of Collaborations',
        'proportion': 'Proportion of Collaborations',
        'cii': 'Collaboration Intensity Index (CII)'
    }
    weight_label = weight_label_mapping.get(selected_weight_metric, 'Weight')
    return f"Threshold ({weight_label} ≥ {value}):"


# Import State for multiple outputs
from dash.dependencies import Input, Output, State


# Update Organization-Level graph and store highlighted nodes
@app.callback(
    [
        Output('organization-graph', 'figure'),
        Output('highlighted-nodes-store', 'data')  # New Output
    ],
    [
        Input('year-slider-organization', 'value'),
        Input('layout-dropdown', 'value'),
        Input('weight-threshold-slider-organization', 'value'),
        Input('node-size-metric-dropdown', 'value'),
        Input('text-toggle', 'data'),  # Corrected to 'data'
        Input('weight-metric-dropdown', 'value'),
    ],
    State('selected-country', 'data')
)
def update_organization_graph(year, layout_type, weight_threshold, node_size_metric, show_text, selected_weight_metric,
                              selected_country):
    """
    Updates the Organization-Level Collaboration Network based on user inputs.
    Highlights organizations from the selected country with pink outlines.
    """
    if not selected_country:
        return create_empty_figure("Organization Collaboration Network - No Plottable Data"), []

    clicked_country = selected_country

    relevant_orgs = country_org_mapping[country_org_mapping['country_name'] == clicked_country][
        'high_level_org_name'].unique()
    if len(relevant_orgs) == 0:
        return create_empty_figure("Organization Collaboration Network - No Plottable Data"), []

    # Load organization-level data
    org_dfs = [
        load_processed_data('organization', year, entity=org)
        for org in relevant_orgs
    ]
    org_dfs = [df for df in org_dfs if not df.empty]
    if not org_dfs:
        return create_empty_figure("Organization Collaboration Network - No Plottable Data"), []

    org_df = pd.concat(org_dfs, ignore_index=True)

    # Compute weights based on selected metric
    if selected_weight_metric == 'raw':
        org_df['computed_weight'] = org_df['weight']
    elif selected_weight_metric == 'proportion':
        org_df = calculate_proportional_weights(org_df)
        org_df['computed_weight'] = org_df['proportional_weight']
    elif selected_weight_metric == 'cii':
        org_df = calculate_cii(org_df)
        # Use 'cii' directly; no need to assign to 'computed_weight'
    else:
        org_df['computed_weight'] = org_df['weight']

    # Thresholding
    if selected_weight_metric == 'cii':
        org_df = org_df[org_df['cii'] >= weight_threshold].copy()
    else:
        org_df = org_df[org_df['computed_weight'] >= weight_threshold].copy()

    if org_df.empty:
        return create_empty_figure(
            "Organization Collaboration Network - No Data Available (Check loading in tab and thresholds for graph)"), []

    # Build graph
    if selected_weight_metric == 'cii':
        G = build_graph(org_df, weight_column='cii')
    else:
        G = build_graph(org_df, weight_column='computed_weight')

    if G.number_of_nodes() == 0:
        return create_empty_figure(
            "Organization Collaboration Network - No Data Available (Check loading in tab and thresholds for graph)"), []

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

    # Identify highlighted nodes (organizations from the selected country)
    highlighted_nodes = relevant_orgs.tolist()

    # Create network figure with highlighting and tooltips
    fig = create_networkx_figure(
        G,
        pos,
        node_sizes,
        node_colors,
        show_text=bool(show_text),  # Convert list to bool
        weight_label=selected_weight_metric,
        org_country_mapping=dict(zip(country_org_mapping['high_level_org_name'], country_org_mapping['country_name'])),
        highlighted_nodes=highlighted_nodes,
        highlight_color='pink'
    )

    return fig, highlighted_nodes  # Return both figure and highlighted nodes


# ------------------------ Callbacks for Within-Country Network ------------------------ #

# Update threshold slider for Within-Country graph
@app.callback(
    [
        Output('weight-threshold-slider-within-country', 'min'),
        Output('weight-threshold-slider-within-country', 'max'),
        Output('weight-threshold-slider-within-country', 'marks'),
        Output('weight-threshold-slider-within-country', 'value'),
        Output('weight-threshold-slider-within-country', 'step'),
    ],
    [
        Input('year-slider-within-country', 'value'),
        Input('weight-metric-dropdown', 'value'),  # Corrected ID here
    ],
    [
        State('weight-threshold-slider-within-country', 'value'),
        State('selected-country', 'data')  # Added State for selected country
    ]
)
def update_within_country_threshold_slider(year, selected_weight_metric, current_value, selected_country):
    min_weight, max_weight, marks, default_value, step = update_threshold_slider(
        'within_country', year, current_value, selected_weight_metric, selected_country)
    return min_weight, max_weight, marks, default_value, step


# Update threshold display for Within-Country graph
@app.callback(
    Output('threshold-display-within-country', 'children'),
    [
        Input('weight-threshold-slider-within-country', 'value'),
        Input('weight-metric-dropdown', 'value'),
    ]
)
def display_within_country_threshold(value, selected_weight_metric):
    weight_label_mapping = {
        'raw': 'Raw Number of Collaborations',
        'proportion': 'Proportion of Collaborations',
        'cii': 'Collaboration Intensity Index (CII)'
    }
    weight_label = weight_label_mapping.get(selected_weight_metric, 'Weight')
    return f"Threshold ({weight_label} ≥ {value}):"


# Update Within-Country graph
@app.callback(
    Output('within-country-graph', 'figure'),
    [
        Input('year-slider-within-country', 'value'),
        Input('layout-dropdown', 'value'),  # Corrected ID
        Input('weight-threshold-slider-within-country', 'value'),
        Input('node-size-metric-dropdown', 'value'),  # Corrected ID
        Input('text-toggle', 'data'),  # Corrected to 'data'
        Input('weight-metric-dropdown', 'value'),  # Correct ID here
    ],
    State('selected-country', 'data')  # This should hold the selected country name
)
def update_within_country_graph(year, layout_type, weight_threshold, node_size_metric, show_text,
                                selected_weight_metric, selected_country):
    """
    Updates the Within-Country Collaboration Network based on user inputs.
    Loads data based on the selected country name.
    """
    logging.info(f"[Callback] Selected Country: {selected_country}")

    if not selected_country:
        logging.info("[Callback] No country selected.")
        return create_empty_figure("Within-Country Collaboration Network - No Country Selected")

    # Set graph_type to 'within_country' and entity to the selected country
    graph_type = 'within_country'
    entity = selected_country  # Ensure this is a country name

    logging.info(f"[Callback] Graph Type: {graph_type}, Entity: {entity}")

    # Load within-country data for the selected country and year
    df = load_processed_data(graph_type, year, entity=selected_country)

    if df.empty:
        logging.info("[Callback] DataFrame is empty after loading.")
        return create_empty_figure("Within-Country Collaboration Network - No Data Available")

    # Compute weights based on selected metric
    if selected_weight_metric == 'raw':
        df['computed_weight'] = df['weight']
    elif selected_weight_metric == 'proportion':
        df = calculate_proportional_weights(df)
        df['computed_weight'] = df['proportional_weight']
    elif selected_weight_metric == 'cii':
        df = calculate_cii(df)
        # 'cii' is now calculated directly in the DataFrame
    else:
        df['computed_weight'] = df['weight']

    # Thresholding
    if selected_weight_metric == 'cii':
        df = df[df['cii'] >= weight_threshold].copy()
    else:
        df = df[df['computed_weight'] >= weight_threshold].copy()

    logging.info(f"[Callback] Threshold Applied: {weight_threshold}")
    logging.info(f"[Callback] Number of Edges After Threshold: {len(df)}")

    if df.empty:
        logging.info("[Callback] DataFrame is empty after applying threshold.")
        return create_empty_figure("Within-Country Collaboration Network - No Data Available")

    # Build the graph
    if selected_weight_metric == 'cii':
        G = build_graph(df, weight_column='cii')
    else:
        G = build_graph(df, weight_column='computed_weight')
    logging.info(f"[Callback] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        logging.info("[Callback] Graph has no nodes.")
        return create_empty_figure("Within-Country Collaboration Network - No Data Available")

    # Compute centralities
    centrality = compute_centrality(G, node_size_metric)

    # Normalize node sizes
    if centrality:
        min_cent = min(centrality.values())
        max_cent = max(centrality.values())
        if max_cent - min_cent != 0:
            node_sizes = [
                10 + 30 * (centrality[node] - min_cent) / (max_cent - min_cent)
                for node in G.nodes()
            ]
            logging.info("[Callback] Normalized node sizes.")
        else:
            node_sizes = [20 for _ in G.nodes()]
            logging.info("[Callback] Uniform node sizes due to zero variance in centrality.")
    else:
        node_sizes = [20 for _ in G.nodes()]
        logging.info("[Callback] Default node sizes applied.")

    # Compute node colors based on betweenness centrality
    betweenness_centrality = compute_centrality(G, 'betweenness')
    if betweenness_centrality:
        min_bet = min(betweenness_centrality.values())
        max_bet = max(betweenness_centrality.values())
        if max_bet - min_bet != 0:
            node_colors = [
                (betweenness_centrality[node] - min_bet) / (max_bet - min_bet)
                for node in G.nodes()
            ]
            logging.info("[Callback] Normalized node colors based on betweenness centrality.")
        else:
            node_colors = [0.5 for _ in G.nodes()]
            logging.info("[Callback] Uniform node colors due to zero variance in betweenness centrality.")
    else:
        node_colors = [0.5 for _ in G.nodes()]
        logging.info("[Callback] Default node colors applied.")

    # Compute layout
    pos = compute_layout(G, layout_type)
    logging.info(f"[Callback] Computed layout: {layout_type}")

    # Highlight nodes if necessary (not needed for within-country)
    highlighted_nodes = []  # Define as needed or leave empty

    # Create network figure
    fig = create_networkx_figure(
        G,
        pos,
        node_sizes,
        node_colors,
        show_text=bool(show_text),  # Convert list to bool
        weight_label=selected_weight_metric,
        org_country_mapping=None,  # No country mapping needed within-country
        highlighted_nodes=highlighted_nodes,
        highlight_color='pink'
    )
    logging.info("[Callback] Network figure created.")

    return fig

# ------------------------ Callbacks to Display Edge Weights ------------------------ #

# Callback to display edge weights for clicked node in Country-Level Network
@app.callback(
    Output('edge-weights-display-country', 'children'),
    [
        Input('country-graph', 'clickData'),
        Input('year-slider-country', 'value'),
        Input('weight-threshold-slider-country', 'value'),
        Input('weight-metric-dropdown', 'value'),
        Input('node-size-metric-dropdown', 'value'),
    ]
)
def display_edge_weights_country(click_data, year, weight_threshold, selected_weight_metric, node_size_metric):
    """
    Displays the connected edge weights for the clicked node in the Country-Level Network.
    Includes node size metric values.
    """
    if not click_data:
        return "Click on a node in the Country-Level graph to see its connected edge weights."

    clicked_node = click_data['points'][0].get('customdata', None)
    if not clicked_node:
        return "No node data available."

    # Load data
    country_df = load_processed_data('country', year)
    if country_df.empty:
        return "No data available (Check loading in tab and thresholds for graph)."

    # Compute weights based on selected metric
    if selected_weight_metric == 'raw':
        country_df['computed_weight'] = country_df['weight']
    elif selected_weight_metric == 'proportion':
        country_df = calculate_proportional_weights(country_df)
        country_df['computed_weight'] = country_df['proportional_weight']
    elif selected_weight_metric == 'cii':
        country_df = calculate_cii(country_df)
        # 'cii' is now calculated directly in the DataFrame
    else:
        country_df['computed_weight'] = country_df['weight']

    # Thresholding
    if selected_weight_metric == 'cii':
        country_df = country_df[country_df['cii'] >= weight_threshold].copy()
    else:
        country_df = country_df[country_df['computed_weight'] >= weight_threshold].copy()

    if country_df.empty:
        return "No data available (Check loading in tab and thresholds for graph)."

    # Build graph
    if selected_weight_metric == 'cii':
        G = build_graph(country_df, weight_column='cii')
    else:
        G = build_graph(country_df, weight_column='computed_weight')

    # Compute centrality for node size metric on thresholded graph
    if node_size_metric == 'cii':
        node_cii = {}
        for node in G.nodes():
            cii_sum = sum([edata.get('weight', 0) for _, _, edata in G.edges(node, data=True)])
            node_cii[node] = cii_sum
        centrality = node_cii
    else:
        centrality = compute_centrality(G, node_size_metric)

    # Find edges connected to the clicked node
    connected_edges = country_df[
        (country_df['node1'] == clicked_node) | (country_df['node2'] == clicked_node)
    ]

    if connected_edges.empty:
        return "No connected edges found for the selected node."

    # Sort edges by computed weight descending
    if selected_weight_metric == 'cii':
        connected_edges = connected_edges.sort_values(by='cii', ascending=False)
    else:
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
        html.Thead(html.Tr([html.Th("Connected Node"), html.Th(weight_label), html.Th(node_size_label)]))
    ]

    table_rows = []
    for _, row in connected_edges.iterrows():
        if row['node1'] == row['node2']:
            continue  # Skip self-loops
        other_node = row['node2'] if row['node1'] == clicked_node else row['node1']

        if selected_weight_metric == 'cii':
            weight_value = row['cii']
            weight_display = f"{weight_value:.4f}"
        else:
            weight_value = row['computed_weight']
            if selected_weight_metric == 'raw':
                weight_display = str(int(weight_value))
            else:
                weight_display = f"{weight_value:.4f}"

        # Get the node size metric value for the connected node
        node_metric_value = centrality.get(other_node, 0)
        node_metric_display = f"{node_metric_value:.4f}"
        table_rows.append(html.Tr([html.Td(other_node), html.Td(weight_display), html.Td(node_metric_display)]))

    table_body = [html.Tbody(table_rows)]

    table = html.Table(table_header + table_body, style={'width': '100%', 'textAlign': 'left'})

    return html.Div([
        html.H4(f"Edge Weights and Node Metrics for {clicked_node}", style={'font-family': 'Arial'}),
        table
    ])


# Callback to display edge weights for clicked node in Organization-Level Network
@app.callback(
    Output('edge-weights-display-organization', 'children'),
    [
        Input('organization-graph', 'clickData'),
        Input('year-slider-organization', 'value'),
        Input('weight-threshold-slider-organization', 'value'),
        Input('weight-metric-dropdown', 'value'),
        Input('node-size-metric-dropdown', 'value'),
    ],
    State('selected-country', 'data')  # Corrected from 'country-dropdown' to 'selected-country'
)
def display_edge_weights_organization(click_data, year, weight_threshold, selected_weight_metric, node_size_metric,
                                      selected_country):
    """
    Displays the connected edge weights for the clicked node in the Organization-Level Network.
    Includes node size metric values.
    """
    if not click_data:
        return "Click on a node in the Organization-Level graph to see its connected edge weights."

    clicked_node = click_data['points'][0].get('customdata', None)
    if not clicked_node:
        return "No node data available."

    if not selected_country:
        return "Please select a country to view organization-level collaborations."

    clicked_country = selected_country

    # Get relevant organizations for the country
    relevant_orgs = country_org_mapping[country_org_mapping['country_name'] == clicked_country][
        'high_level_org_name'].unique()
    if len(relevant_orgs) == 0:
        return "No organizations found for the country."

    # Load data for relevant organizations
    org_dfs = [
        load_processed_data('organization', year, entity=org)
        for org in relevant_orgs
    ]
    org_dfs = [df for df in org_dfs if not df.empty]
    if not org_dfs:
        return "No data available (Check loading in tab and thresholds for graph)."

    org_df = pd.concat(org_dfs, ignore_index=True)

    # Compute weights based on selected metric
    if selected_weight_metric == 'raw':
        org_df['computed_weight'] = org_df['weight']
    elif selected_weight_metric == 'proportion':
        org_df = calculate_proportional_weights(org_df)
        org_df['computed_weight'] = org_df['proportional_weight']
    elif selected_weight_metric == 'cii':
        org_df = calculate_cii(org_df)
        # Use 'cii' directly; no need to assign to 'computed_weight'
    else:
        org_df['computed_weight'] = org_df['weight']

    # Thresholding
    if selected_weight_metric == 'cii':
        org_df = org_df[org_df['cii'] >= weight_threshold].copy()
    else:
        org_df = org_df[org_df['computed_weight'] >= weight_threshold].copy()

    if org_df.empty:
        return "No data available (Check loading in tab and thresholds for graph)."

    # Build graph
    if selected_weight_metric == 'cii':
        G = build_graph(org_df, weight_column='cii')
    else:
        G = build_graph(org_df, weight_column='computed_weight')

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
    connected_edges = org_df[
        (org_df['node1'] == clicked_node) | (org_df['node2'] == clicked_node)
    ]

    if connected_edges.empty:
        return "No connected edges found for the selected node."

    # Sort edges by weight descending
    if selected_weight_metric == 'cii':
        connected_edges = connected_edges.sort_values(by='cii', ascending=False)
    else:
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
        html.Thead(html.Tr([html.Th("Connected Node"), html.Th(weight_label), html.Th(node_size_label)]))
    ]

    table_rows = []
    for _, row in connected_edges.iterrows():
        if row['node1'] == row['node2']:
            continue  # Skip self-loops
        other_node = row['node2'] if row['node1'] == clicked_node else row['node1']

        if selected_weight_metric == 'cii':
            weight_value = row['cii']
            weight_display = f"{weight_value:.4f}"
        else:
            weight_value = row['computed_weight']
            if selected_weight_metric == 'raw':
                weight_display = str(int(weight_value))
            else:
                weight_display = f"{weight_value:.4f}"

        # Get the node size metric value for the connected node
        node_metric_value = centrality.get(other_node, 0)
        node_metric_display = f"{node_metric_value:.4f}"
        table_rows.append(html.Tr([html.Td(other_node), html.Td(weight_display), html.Td(node_metric_display)]))

    table_body = [html.Tbody(table_rows)]

    table = html.Table(table_header + table_body, style={'width': '100%', 'textAlign': 'left'})

    return html.Div([
        html.H4(f"Edge Weights and Node Metrics for {clicked_node}", style={'font-family': 'Arial'}),
        table
    ])


# ------------------------ Callbacks to Display Edge Weights for Within-Country Network ------------------------ #

# Callback to display edge weights for clicked node in Within-Country Network
@app.callback(
    Output('edge-weights-display-within-country', 'children'),
    [
        Input('within-country-graph', 'clickData'),
        Input('year-slider-within-country', 'value'),
        Input('weight-threshold-slider-within-country', 'value'),
        Input('weight-metric-dropdown', 'value'),
        Input('node-size-metric-dropdown', 'value'),
    ],
    State('selected-country', 'data')  # Corrected from 'country-dropdown' to 'selected-country'
)
def display_edge_weights_within_country(click_data, year, weight_threshold, selected_weight_metric, node_size_metric,
                                        selected_country):
    """
    Displays the connected edge weights for the clicked node in the Within-Country Network.
    Includes node size metric values.
    """
    if not click_data:
        return "Click on a node in the Within-Country graph to see its connected edge weights."

    clicked_node = click_data['points'][0].get('customdata', None)
    if not clicked_node:
        return "No node data available."

    if not selected_country:
        return "Please select a country to view within-country collaborations."

    clicked_country = selected_country

    # Load data for the selected country and year
    df = load_processed_data('within_country', year, entity=clicked_country)
    if df.empty:
        return "No data available (Check loading in tab and thresholds for graph)."

    # Compute weights based on selected metric
    if selected_weight_metric == 'raw':
        df['computed_weight'] = df['weight']
    elif selected_weight_metric == 'proportion':
        df = calculate_proportional_weights(df)
        df['computed_weight'] = df['proportional_weight']
    elif selected_weight_metric == 'cii':
        df = calculate_cii(df)
        # Use 'cii' directly; no need to assign to 'computed_weight'
    else:
        df['computed_weight'] = df['weight']

    # Thresholding
    if selected_weight_metric == 'cii':
        df = df[df['cii'] >= weight_threshold].copy()
    else:
        df = df[df['computed_weight'] >= weight_threshold].copy()

    if df.empty:
        return "No data available (Check loading in tab and thresholds for graph)."

    # Build graph
    if selected_weight_metric == 'cii':
        G = build_graph(df, weight_column='cii')
    else:
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

    # Sort edges by weight descending
    if selected_weight_metric == 'cii':
        connected_edges = connected_edges.sort_values(by='cii', ascending=False)
    else:
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
        html.Thead(html.Tr([html.Th("Connected Node"), html.Th(weight_label), html.Th(node_size_label)]))
    ]

    table_rows = []
    for _, row in connected_edges.iterrows():
        if row['node1'] == row['node2']:
            continue  # Skip self-loops
        other_node = row['node2'] if row['node1'] == clicked_node else row['node1']

        if selected_weight_metric == 'cii':
            weight_value = row['cii']
            weight_display = f"{weight_value:.4f}"
        else:
            weight_value = row['computed_weight']
            if selected_weight_metric == 'raw':
                weight_display = str(int(weight_value))
            else:
                weight_display = f"{weight_value:.4f}"

        # Get the node size metric value for the connected node
        node_metric_value = centrality.get(other_node, 0)
        node_metric_display = f"{node_metric_value:.4f}"
        table_rows.append(html.Tr([html.Td(other_node), html.Td(weight_display), html.Td(node_metric_display)]))

    table_body = [html.Tbody(table_rows)]

    table = html.Table(table_header + table_body, style={'width': '100%', 'textAlign': 'left'})

    return html.Div([
        html.H4(f"Edge Weights and Node Metrics for {clicked_node}", style={'font-family': 'Arial'}),
        table
    ])


# ------------------------ Run the App ------------------------ #

# if __name__ == '__main__':
#     app.run_server(debug=True)

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080)
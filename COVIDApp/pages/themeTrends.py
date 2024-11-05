# ./pages/themeTrends.py

from dash import html, dcc, callback, Output, Input
import plotly.graph_objs as go
import pandas as pd
import gcsfs
import numpy as np
from dash.exceptions import PreventUpdate
from cache import cache
import config  # Adjust the import path based on your project structure
import ast
import plotly.graph_objects as go

# Define a list of 20 distinct colors (same as in themes.py)
DISTINCT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# Define a mapping from cluster numbers (1-20) to cluster names
CLUSTER_MAPPING = {
    1: "Social Impacts and Public Health",
    2: "Medical and Patient-Focused Research",
    3: "Nutritional and Prevention Research",
    4: "Policy and Governance",
    5: "Clinical Trials and Patient Outcomes",
    6: "Educational Changes",
    7: "Digital and Remote Learning",
    8: "Social Media and Information Dissemination",
    9: "Disease Spread and Impact",
    10: "Health Risks and Vulnerabilities",
    11: "Pandemic Response and Learning Systems",
    12: "COVID and Sociocultural Dynamics",
    13: "Environmental Impact and Behavior",
    14: "Psychological and Social Support",
    15: "Public Health and Healthcare Systems",
    16: "Artificial Intelligence and Data Analysis",
    17: "Social Distancing and Community Response",
    18: "Digital Health and Telemedicine",
    19: "Clinical Evidence and Treatment",
    20: "Global and Policy Impact"
}

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

def load_theme_trends_data():
    """
    Loads the topicTrends.parquet data from Google Cloud Storage.

    Returns:
        pd.DataFrame: DataFrame containing yearly trends for each cluster.
    """
    @cache.memoize(timeout=600)  # Cache for 10 minutes
    def _load():
        try:
            fs = gcsfs.GCSFileSystem()
            file_path = 'gs://covid-dash-app/wordClouds/topicTrends.parquet'
            with fs.open(file_path, 'rb') as f:
                df_trends = pd.read_parquet(f)
            print(f"Loaded topicTrends.parquet with shape: {df_trends.shape}")
            print("Columns in topicTrends.parquet:", df_trends.columns.tolist())
            return df_trends
        except Exception as e:
            print(f"Error loading topicTrends.parquet: {e}")
            return pd.DataFrame()

    return _load()

def process_trends_data(df_trends):
    """
    Processes the trends DataFrame by ensuring proper data types and mapping.

    Parameters:
        df_trends (pd.DataFrame): Raw trends data.

    Returns:
        pd.DataFrame: Processed trends data with cluster names.
    """
    if df_trends.empty:
        print("Trends DataFrame is empty.")
        return df_trends

    # Map ClusterNumber to names using CLUSTER_MAPPING, and drop any unmapped clusters
    df_trends = df_trends[df_trends['finalClustersAll'].isin(CLUSTER_MAPPING.keys())]
    df_trends['ClusterName'] = df_trends['finalClustersAll'].map(CLUSTER_MAPPING)

    # Ensure 'year' is integer
    df_trends['year'] = df_trends['year'].astype(int)

    print("Processed trends data successfully with mapped cluster names.")
    return df_trends
def create_growth_rate_plot(df_trends):
    """
    Creates a Plotly line plot for growth rate over time for each cluster, starting from 2021.

    Parameters:
        df_trends (pd.DataFrame): Processed trends data.

    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        fig = go.Figure()

        # Filter data to include only years >= 2021
        df_trends = df_trends[df_trends['year'] >= 2021]

        for cluster_name in sorted(df_trends['ClusterName'].unique()):
            cluster_data = df_trends[df_trends['ClusterName'] == cluster_name].sort_values('year')

            fig.add_trace(go.Scatter(
                x=cluster_data['year'],
                y=cluster_data['growth_rate'],
                mode='lines+markers',
                name=cluster_name,
                line=dict(color=DISTINCT_COLORS[list(CLUSTER_MAPPING.values()).index(cluster_name)]),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title='Growth Rate Over Time by Cluster ',
            xaxis_title='Year',
            yaxis_title='Growth Rate',
            xaxis=dict(range=[2021, df_trends['year'].max()]),  # Set x-axis to start at 2021
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        print("Growth rate plot created successfully with correct cluster names.")
        return fig
    except Exception as e:
        print(f"Error creating growth rate plot: {e}")
        return create_empty_figure("Growth Rate Plotting Error")

def create_relative_share_plot(df_trends):
    """
    Creates a Plotly line plot for relative share of each cluster over time, labeled by cluster name.

    Parameters:
        df_trends (pd.DataFrame): Processed trends data with mapped cluster names.

    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        fig = go.Figure()

        # Loop over unique cluster names for plotting
        for cluster_name in sorted(df_trends['ClusterName'].unique()):
            cluster_data = df_trends[df_trends['ClusterName'] == cluster_name].sort_values('year')

            fig.add_trace(go.Scatter(
                x=cluster_data['year'],
                y=cluster_data['relative_share'],
                mode='lines+markers',
                name=cluster_name,
                line=dict(color=DISTINCT_COLORS[list(CLUSTER_MAPPING.values()).index(cluster_name)]),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title='Relative Share of Each Cluster Over Time',
            xaxis_title='Year',
            yaxis_title='Relative Share',
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        print("Relative share plot created successfully.")
        return fig
    except Exception as e:
        print(f"Error creating relative share plot: {e}")
        return create_empty_figure("Relative Share Plotting Error")
def create_slope_plot(df_trends):
    """
    Creates a Plotly bar plot for the slope of relative share over time for each cluster, labeled by cluster name.

    Parameters:
        df_trends (pd.DataFrame): Processed trends data with mapped cluster names.

    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        # Calculate slope using linear regression for each cluster name
        slopes = {}
        for cluster_name in sorted(df_trends['ClusterName'].unique()):
            cluster_data = df_trends[df_trends['ClusterName'] == cluster_name].sort_values('year')
            if len(cluster_data) < 2:
                slopes[cluster_name] = 0
                continue
            x = cluster_data['year']
            y = cluster_data['relative_share']
            slope = np.polyfit(x, y, 1)[0]
            slopes[cluster_name] = slope

        # Prepare data for bar plot
        cluster_names = list(slopes.keys())
        slope_values = [slopes[cluster] for cluster in cluster_names]
        colors = [DISTINCT_COLORS[list(CLUSTER_MAPPING.values()).index(cluster)] for cluster in cluster_names]

        fig = go.Figure(data=[
            go.Bar(
                x=cluster_names,
                y=slope_values,
                marker_color=colors
            )
        ])

        fig.update_layout(
            title='Slope of Relative Share Over Time by Cluster',
            xaxis_title='Cluster Name',
            yaxis_title='Slope',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        print("Slope plot created successfully.")
        return fig
    except Exception as e:
        print(f"Error creating slope plot: {e}")
        return create_empty_figure("Slope Plotting Error")
def create_theme_trends_layout():
    """
    Creates the layout for the Theme Trends page.

    Returns:
        dash_html_components.Div: The layout for the Theme Trends page.
    """
    return html.Div([
        # Header
        html.H2("Theme Trends Analysis", style={'textAlign': 'center'}),

        # Description Text Area
        dcc.Textarea(
            id='theme-trends-description',
            value="""
            This section presents the yearly trends for each theme cluster. The visualizations include:

            1. **Growth Rate Over Time**: Displays how the growth rate of each cluster has evolved annually.
            2. **Relative Share of Each Cluster Over Time**: Shows the proportion of each cluster relative to the total over the years.
            3. **Slope of Relative Share Over Time**: Illustrates the trend (increase or decrease) in the relative share of each cluster.

            Helps highlight the dynamics and evolving focus areas within the themes.
            """,
            style={
                'width': '100%',
                'height': 150,
                'padding': '10px',
                'font-family': 'Arial',
                'font-size': '16px',
                'resize': 'none'
            },
            disabled=True  # Make it read-only
        ),

        html.Br(),

        # Growth Rate Plot
        html.Div([
            dcc.Graph(id='theme-trends-growth-rate-plot')
        ], style={'width': '100%', 'display': 'inline-block'}),

        html.Br(),

        # Relative Share Plot
        html.Div([
            dcc.Graph(id='theme-trends-relative-share-plot')
        ], style={'width': '100%', 'display': 'inline-block'}),

        html.Br(),

        # Slope Plot
        html.Div([
            dcc.Graph(id='theme-trends-slope-plot')
        ], style={'width': '100%', 'display': 'inline-block'}),

    ], style={'padding': '20px', 'font-family': 'Arial'})

@callback(
    Output('theme-trends-growth-rate-plot', 'figure'),
    Input('url', 'pathname')
)
def update_growth_rate_plot(pathname):
    print(f"Callback Triggered: update_growth_rate_plot with pathname={pathname}")
    if pathname != '/theme-trends':
        print("Not the Theme Trends page. Preventing update.")
        raise PreventUpdate

    df_trends = load_theme_trends_data()

    if df_trends.empty:
        print("topicTrends.parquet is empty. Returning empty growth rate plot.")
        return create_empty_figure("Growth Rate Over Time - No Data Available")

    df_trends = process_trends_data(df_trends)

    if df_trends.empty:
        print("Processed trends DataFrame is empty after remapping. Returning empty growth rate plot.")
        return create_empty_figure("Growth Rate Over Time - No Valid Data")

    fig = create_growth_rate_plot(df_trends)
    return fig

@callback(
    Output('theme-trends-relative-share-plot', 'figure'),
    Input('url', 'pathname')
)
def update_relative_share_plot(pathname):
    print(f"Callback Triggered: update_relative_share_plot with pathname={pathname}")
    if pathname != '/theme-trends':
        print("Not the Theme Trends page. Preventing update.")
        raise PreventUpdate

    df_trends = load_theme_trends_data()

    if df_trends.empty:
        print("topicTrends.parquet is empty. Returning empty relative share plot.")
        return create_empty_figure("Relative Share Over Time - No Data Available")

    df_trends = process_trends_data(df_trends)

    if df_trends.empty:
        print("Processed trends DataFrame is empty after remapping. Returning empty relative share plot.")
        return create_empty_figure("Relative Share Over Time - No Valid Data")

    fig = create_relative_share_plot(df_trends)
    return fig

@callback(
    Output('theme-trends-slope-plot', 'figure'),
    Input('url', 'pathname')
)
def update_slope_plot(pathname):
    print(f"Callback Triggered: update_slope_plot with pathname={pathname}")
    if pathname != '/theme-trends':
        print("Not the Theme Trends page. Preventing update.")
        raise PreventUpdate

    df_trends = load_theme_trends_data()

    if df_trends.empty:
        print("topicTrends.parquet is empty. Returning empty slope plot.")
        return create_empty_figure("Slope of Relative Share - No Data Available")

    df_trends = process_trends_data(df_trends)

    if df_trends.empty:
        print("Processed trends DataFrame is empty after remapping. Returning empty slope plot.")
        return create_empty_figure("Slope of Relative Share - No Valid Data")

    fig = create_slope_plot(df_trends)
    return fig

# Register the layout
def register_theme_trends_page(app):
    """
    Registers the Theme Trends page layout with the Dash app.

    Parameters:
        app (dash.Dash): The Dash application instance.
    """
    app.layout = create_theme_trends_layout()

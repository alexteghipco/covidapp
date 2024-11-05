# pages/main_page.py

from dash import html, dcc

# Assuming you have pre-defined helper functions and imports
def create_main_page_layout():
    return html.Div([
        # Header Section
        html.H1("Collaboration Networks Over Time (please be patient when clicking on nodes)", style={'font-family': 'Arial', 'textAlign': 'center'}),

        # Controls arranged horizontally
        html.Div([
            html.Button('Toggle Text', id='toggle-text-button', n_clicks=0, style={'margin-right': '20px'}),
            html.Label("Select Layout Algorithm:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='layout-dropdown',
                options=[
                    {'label': 'Spring Layout', 'value': 'spring_layout'},
                    {'label': 'Kamada-Kawai Layout', 'value': 'kamada_kawai_layout'},
                    {'label': 'Circular Layout', 'value': 'circular_layout'},
                    {'label': 'Shell Layout', 'value': 'shell_layout'},
                    {'label': 'Spectral Layout', 'value': 'spectral_layout'}
                ],
                value='spring_layout',  # Default layout
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle', 'margin-right': '20px'}
            ),
            html.Label("Select Node Size Metric:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='node-size-metric-dropdown',
                options=[
                    {'label': 'Degree Centrality', 'value': 'degree'},
                    {'label': 'Betweenness Centrality', 'value': 'betweenness'},
                    {'label': 'Closeness Centrality', 'value': 'closeness'},
                    {'label': 'Eigenvector Centrality', 'value': 'eigenvector'},
                    {'label': 'Collaboration Intensity Index (CII)', 'value': 'cii'},
                ],
                value='cii',
                clearable=False,
                style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle', 'margin-right': '20px'}
            ),
            html.Label("Select Weight Metric:", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='weight-metric-dropdown',
                options=[
                    {'label': 'Raw Number of Collaborations', 'value': 'raw'},
                    {'label': 'Proportion of Collaborations', 'value': 'proportion'},
                    {'label': 'Collaboration Intensity Index (CII)', 'value': 'cii'}
                ],
                value='proportion',
                clearable=False,
                style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle', 'margin-right': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'padding': '10px', 'font-family': 'Arial'}),

        # Country-Level Network Section with its own Sliders
        html.Div([
            html.H2("Country-Level Collaboration Network", style={'font-family': 'Arial', 'textAlign': 'center'}),
            html.Div([
                html.Label("Country-Level Minimum Collaborations:", style={'font-family': 'Arial'}),
                html.Div(id='threshold-display-country', style={'font-family': 'Arial', 'textAlign': 'center'}),
                dcc.Slider(
                    id='weight-threshold-slider-country',
                    min=1,
                    max=100,
                    step=1,
                    value=5,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'padding': '20px'}),

            html.Div([
                html.Label("Select Year for Country-Level:", style={'font-family': 'Arial'}),
                dcc.Slider(
                    id='year-slider-country',
                    min=2020,  # Replace with dynamic minimum year
                    max=2024,  # Replace with dynamic maximum year
                    value=2020,  # Replace with dynamic default year
                    marks={year: str(year) for year in range(2020, 2025)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': '20px'}),

            dcc.Loading(
                dcc.Graph(id='country-graph'),
                type='default'
            ),

            # Display Edge Weights for Clicked Node in Country-Level Network
            html.Div([
                html.H3("Connected Edge Weights", style={'font-family': 'Arial'}),
                html.Div(id='edge-weights-display-country', style={'font-family': 'Arial'})
            ], style={'padding': '20px'})
        ], style={'padding': '20px', 'border': '3px solid #ccc', 'margin': '20px'}),

        # Organization-Level Network Section with its own Sliders
        html.Div([
            html.H2("Organization-Level Collaboration Network", style={'font-family': 'Arial', 'textAlign': 'center'}),
            html.Div([
                html.Label("Organization-Level Minimum Collaborations:", style={'font-family': 'Arial'}),
                html.Div(id='threshold-display-organization', style={'font-family': 'Arial', 'textAlign': 'center'}),
                dcc.Slider(
                    id='weight-threshold-slider-organization',
                    min=1,
                    max=100,
                    step=1,
                    value=5,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'padding': '20px'}),

            html.Div([
                html.Label("Select Year for Organization-Level:", style={'font-family': 'Arial'}),
                dcc.Slider(
                    id='year-slider-organization',
                    min=2020,  # Replace with dynamic minimum year
                    max=2024,  # Replace with dynamic maximum year
                    value=2020,  # Replace with dynamic default year
                    marks={year: str(year) for year in range(2020, 2025)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': '20px'}),

            dcc.Loading(
                dcc.Graph(id='organization-graph'),
                type='default'
            ),

            # Display Edge Weights for Clicked Node in Organization-Level Network
            html.Div([
                html.H3("Connected Edge Weights", style={'font-family': 'Arial'}),
                html.Div(id='edge-weights-display-organization', style={'font-family': 'Arial'})
            ], style={'padding': '20px'})
        ], style={'padding': '20px', 'border': '3px solid #ccc', 'margin': '20px'}),

        # Within-Country Network Section with its own Sliders
        html.Div([
            html.H2("Within-Country Collaboration Network", style={'font-family': 'Arial', 'textAlign': 'center'}),
            html.Div([
                html.Label("Within-Country Minimum Collaborations:", style={'font-family': 'Arial'}),
                html.Div(id='threshold-display-within-country', style={'font-family': 'Arial', 'textAlign': 'center'}),
                dcc.Slider(
                    id='weight-threshold-slider-within-country',
                    min=0,  # Set to 0 for 'proportion' and 'cii'
                    max=1,  # Initial max; will be updated by callback
                    step=0.01,  # Finer steps for 'proportion' and 'cii'
                    value=0.04,  # Initial value; will be updated by callback
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'padding': '20px'}),

            html.Div([
                html.Label("Select Year for Within-Country:", style={'font-family': 'Arial'}),
                dcc.Slider(
                    id='year-slider-within-country',
                    min=2020,  # Replace with dynamic minimum year
                    max=2024,  # Replace with dynamic maximum year
                    value=2020,  # Replace with dynamic default year
                    marks={year: str(year) for year in range(2020, 2025)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': '20px'}),

            dcc.Loading(
                dcc.Graph(id='within-country-graph'),
                type='default'
            ),

            # Display Edge Weights for Clicked Node in Within-Country Network
            html.Div([
                html.H3("Connected Edge Weights", style={'font-family': 'Arial'}),
                html.Div(id='edge-weights-display-within-country', style={'font-family': 'Arial'})
            ], style={'padding': '20px'})
        ], style={'padding': '20px', 'border': '3px solid #ccc', 'margin': '20px'}),

        # Store components for inter-page interactions
        dcc.Store(id='selected-node', data={'node': None, 'type': None}),
        dcc.Store(id='selected-country', data=None),
        dcc.Store(id='text-toggle', data=True),
        dcc.Store(id='highlighted-nodes-store', data=[]),  # New Store for highlighted nodes

    ])

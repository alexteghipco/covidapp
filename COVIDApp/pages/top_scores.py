from dash import html, dcc, callback, Output, Input
import plotly.graph_objs as go
import pandas as pd
import gcsfs
from cache import cache
from dash.exceptions import PreventUpdate

# Define color scales
VIRIDIS_COLORS = ['#FDE725', '#B4DE2C', '#6DCD59', '#35B779', '#1F9E89', '#26828E', '#31688E', '#3E4989', '#482878', '#440154']
THERMAL_COLORS = ['#FCFFA4', '#F7D03C', '#FB9006', '#ED6925', '#CF4446', '#A52C60', '#781C6D', '#4B0C6B', '#1B0C41', '#000003']

@cache.memoize(timeout=600)
def load_data(file_name):
    """
    Loads data from Google Cloud Storage using gcsfs.
    """
    try:
        fs = gcsfs.GCSFileSystem()
        if file_name == 'top_30_authors_by_total_score.csv':
            file_path = 'gs://covid-dash-app/top/top/top_30_authors_by_total_score.csv'
        elif file_name == 'top_30_organizations_by_total_score.csv':
            file_path = 'gs://covid-dash-app/top/top/top_30_organizations_by_total_score.csv'
        else:
            raise ValueError("Unexpected file name provided")

        print(f"Loading data from file: {file_name}")
        with fs.open(file_path, 'r') as f:
            df = pd.read_csv(f)

        # Take the top 25 rows for display
        df = df.iloc[1:26]
        print(f"Loaded {file_name} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_name}: {str(e)}")
        return pd.DataFrame()

@callback(
    [Output('authors-bar-chart', 'figure'),
     Output('organizations-bar-chart', 'figure')],
    [Input('author-score-type', 'value'),
     Input('org-score-type', 'value')]
)
def update_charts(author_score_type, org_score_type):
    # Load the authors data for the authors chart
    print("\nLOADING AUTHORS DATA:")
    authors_df = load_data('top_30_authors_by_total_score.csv')
    if authors_df.empty:
        raise PreventUpdate  # Stop if author data isn't loaded correctly
    print("Authors dataframe loaded:")
    print(authors_df.head())

    # Create authors chart using the correct author data
    authors_fig = create_bar_chart(
        df=authors_df,
        score_type=author_score_type,
        color_scale=VIRIDIS_COLORS,
        title=f"Top 25 Authors by {'Total' if author_score_type == 'total' else 'Median'} Impact Score",
        is_author=True
    )

    # Load the organizations data for the organizations chart
    print("\nLOADING ORGANIZATIONS DATA:")
    orgs_df = load_data('top_30_organizations_by_total_score.csv')
    if orgs_df.empty:
        raise PreventUpdate  # Stop if organization data isn't loaded correctly
    print("Organizations dataframe loaded:")
    print(orgs_df.head())

    # Create organizations chart using the correct organization data
    orgs_fig = create_bar_chart(
        df=orgs_df,
        score_type=org_score_type,
        color_scale=THERMAL_COLORS,
        title=f"Top 25 Organizations by {'Total' if org_score_type == 'total' else 'Average per Author'} Impact Score",
        is_author=False  # Set to False to use organization-specific parameters
    )

    return authors_fig, orgs_fig


def create_bar_chart(df, score_type, color_scale, title, is_author=False):
    """
    Creates a bar chart visualization.
    """
    if df.empty:
        return go.Figure()

    print(f"Available columns: {df.columns.tolist()}")  # Debug print

    # Set the prefix based on whether the chart is for authors or organizations
    prefix = "Author" if is_author else "Organization"

    # Get the name column (first column in each dataset)
    name_col = df.columns[0]

    # Get the score column based on selection and type
    if is_author:
        score_col = 'total_impact_score' if score_type == 'total' else 'median_impact_score'
    else:
        score_col = 'total_impact_score' if score_type == 'total' else df.columns[2]  # For organizations

    print(f"Using score column: {score_col}, name column: {name_col}, prefix: {prefix}")  # Debug print

    # Sort by score while keeping original order for colors
    df = df.copy()
    df['color_index'] = range(len(df))
    df_sorted = df.sort_values(score_col, ascending=False)

    # Create hover text
    score_label = score_col.replace('_', ' ').title()
    hover_text = [
        f"{prefix}: {name}<br>{score_label}: {score:.2f}"
        for name, score in zip(df_sorted[name_col], df_sorted[score_col])
    ]

    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(df_sorted))),
        y=df_sorted[score_col],
        marker=dict(
            color=df_sorted['color_index'],
            colorscale=color_scale,
            showscale=False
        ),
        hovertext=hover_text,
        hoverinfo='text'
    ))

    # Update layout with minimal bar gaps
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            title=None,
            showgrid=False
        ),
        yaxis=dict(
            title='Impact Score',
            gridcolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Arial'
        ),
        height=400,
        margin=dict(t=50, r=20, b=20, l=60),
        bargap=0.05,
        bargroupgap=0.05
    )

    return fig

def create_top_scores_layout():
    """
    Creates the layout for the Top Scores page.
    """
    return html.Div([
        # Description text box
        dcc.Textarea(
            id='top-scores-description',
            value="""
                This page shows the top 25 authors and organizations by impact scores. 
                Impact scores are calculated based on article-level metrics and citation patterns. 
                You can view either total impact scores or median impact scores (authors) / average impact per author (organizations).

                After normalization, each component is given a weight that reflects its importance in the overall impact metric. Here are the weights used:

                - Field Citation Ratio: 0.2
                - Altmetric Score: 0.2
                - Altmetric Residual (difference between predicted and true altmetric scores; see prediction tab): 0.2
                - Number of Other Author Publications: 0.2
                - Patent Count, Dataset Count, Grant Count, and Unique Organizations On Publication: 0.05 each
                
                The impact score for each author-publication entry is computed by multiplying each normalized component by its weight and summing these products.
                
            """
            ,
            style={
                'width': '100%',
                'height': 100,
                'padding': '10px',
                'margin-bottom': '20px',
                'font-family': 'Arial',
                'font-size': '16px',
                'border': '1px solid #ddd',
                'border-radius': '5px',
                'background-color': '#f8f9fa',
                'color': '#212529'
            }
        ),

        # Authors Section
        html.Div([
            html.H3("Top Authors by Impact Score", style={'margin-bottom': '20px'}),
            dcc.Dropdown(
                id='author-score-type',
                options=[
                    {'label': 'Total Impact Score', 'value': 'total'},
                    {'label': 'Median Impact Score', 'value': 'median'}
                ],
                value='total',
                style={'width': '200px', 'margin-bottom': '20px'}
            ),
            dcc.Loading(
                id="loading-authors",
                type="circle",
                children=dcc.Graph(id='authors-bar-chart')
            )
        ], style={'margin-bottom': '40px'}),

        # Organizations Section
        html.Div([
            html.H3("Top Organizations by Impact Score", style={'margin-bottom': '20px'}),
            dcc.Dropdown(
                id='org-score-type',
                options=[
                    {'label': 'Total Impact Score', 'value': 'total'},
                    {'label': 'Average Impact per Author', 'value': 'average'}
                ],
                value='total',
                style={'width': '250px', 'margin-bottom': '20px'}
            ),
            dcc.Loading(
                id="loading-orgs",
                type="circle",
                children=dcc.Graph(id='organizations-bar-chart')
            )
        ])
    ], style={'padding': '20px', 'font-family': 'Arial', 'max-width': '1200px', 'margin': '0 auto'})

__all__ = ['create_top_scores_layout']

# pages/altmetricPred.py

import pandas as pd
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import gcsfs
fs = gcsfs.GCSFileSystem()

def create_altmetric_predictions_layout():
    data_path = 'gs://covid-dash-app/ML/preds.parquet'

    # Load data from GCS
    with fs.open(data_path, 'rb') as f:
        df = pd.read_parquet(f)

    # Create scatter plot with larger dots, color #257180, and crop axes
    fig = px.scatter(
        df,
        x='Actual Altmetric Score',
        y='Predicted Altmetric Score',
        trendline=None
    )

    # Customize scatter plot points with larger size
    fig.update_traces(
        marker=dict(size=12, color='#257180')  # Increase dot size
    )

    # Add diagonal line (y = x) for ideal predictions and ensure it appears on top
    min_val = min(df['Actual Altmetric Score'].min(), df['Predicted Altmetric Score'].min())
    max_val = max(df['Actual Altmetric Score'].max(), df['Predicted Altmetric Score'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal Prediction',
        line=dict(color='red', dash='dash', width=2),
        visible=True  # This trace will appear above the scatter plot points
    ))

    # Crop axes, set white background, grey gridlines, and labels
    fig.update_layout(
        xaxis=dict(range=[0, 12000], title='Actual Altmetric Score', showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(range=[0, 12000], title='Predicted Altmetric Score', showgrid=True, gridcolor='lightgrey'),
        plot_bgcolor='white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Add an annotation to explain the area above the red line
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.05, showarrow=False,
        text="Above line: Higher than anticipated impact",
        font=dict(size=12, color="black"),
        align="right"
    )

    # Text for statistical metrics
    stats_text = "Pearson's r = 0.9, p = 0; Spearman's rho = 0.9, p = 0."

    # Create layout
    layout = html.Div([
        html.H3("Altmetric Predictions"),
        # Text box at the top
        dcc.Textarea(
            id='altmetric-textbox',
            value="""Altmetric scores for individual publications were predicted from a series of variables:
            - Number of times cited
            - Number of authors on the paper
            - Publication type
            - Days that elapsed between publication and access
            - Recent citations
            - Open access status
            - Number of collaborating organizations on the publication
            - Field citation ratio
            - Citation count
            - Relative citation ratio
            - Number of associated datasets
            - Number of associated grants
            - Number of associated patents
            - Number of clinical trials associated
            - Number of unique cities, countries, and states represented by the authors

            An XGBoost model was initialized with decision trees robust to missing values and mixed tabular data. A Bayesian optimization approach was employed to optimize hyperparameters. The following search space was defined for the optimization process:
            - **n_estimators**: Integer values ranging from 50 to 5000, controlling the number of boosting rounds.
            - **max_depth**: Integer values from 3 to 15, specifying the maximum depth of each tree.
            - **learning_rate**: A real-valued parameter with a range of 0.01 to 0.2 on a logarithmic scale, controlling the rate at which the model learns.
            - **subsample**: Real values between 0.5 and 1.0, dictating the fraction of samples used per tree, thereby controlling overfitting.

            To evaluate model performance, nested cross-validation was implemented with two 4-fold CV.
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
        # Scatter plot
        dcc.Graph(figure=fig),
        # Stats text
        html.Div(stats_text, style={'margin-top': '20px', 'font-weight': 'bold'})
    ])

    return layout

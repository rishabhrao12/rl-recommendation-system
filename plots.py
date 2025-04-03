# streamlit_rl_news_app/plots.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_interactive_user_insights(log_path="interaction_log.csv", force_reload=True):
    # Reload CSV fresh every time
    if force_reload or not os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
        except Exception:
            return go.Figure()  # Return empty if log is missing or corrupted
    else:
        log_df = pd.read_csv(log_path)

    if log_df.empty:
        return go.Figure()

    # Prepare data
    action_counts = log_df['Action'].value_counts()
    category_rewards = log_df.groupby('Category')['Reward'].sum().sort_values(ascending=False)
    liked_df = log_df[log_df['Action'] == 'like']
    liked_counts = liked_df['Category'].value_counts()
    log_df['Cumulative Reward'] = log_df['Reward'].cumsum()
    top_liked = liked_df['Title'].value_counts().head(5)

    # Subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "User Actions Distribution",
            "Total Reward per Category",
            "Articles Liked per Category",
            "Cumulative Reward Over Time",
            ""
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # Add plots
    fig.add_trace(go.Bar(x=action_counts.index, y=action_counts.values, marker_color='skyblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=category_rewards.index, y=category_rewards.values, marker_color='lightgreen'), row=1, col=2)
    fig.add_trace(go.Bar(x=liked_counts.index, y=liked_counts.values, marker_color='coral'), row=2, col=1)
    fig.add_trace(go.Scatter(y=log_df['Cumulative Reward'], mode='lines+markers'), row=2, col=2)

    fig.update_layout(height=900, width=1000, title_text="📊 User Recommendation Insights", showlegend=False)
    return fig

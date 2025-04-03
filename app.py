# streamlit_rl_news_app/app.py

import streamlit as st
from utils import load_data, display_article, get_action_buttons
from rl_agent import RLAgent
from plots import create_interactive_user_insights

# Load data and embeddings
df, embeddings = load_data()

# Session state management
if "agent" not in st.session_state:
    st.session_state.agent = RLAgent()
if "current_index" not in st.session_state:
    st.session_state.current_index = st.session_state.agent.recommend_article()
if "selected_action" not in st.session_state:
    st.session_state.selected_action = None

agent = st.session_state.agent

# Streamlit layout
st.set_page_config(page_title="News Recommender", layout="wide")

page = st.sidebar.selectbox("Navigate", ["📄 Article Feed", "📊 User Insights"])

if page == "📄 Article Feed":
    st.title("📰 Personalized News Recommender")

    idx = st.session_state.current_index
    article = df.iloc[idx]
    display_article(article)

    st.markdown("---")
    st.subheader("How do you feel about this article?")

    selected = get_action_buttons()
    if selected:
        st.session_state.selected_action = selected

    if st.session_state.selected_action:
        if st.button("➡️ Next Article"):
            agent.update(idx, st.session_state.selected_action)
            next_idx = agent.recommend_article()
            st.session_state.current_index = next_idx
            st.session_state.selected_action = None
            st.rerun()

elif page == "📊 User Insights":
    st.title("📈 Recommendation Statistics")
    fig = create_interactive_user_insights(force_reload=True)
    st.plotly_chart(fig, use_container_width=True)

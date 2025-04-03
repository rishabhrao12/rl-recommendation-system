import streamlit as st
import pandas as pd
import numpy as np

def load_data(df_path="news_subset_2k.csv", emb_path="embeddings_2k.npy"):
    df = pd.read_csv(df_path)
    embeddings = np.load(emb_path)
    return df, embeddings

def display_article(article):
    st.subheader(article['title'])
    st.markdown(f"<div style='color: gray; font-size: 0.9em;'>Category: {article['category']}</div>", unsafe_allow_html=True)
    
    st.text_area("Article Content", article['body'], height=250, disabled=True)

def get_action_buttons():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("👍 Like"):
            return "like"
    with col2:
        if st.button("🔄 Share"):
            return "share"
    with col3:
        if st.button("📖 Read"):
            return "read"
    with col4:
        if st.button("🚫 Not Interested"):
            return "not_interested"
    return None

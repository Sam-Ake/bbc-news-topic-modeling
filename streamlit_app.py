import streamlit as st
import requests
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="BBC News Topic Modeling & Analytics", page_icon="📰", layout="wide")

st.title("📰 BBC News Topic Modeling & Summarization Dashboard")
st.markdown("Explore latent topics across the BBC News dataset, visualize clusters, and analyze custom news text in real time.")

# Sidebar navigation
option = st.sidebar.selectbox("Navigation", ["Text Predictor", "Interactive LDA Visualizer"])

if option == "Text Predictor":
    st.header("🔍 Real-Time Topic Prediction")
    user_input = st.text_area("Paste a news article or paragraph below:", height=200, placeholder="Enter news text here...")
    
    if st.button("Analyze Topic"):
        if user_input.strip():
            with st.spinner("Analyzing text..."):
                # Call FastAPI backend or evaluate locally if API isn't running
                try:
                    import joblib
                    import numpy as np
                    
                    # Fallback direct evaluation if backend isn't up
                    lda = joblib.load('lda_model.pkl')
                    vectorizer = joblib.load('vectorizer.pkl')
                    
                    transformed = vectorizer.transform([user_input])
                    dist = lda.transform(transformed)[0]
                    dom_topic = int(np.argmax(dist)) + 1
                    
                    st.success(f"**Dominant Topic:** Topic #{dom_topic} (Confidence: {round(dist[dom_topic-1]*100, 2)}%)")
                    
                    st.subheader("Topic Distribution Breakdown")
                    st.bar_chart({f"Topic {i+1}": float(score) for i, score in enumerate(dist)})
                    
                except Exception as err:
                    st.error(f"Error processing text: {err}")
        else:
            st.warning("Please enter some text to analyze.")

elif option == "Interactive LDA Visualizer":
    st.header("📊 Interactive pyLDAvis Topic Clusters")
    html_path = 'lda_visualization.html'
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.warning("pyLDAvis HTML file not found. Please run your Jupyter notebook cell to generate it.")
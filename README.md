# 📰 BBC News Topic Modeling & Summarization App

An end-to-end Machine Learning and Data Engineering project that extracts latent semantic topics from the BBC News dataset using Scikit-learn's Latent Dirichlet Allocation (LDA), interactive pyLDAvis visualizations, a FastAPI backend, and a Streamlit frontend dashboard.

---

## 🚀 Project Overview

This project processes **2,255 BBC articles** categorized across five domains (business, entertainment, politics, sport, and tech). It provides:
1. **Exploratory Data Analysis (EDA)**: Text length distributions, keyword unigrams, and bigram frequency charts.
2. **Unsupervised Topic Modeling**: A trained LDA model identifying 5 latent thematic clusters.
3. **Interactive Visualizations**: An embedded `pyLDAvis` dashboard displaying inter-topic distance maps and salient keyword weights.
4. **FastAPI Backend**: REST API endpoints for real-time topic prediction and confidence scoring.
5. **Streamlit Dashboard**: A user-friendly web interface allowing users to paste custom news text and analyze its topic breakdown instantly.

---

## 🛠️ Tech Stack

* **Language**: Python 3.14
* **Machine Learning & NLP**: Scikit-learn (LDA, CountVectorizer), Joblib
* **Visualization**: pyLDAvis, Matplotlib, Seaborn
* **Backend**: FastAPI, Uvicorn
* **Frontend**: Streamlit, Streamlit Components
* **Version Control**: Git, GitHub

---

## 📂 Project Structure

```text
├── BBC News Summary/             # Raw dataset folder
├── app.py                        # FastAPI backend service
├── streamlit_app.py              # Streamlit dashboard frontend
├── eda_and_preprocessing.ipynb   # Jupyter Notebook for EDA & model training
├── lda_model.pkl                 # Serialized scikit-learn LDA model
├── vectorizer.pkl                # Serialized CountVectorizer
├── lda_visualization.html        # Interactive pyLDAvis HTML dashboard
├── requirements.txt              # Project dependencies
└── .gitignore                    # Git ignore file
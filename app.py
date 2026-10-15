from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="BBC News Topic Modeling & Summarization API",
    description="API for extracting latent topics and summaries from news text using scikit-learn LDA.",
    version="1.0.0"
)

# Load serialized model components
try:
    lda_model = joblib.load('lda_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"Warning: Model files not found. Ensure eda_and_preprocessing.ipynb has been run. Error: {e}")

class ArticleRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the BBC News Topic Modeling & Summarization API!"}

@app.post("/predict-topic")
def predict_topic(request: ArticleRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    # Transform input text using the fitted vectorizer
    transformed_text = vectorizer.transform([request.text])
    topic_distribution = lda_model.transform(transformed_text)[0]
    
    # Identify dominant topic
    dominant_topic_id = int(np.argmax(topic_distribution))
    confidence = float(topic_distribution[dominant_topic_id])
    
    # Extract top keywords for the dominant topic
    feature_names = vectorizer.get_feature_names_out()
    topic_components = lda_model.components_[dominant_topic_id]
    top_keywords = [feature_names[i] for i in topic_components.argsort()[:-6:-1]]
    
    return {
        "dominant_topic": dominant_topic_id + 1,
        "confidence": round(confidence * 100, 2),
        "topic_distribution": {f"Topic {i+1}": round(float(score)*100, 2) for i, score in enumerate(topic_distribution)},
        "top_keywords": top_keywords
    }
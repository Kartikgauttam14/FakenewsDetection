from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from model import FakeNewsClassifier
from explainer import FakeNewsExplainer
from utils import preprocess_text, extract_article_from_url
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="API for detecting fake news using BERT/RoBERTa models with explainable AI",
    version="1.0.0"
)

# Request/Response models
class NewsArticle(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    
class ExplanationResponse(BaseModel):
    prediction: str
    confidence: float
    explanations: dict

# Global variables
classifier = None
explainer = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global classifier, explainer
    
    logger.info("Loading fake news detection model...")
    config = Config()
    
    try:
        classifier = FakeNewsClassifier(
            model_path=config.MODEL_PATH,
            model_name=config.MODEL_NAME
        )
    except:
        classifier = FakeNewsClassifier(model_name=config.MODEL_NAME)
    
    explainer = FakeNewsExplainer(classifier, classifier.tokenizer)
    logger.info("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fake News Detection API",
        "endpoints": {
            "/predict": "Get prediction for news article",
            "/explain": "Get prediction with explanation",
            "/health": "Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(article: NewsArticle):
    """Predict if news is fake or real"""
    
    # Extract text
    if article.url:
        text = extract_article_from_url(article.url)
        if "Error" in text:
            raise HTTPException(status_code=400, detail=text)
    elif article.text:
        text = article.text
    else:
        raise HTTPException(status_code=400, detail="Please provide either text or URL")
    
    # Preprocess and predict
    processed_text = preprocess_text(text)
    
    if len(processed_text.split()) < 10:
        raise HTTPException(status_code=400, detail="Text too short for analysis")
    
    try:
        result = classifier.predict(processed_text)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/explain", response_model=ExplanationResponse)
async def explain(article: NewsArticle, method: str = "lime"):
    """Get prediction with explanation"""
    
    # Extract text
    if article.url:
        text = extract_article_from_url(article.url)
        if "Error" in text:
            raise HTTPException(status_code=400, detail=text)
    elif article.text:
        text = article.text
    else:
        raise HTTPException(status_code=400, detail="Please provide either text or URL")
    
    # Preprocess
    processed_text = preprocess_text(text)
    
    if len(processed_text.split()) < 10:
        raise HTTPException(status_code=400, detail="Text too short for analysis")
    
    try:
        # Get prediction
        result = classifier.predict(processed_text)
        
        # Get explanation
        if method.lower() == "lime":
            explanation = explainer.explain_with_lime(processed_text, num_features=10)
            explanations = {
                "method": "LIME",
                "important_features": explanation['feature_importance']
            }
        elif method.lower() == "attention":
            explanation = explainer.explain_with_shap(processed_text, num_features=10)
            explanations = {
                "method": "Attention",
                "important_words": explanation['important_words']
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid explanation method. Use 'lime' or 'attention'")
        
        return ExplanationResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            explanations=explanations
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Explanation generation failed")

@app.post("/batch_predict")
async def batch_predict(articles: List[NewsArticle]):
    """Batch prediction for multiple articles"""
    
    results = []
    
    for i, article in enumerate(articles):
        try:
            # Extract text
            if article.url:
                text = extract_article_from_url(article.url)
            elif article.text:
                text = article.text
            else:
                results.append({"error": "No text or URL provided", "index": i})
                continue
            
            # Predict
            processed_text = preprocess_text(text)
            result = classifier.predict(processed_text)
            result['index'] = i
            results.append(result)
            
        except Exception as e:
            results.append({"error": str(e), "index": i})
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
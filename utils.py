import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import re

def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_article_from_url(url: str) -> str:
    """Extract article text from URL"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return preprocess_text(text)
    
    except Exception as e:
        return f"Error extracting article: {str(e)}"

def create_evaluation_report(predictions: List[int], labels: List[int], 
                           class_names: List[str] = ['Real', 'Fake']) -> Dict:
    """Create comprehensive evaluation report"""
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot metrics
    metrics = ['precision', 'recall', 'f1-score']
    real_scores = [report['Real'][m] for m in metrics]
    fake_scores = [report['Fake'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, real_scores, width, label='Real')
    ax2.bar(x + width/2, fake_scores, width, label='Fake')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Classification Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    return {
        'report': report,
        'confusion_matrix': cm,
        'figure': fig
    }

def load_sample_data() -> pd.DataFrame:
    """Load sample fake news data for testing"""
    # Sample data for demonstration
    data = {
        'text': [
            "Scientists discover new planet in solar system with potential for life",
            "BREAKING: Celebrity secretly replaced by clone, sources confirm",
            "New study shows benefits of regular exercise for mental health",
            "Aliens landed in my backyard last night and told me the truth about everything",
            "Government announces new infrastructure spending bill",
            "Secret society controls all world governments through mind control"
        ],
        'label': [0, 1, 0, 1, 0, 1],  # 0: Real, 1: Fake
        'source': ['Science Daily', 'Conspiracy Blog', 'Health Journal', 
                  'Anonymous Blog', 'Reuters', 'Conspiracy Forum']
    }
    
    return pd.DataFrame(data)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def download_and_convert_model():
    """Download a pre-trained model from Hugging Face"""
    
    os.makedirs('saved_models', exist_ok=True)
    
    # You can use a pre-trained fake news detection model from Hugging Face
    # For example: "hamzab/roberta-fake-news-classification"
    model_name = "hamzab/roberta-fake-news-classification"
    
    try:
        # Download model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': model.config
        }
        
        torch.save(checkpoint, 'saved_models/fake_news_detector.pt')
        print(f"Model downloaded and saved to saved_models/fake_news_detector.pt")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Creating dummy model instead...")
        create_dummy_model()

if __name__ == "__main__":
    download_and_convert_model()
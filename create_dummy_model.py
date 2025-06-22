import torch
import os
from model import FakeNewsDetector
from config import Config

def create_dummy_model():
    """Create a dummy model file for testing"""
    
    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Initialize model
    config = Config()
    model = FakeNewsDetector(config.MODEL_NAME)
    
    # Create dummy checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 0,
        'loss': 0.5,
        'accuracy': 0.5
    }
    
    # Save model
    torch.save(checkpoint, 'saved_models/fake_news_detector.pt')
    print("Dummy model saved to saved_models/fake_news_detector.pt")

if __name__ == "__main__":
    create_dummy_model()
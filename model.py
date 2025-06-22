import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import numpy as np

class FakeNewsDetector(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.3):
        super(FakeNewsDetector, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool the outputs
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        
        # Classify
        logits = self.classifier(pooled_output)
        
        return logits

class FakeNewsClassifier:
    def __init__(self, model_path: str = None, model_name: str = 'roberta-base'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FakeNewsDetector(model_name)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, path: str):
        """Load trained model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def predict(self, text: str) -> Dict:
        """Predict if news is fake or real"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        
        # Convert to numpy
        probs = probabilities.cpu().numpy()[0]
        pred_label = prediction.cpu().numpy()[0]
        
        return {
            'prediction': 'Fake' if pred_label == 1 else 'Real',
            'confidence': float(probs[pred_label]),
            'probabilities': {
                'Real': float(probs[0]),
                'Fake': float(probs[1])
            }
        }
import shap
import lime
import lime.lime_text
import torch
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

class FakeNewsExplainer:
    def __init__(self, classifier, tokenizer):
        self.classifier = classifier
        self.tokenizer = tokenizer
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=['Real', 'Fake']
        )
        
    def predict_proba_for_lime(self, texts: List[str]) -> np.ndarray:
        """Prediction function for LIME"""
        probs_list = []
        
        for text in texts:
            result = self.classifier.predict(text)
            probs = [result['probabilities']['Real'], result['probabilities']['Fake']]
            probs_list.append(probs)
        
        return np.array(probs_list)
    
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict:
        """Generate LIME explanation for a single text"""
        # Get prediction
        prediction = self.classifier.predict(text)
        
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            text,
            self.predict_proba_for_lime,
            num_features=num_features,
            num_samples=1000
        )
        
        # Extract explanation
        explanation = {
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'feature_importance': exp.as_list(),
            'lime_explanation': exp
        }
        
        return explanation
    
    def explain_with_shap(self, text: str, num_features: int = 10) -> Dict:
        """Generate SHAP explanation using transformer interpretability"""
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model predictions and attention weights
        self.classifier.model.eval()
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(self.classifier.device)
            attention_mask = inputs['attention_mask'].to(self.classifier.device)
            
            # Get transformer outputs with attention weights
            transformer_outputs = self.classifier.model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Get final prediction
            pooled_output = transformer_outputs.last_hidden_state.mean(dim=1)
            logits = self.classifier.model.classifier(pooled_output)
            probs = torch.softmax(logits, dim=-1)
        
        # Extract attention weights from last layer
        attention_weights = transformer_outputs.attentions[-1]
        # Average over heads and layers
        avg_attention = attention_weights.mean(dim=1).squeeze(0)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        # Calculate token importance based on attention
        token_importance = avg_attention.sum(dim=0).cpu().numpy()
        
        # Create word-level importance by aggregating subword tokens
        word_importance = self._aggregate_token_importance(tokens, token_importance)
        
        # Sort by importance
        sorted_importance = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:num_features]
        
        return {
            'prediction': 'Fake' if probs[0][1] > probs[0][0] else 'Real',
            'confidence': float(probs.max()),
            'important_words': sorted_importance,
            'attention_weights': avg_attention.cpu().numpy()
        }
    
    def _aggregate_token_importance(self, tokens: List[str], importance: np.ndarray) -> Dict[str, float]:
        """Aggregate subword token importance to word level"""
        word_importance = {}
        current_word = ""
        current_importance = 0
        
        for token, imp in zip(tokens, importance):
            if token.startswith("##"):  # BERT subword token
                current_word += token[2:]
                current_importance += imp
            elif token.startswith("Ġ"):  # RoBERTa subword token
                if current_word:
                    word_importance[current_word] = current_importance
                current_word = token[1:]
                current_importance = imp
            else:
                if current_word:
                    word_importance[current_word] = current_importance
                current_word = token
                current_importance = imp
        
        if current_word:
            word_importance[current_word] = current_importance
        
        return word_importance
    
    def visualize_lime_explanation(self, explanation: Dict) -> plt.Figure:
        """Visualize LIME explanation"""
        fig = explanation['lime_explanation'].as_pyplot_figure()
        return fig
    
    def visualize_attention_heatmap(self, text: str, explanation: Dict) -> plt.Figure:
        """Visualize attention weights as heatmap"""
        tokens = text.split()[:50]  # Limit to first 50 tokens for visualization
        attention = explanation['attention_weights'][:len(tokens), :len(tokens)]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(attention, cmap='Blues', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        
        # Add colorbar
        plt.colorbar(im)
        
        ax.set_title(f"Attention Heatmap - Prediction: {explanation['prediction']}")
        plt.tight_layout()
        
        return fig
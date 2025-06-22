import torch
from model import FakeNewsDetector
from transformers import AutoTokenizer
import pickle
import os

def save_model_for_deployment(model_path: str, output_dir: str):
    """Prepare model for deployment"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = FakeNewsDetector('roberta-base')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Save model in ONNX format for better compatibility
    dummy_input = torch.randint(0, 1000, (1, 512))
    dummy_mask = torch.ones(1, 512)
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        os.path.join(output_dir, 'model.onnx'),
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved for deployment in {output_dir}")

if __name__ == "__main__":
    save_model_for_deployment('./saved_models/fake_news_detector.pt', './deployment')
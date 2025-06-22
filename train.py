import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config
from model import FakeNewsDetector

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(train_dataloader, val_dataloader, model, device, config):
    """Train the fake news detection model"""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Scheduler
    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}')
        
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_dataloader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
    return model

def prepare_data(df):
    """Prepare data for training"""
    # Assuming df has 'text' and 'label' columns
    # where label is 0 for real and 1 for fake
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    config = Config()
    
    # Load your dataset
    # df = pd.read_csv('path_to_your_dataset.csv')
    # X_train, X_val, y_train, y_val = prepare_data(df)
    
    # For demo purposes, using dummy data
    X_train = ["Real news article"] * 100 + ["Fake news article"] * 100
    y_train = [0] * 100 + [1] * 100
    X_val = ["Real news article"] * 20 + ["Fake news article"] * 20
    y_val = [0] * 20 + [1] * 20
    
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_dataset = NewsDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
    val_dataset = NewsDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    model = FakeNewsDetector(config.MODEL_NAME)
    model.to(config.DEVICE)
    
    # Train model
    trained_model = train_model(train_dataloader, val_dataloader, model, config.DEVICE, config)
    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config
    }, config.MODEL_PATH)
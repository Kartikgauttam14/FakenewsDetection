import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import requests
import zipfile
import os

def download_fake_news_dataset():
    """Download popular fake news datasets"""
    
    datasets = {
        'fake_news_detection': 'https://www.kaggle.com/c/fake-news/download/train.csv',
        'liar_dataset': 'https://www.cs.ucsb.edu/~william/data/liar_dataset.zip'
    }
    
    os.makedirs('data', exist_ok=True)
    
    # Note: You'll need to manually download from Kaggle or use Kaggle API
    print("Please download datasets manually from:")
    for name, url in datasets.items():
        print(f"{name}: {url}")

def prepare_combined_dataset(data_path: str) -> pd.DataFrame:
    """Combine and prepare multiple fake news datasets"""
    
    all_data = []
    
    # Load different dataset formats
    # Example for a typical fake news dataset
    if os.path.exists(os.path.join(data_path, 'train.csv')):
        df1 = pd.read_csv(os.path.join(data_path, 'train.csv'))
        if 'title' in df1.columns and 'text' in df1.columns:
            df1['combined_text'] = df1['title'].fillna('') + ' ' + df1['text'].fillna('')
            df1['label'] = df1['label'].map({0: 0, 1: 1})  # 0: reliable, 1: unreliable
            all_data.append(df1[['combined_text', 'label']].rename(columns={'combined_text': 'text'}))
    
    # Combine all datasets
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'])
        
        # Remove empty texts
        combined_df = combined_df[combined_df['text'].str.len() > 50]
        
        # Balance dataset
        combined_df = balance_dataset(combined_df)
        
        return combined_df
    else:
        # Return sample data if no dataset available
        return create_sample_dataset()

def balance_dataset(df: pd.DataFrame, target_col: str = 'label') -> pd.DataFrame:
    """Balance the dataset to have equal number of real and fake news"""
    
    # Separate classes
    df_majority = df[df[target_col] == df[target_col].value_counts().idxmax()]
    df_minority = df[df[target_col] != df[target_col].value_counts().idxmax()]
    
    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    
    # Combine
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing"""
    
    real_news_samples = [
        "Scientists at NASA have discovered a new exoplanet in the habitable zone of a distant star system. The planet, designated Kepler-452b, shows promising signs for potential life.",
        "The World Health Organization announced new guidelines for COVID-19 vaccination boosters, recommending additional doses for high-risk populations.",
        "Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue growth.",
        "Climate scientists warn that global temperatures are rising faster than previously predicted, urging immediate action on carbon emissions.",
        "Researchers at MIT have developed a new artificial intelligence system capable of detecting cancer cells with 95% accuracy.",
    ]
    
    fake_news_samples = [
        "BREAKING: Government admits to hiding alien technology for decades! Secret documents reveal shocking truth about UFO encounters.",
        "Miracle cure discovered! This one simple trick doctors don't want you to know cures all diseases instantly.",
        "Scientists confirm the Earth is actually flat! NASA exposed in massive conspiracy to hide the truth from the public.",
        "5G towers are mind control devices designed by the government to control the population through radio waves.",
        "Celebrity found to be reptilian shapeshifter! Shocking video evidence reveals the truth about Hollywood elite.",
    ]
    
    # Create more samples by adding variations
    all_real = real_news_samples * 20
    all_fake = fake_news_samples * 20
    
    df = pd.DataFrame({
        'text': all_real + all_fake,
        'label': [0] * len(all_real) + [1] * len(all_fake)
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def create_data_splits(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    """Create train, validation, and test splits"""
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['text'], 
        df['label'], 
        test_size=test_size, 
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, 
        y_temp, 
        test_size=val_size_adjusted, 
        random_state=42,
        stratify=y_temp
    )
    
    # Save splits
    splits = {
        'train': pd.DataFrame({'text': X_train, 'label': y_train}),
        'val': pd.DataFrame({'text': X_val, 'label': y_val}),
        'test': pd.DataFrame({'text': X_test, 'label': y_test})
    }
    
    for split_name, split_df in splits.items():
        split_df.to_csv(f'data/{split_name}.csv', index=False)
        print(f"{split_name} set: {len(split_df)} samples")
    
    return splits

if __name__ == "__main__":
    # Prepare dataset
    print("Preparing fake news dataset...")
    
    # Try to load existing dataset or create sample
    df = prepare_combined_dataset('./data')
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Create splits
    splits = create_data_splits(df)
    
    print("\nDataset preparation complete!")
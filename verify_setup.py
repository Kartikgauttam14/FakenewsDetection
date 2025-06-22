import os
import pandas as pd

def verify_setup():
    """Verify that all required files and directories are in place"""
    
    print("Verifying Fake News Detector Setup...")
    print("=" * 50)
    
    # Check directories
    directories = ['data', 'saved_models']
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ Directory '{directory}' exists")
        else:
            print(f"✗ Directory '{directory}' is missing")
            os.makedirs(directory, exist_ok=True)
            print(f"  → Created '{directory}' directory")
    
    # Check dataset files
    dataset_files = ['data/train.csv', 'data/val.csv', 'data/test.csv']
    for file in dataset_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"✓ File '{file}' exists ({len(df)} samples)")
            except Exception as e:
                print(f"✗ Error reading '{file}': {e}")
        else:
            print(f"✗ File '{file}' is missing")
    
    # Check model file
    model_file = 'saved_models/fake_news_detector.pt'
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✓ Model file exists ({size_mb:.2f} MB)")
    else:
        print(f"✗ Model file is missing")
        print("  → Run 'python create_dummy_model.py' to create a dummy model")
        print("  → Or run 'python train.py' to train a real model")
    
    # Check Python files
    python_files = [
        'app.py', 'model.py', 'train.py', 'explainer.py', 
        'utils.py', 'config.py', 'api.py'
    ]
    
    print("\nPython Files:")
    for file in python_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} is missing")
    
    print("\n" + "=" * 50)
    print("Setup verification complete!")

if __name__ == "__main__":
    verify_setup()
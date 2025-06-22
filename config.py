import torch

class Config:
    # Model settings
    MODEL_NAME = 'roberta-base'  # Can also use 'bert-base-uncased'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    
    # Paths
    MODEL_PATH = './saved_models/fake_news_detector.pt'
    DATA_PATH = './data/'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Labels
    LABELS = ['Real', 'Fake']
    LABEL2ID = {'Real': 0, 'Fake': 1}
    ID2LABEL = {0: 'Real', 1: 'Fake'}
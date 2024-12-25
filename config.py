# config.py
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 10
INPUT_SIZE = 32
SEED = 42
PRINT_FREQ = 100
OPTIMIZER = "sgd"
NUM_WORKERS=20
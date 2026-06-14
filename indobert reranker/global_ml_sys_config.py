# konfigurasi central ke system

import torch

MODEL_NAME = "indobenchmark/indobert-base-p1"

MAX_LEN = 256

HIDDEN_DIM = 256

VECTOR_DIM = 128

# parameter training
DROPOUT = 0.1

LR = 2e-5

BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15

WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "train_checkpoints"

import torch
from model import GestureModelConfig

GESTURE_MAP = {
    # Static Gestures
    "ONE": 0, "TWO": 1, "THREE": 2,"FOUR": 3,"OK": 4,"MENU": 5,
    # Dynamic Gestures
    "LEFT": 6, "RIGHT": 7, "CIRCLE": 8, "V": 9, "CROSS": 10,
    # Fine-grained dynamic Gestures
    "GRAB": 11,"PINCH": 12,        
    # Dynamic-periodic Gestures
    "DENY": 13, "WAVE": 14, "KNOB": 15,

    "NO_GESTURE": -1 
}

TRAIN_DATA_DIR = "data/training_set_csv"
TRAIN_ANNOTATIONS_FILE = "data/training_set/annotations.txt"

TEST_DATA_DIR = "data/test_set_csv"
TEST_ANNOTATIONS_FILE = "data/test_set/test_annotations_fixed.txt"

STRIDE = 12
BATCH_SIZE = 128
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 128
num_epochs = 20

modelConfig = GestureModelConfig(
    num_gestures=len(GESTURE_MAP),
    device=DEVICE.type,
    n_embd=256,
    seq_len=seq_len,
    n_heads=4,
    n_layer=2,
    dout=0.3,
)
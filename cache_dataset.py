import torch
import time
from dataset import GestureDataset, CachedDataset
from config import *

start_total = time.perf_counter()

# -------- Train --------
start = time.perf_counter()

dataset = GestureDataset(TRAIN_DATA_DIR, TRAIN_ANNOTATIONS_FILE, seq_len=seq_len, stride=STRIDE)
dataset = CachedDataset(dataset)
cached_data = [dataset[i] for i in range(len(dataset))]
torch.save(cached_data, "train_dataset.pt")

print(f"Saved train dataset to train_dataset.pt")
print(f"Train caching took: {time.perf_counter() - start:.2f} seconds")

# -------- Test --------
start = time.perf_counter()

dataset = GestureDataset(TEST_DATA_DIR, TEST_ANNOTATIONS_FILE, seq_len=seq_len, stride=STRIDE)
dataset = CachedDataset(dataset)
cached_data = [dataset[i] for i in range(len(dataset))]
torch.save(cached_data, "test_dataset.pt")

print(f"Saved test dataset to test_dataset.pt")
print(f"Test caching took: {time.perf_counter() - start:.2f} seconds")

print(f"Total time: {time.perf_counter() - start_total:.2f} seconds")
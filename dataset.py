import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from meth import *
from config import GESTURE_MAP

class GestureDataset(Dataset):
    def __init__(self, in_path: str, annotations_file: str, seq_len=128, stride=16):
        self.seq_len = seq_len
        self.stride = stride
        self.index = []
        self.cache_file = None
        self.cache_data = None
        self.annotations = {}

        # Load annotations
        with open(annotations_file) as f:
            for line in f:
                parts = line.strip().split(";")
                file_id = parts[0]
                gestures = [(parts[i], int(parts[i+1]), int(parts[i+2]))
                            for i in range(1, len(parts), 3) if i+2 < len(parts)]
                self.annotations[file_id] = gestures

        # Collect CSV files and build sequence indices
        txt_files = sorted([f for f in Path(in_path).glob("*.csv") if f.name[0].isdigit()],
                           key=lambda x: int(x.stem))
        for txt_file in txt_files:
            total_frames = sum(1 for _ in open(txt_file))
            if total_frames < seq_len:
                continue
            self.index.extend((txt_file, start) for start in range(0, total_frames - seq_len + 1, stride))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        txt_file, seq_start = self.index[idx]

        # Load cache if needed
        if self.cache_file != txt_file:
            self.cache_data = torch.from_numpy(
                pd.read_csv(txt_file, header=None).values
            ).float()  # [total_frames, C]
            self.cache_file = txt_file

        seq_end = seq_start + self.seq_len
        data = self.cache_data[seq_start:seq_end]
        frame_indices = data[:, 1]  # [T, 1]
        timestamps = data[:, 2]     # [T, 1]
        raw_sequence = data[:, 2:]  # [T, C-2]

        # Compute multi-view sequences
        sequence_views = {
            "raw": raw_sequence.clone(),                # [T, C]
            "jcd": compute_jcd(raw_sequence),           # [T, C]
            "slow_diff": compute_td(raw_sequence, 7),   # [T, C]
            "fast_diff": compute_td(raw_sequence, 3)    # [T, C]
        }
        sequence_views["fast_acc"] = compute_acc(sequence_views["fast_diff"])
        sequence_views["slow_acc"] = compute_acc(sequence_views["slow_diff"])
        

        # initialize labels
        gesture_labels = torch.full((self.seq_len,), GESTURE_MAP["NO_GESTURE"], dtype=torch.long)
        start_labels   = torch.zeros(self.seq_len, dtype=torch.float32)
        end_labels     = torch.zeros(self.seq_len, dtype=torch.float32)

        # fill labels
        for g_name, g_start, g_end in self.annotations[txt_file.stem]:
            if g_end < seq_start or g_start >= seq_end:
                continue

            gesture = GESTURE_MAP[g_name]
            local_start = max(g_start, seq_start) - seq_start
            local_end   = min(g_end, seq_end - 1) - seq_start
            gesture_labels[local_start:local_end+1] = gesture

            if seq_start <= g_start < seq_end:
                start_labels[g_start - seq_start] = 1.0
            if seq_start <= g_end < seq_end:
                end_labels[g_end - seq_start] = 1.0

        return sequence_views, gesture_labels, start_labels, end_labels
    
class CachedDataset(Dataset):
    def __init__(self, orig_dataset):
        self.orig_dataset = orig_dataset
        self.cache = []

        print("Caching dataset into RAM...")
        for i in range(len(orig_dataset)):
            self.cache.append(orig_dataset[i])
        print(f"Cached {len(self.cache)} items.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]
    
class RAMDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file, weights_only=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
import matplotlib.pyplot as plt
def plot_sequence(gesture_labels, start_labels, end_labels, seq_idx=0):
    """
    gesture_labels: [seq_len] or [B, seq_len]
    start_labels, end_labels: [seq_len] or [B, seq_len]
    seq_idx: which sequence in batch to plot
    """
    if gesture_labels.ndim == 2:  # batch dimension
        gesture_labels = gesture_labels[seq_idx].cpu().numpy()
        start_labels   = start_labels[seq_idx].cpu().numpy()
        end_labels     = end_labels[seq_idx].cpu().numpy()
    else:
        gesture_labels = gesture_labels.cpu().numpy()
        start_labels   = start_labels.cpu().numpy()
        end_labels     = end_labels.cpu().numpy()

    seq_len = len(gesture_labels)
    frames = range(seq_len)

    plt.figure(figsize=(12,3))
    plt.plot(frames, gesture_labels, drawstyle='steps-post', label='Gesture class', color='blue')
    plt.scatter(frames, start_labels*gesture_labels.max(), color='green', label='Start', marker='^', s=80)
    plt.scatter(frames, end_labels*gesture_labels.max(), color='red', label='End', marker='v', s=80)
    plt.xlabel("Frame")
    plt.ylabel("Class / Start-End")
    plt.title(f"Sequence {seq_idx}")
    plt.legend()
    plt.show()


def main():
    dataset = GestureDataset("data/training_set_csv", "data/training_set/annotations.txt")

    """for i in range(10):
        seq, gesture_labels, start_labels, end_labels = dataset[i]
        txt_file, seq_start = dataset.index[i]
        print(f"Sequence {i}: file={txt_file.stem}, seq_start={seq_start}")
        print("Gesture labels:", gesture_labels)
        print("Start labels:", start_labels)
        print("End labels:", end_labels)
        plot_sequence(gesture_labels, start_labels, end_labels, seq_idx=0)"""

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for sequences, gesture_labels, start_labels, end_labels in loader:
        sequences = {k: v.to(device) for k, v in sequences.items()}
        gesture_labels = gesture_labels.to(device)
        start_labels = start_labels.to(device)
        end_labels = end_labels.to(device)
  

if __name__ == "__main__":
    main()
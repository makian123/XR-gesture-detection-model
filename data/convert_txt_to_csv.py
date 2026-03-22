# convert_txt_to_csv.py
import os
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = "data/test_set"
OUTPUT_DIR = "data/test_set_csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
txt_files = sorted(
    [f for f in Path(INPUT_DIR).glob("*.txt") if f.name[0].isdigit()],
    key=lambda x: int(x.stem)
)

for txt_file in txt_files:
    print(f"Processing {txt_file.name}...")
    df = pd.read_csv(txt_file, sep=";", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    df = df.fillna(0)

    out_csv = Path(OUTPUT_DIR) / (txt_file.stem + ".csv")
    df.to_csv(out_csv, index=False, header=False)

    out_npy = Path(OUTPUT_DIR) / (txt_file.stem + ".npy")
    np.save(out_npy, df.values)
print("done")
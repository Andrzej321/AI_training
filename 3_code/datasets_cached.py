import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VehicleSpeedDatasetLongCached(Dataset):
    """
    Faster replacement for VehicleSpeedDatasetLong:
    - Reads each CSV ONCE in __init__
    - Drops columns ONCE
    - Keeps feature arrays (float32) and targets in memory
    - __getitem__ slices from in-memory arrays (no pandas or file IO per sample)
    """

    def __init__(
        self,
        data_path: str,
        extension: str = "*.csv",
        seq_length: int = 100,
        step_size: int = 5,
        drop_columns: Optional[List[str]] = None,
        target_column: str = "veh_u",
        dtype: np.dtype = np.float32,
    ):
        self.seq_length = int(seq_length)
        self.step_size = int(step_size)
        self.drop_columns = list(drop_columns) if drop_columns else []
        self.target_column = target_column

        self.csv_files = sorted(glob.glob(os.path.join(data_path, extension)))
        if len(self.csv_files) == 0:
            raise RuntimeError(f"No files found in directory '{data_path}' with extension '{extension}'.")

        # Per-file cached arrays
        self.features_per_file: List[np.ndarray] = []
        self.target_per_file: List[np.ndarray] = []
        # Global index: (file_idx, start_idx)
        self.index_map: List[Tuple[int, int]] = []

        # Preload all files once
        for fidx, file in enumerate(self.csv_files):
            df = pd.read_csv(file)

            # Sanity check target column
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in {file}")

            # Drop once
            to_drop = [c for c in self.drop_columns if c in df.columns]
            feats_df = df.drop(columns=to_drop, errors="ignore")

            # Extract features: everything except target column
            # Keep column order stable: drop target afterwards if present
            if self.target_column in feats_df.columns:
                feats_df = feats_df.drop(columns=[self.target_column])

            feats = feats_df.to_numpy(dtype=dtype, copy=False)
            targ = df[self.target_column].to_numpy(dtype=dtype, copy=False)

            # Skip too-short files
            if len(feats) < self.seq_length:
                continue

            # Build windows for this file (only indices; slicing done in __getitem__)
            for start in range(0, len(feats) - self.seq_length + 1, self.step_size):
                self.index_map.append((fidx, start))

            self.features_per_file.append(feats)
            self.target_per_file.append(targ)

        if len(self.index_map) == 0:
            raise RuntimeError(
                f"No valid sequences produced. Check seq_length={self.seq_length} and data in {data_path}"
            )

        # Record input feature size from first file
        self.input_size = self.features_per_file[0].shape[1]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, start = self.index_map[idx]
        feats = self.features_per_file[file_idx]
        targ = self.target_per_file[file_idx]

        end = start + self.seq_length
        window = feats[start:end, :]  # [T, F]
        y = targ[end - 1]             # last step target

        # Convert to torch tensors
        # Model expects batch_first=True => [B, T, F]; DataLoader will add batch dim
        x_tensor = torch.from_numpy(window)              # [T, F], float32
        y_tensor = torch.tensor([y], dtype=x_tensor.dtype)  # [1]

        return x_tensor, y_tensor
from typing import List, Tuple, Dict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GraphDemandDataset(Dataset):
    """
    Spatio-temporal sliding window dataset for multi-horizon forecasting.

    - Nodes are (store_nbr, family) pairs defined by data/processed/node_index.csv
    - Features X: [T, N, F] built from panel.csv
    - Target Y (sales): [T, N]

    Each sample returns:
      x_hist:  [L_enc, N, F]
      y_fut:   [H, N]
      t_end:   pandas.Timestamp (decoder end date, optional for plotting)

    Splits by date using inclusive decoder end date:
      train: dec_end <= train_end
      val:   train_end < dec_end <= val_end
      test:  val_end < dec_end <= test_end
    """
    def __init__(
        self,
        panel_csv: str,
        node_index_csv: str,
        enc_len: int,
        horizon: int,
        split_bounds: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp],
        split: str,
        feature_cols: List[str],
        target_col: str = "sales",
        date_col: str = "date",
    ):
        super().__init__()
        self.enc_len = enc_len
        self.horizon = horizon
        self.split = split
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col

        df = pd.read_csv(panel_csv, parse_dates=[date_col])
        nodes = pd.read_csv(node_index_csv)
        # Build node map
        self.node_df = nodes.copy()
        key = df.merge(nodes, on=["store_nbr", "family"], how="inner")
        # Keep only rows present in node_index mapping
        df = key[["date", "store_nbr", "family"] + self.feature_cols + [self.target_col, "node_id"]].copy()

        # Build continuous date index
        dates = pd.date_range(df[self.date_col].min(), df[self.date_col].max(), freq="D")
        self.dates = dates

        # Pivot per feature into [T, N] and stack into [T, N, F]
        N = len(nodes)
        T = len(dates)
        X_parts = []
        df = df.sort_values([self.date_col, "node_id"])
        for col in self.feature_cols:
            mat = (
                df.pivot_table(index=self.date_col, columns="node_id", values=col, aggfunc="first")
                  .reindex(dates)
                  .sort_index()
                  .to_numpy(dtype=np.float32)
            )
            if mat.shape[1] != N:
                # Missing some nodes for feature; fill to N
                pad = np.zeros((mat.shape[0], N), dtype=np.float32)
                pad[:, :mat.shape[1]] = mat
                mat = pad
            X_parts.append(mat[:, :, None])  # [T, N, 1]
        X = np.concatenate(X_parts, axis=2)  # [T, N, F]

        Y = (
            df.pivot_table(index=self.date_col, columns="node_id", values=self.target_col, aggfunc="first")
              .reindex(dates)
              .sort_index()
              .to_numpy(dtype=np.float32)
        )  # [T, N]

        # Fill NaNs
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)

        self.X = X
        self.Y = Y
        self.N = N
        self.F = X.shape[2]

        # Build indices by split
        train_end, val_end, test_end = split_bounds
        self.index = []
        # anchor t is last encoder index (inclusive)
        for t in range(enc_len - 1, T - horizon - 1):
            dec_end_date = dates[t + horizon]
            if split == "train" and dec_end_date <= train_end:
                self.index.append(t)
            elif split == "val" and (dec_end_date > train_end) and (dec_end_date <= val_end):
                self.index.append(t)
            elif split == "test" and (dec_end_date > val_end) and (dec_end_date <= test_end):
                self.index.append(t)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t = self.index[i]
        x_hist = self.X[t - self.enc_len + 1: t + 1]      # [L_enc, N, F]
        y_fut = self.Y[t + 1: t + 1 + self.horizon]       # [H, N]
        return {
            "x_hist": torch.from_numpy(x_hist),           # [L_enc, N, F]
            "y_fut": torch.from_numpy(y_fut),             # [H, N]
            "dec_end_date": str(self.dates[t + self.horizon].date()),
        }
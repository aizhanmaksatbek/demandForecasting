from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TFTWindowDataset(Dataset):
    """
    Sliding-window dataset for TFT.
    - Works per (store_nbr, family) series.
    - For each anchor t, returns:
      past_inputs:  [L_enc, sum(enc_dims)]   (encoder features)
      future_inputs:[L_dec, sum(dec_dims)]   (decoder known features)
      static_inputs:[sum(static_dims)]       (one-hot concatenation)
      target:       [L_dec]                  (sales for decoder horizon)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        enc_len: int,
        dec_len: int,
        enc_vars: List[str],
        dec_vars: List[str],
        static_cols: List[str],
        split_bounds: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp],
        split: str = "train",
        stride: int = 1,
        store_col: str = "store_nbr",
        fam_col: str = "family",
        date_col: str = "date",
        target_col: str = "sales",
        static_onehot_maps: Dict[str, Dict[Any, np.ndarray]] = None,
    ):
        self.df = df.copy()
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.enc_vars = enc_vars
        self.dec_vars = dec_vars
        self.static_cols = static_cols
        self.store_col = store_col
        self.fam_col = fam_col
        self.date_col = date_col
        self.target_col = target_col
        self.stride = stride
        self.static_onehot_maps = static_onehot_maps or {}

        train_end, val_end, test_end = split_bounds
        assert split in ["train", "val", "test"]
        self.split = split
        self.train_end, self.val_end, self.test_end = train_end, val_end, test_end

        # group arrays
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values([self.store_col, self.fam_col, self.date_col])
        self.groups = []
        self.index = []  # (group_id, anchor_idx)
        for (s, f), g in self.df.groupby([self.store_col, self.fam_col], sort=False):
            g = g.reset_index(drop=True)
            dates = g[self.date_col].values
            n = len(g)
            if n < (enc_len + dec_len + 1):
                continue
            # precompute static vector for group
            static_vec = self._make_static_vector(g.iloc[0])
            self.groups.append({"key": (s, f), "df": g, "static": static_vec})
            gi = len(self.groups) - 1
            # sliding anchors
            # anchor t is last encoder index (inclusive)
            for t in range(enc_len - 1, n - dec_len - 1, self.stride):
                dec_end_date = g.loc[t + dec_len, self.date_col]
                if split == "train" and dec_end_date <= train_end:
                    self.index.append((gi, t))
                elif split == "val" and (dec_end_date > train_end) and (dec_end_date <= val_end):
                    self.index.append((gi, t))
                elif split == "test" and (dec_end_date > val_end) and (dec_end_date <= test_end):
                    self.index.append((gi, t))

    def _make_static_vector(self, row: pd.Series) -> np.ndarray:
        parts = []
        for c in self.static_cols:
            oh = self.static_onehot_maps[c][row[c]]
            parts.append(oh)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        gi, t = self.index[idx]
        g = self.groups[gi]["df"]
        static = self.groups[gi]["static"]
        enc_slice = slice(t - self.enc_len + 1, t + 1)
        dec_slice = slice(t + 1, t + 1 + self.dec_len)

        enc_x = g.loc[enc_slice, self.enc_vars].to_numpy(dtype=np.float32)   # [L_enc, E]
        dec_x = g.loc[dec_slice, self.dec_vars].to_numpy(dtype=np.float32)   # [L_dec, D]
        y = g.loc[dec_slice, self.target_col].to_numpy(dtype=np.float32)     # [L_dec]

        return {
            "past_inputs": torch.from_numpy(enc_x),      # [L_enc, E]
            "future_inputs": torch.from_numpy(dec_x),    # [L_dec, D]
            "static_inputs": torch.from_numpy(static),   # [S]
            "target": torch.from_numpy(y),               # [L_dec]
        }


def tft_collate(batch):
    # stack and ensure shapes [B, T, F]
    past = torch.stack([b["past_inputs"] for b in batch], dim=0)
    future = torch.stack([b["future_inputs"] for b in batch], dim=0)
    static = torch.stack([b["static_inputs"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    return {"past_inputs": past, "future_inputs": future, "static_inputs": static, "target": target}
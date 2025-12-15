import random
import numpy as np
import torch
import pandas as pd
from typing import Dict, List


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_onehot_maps(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict]:
    """Builds one-hot maps for static features"""
    maps = {}
    for c in cols:
        cats = sorted(df[c].dropna().unique().tolist())
        idx = {v: i for i, v in enumerate(cats)}
        dim = len(cats)
        maps[c] = {}
        eye = np.eye(dim, dtype=np.float32)
        for v in cats:
            maps[c][v] = eye[idx[v]]
    return maps


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Calculates Mean Absolute Error"
    return float(np.abs(y_true - y_pred).mean())


def calc_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Calculates Weighted Absolute Percentage Error"
    denom = np.abs(y_true).sum() + 1e-8
    return float(np.abs(y_true - y_pred).sum() / denom)


def calc_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Calculates Symmetric Mean Absolute Percentage Error"
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    return float((2.0 * np.abs(y_true - y_pred) / denom).mean())

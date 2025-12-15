import random
import numpy as np
import os
import torch
import pandas as pd
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter


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


def compute_metrics(y_true, y_pred):
    """
    Metric computation for classification tasks.
    MAE, WAPE, sMAPE
    Computes ROC AUC and PR AUC using continuous scores.
    """
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    metrics = {}
    metrics["mae"] = calc_mae(y_true, y_pred)
    metrics["wape"] = calc_wape(y_true, y_pred)
    metrics["smape"] = calc_smape(y_true, y_pred)

    return metrics


def write_metrics_to_tensorboard(tensorboard_writer, metrics, epoch):
    tensorboard_writer.write("train_mae", metrics.get("mae"), epoch)
    tensorboard_writer.write("train_wape", metrics.get("wape"), epoch)
    tensorboard_writer.write("train_smape", metrics.get("smape"), epoch)


class TensorboardConfig:
    def __init__(self, store_flag, log_dir):
        if store_flag:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def write(self, naming, loss, epoch):
        if self.writer:
            self.writer.add_scalar(f"{naming}", loss, epoch)

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()


def get_date_splits(df, dec_len):
    "Determine split dates"
    max_date = df["date"].max()
    test_days = pd.Timedelta(days=dec_len)  # hold-out horizon for test
    val_days = pd.Timedelta(days=dec_len)   # validation horizon
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days

    return train_end, val_end, test_end

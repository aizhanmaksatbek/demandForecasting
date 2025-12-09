import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List


class NodeIndexer:
    """Maps (store_nbr, family) to contiguous node ids and back."""

    def __init__(self, keys: List[Tuple[int, str]]):
        uniq = sorted(set(keys))
        self.key2id: Dict[Tuple[int, str], int] = {k: i for i, k in enumerate(uniq)}
        self.id2key: List[Tuple[int, str]] = uniq

    def __len__(self) -> int:
        return len(self.id2key)

    def id(self, key: Tuple[int, str]) -> int:
        return self.key2id[key]

    def key(self, idx: int) -> Tuple[int, str]:
        return self.id2key[idx]


def build_node_indexer(df: pd.DataFrame) -> NodeIndexer:
    keys = list(zip(df["store_nbr"].astype(int), df["family"].astype(str)))
    return NodeIndexer(keys)


def build_static_node_features(
    df: pd.DataFrame,
    indexer: NodeIndexer,
    static_cols: List[str],
    static_onehot_maps: Dict[str, Dict],
) -> torch.Tensor:
    """Create node static feature matrix X_nodes [N, S] from one-hots used by TFT."""
    rows = []
    for key in indexer.id2key:
        s, f = key
        row = df[(df.store_nbr == s) & (df.family == f)].iloc[0]
        parts = []
        for c in static_cols:
            oh = static_onehot_maps[c][row[c]]
            parts.append(oh)
        x = np.concatenate(parts, axis=0).astype(np.float32)
        rows.append(x)
    X = np.vstack(rows)
    return torch.from_numpy(X)


def build_product_graph_adjacency(
    df: pd.DataFrame,
    indexer: NodeIndexer,
    train_end: pd.Timestamp,
    top_k: int = 10,
    min_corr: float = 0.2,
) -> torch.Tensor:
    """Build adjacency based on Pearson correlation of sales over the train window.

    - For each node (store, family), compute its daily sales series up to train_end
    - Compute correlation matrix across nodes
    - For each node, keep top_k neighbors with corr >= min_corr (symmetrized)
    """
    # Build panel pivot [date x node]
    df_train = df[df["date"] <= train_end].copy()
    df_train["node"] = list(zip(df_train.store_nbr.astype(int), df_train.family))
    pivot = df_train.pivot_table(index="date", columns="node", values="sales")
    # Align columns with indexer order
    cols = indexer.id2key
    pivot = pivot.reindex(columns=cols)
    # Fill NaNs with 0 for correlation
    pivot = pivot.fillna(0.0)
    C = pivot.corr(method="pearson").to_numpy(dtype=np.float32)  # [N, N]
    N = C.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    # For each node, select neighbors
    for i in range(N):
        # Exclude self, sort by corr
        corr_i = C[i].copy()
        corr_i[i] = -np.inf
        # Indices sorted descending
        nbr_idx = np.argsort(-corr_i)
        kept = 0
        for j in nbr_idx:
            if corr_i[j] < min_corr:
                break
            A[i, j] = 1.0
            kept += 1
            if kept >= top_k:
                break
    # Symmetrize
    A = np.maximum(A, A.T)
    return torch.from_numpy(A)

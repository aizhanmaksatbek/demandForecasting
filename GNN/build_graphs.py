"""
Build store- and family-level graphs and combine them
into a Kronecker-sum adjacency for (store_nbr, family) nodes.

Outputs:
- GNN/data/processed/store_graph_edges.csv
- GNN/data/processed/family_graph_edges.csv
- GNN/data/processed/node_index.csv
- GNN/data/processed/adjacency.npy  (row-normalized)
    where A_hat = D^{-1/2}(A + I)D^{-1/2}

Assumptions:
- raw_data/train.csv, stores.csv, transactions.csv (transactions optional)
- raw_data/holidays_events.csv optional (not used here directly)
"""

import os
import numpy as np
import pandas as pd

RAW = "raw_data"
PROC = "GNN/data/processed"
os.makedirs(PROC, exist_ok=True)


def build_store_graph(train: pd.DataFrame, stores: pd.DataFrame, trans: pd.DataFrame | None, corr_min=0.2):
    stores = stores.rename(columns={"type": "store_type"})
    base = stores[["store_nbr", "state", "cluster"]].copy()

    corr = None
    if trans is not None and len(trans) > 0:
        pivot = trans.pivot(index="date", columns="store_nbr", values="transactions")
        corr = pivot.corr(min_periods=30)

    edges = []
    snbrs = base["store_nbr"].unique()
    for i, a in enumerate(snbrs):
        for b in snbrs[i + 1:]:
            w = 0.0
            ra = base.loc[base.store_nbr == a].iloc[0]
            rb = base.loc[base.store_nbr == b].iloc[0]
            if pd.notna(ra["cluster"]) and pd.notna(rb["cluster"]) and ra["cluster"] == rb["cluster"]:
                w += 0.5
            if pd.notna(ra["state"]) and pd.notna(rb["state"]) and ra["state"] == rb["state"]:
                w += 0.3
            if corr is not None and a in corr.index and b in corr.columns:
                c = corr.loc[a, b]
                if pd.notna(c) and c >= corr_min:
                    w += float(c)
            if w > 0:
                edges.append((int(a), int(b), float(w)))
    return pd.DataFrame(edges, columns=["store_nbr_src", "store_nbr_dst", "weight"])


def build_family_graph(train: pd.DataFrame, corr_min=0.3):
    fam_daily = train.groupby(["date", "family"])["sales"].sum().unstack("family").fillna(0.0)
    C = fam_daily.corr(min_periods=60)
    families = list(C.columns)
    edges = []
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            c = C.loc[a, b]
            if pd.notna(c) and c >= corr_min:
                edges.append((a, b, float(c)))
    return pd.DataFrame(edges, columns=["family_src", "family_dst", "weight"])


def edges_to_adj(nodes, edges_df, src_col, dst_col, weight_col, undirected=True):
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for _, r in edges_df.iterrows():
        i = idx.get(r[src_col])
        j = idx.get(r[dst_col])
        if i is None or j is None: 
            continue
        w = float(r[weight_col])
        A[i, j] += w
        if undirected:
            A[j, i] += w
    return A


def normalize_adj(A: np.ndarray, add_self_loops=True) -> np.ndarray:
    N = A.shape[0]
    if add_self_loops:
        A = A + np.eye(N, dtype=A.dtype)
    d = A.sum(axis=1)
    d[d == 0.0] = 1.0
    D_inv_sqrt = np.diag((1.0 / np.sqrt(d)).astype(A.dtype))
    return D_inv_sqrt @ A @ D_inv_sqrt


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(RAW, "train.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(RAW, "stores.csv"))
    trans_path = os.path.join(RAW, "transactions.csv")
    trans = pd.read_csv(trans_path, parse_dates=["date"]) if os.path.exists(trans_path) else None

    # Build edge lists
    store_edges = build_store_graph(train, stores, trans)
    fam_edges = build_family_graph(train)

    store_edges.to_csv(os.path.join(PROC, "store_graph_edges.csv"), index=False)
    fam_edges.to_csv(os.path.join(PROC, "family_graph_edges.csv"), index=False)

    # Node universe
    stores_u = sorted(train["store_nbr"].dropna().unique().tolist())
    families_u = sorted(train["family"].dropna().unique().tolist())

    # Store and family adjacency
    A_s = edges_to_adj(stores_u, store_edges, "store_nbr_src", "store_nbr_dst", "weight", undirected=True)
    A_f = edges_to_adj(families_u, fam_edges, "family_src", "family_dst", "weight", undirected=True)

    A_s_hat = normalize_adj(A_s, add_self_loops=True)
    A_f_hat = normalize_adj(A_f, add_self_loops=True)

    # Kronecker-sum adjacency:
    # A = kron(A_s_hat, I_F) + kron(I_S, A_f_hat)
    I_s = np.eye(len(stores_u), dtype=np.float32)
    I_f = np.eye(len(families_u), dtype=np.float32)
    A_kron = np.kron(A_s_hat, I_f) + np.kron(I_s, A_f_hat)
    A_kron = normalize_adj(A_kron, add_self_loops=True)

    # Node index mapping
    rows = []
    node_id = 0
    for s in stores_u:
        for f in families_u:
            rows.append((node_id, s, f))
            node_id += 1
    node_index = pd.DataFrame(rows, columns=["node_id", "store_nbr", "family"])
    node_index.to_csv(os.path.join(PROC, "node_index.csv"), index=False)

    # Save adjacency
    np.save(os.path.join(PROC, "adjacency.npy"), A_kron.astype(np.float32))
    print(f"Saved adjacency.npy with shape {A_kron.shape} and node_index.csv with {len(node_index)} nodes.")
"""
STGNN explainer and evaluator.
- Evaluates the STGNN on the test set (pinball, WAPE, sMAPE).
- Explains a target node using either PyG GNNExplainer (if compatible)
  or a custom adjacency-mask explainer that works with this STGNN.
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from graph_dataset import GraphDemandDataset

from GNN.architecture.stgnn import STGNN, QuantileLoss
from GNN.train_stgnn import eval_gnn_model
from config.settings import GNN_CHECKPOINTS_PATH, GNN_DATA_PATH
from utils.utils import get_date_splits
from config.settings import ENC_VARS as feature_cols


def build_static_graph_for_node(A_np: np.ndarray, target_idx: int):
    # Lazy import to avoid hard dependency when using --algo custom
    edges = np.argwhere(A_np[target_idx] > 0).reshape(-1)
    src = np.concatenate([edges, np.full_like(edges, target_idx)])
    dst = np.concatenate([np.full_like(edges, target_idx), edges])
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    x = torch.eye(A_np.shape[0], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)


class STGNNNodeAdapter(torch.nn.Module):
    def __init__(self, stgnn: STGNN, target_idx: int, median_idx: int = 1):
        super().__init__()
        self.stgnn = stgnn
        self.target_idx = int(target_idx)
        self.median_idx = int(median_idx)
        self.x_hist = None  # set by caller

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        yhat = self.stgnn(self.x_hist)[..., self.median_idx]  # [B,H,N]
        node_score = yhat[..., self.target_idx].mean(dim=(0, 1), keepdim=True)
        return node_score.squeeze()


def run_custom_edge_explainer_for_node(
    stgnn: STGNN,
    x_hist: torch.Tensor,
    target_idx: int,
    steps: int,
    lr: float,
    l1_coeff: float,
    ent_coeff: float,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    device = next(stgnn.parameters()).device
    stgnn.eval()

    with torch.no_grad():
        y0 = stgnn(x_hist)[..., 1]
        y0_scalar = y0[..., target_idx].mean()

    gconv = None
    for blk in stgnn.blocks:
        gconv = blk.gconv
        break
    assert gconv is not None, "STGNN has no GraphConv blocks"
    A = gconv.A_hat.to(device)
    nb_mask = (A[target_idx] > 0).float()
    alpha = torch.nn.Parameter(torch.zeros_like(A[target_idx]))
    optimizer = torch.optim.Adam([alpha], lr=lr)

    def masked_forward(alpha_logits: torch.Tensor):
        m_row = torch.sigmoid(alpha_logits) * nb_mask
        M = torch.ones_like(A)
        M[target_idx] = m_row
        orig_forwards = []
        for blk in stgnn.blocks:
            gc = blk.gconv

            def gc_forward_with_mask(x, gc=gc, M=M):
                B, T, Nloc, C = x.shape
                x_agg = torch.einsum("ij,btjc->btic", gc.A_hat * M, x)
                yloc = gc.lin(x_agg)
                return gc.act(yloc)
            orig_forwards.append((gc, gc.forward))
            gc.forward = gc_forward_with_mask  # type: ignore
        out = stgnn(x_hist)[..., 1]
        for gc, f in orig_forwards:
            gc.forward = f
        return out

    for _ in range(steps):
        optimizer.zero_grad()
        y_mask = masked_forward(alpha)
        y_scalar = y_mask[..., target_idx].mean()
        fidelity = (y_scalar - y0_scalar).pow(2)
        m = torch.sigmoid(alpha) * nb_mask
        size = m.sum()
        ent = -(m * (m + 1e-8).log() + (1 - m) * (1 - m + 1e-8).log()).mean()
        loss = fidelity + l1_coeff * size + ent_coeff * ent
        loss.backward()
        optimizer.step()

    m_final = (torch.sigmoid(alpha) * nb_mask).detach().cpu().numpy()
    np.save(
        os.path.join(
            out_dir, f"gnn_node_{target_idx}_mask_.npy"
            ),
        m_final)

    return {
        'num_neighbors': int(nb_mask.sum().item()),
        'mask_min': (float(m_final[m_final > 0].min())
                     if (m_final > 0).any() else 0.0),
        'mask_max': float(m_final.max()) if m_final.size else 0.0,
    }


def save_neighbor_importances_csv(
    mask_path: str,
    node_index_csv: str,
    target_node: int,
    out_dir: str,
    top_k: int = 20,
):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    m = np.load(mask_path)
    node_map = pd.read_csv(node_index_csv)
    node_map = node_map.drop_duplicates("node_id").set_index("node_id")
    idxs = np.argsort(-m)[:top_k]
    rows = []
    for nid in idxs:
        meta = node_map.loc[nid].to_dict() if nid in node_map.index else {}
        rows.append(
            {"neighbor_node_id": int(nid), "importance": float(m[nid]),
             **meta}
             )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        out_dir, f"xai_custom_top{top_k}_neighbors_node{target_node}.csv"
        )
    df.to_csv(csv_path, index=False)
    print(f"[CustomExplainer] Saved labeled top-{top_k} CSV -> {csv_path}")
    return csv_path


def save_least_important_neighbors_csv(
    mask_path: str,
    node_index_csv: str,
    target_node: int,
    out_dir: str,
    bottom_k: int = 20,
):
    """
    Save the least important neighbors (lowest mask values) to a CSV file.
    """
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    m = np.load(mask_path)
    node_map = pd.read_csv(node_index_csv)
    node_map = node_map.drop_duplicates("node_id").set_index("node_id")
    idxs = np.argsort(m)[:bottom_k]
    rows = []
    for nid in idxs:
        meta = node_map.loc[nid].to_dict() if nid in node_map.index else {}
        rows.append({"neighbor_node_id": int(nid), "importance": float(m[nid]), **meta})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"xai_custom_bottom{bottom_k}_neighbors_node{target_node}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[CustomExplainer] Saved labeled bottom-{bottom_k} CSV -> {csv_path}")
    return csv_path


def save_neighbor_importances_plot(
    mask_path: str,
    node_index_csv: str,
    target_node: int,
    out_dir: str,
    top_k: int = 20,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    m = np.load(mask_path)
    node_map = pd.read_csv(node_index_csv)
    node_map = node_map.drop_duplicates("node_id").set_index("node_id")
    idxs = np.argsort(-m)[:top_k]
    labels = []
    vals = []
    for nid in idxs:
        meta = node_map.loc[nid] if nid in node_map.index else None
        label = (f"id={nid}, store={meta['store_nbr']}, {meta['family']}")
        labels.append(label)
        vals.append(m[nid])
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel("Neighbor importance")
    plt.title(f"Top-{top_k} neighbors for target node {target_node}")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"GNN_node_{target_node}_neighbors.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[CustomExplainer] Saved plot -> {png_path}")
    return png_path


def save_least_important_neighbors_plot(
    mask_path: str,
    node_index_csv: str,
    target_node: int,
    out_dir: str,
    bottom_k: int = 20,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    m = np.load(mask_path)
    node_map = pd.read_csv(node_index_csv)
    node_map = node_map.drop_duplicates("node_id").set_index("node_id")
    idxs = np.argsort(m)[:bottom_k]
    labels = []
    vals = []
    for nid in idxs:
        meta = node_map.loc[nid] if nid in node_map.index else None
        label = (f"id={nid}, store={meta['store_nbr']}, {meta['family']}")
        labels.append(label)
        vals.append(m[nid])
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel("Neighbor importance")
    plt.title(f"Bottom-{bottom_k} neighbors for target node {target_node}")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"GNN_node_{target_node}_bottom_neighbors.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[CustomExplainer] Saved plot -> {png_path}")
    return png_path


def main():
    parser = argparse.ArgumentParser(description="STGNN explainer")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(
                            GNN_CHECKPOINTS_PATH, "stgnn_best.pt"))
    parser.add_argument("--panel-csv", type=str,
                        default=os.path.join(GNN_DATA_PATH, "panel.csv"))
    parser.add_argument("--node-index-csv", type=str,
                        default=os.path.join(GNN_DATA_PATH, "node_index.csv"))
    parser.add_argument("--adj", type=str,
                        default=os.path.join(GNN_DATA_PATH, "adjacency.npy"))
    parser.add_argument("--target-node", type=int, default=10)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(GNN_CHECKPOINTS_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K neighbors of target node for explainer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    assert os.path.exists(args.panel_csv), f"{args.panel_csv} not found"
    assert os.path.exists(args.node_index_csv), "node_index.csv not found"
    assert os.path.exists(args.adj), f"adjacency.npy not found: {args.adj}"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    quantiles = ckpt.get("quantiles", [0.1, 0.5, 0.9])
    enc_len = int(cfg.get("enc_len", 56))
    horizon = int(cfg.get("horizon", 28))

    # Split bounds via panel
    df = pd.read_csv(args.panel_csv, parse_dates=["date"])
    train_end, val_end, test_end = get_date_splits(df, horizon)
    split_bounds = (train_end, val_end, test_end)

    test_ds = GraphDemandDataset(
        args.panel_csv, args.node_index_csv, enc_len, horizon,
        split_bounds, split="test", feature_cols=feature_cols,
    )
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset is empty")
    idx = max(0, min(args.batch_index, len(test_ds) - 1))
    sample = test_ds[idx]
    x_hist = sample["x_hist"].unsqueeze(0).to(device)  # [1,L,N,F]

    A_hat = torch.from_numpy(np.load(args.adj)).to(device)
    num_nodes = A_hat.shape[0]
    model = STGNN(
        num_nodes=num_nodes,
        in_features=len(feature_cols),
        horizon=horizon,
        A_hat=A_hat,
        hidden_channels=int(cfg.get("hidden", 64)),
        num_blocks=int(cfg.get("blocks", 3)),
        kernel_size=int(cfg.get("kernel", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        quantiles=quantiles,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.eval()

    bs = int(args.batch_size or cfg.get("batch_size", 32))
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True
        )
    criterion = QuantileLoss(quantiles=quantiles)
    # eval_gnn_model(model, test_loader, criterion, args.ckpt, quantiles)

    # GNN Explainer
    stats = run_custom_edge_explainer_for_node(
        stgnn=model, x_hist=x_hist, target_idx=args.target_node,
        steps=20, lr=0.1, l1_coeff=5e-3, ent_coeff=1e-3,
        out_dir=args.out_dir,
    )
    print({"explainer": "custom", "target_node": args.target_node, **stats})
    # Save labeled CSV and optional plot
    mask_path = os.path.join(
        args.out_dir, f"gnn_node_{args.target_node}_mask_.npy"
    )
    csv_path = save_neighbor_importances_csv(
        mask_path, args.node_index_csv, args.target_node,
        args.out_dir, top_k=args.top_k
    )
    print(f"[GNNCustomExplainer] Saved top-K neighbors CSV -> {csv_path}")
    save_neighbor_importances_plot(
        mask_path, args.node_index_csv, args.target_node,
        args.out_dir, top_k=args.top_k
    )
    print("[GNNCustomExplainer] Saved labeled top-K neighbors plot.")

    # Save and plot least important neighbors
    bottom_k = args.top_k  # Use same k for bottom as for top
    bottom_csv_path = save_least_important_neighbors_csv(
        mask_path, args.node_index_csv, args.target_node,
        args.out_dir, bottom_k=bottom_k
    )
    print(f"[GNNCustomExplainer] Saved bottom-K neighbors CSV -> {bottom_csv_path}")
    save_least_important_neighbors_plot(
        mask_path, args.node_index_csv, args.target_node,
        args.out_dir, bottom_k=bottom_k
    )
    print("[GNNCustomExplainer] Saved labeled bottom-K neighbors plot.")


if __name__ == "__main__":
    main()

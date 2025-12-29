import os
import numpy as np
import torch
import pandas as pd
import shap
from architecture.stgnn import STGNN
from torch.utils.data import DataLoader
import argparse
from graph_dataset import GraphDemandDataset
from config.settings import GNN_DATA_PATH, GNN_CHECKPOINTS_PATH
from utils.utils import get_date_splits
from config.settings import ENC_VARS as feature_cols
import matplotlib.pyplot as plt

feature_names = feature_cols


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
    parser.add_argument("--target-node", type=int, default=2)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(GNN_CHECKPOINTS_PATH))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-background", type=int, default=8,
                        help="Number of background samples for SHAP")
    parser.add_argument("--n-test", type=int, default=16,
                        help="Number of test samples to explain")
    parser.add_argument("--forecast-index", type=int, default=-1,
                        help="Which forecast step to explain (default last)")
    parser.add_argument(
        "--quantile-index",
        type=int,
        default=1,
        help="Which quantile to explain (default median index 1)"
    )
    args = parser.parse_args()

    # Prefer CUDA, then Apple Metal (MPS), else CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps"
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            else "cpu"
        )
    )

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

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

    # criterion = QuantileLoss(quantiles=quantiles)
    # eval_gnn_model(model, test_loader, criterion, args.ckpt, quantiles)

    # --- Collect background and test tensors (full sequences, not zero-padded)
    background_tensors = []
    test_tensors = []
    for i, batch in enumerate(test_loader):
        x = batch["x_hist"].to(device)
        if len(background_tensors) < args.n_background:
            background_tensors.append(x)
        if len(test_tensors) < args.n_test:
            test_tensors.append(x)
        if (
            len(background_tensors) >= args.n_background
            and len(test_tensors) >= args.n_test
        ):
            break

    if len(background_tensors) == 0 or len(test_tensors) == 0:
        raise RuntimeError(
            "Insufficient data for SHAP: background or test tensors empty."
        )

    background = torch.cat(background_tensors, dim=0).to(device)
    # shape: [B_bg, L, N, F]
    X_test = torch.cat(test_tensors, dim=0).to(device)
    # shape: [B_te, L, N, F]

    target_node = int(args.target_node)
    q_idx = int(args.quantile_index)
    f_idx = int(args.forecast_index)

    # --- Wrap model to expose target scalar for SHAP
    class TargetOutput(torch.nn.Module):
        def __init__(self, base_model, node_idx, q_index, f_index):
            super().__init__()
            self.base_model = base_model
            self.node_idx = node_idx
            self.q_index = q_index
            self.f_index = f_index

        def forward(self, x):
            y = self.base_model(x)
            step = self.f_index if self.f_index >= 0 else (
                y.shape[1] + self.f_index
            )
            # Return shape [B, 1] to satisfy DeepExplainer expectations
            return y[:, step, self.node_idx, self.q_index].unsqueeze(1)

    target_model = TargetOutput(model, target_node, q_idx, f_idx)

    # Use DeepExplainer for PyTorch with Torch tensors
    explainer = shap.DeepExplainer(target_model, background)

    # Compute SHAP values on test set: returns list/array matching inputs
    # Disable additivity check to avoid unsupported-op issues in deep graphs
    sv_any = explainer.shap_values(X_test, check_additivity=False)
    if isinstance(sv_any, list):
        sv_np = sv_any[0]
    else:
        sv_np = sv_any

    # --- Aggregate SHAP for the target node across time: [B, F]
    sv_node_time = sv_np[:, :, target_node, :]              # [B, L, F] or [B, L, F, 1]
    sv_node_feat = sv_node_time.mean(axis=1)                # [B, F] or [B, F, 1]
    # Squeeze trailing singleton dim if present (DeepExplainer often adds output dim)
    if hasattr(sv_node_feat, "ndim") and sv_node_feat.ndim == 3:
        sv_node_feat = np.squeeze(sv_node_feat, -1)

    # Aggregate feature values for visualization (mean over time at node)
    X_np = X_test.detach().cpu().numpy()
    X_node_time = X_np[:, :, target_node, :]                # [B, L, F]
    X_node_feat = X_node_time.mean(axis=1)                  # [B, F]

    # --- Save SHAP values (per-sample, per-feature) to CSV
    shap_df = pd.DataFrame(sv_node_feat, columns=feature_names)
    out_csv = os.path.join(args.out_dir, "shap_values.csv")
    shap_df.to_csv(out_csv, index=False)
    print(f"Saved SHAP values to {out_csv}")

    # --- Visualize local explanation (force plot for first sample)
    shap.initjs()
    # Base value: mean prediction on background
    with torch.no_grad():
        base_pred = target_model(background).mean().item()
    force = shap.force_plot(
        base_pred,
        sv_node_feat[0],
        X_node_feat[0],
        feature_names=feature_names
    )
    shap.save_html(os.path.join(args.out_dir, "force_plot.html"), force)

    # --- Visualize global feature importance (summary plot)
    plt.figure()
    shap.summary_plot(
        sv_node_feat,
        X_node_feat,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(os.path.join(args.out_dir, "summary_plot.png"), dpi=300)
    plt.close()


def plot_shap_from_file(
    csv_path,
    feature_names=None,
    out_path="summary_plot_from_file.png",
):
    shap_df = pd.read_csv(csv_path)
    if feature_names is None:
        feature_names = list(shap_df.columns)
    plt.figure()
    shap_vals = shap_df.values
    # Plot mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    plt.bar(range(len(feature_names)), mean_abs_shap)
    plt.xticks(
        range(len(feature_names)), feature_names, rotation=45, ha='right'
    )
    plt.ylabel("Mean |SHAP value|")
    plt.title("Global SHAP Feature Importance (from file)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot from file to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
    # Plot from file path in out-dir for convenience
    plot_shap_from_file(
        os.path.join(GNN_CHECKPOINTS_PATH, "shap_values.csv"),
        feature_names=feature_names,
    )

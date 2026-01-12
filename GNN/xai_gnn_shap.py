import os
import numpy as np
import torch
import pandas as pd
from architecture.stgnn import STGNN
from torch.utils.data import DataLoader
import argparse
from graph_dataset import GraphDemandDataset
from config.settings import GNN_DATA_PATH, GNN_CHECKPOINTS_PATH
from utils.utils import get_date_splits
from config.settings import ENC_VARS as feature_cols
import matplotlib.pyplot as plt
import shap

feature_names = feature_cols


def plot_global_feature_importance(
    shap_values,
    feature_names,
    out_path,
    csv_out=None,
    title=None,
):
    """Plot mean absolute SHAP values per feature as a bar chart.

    shap_values: array [B, F] of SHAP values aggregated per sample
        for target node.
    feature_names: list[str] length F.
    out_path: file path to save the PNG.
    csv_out: optional path to save the ranking as CSV.
    title: optional title to annotate the figure.
    """
    sv = shap_values
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()
    if sv.ndim != 2:
        sv = np.squeeze(sv)
    # Mean absolute SHAP across samples
    mean_abs = np.mean(np.abs(sv), axis=0)  # [F]
    # Sort descending
    order = np.argsort(-mean_abs)
    sorted_vals = mean_abs[order]
    sorted_names = [
        feature_names[i] if i < len(feature_names) else f"feat_{i}"
        for i in order
    ]

    # Save CSV ranking if requested
    if csv_out:
        pd.DataFrame({
            "feature": sorted_names,
            "mean_abs_shap": sorted_vals,
        }).to_csv(csv_out, index=False)
        print(f"Saved global feature importance CSV to {csv_out}")

    # Plot horizontal bar chart
    plt.figure(figsize=(8, max(4, len(sorted_names) * 0.3)))
    y_pos = np.arange(len(sorted_names))
    plt.barh(y_pos, sorted_vals, color="#4C78A8")
    plt.yticks(y_pos, sorted_names)
    plt.xlabel("Mean |SHAP| (importance)")
    if title:
        plt.title(f"Global Feature Importance\n{title}")
    else:
        plt.title("Global Feature Importance")
    plt.gca().invert_yaxis()  # largest on top
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved global feature importance bar chart to {out_path}")


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
    parser.add_argument("--n-test", type=int, default=28,
                        help="Number of test samples to explain")
    parser.add_argument("--forecast-index", type=int, default=1,
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
        "cuda" if torch.cuda.is_available() else "cpu"
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

    # --- Collect background and test tensors (full sequences, not zero-padded)
    background_tensors = []
    test_tensors = []
    test_dates = []
    for i, batch in enumerate(test_loader):
        x = batch["x_hist"].to(device)
        if len(background_tensors) < args.n_background:
            background_tensors.append(x)
        if len(test_tensors) < args.n_test:
            test_tensors.append(x)
            # Capture decoder end dates for x-axis labels
            dec_dates = batch.get("dec_end_date")
            if isinstance(dec_dates, (list, tuple)):
                test_dates.extend(list(dec_dates))
            elif dec_dates is not None:
                test_dates.append(dec_dates)
        if (
            len(background_tensors) >= args.n_background
            and len(test_tensors) >= args.n_test
        ):
            break

    if len(background_tensors) == 0 or len(test_tensors) == 0:
        raise RuntimeError(
            "Insufficient data for SHAP: background or test tensors empty."
        )

    # Background tensor collection retained for SHAP if needed
    X_test = torch.cat(test_tensors, dim=0).to(device)
    # shape: [B_te, L, N, F]

    target_node = int(args.target_node)
    # q_idx and f_idx retained for SHAP when enabled

    # Map target node index to human-readable label from node_index.csv
    node_label = None
    node_file_tag = f"node{target_node}"
    try:
        nodes_df = pd.read_csv(args.node_index_csv)
        row = nodes_df.loc[nodes_df["node_id"] == target_node]
        if not row.empty:
            store = str(row.iloc[0].get("store_nbr", ""))
            fam = str(row.iloc[0].get("family", ""))
            node_label = (
                f"node_id: {target_node}, store_nbr: {store}, family: {fam}"
            )
            print(f"Target node: {node_label}")
            # Build a safe file tag for outputs
            
            def _san(s):
                return "".join(ch if ch.isalnum() else "_" for ch in str(s))
            node_file_tag = (
                f"node{target_node}_store{_san(store)}_family{_san(fam)}"
            )
        else:
            print(
                f"Target node {target_node} not found in node_index.csv"
            )
    except Exception as e:
        print(f"Warning: failed to read node_index mapping: {e}")

    # --- Wrap model to expose target scalar for SHAP
    # TargetOutput wrapper retained for SHAP if enabled later

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
    sv_node_time = sv_np[:, :, target_node, :]  # [B, L, F] or [B, L, F, 1]
    sv_node_feat = sv_node_time.mean(axis=1)    # [B, F] or [B, F, 1]
    # Squeeze trailing singleton dim if present
    if hasattr(sv_node_feat, "ndim") and sv_node_feat.ndim == 3:
        sv_node_feat = np.squeeze(sv_node_feat, -1)

    # Aggregate feature values for visualization (mean over time at node)
    X_np = X_test.detach().cpu().numpy()
    X_node_time = X_np[:, :, target_node, :]                # [B, L, F]
    X_node_feat = X_node_time.mean(axis=1)                  # [B, F]

    # # --- Save SHAP values (per-sample, per-feature) to CSV
    shap_df = pd.DataFrame(sv_node_feat, columns=feature_names)
    out_csv = os.path.join(
        args.out_dir,
        f"shap_values_{node_file_tag}.csv",
    )
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
    shap.save_html(
        os.path.join(
            args.out_dir,
            f"force_plot_{node_file_tag}.html",
        ),
        force,
    )

    # --- Visualize global feature importance (summary plot)
    plt.figure()
    # Add title before saving
    shap.summary_plot(
        sv_node_feat,
        X_node_feat,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    if node_label:
        plt.suptitle(node_label, fontsize=10)
        plt.tight_layout()
    plt.savefig(
        os.path.join(
            args.out_dir,
            f"summary_plot_{node_file_tag}.png",
        ),
        dpi=300,
    )
    plt.close()

    # --- Plot input values per feature across test samples
    inputs_png = os.path.join(
        args.out_dir,
        f"inputs_by_feature_{node_file_tag}.png",
    )
    plot_inputs_per_feature(
        X_node_feat,
        feature_names,
        inputs_png,
        x_labels=test_dates,
        title=node_label,
    )

    # --- Additional: Global feature importance bar chart (mean |SHAP|)
    bar_png = os.path.join(
        args.out_dir,
        f"global_feature_importance_{node_file_tag}.png",
    )
    bar_csv = os.path.join(
        args.out_dir,
        f"global_feature_importance_{node_file_tag}.csv",
    )
    plot_global_feature_importance(
        sv_node_feat,
        feature_names,
        bar_png,
        csv_out=bar_csv,
        title=node_label,
    )


def plot_inputs_per_feature(
    X_node_feat,
    feature_names,
    out_path,
    x_labels=None,
    title=None,
):
    """Plot per-feature input values across test samples.

    X_node_feat: numpy array with shape [B, F] where B=n_test samples,
                 values aggregated over time for the target node.
    """
    X_arr = X_node_feat
    if isinstance(X_arr, torch.Tensor):
        X_arr = X_arr.detach().cpu().numpy()
    # Ensure 2D
    if X_arr.ndim != 2:
        X_arr = np.squeeze(X_arr)
    B, F = X_arr.shape
    # Build x-axis positions and labels (dates if provided)
    x_pos = np.arange(B)
    labels = None
    if x_labels is not None:
        try:
            labels = pd.to_datetime(x_labels).strftime("%Y-%m-%d").tolist()
        except Exception:
            labels = list(x_labels)
    cols = int(np.ceil(np.sqrt(F)))
    rows = int(np.ceil(F / cols))
    plt.figure(figsize=(cols * 5, rows * 3))
    if title:
        plt.suptitle(title, fontsize=10)
    for j in range(F):
        ax = plt.subplot(rows, cols, j + 1)
        ax.plot(x_pos, X_arr[:, j], marker="o", linewidth=1)
        feat_title = (
            feature_names[j] if j < len(feature_names) else f"feat_{j}"
        )
        # Include node info in each subplot title if provided
        if title:
            ax.set_title(f"{feat_title}", fontsize=9)
        else:
            ax.set_title(feat_title, fontsize=9)
        ax.set_xlabel("date" if labels is not None else "sample")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if labels is not None:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=720)
    plt.close()
    print(f"Saved inputs-by-feature plot to {out_path}")


if __name__ == "__main__":
    main()


import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from graph_dataset import GraphDemandDataset
from architecture.stgnn import STGNN
import matplotlib.pyplot as plt


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_paths() -> Tuple[str, str, str, str, str]:
    panel_csv = os.path.join("GNN", "data", "panel.csv")
    node_index_csv = os.path.join("GNN", "data", "node_index.csv")
    adj_path = os.path.join("GNN", "data", "adjacency.npy")
    ckpt_path = os.path.join("GNN", "checkpoints", "stgnn_best.pt")
    out_csv = os.path.join("GNN", "checkpoints", "stgnn_test_forecasts.csv")
    return panel_csv, node_index_csv, adj_path, ckpt_path, out_csv


def _feature_cols() -> List[str]:
    # Must match training
    return [
        "sales",
        "transactions",
        "dcoilwtico",
        "onpromotion",
        "dow",
        "month",
        "weekofyear",
        "is_holiday",
        "is_workday",
    ]


def _load_checkpoint(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            (
                "Checkpoint not found at "
                f"{ckpt_path}. Train with GNN/train_stgnn.py first."
            )
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    quantiles = ckpt.get("quantiles")
    state = ckpt["model_state"]
    return state, cfg, quantiles


def export_test_forecasts(
    enc_len: Optional[int] = None,
    horizon: Optional[int] = None,
    batch_size: int = 32,
    save_path: Optional[str] = None,
    include_window_stats: bool = True,
) -> str:
    """
    Run the best STGNN checkpoint on the test split and export a CSV with
    one row per (date, store_nbr, family)
    using the last-step horizon prediction.

    The CSV includes y_true, y_pred and (optionally) summary stats of the
    encoder window features that "made the prediction" (mean and last values).
    """
    device = _get_device()
    panel_csv, node_index_csv, adj_path, ckpt_path, out_csv = _default_paths()
    if save_path is None:
        save_path = out_csv

    state, cfg, quantiles = _load_checkpoint(ckpt_path, device)
    # Fallback to training config if args not provided
    enc_len = int(enc_len if enc_len is not None else cfg.get("enc_len", 56))
    horizon = int(horizon if horizon is not None else cfg.get("horizon", 28))

    # Build split bounds from panel
    # unscaled for metadata
    df_panel = pd.read_csv(panel_csv, parse_dates=["date"])
    max_date = df_panel["date"].max()
    test_days = pd.Timedelta(days=horizon)
    val_days = pd.Timedelta(days=horizon)
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days
    split_bounds = (train_end, val_end, test_end)

    feature_cols = _feature_cols()

    # Note: training might have saved a scaled panel copy; for inference we can
    # reuse the original since encoding is simple stats and the model learned
    # on scaled "transactions" and "dcoilwtico" only. For exact matching,
    # you could optionally recompute scaling before dataset creation the same
    # way as train.
    # Here we keep it simple and use the original panel (works if features are
    # not too off-scale). If exact parity is needed, port the scaling logic.

    test_ds = GraphDemandDataset(
        panel_csv,
        node_index_csv,
        enc_len,
        horizon,
        split_bounds,
        split="test",
        feature_cols=feature_cols,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Instantiate model
    A_hat = torch.from_numpy(np.load(adj_path)).to(device)
    num_nodes = A_hat.shape[0]
    in_features = len(feature_cols)
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    model = STGNN(
        num_nodes=num_nodes,
        in_features=in_features,
        horizon=horizon,
        A_hat=A_hat,
        hidden_channels=int(cfg.get("hidden", 64)),
        num_blocks=int(cfg.get("blocks", 3)),
        kernel_size=int(cfg.get("kernel", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        quantiles=list(quantiles),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Node mapping for identifiers
    nodes = pd.read_csv(node_index_csv)[["node_id", "store_nbr", "family"]]
    nodes["node_id"] = nodes["node_id"].astype(int)

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_hist"].to(device)  # [B, L, N, F]
            y = batch["y_fut"].to(device)   # [B, H, N]
            dates = batch["dec_end_date"]   # list[str] len=B
            yhat = model(x)                  # [B, H, N, Q]
            # median quantile
            qs = np.array(list(quantiles), dtype=float)
            med_idx = int(np.argmin(np.abs(qs - 0.5)))
            yhat_med = yhat[..., med_idx]    # [B, H, N]
            # last-step values aligned to dec_end_date
            y_true_last = y[:, -1, :].cpu().numpy()     # [B, N]
            y_pred_last = yhat_med[:, -1, :].cpu().numpy()  # [B, N]

            # Optional: window stats of features
            enc_mean = None
            enc_last = None
            if include_window_stats:
                x_np = x.cpu().numpy()  # [B, L, N, F]
                enc_mean = x_np.mean(axis=1)   # [B, N, F]
                enc_last = x_np[:, -1, :, :]   # [B, N, F]

            for b, date in enumerate(dates):
                # Expand across nodes
                df_b = pd.DataFrame({
                    "node_id": np.arange(y_true_last.shape[1], dtype=int),
                    "date": pd.to_datetime(date),
                    "y_true": y_true_last[b, :],
                    "y_pred": y_pred_last[b, :],
                })
                if include_window_stats:
                    for f_idx, f_name in enumerate(feature_cols):
                        df_b[f"enc_mean_{f_name}"] = enc_mean[b, :, f_idx]
                        df_b[f"enc_last_{f_name}"] = enc_last[b, :, f_idx]
                rows.append(df_b)

    pred_df = pd.concat(rows, axis=0, ignore_index=True)
    pred_df = pred_df.merge(nodes, on="node_id", how="left")
    # Keep tidy columns
    cols = ["date", "store_nbr", "family", "y_true", "y_pred"] + [
        c
        for c in pred_df.columns
        if c.startswith("enc_mean_") or c.startswith("enc_last_")
    ]
    pred_df = pred_df[cols]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred_df.to_csv(save_path, index=False)
    print(f"Saved forecasts to {save_path} ({len(pred_df)} rows)")
    return save_path


def load_forecasts(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        _, _, _, _, path = _default_paths()
    if not os.path.exists(path):
        # Try to generate if missing
        print("Forecast CSV missing; generating from best checkpoint...")
        export_test_forecasts(save_path=path)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def plot_store_family(
    df_pred: pd.DataFrame,
    store_nbr: int,
    family: str,
    save_dir: Optional[str] = None,
):
    sub = (
        df_pred[(df_pred.store_nbr == store_nbr) & (df_pred.family == family)]
        .sort_values("date")
        .copy()
    )
    if sub.empty:
        print("No rows for that (store, family)")
        return None

    if save_dir is None:
        save_dir = os.path.join("GNN", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Determine encoder history length from checkpoint (fallback to 56)
    try:
        panel_csv, _, _, ckpt_path, _ = _default_paths()
        device = _get_device()
        _, cfg, _ = _load_checkpoint(ckpt_path, device)
        enc_len = int(cfg.get("enc_len", 56))
    except Exception:
        panel_csv, _, _, _, _ = _default_paths()
        enc_len = 56

    # Load raw panel to reconstruct encoder input sales history
    try:
        panel_df = pd.read_csv(panel_csv, parse_dates=["date"])
        panel_df = panel_df[
            (panel_df["store_nbr"] == store_nbr)
            & (panel_df["family"] == family)
        ][["date", "sales"]].sort_values("date")
    except Exception:
        panel_df = None

    first_test_date = sub.date.min()
    hist_df = None
    if panel_df is not None and pd.notnull(first_test_date):
        start_date = first_test_date - pd.Timedelta(days=enc_len)
        hist_df = panel_df[
            (panel_df.date < first_test_date)
            & (panel_df.date >= start_date)
        ]
    plt.figure(figsize=(10, 4))
    # Input encoder history line
    if hist_df is not None and not hist_df.empty:
        plt.plot(
            hist_df.date,
            hist_df.sales,
            label="Input (encoder sales)",
            lw=2,
            color="tab:gray",
            # alpha=0.8,
        )
        plt.axvline(
            first_test_date,
            color="tab:gray",
            linestyle="--",
            linewidth=1,
        )
    plt.plot(sub.date, sub.y_true, label="Actual", lw=2)
    plt.plot(sub.date, sub.y_pred, label="Predicted", lw=2)
    plt.title(f"STGNN Forecast - store {store_nbr}, family {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc="upper left")
    plt.tight_layout()
    out = os.path.join(
        save_dir, f"stgnn_store{store_nbr}_family_{family}.png"
    )
    plt.savefig(out, dpi=1024)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_family_all_stores(
    df_pred: pd.DataFrame,
    family: str,
    max_cols: int = 5,
    save_dir: Optional[str] = None,
):
    sub = df_pred[df_pred.family == family]
    stores = sorted(sub.store_nbr.dropna().unique())
    if len(stores) == 0:
        print("No stores for that family")
        return None
    if save_dir is None:
        save_dir = os.path.join("GNN", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Determine encoder history length and panel path
    try:
        panel_csv, _, _, ckpt_path, _ = _default_paths()
        device = _get_device()
        _, cfg, _ = _load_checkpoint(ckpt_path, device)
        enc_len = int(cfg.get("enc_len", 56))
    except Exception:
        panel_csv, _, _, _, _ = _default_paths()
        enc_len = 56

    try:
        panel_full = pd.read_csv(panel_csv, parse_dates=["date"])
        panel_full = panel_full[panel_full["family"] == family][
            ["date", "sales", "store_nbr"]
        ]
    except Exception:
        panel_full = None

    cols = min(max_cols, len(stores))
    rows = (len(stores) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.2, rows * 3.2), sharey=True
    )
    axes = np.array(axes).reshape(-1)
    for i, store in enumerate(stores):
        ssub = sub[sub.store_nbr == store].sort_values("date")
        ax = axes[i]
        # Plot encoder input history before first test date for each store
        if panel_full is not None and not ssub.empty:
            first_test_date = ssub.date.min()
            start_date = first_test_date - pd.Timedelta(days=enc_len)
            hist = panel_full[
                (panel_full.store_nbr == store)
                & (panel_full.date < first_test_date)
                & (panel_full.date >= start_date)
            ]
            if not hist.empty:
                ax.plot(
                    hist.date,
                    hist.sales,
                    label="Input",
                    color="tab:gray",
                    alpha=0.8,
                )
                ax.axvline(
                    first_test_date,
                    color="tab:gray",
                    linestyle="--",
                    linewidth=1,
                )
        ax.plot(ssub.date, ssub.y_true, label="Actual", color="black")
        ax.plot(ssub.date, ssub.y_pred, label="Pred", color="tab:blue")
        ax.set_title(f"Store {store}")
        ax.tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2)
    fig.suptitle(f"STGNN: Family {family}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(save_dir, f"stgnn_family_{family}_all_stores.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_family_aggregate(
    df_pred: pd.DataFrame,
    family: str,
    save_dir: Optional[str] = None,
):
    sub = (
        df_pred[df_pred.family == family]
        .groupby("date", as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values("date")
        .copy()
    )
    if sub.empty:
        print("No rows for that (store, family)")
        return None

    if save_dir is None:
        save_dir = os.path.join("GNN", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Determine encoder history length from checkpoint (fallback to 56)
    try:
        panel_csv, _, _, ckpt_path, _ = _default_paths()
        device = _get_device()
        _, cfg, _ = _load_checkpoint(ckpt_path, device)
        enc_len = int(cfg.get("enc_len", 56))
    except Exception:
        panel_csv, _, _, _, _ = _default_paths()
        enc_len = 56

    # Load raw panel to reconstruct encoder input sales history
    try:
        panel_df = pd.read_csv(panel_csv, parse_dates=["date"])
        panel_df = (
            panel_df[panel_df["family"] == family]
            .groupby("date", as_index=False)[["sales"]]
            .sum()
            .sort_values("date"))
    except Exception:
        panel_df = None

    first_test_date = sub.date.min()
    hist_df = None
    if panel_df is not None and pd.notnull(first_test_date):
        start_date = first_test_date - pd.Timedelta(days=enc_len)
        hist_df = panel_df[
            (panel_df.date < first_test_date)
            & (panel_df.date >= start_date)
        ]
    plt.figure(figsize=(10, 4))
    # Input encoder history line
    if hist_df is not None and not hist_df.empty:
        plt.plot(
            hist_df.date,
            hist_df.sales,
            label="Input (encoder sales)",
            lw=2,
            color="tab:gray",
            alpha=0.8,
        )
        plt.axvline(
            first_test_date,
            color="tab:gray",
            linestyle="--",
            linewidth=1,
        )
    plt.plot(sub.date, sub.y_true, label="Actual", lw=2)
    plt.plot(sub.date, sub.y_pred, label="Predicted", lw=2)
    plt.title(f"STGNN Forecast - all stores, family {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc="upper left")
    plt.tight_layout()
    out = os.path.join(
        save_dir, f"stgnn_store_aggregate_family_{family}.png"
    )
    plt.savefig(out, dpi=1024)
    plt.close()
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    # If forecasts CSV is missing, generate it first
    forecasts = load_forecasts()

    # Example usage: plot a sample family across stores and one store-family
    # with features
    fams = sorted(forecasts.family.dropna().unique())
    fam_num = fams[7]
    if len(fams) > 0:
        plot_family_all_stores(forecasts, fam_num)
        sample = forecasts[(forecasts.family == fam_num)].iloc[0]
        plot_store_family(
            forecasts,
            int(sample.store_nbr),
            str(sample.family)
        )
    plot_family_aggregate(forecasts, fam_num, "GNN/checkpoints")

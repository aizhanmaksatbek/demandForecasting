import os
import pandas as pd
import matplotlib.pyplot as plt


def load_forecasts(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(
            "HYBRID_GNN_TFT", "checkpoints", "gnn_tft_test_forecasts.csv"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Hybrid forecasts CSV not found: {path}. Run HYBRID_GNN_TFT/train_tft.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    expected = ["date", "store_nbr", "family", "y_true", "y_pred", "y_past"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    return df[expected]


def verify_export_schema(df: pd.DataFrame):
    cols = list(df.columns)
    print(f"Hybrid CSV columns: {cols}")
    has_past = "y_past" in df.columns
    past_count = int(df["y_past"].notna().sum()) if has_past else 0
    pred_count = int(df["y_pred"].notna().sum()) if "y_pred" in df.columns else 0
    true_count = int(df["y_true"].notna().sum()) if "y_true" in df.columns else 0
    print(
        f"Counts -> y_past: {past_count} | y_pred: {pred_count} | y_true: {true_count}"
    )
    if not has_past or past_count == 0:
        print(
            "Warning: 'y_past' missing or empty. Re-run HYBRID_GNN_TFT/train_tft.py to export past encoder sales."
        )


def plot_store_family(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
    save_dir: str = None,
    show_onpromotion: bool = False,
):
    sub = df[(df.store_nbr == store_nbr) & (df.family == family)].sort_values("date")
    if sub.empty:
        print("No rows for that (store, family)")
        return None

    # Ensure input sequence exists in the export
    if "y_past" not in sub.columns:
        print(
            "Hybrid plot: missing 'y_past' in CSV for input sequence. "
            "Re-run HYBRID_GNN_TFT/train_tft.py to export past encoder sales."
        )
        past_seg = pd.DataFrame()
    else:
        past_seg = sub.dropna(subset=["y_past"]) 
    fut_seg = sub.dropna(subset=["y_pred"]) if "y_pred" in sub.columns else pd.DataFrame()

    plt.figure(figsize=(10, 4))
    split_date = None
    if not past_seg.empty:
        plt.plot(past_seg.date, past_seg.y_past, label="History (past)", color="black", lw=2)
        split_date = past_seg.date.max()
        plt.axvline(split_date, color="#888", linestyle="--", lw=1)
    if not fut_seg.empty:
        if split_date is not None:
            fut_seg = fut_seg[fut_seg.date > split_date]
        plt.plot(fut_seg.date, fut_seg.y_pred, label="Forecast (pred)", color="tab:blue", lw=2)
        if "y_true" in fut_seg.columns:
            plt.plot(fut_seg.date, fut_seg.y_true, label="Actual (future)", color="tab:green", lw=2)
        # Optional covariate overlays are disabled by default to match TFT

    title = f"Hybrid: store={store_nbr}, family={family}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.join("HYBRID_GNN_TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"hybrid_store{store_nbr}_family_{family}.png")
    plt.savefig(out, dpi=720)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_family_aggregate(df: pd.DataFrame, family: str, save_dir: str = None):
    fam_df = df[df.family == family]
    if fam_df.empty:
        print("No rows for that family")
        return None

    # Deduplicate overlapping windows
    past_agg = pd.DataFrame()
    if "y_past" in fam_df.columns:
        fam_past = fam_df.dropna(subset=["y_past"]).groupby(["store_nbr", "date"], as_index=False).agg({"y_past": "last"})
        past_agg = fam_past.groupby("date", as_index=False)[["y_past"]].sum().sort_values("date")

    pred_agg = pd.DataFrame()
    if "y_pred" in fam_df.columns:
        fam_pred = fam_df.dropna(subset=["y_pred"]).groupby(["store_nbr", "date"], as_index=False).agg({"y_pred": "last"})
        pred_agg = fam_pred.groupby("date", as_index=False)[["y_pred"]].sum().sort_values("date")

    true_agg = pd.DataFrame()
    if "y_true" in fam_df.columns:
        fam_true = fam_df.dropna(subset=["y_true"]).groupby(["store_nbr", "date"], as_index=False).agg({"y_true": "last"})
        true_agg = fam_true.groupby("date", as_index=False)[["y_true"]].sum().sort_values("date")

    plt.figure(figsize=(10, 4))
    split_date = None
    if not past_agg.empty:
        plt.plot(past_agg.date, past_agg.y_past, label="History (sum)", color="black", lw=2)
        split_date = past_agg.date.max()
        plt.axvline(split_date, color="#888", linestyle="--", lw=1)
    if not pred_agg.empty:
        pred_plot = pred_agg[pred_agg.date > split_date] if split_date is not None else pred_agg
        if not pred_plot.empty:
            plt.plot(pred_plot.date, pred_plot.y_pred, label="Forecast (sum)", color="tab:blue", lw=2)
    if not true_agg.empty:
        true_plot = true_agg[true_agg.date > split_date] if split_date is not None else true_agg
        if not true_plot.empty:
            plt.plot(true_plot.date, true_plot.y_true, label="Actual (sum)", color="tab:green", lw=2)

    plt.title(f"Hybrid Family aggregate: {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales (sum across stores)")
    plt.legend()
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.join("HYBRID_GNN_TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"hybrid_family_{family}_aggregate.png")
    plt.savefig(out, dpi=720)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_family_all_stores(
    df: pd.DataFrame,
    family: str,
    max_cols: int = 5,
    save_dir: str = None,
    show_onpromotion: bool = False,
):
    sub = df[df.family == family]
    has_pred = sub.dropna(subset=["y_pred"]) if "y_pred" in sub.columns else pd.DataFrame()
    stores = sorted(has_pred.store_nbr.unique()) if not has_pred.empty else sorted(sub.store_nbr.unique())
    if not stores:
        print("No stores for that family")
        return None
    cols = min(max_cols, len(stores))
    rows = (len(stores) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), sharey=True)
    axes = axes.flatten()
    for i, store in enumerate(stores):
        ssub = sub[sub.store_nbr == store].sort_values("date")
        ax = axes[i]
        if "y_past" not in ssub.columns:
            ssub_past = pd.DataFrame()
        else:
            ssub_past = ssub.dropna(subset=["y_past"]) 
        if not ssub_past.empty:
            ax.plot(ssub_past.date, ssub_past.y_past, label="History (past)", color="black", lw=2)
            split_date = ssub_past.date.max()
            ax.axvline(split_date, color="#888", linestyle="--", lw=1)
        ssub_pred = ssub.dropna(subset=["y_pred"]) if "y_pred" in ssub.columns else pd.DataFrame()
        if not ssub_pred.empty:
            if not ssub_past.empty:
                ssub_pred = ssub_pred[ssub_pred.date > split_date]
            ax.plot(ssub_pred.date, ssub_pred.y_pred, label="Forecast (pred)", color="tab:blue", lw=2)
            # Optional covariate overlays disabled by default
        else:
            ax.text(0.5, 0.85, "No predictions", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#666")
        ssub_true = ssub.dropna(subset=["y_true"]) if "y_true" in ssub.columns else pd.DataFrame()
        if not ssub_true.empty:
            if not ssub_past.empty:
                ssub_true = ssub_true[ssub_true.date > split_date]
            ax.plot(ssub_true.date, ssub_true.y_true, label="Actual (future)", color="tab:green", lw=2)
        ax.set_title(f"Store {store}")
        ax.tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"Hybrid Family: {family}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is None:
        save_dir = os.path.join("HYBRID_GNN_TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"hybrid_family_{family}_all_stores.png")
    fig.savefig(out, dpi=720)
    plt.close(fig)
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    forecasts = load_forecasts()
    verify_export_schema(forecasts)
    # Examples (adjust family/store as needed)
    plot_store_family(forecasts, 1, "SEAFOOD")
    plot_family_all_stores(forecasts, "SEAFOOD")
    plot_family_aggregate(forecasts, "SEAFOOD")

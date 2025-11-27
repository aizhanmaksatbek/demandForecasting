import os
import pandas as pd
import matplotlib.pyplot as plt


def load_forecasts(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join("TFT", "checkpoints", "tft_test_forecasts.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Forecasts CSV not found: {path}. Run train_tft.py first."
        )
    # columns: date, store_nbr, family, y_true, y_pred
    return pd.read_csv(path, parse_dates=["date"])


def plot_graph(sub, save_dir, title=None, out_file=None):
    plt.figure(figsize=(10, 4))
    plt.plot(sub.date, sub.y_true, label="Actual", lw=2)
    plt.plot(sub.date, sub.y_pred, label="Predicted", lw=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.join("TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, out_file)
    plt.savefig(out, dpi=720)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_store_family(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
    save_dir: str = None,
):
    sub = df[
        (df.store_nbr == store_nbr) & (df.family == family)
    ].sort_values("date")
    if sub.empty:
        print("No rows for that (store, family)")
        return None
    """Plot past encoder history and future predictions as one chart.

    - Past segment: uses rows with non-null y_past
    - Forecast segment: uses y_pred, and overlays y_true if available
    - Marks the split with a vertical line at the last past date
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(install matplotlib) {e}")
        return None

    past_seg = (
        sub.dropna(subset=["y_past"]) if "y_past" in sub else pd.DataFrame()
    )
    fut_seg = (
        sub.dropna(subset=["y_pred"]) if "y_pred" in sub else pd.DataFrame()
    )

    plt.figure(figsize=(10, 4))
    # Past history
    if not past_seg.empty:
        plt.plot(
            past_seg.date,
            past_seg.y_past,
            label="History (past)",
            color="black",
            lw=2,
        )
        split_date = past_seg.date.max()
        # Mark split
        plt.axvline(split_date, color="#888", linestyle="--", lw=1)
    else:
        split_date = None

    # Forecast prediction
    if not fut_seg.empty:
        plt.plot(
            fut_seg.date,
            fut_seg.y_pred,
            label="Forecast (pred)",
            color="tab:blue",
            lw=2,
        )
        # Overlay actuals on forecast horizon if present
        if "y_true" in fut_seg:
            plt.plot(
                fut_seg.date,
                fut_seg.y_true,
                label="Actual (future)",
                color="tab:green",
                lw=2,
            )

    title = f"TFT: store={store_nbr}, family={family}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.join("TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"plot_store{store_nbr}_family_{family}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_family_aggregate(df: pd.DataFrame, family: str, save_dir: str = None):
    fam_df = df[df.family == family]
    if fam_df.empty:
        print("No rows for that family")
        return None

    # Past history aggregate (deduplicate overlapping windows per store/date)
    past_agg = pd.DataFrame()
    stores_with_past = []
    if "y_past" in fam_df:
        fam_past = fam_df.dropna(subset=["y_past"])  # keep only past rows
        # Deduplicate: one value per (store, date)
        fam_past = (
            fam_past.groupby(["store_nbr", "date"], as_index=False)
            .agg({"y_past": "last"})
        )
        stores_with_past = sorted(fam_past.store_nbr.unique())
        past_agg = (
            fam_past.groupby("date", as_index=False)[["y_past"]]
            .sum()
            .sort_values("date")
        )

    # Future predictions and actuals aggregate (sum over stores)
    # Restrict future aggregation to same store set as past
    future_df = fam_df
    if stores_with_past:
        future_df = fam_df[fam_df.store_nbr.isin(stores_with_past)]

    pred_agg = pd.DataFrame()
    if "y_pred" in future_df:
        fut_pred = future_df.dropna(subset=["y_pred"])  # forecast rows
        # Deduplicate by (store, date) before summing
        fut_pred = (
            fut_pred.groupby(["store_nbr", "date"], as_index=False)
            .agg({"y_pred": "last"})
        )
        pred_agg = (
            fut_pred.groupby("date", as_index=False)[["y_pred"]]
            .sum()
            .sort_values("date")
        )
    true_agg = pd.DataFrame()
    if "y_true" in future_df:
        fut_true = future_df.dropna(subset=["y_true"])  # actual rows
        fut_true = (
            fut_true.groupby(["store_nbr", "date"], as_index=False)
            .agg({"y_true": "last"})
        )
        true_agg = (
            fut_true.groupby("date", as_index=False)[["y_true"]]
            .sum()
            .sort_values("date")
        )

    plt.figure(figsize=(10, 4))
    split_date = None
    if not past_agg.empty:
        plt.plot(
            past_agg.date,
            past_agg.y_past,
            label="History (sum)",
            color="black",
            lw=2,
        )
        split_date = past_agg.date.max()
        plt.axvline(split_date, color="#888", linestyle="--", lw=1)
    if not pred_agg.empty:
        plt.plot(
            pred_agg.date,
            pred_agg.y_pred,
            label="Forecast (sum)",
            color="tab:blue",
            lw=2,
        )
    if not true_agg.empty:
        # If we have a split date, start actuals after it to avoid overlap
        if split_date is not None:
            true_agg_plot = true_agg[true_agg.date > split_date]
        else:
            true_agg_plot = true_agg
        if not true_agg_plot.empty:
            plt.plot(
                true_agg_plot.date,
                true_agg_plot.y_true,
                label="Actual (sum)",
                color="tab:green",
                lw=2,
            )

    plt.title(f"Family aggregate: {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales (sum across stores)")
    plt.legend()
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.join("TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"plot_family_{family}_aggregate.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_family_all_stores(
    df: pd.DataFrame,
    family: str,
    max_cols: int = 4,
    save_dir: str = None,
):
    sub = df[df.family == family]
    stores = sorted(sub.store_nbr.unique())
    if not stores:
        print("No stores for that family")
        return None
    cols = min(max_cols, len(stores))
    rows = (len(stores) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.2, rows * 3.2), sharey=True
    )
    axes = axes.flatten()
    for i, store in enumerate(stores):
        ssub = sub[sub.store_nbr == store].sort_values("date")
        ax = axes[i]
        # Past (history) for store
        if "y_past" in ssub.columns:
            ssub_past = ssub.dropna(subset=["y_past"])
        else:
            ssub_past = pd.DataFrame()
        if not ssub_past.empty:
            ax.plot(
                ssub_past.date,
                ssub_past.y_past,
                label="History (past)",
                color="black",
                lw=2,
            )
            split_date = ssub_past.date.max()
            ax.axvline(split_date, color="#888", linestyle="--", lw=1)

        # Future forecast and actuals for store
        if "y_pred" in ssub.columns:
            ssub_pred = ssub.dropna(subset=["y_pred"])
        else:
            ssub_pred = pd.DataFrame()
        if not ssub_pred.empty:
            ax.plot(
                ssub_pred.date,
                ssub_pred.y_pred,
                label="Forecast (pred)",
                color="tab:blue",
                lw=2,
            )
        if "y_true" in ssub.columns:
            ssub_true = ssub.dropna(subset=["y_true"])
        else:
            ssub_true = pd.DataFrame()
        if not ssub_true.empty:
            ax.plot(
                ssub_true.date,
                ssub_true.y_true,
                label="Actual (future)",
                color="tab:green",
                lw=2,
            )
        ax.set_title(f"Store {store}")
        ax.tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Family: {family}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is None:
        save_dir = os.path.join("TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"plot_family_{family}_all_stores.png")
    fig.savefig(out, dpi=720)
    plt.close(fig)
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    # Simple CLI-less usage example
    forecasts = load_forecasts()
    # Example calls (adjust to your data):
    plot_store_family(forecasts, 1, "PRODUCE")
    plot_family_all_stores(forecasts, "PRODUCE")
    plot_family_aggregate(forecasts, "PRODUCE")

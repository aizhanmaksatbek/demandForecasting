import os
import pandas as pd


def load_forecasts(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join("TFT", "checkpoints", "tft_test_forecasts.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Forecasts CSV not found: {path}. Run train_tft.py first."
        )
    # columns: date, store_nbr, family, y_true, y_pred
    return pd.read_csv(path, parse_dates=["date"])


def plot_store_family(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
    save_dir: str = None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(install matplotlib) {e}")
        return None

    sub = df[
        (df.store_nbr == store_nbr) & (df.family == family)
    ].sort_values("date")
    if sub.empty:
        print("No rows for that (store, family)")
        return None

    plt.figure(figsize=(10, 4))
    plt.plot(sub.date, sub.y_true, label="Actual", lw=2)
    plt.plot(sub.date, sub.y_pred, label="Predicted", lw=2)
    plt.title(f"TFT Test Forecast (store={store_nbr}, family={family})")
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


def plot_family_all_stores(
    df: pd.DataFrame,
    family: str,
    max_cols: int = 4,
    save_dir: str = None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(install matplotlib) {e}")
        return None

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
        ax.plot(ssub.date, ssub.y_true, label="Actual", color="black")
        ax.plot(ssub.date, ssub.y_pred, label="Pred", color="tab:blue")
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
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_family_aggregate(df: pd.DataFrame, family: str, save_dir: str = None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(install matplotlib) {e}")
        return None

    sub = (
        df[df.family == family]
        .groupby("date", as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values("date")
    )
    if sub.empty:
        print("No rows for that family")
        return None

    plt.figure(figsize=(10, 4))
    plt.plot(sub.date, sub.y_true, label="Actual (sum)", lw=2)
    plt.plot(sub.date, sub.y_pred, label="Predicted (sum)", lw=2)
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


if __name__ == "__main__":
    # Simple CLI-less usage example
    forecasts = load_forecasts()
    # Example calls (adjust to your data):
    # plot_store_family(forecasts, 1, "AUTOMOTIVE")
    plot_family_all_stores(forecasts, "BOOKS")
    # plot_family_aggregate(forecasts, "BOOKS")
    pass


# # Quick example plot (first family/store) if matplotlib available
# try:
#     import matplotlib.pyplot as plt
#     first = test_forecasts_df.iloc[0]
#     fam0 = first.family
#     store0 = first.store_nbr
#     subset = test_forecasts_df[
#         (test_forecasts_df.family == fam0)
#         & (test_forecasts_df.store_nbr == store0)
#     ]
#     plt.figure(figsize=(10, 4))
#     plt.plot(subset.date, subset.y_true, label="Actual", lw=2)
#     plt.plot(subset.date, subset.y_pred, label="Pred", lw=2)
#     plt.title(
#         f"TFT Test Forecast (store={store0}, family={fam0})"
#     )
#     plt.xlabel("Date")
#     plt.ylabel("Sales")
#     plt.legend()
#     plt.tight_layout()
#     out_png = os.path.join(
#         "TFT",
#         "checkpoints",
#         f"example_plot_store{store0}_family_{fam0}.png",
#     )
#     plt.savefig(out_png, dpi=150)
#     plt.close()
#     print(f"Saved example plot -> {out_png}")
# except Exception as e:
#     print(f"(skip example plot) {e}")

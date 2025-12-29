import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def load_lime_csv(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(
            "HYBRID_GNN_TFT", "checkpoints", "gnn_tft_lime_explanation.csv"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LIME CSV not found: {path}. Run HYBRID_GNN_TFT/xai_lime.py first."
        )
    df = pd.read_csv(path)
    expected = ["feature", "weight"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    return df[expected]


def _parse_feature_name(name: str):
    """Return (category, time_idx, var_name) for feature string.
    Examples:
      static.12 -> ("static", None, "static")
      past[7].sales -> ("past", 7, "sales")
      future[3].month -> ("future", 3, "month")
    """
    if name.startswith("static"):
        return ("static", None, "static")
    m = re.match(r"^(past|future)\[(\d+)\]\.(.+)$", name)
    if m:
        grp = m.group(1)
        t = int(m.group(2))
        var = m.group(3)
        return (grp, t, var)
    return ("unknown", None, name)


def plot_top_features(df: pd.DataFrame, top_n: int = 25, save_dir: str = None):
    df_top = df.sort_values("weight", ascending=False).head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * top_n)))
    plt.barh(df_top["feature"][::-1], df_top["weight"][::-1], color="tab:blue")
    plt.xlabel("Contribution (weight)")
    plt.title(f"LIME Top {top_n} Feature Contributions")
    plt.tight_layout()
    if save_dir is None:
        save_dir = os.path.join("HYBRID_GNN_TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"lime_top_{top_n}_features.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return out


def plot_aggregated_variables(df: pd.DataFrame, save_dir: str = None):
    # Sum absolute weights by variable (across time indices), grouped by category
    rows = []
    for _, r in df.iterrows():
        cat, t, var = _parse_feature_name(r["feature"])
        rows.append({"category": cat, "var": var, "abs_weight": abs(float(r["weight"]))})
    agg = (
        pd.DataFrame(rows)
        .groupby(["category", "var"], as_index=False)["abs_weight"]
        .sum()
        .sort_values(["category", "abs_weight"], ascending=[True, False])
    )

    # Plot per category
    if save_dir is None:
        save_dir = os.path.join("HYBRID_GNN_TFT", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    for cat in ["static", "past", "future", "unknown"]:
        sub = agg[agg.category == cat]
        if sub.empty:
            continue
        plt.figure(figsize=(10, max(4, 0.4 * len(sub))))
        # Use column indexing to avoid colliding with DataFrame.var method
        plt.barh(sub["var"][::-1], sub["abs_weight"][::-1], color="tab:orange")
        plt.xlabel("Sum |weight|")
        plt.title(f"LIME Aggregated Contributions — {cat}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"lime_aggregated_{cat}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    df_lime = load_lime_csv()
    plot_top_features(df_lime, top_n=25)
    plot_aggregated_variables(df_lime)

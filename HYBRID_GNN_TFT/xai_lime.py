import argparse
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import torch
from lime.lime_tabular import LimeTabularExplainer
from HYBRID_GNN_TFT.hybrid_model import HybridGNNTFT, normalize_adjacency
from HYBRID_GNN_TFT.hybrid_utils import (
    build_node_indexer,
    build_static_node_features,
    build_product_graph_adjacency,
)
from TFT.train_tft import get_data_split
from config.settings import (
    ENC_VARS,
    DEC_VARS,
    STATIC_COLS,
    TFT_DATA_DIR,
    HYBRID_CHECKPOINTS_PATH,
)
from utils.utils import set_seed, get_date_splits, build_onehot_maps


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


def _flatten_sample(past_np: np.ndarray,
                    future_np: np.ndarray,
                    static_np: np.ndarray) -> np.ndarray:
    """Flatten tensors into a single 1D feature vector."""
    return np.concatenate([
        static_np.reshape(-1),
        past_np.reshape(-1),
        future_np.reshape(-1),
    ], axis=0)


def _unflatten_sample(vec: np.ndarray,
                      enc_len: int,
                      dec_len: int,
                      n_enc_vars: int,
                      n_dec_vars: int,
                      n_static: int,
                      device: torch.device):
    """Inverse of flatten: return torch tensors for model input."""
    s0 = n_static
    s1 = s0 + (enc_len * n_enc_vars)
    s2 = s1 + (dec_len * n_dec_vars)
    static = torch.from_numpy(vec[:s0].astype(np.float32)).to(device)
    past = torch.from_numpy(
        vec[s0:s1].astype(np.float32).reshape(enc_len, n_enc_vars)
    ).to(device)
    future = torch.from_numpy(
        vec[s1:s2].astype(np.float32).reshape(dec_len, n_dec_vars)
    ).to(device)
    return past, future, static


def build_feature_names(enc_len: int,
                        dec_len: int,
                        n_static: int) -> list:
    names = []
    names += [f"static.{i}" for i in range(n_static)]
    for t in range(enc_len):
        for j, v in enumerate(ENC_VARS):
            names.append(f"past[{t}].{v}")
    for t in range(dec_len):
        for j, v in enumerate(DEC_VARS):
            names.append(f"future[{t}].{v}")
    return names


def explain_hybrid_with_lime(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    panel_path = os.path.join(TFT_DATA_DIR, "panel.csv")
    assert os.path.exists(panel_path), (
        "Run data preprocessing first: python src/data/preprocess_favorita.py"
    )
    df = pd.read_csv(panel_path, parse_dates=["date"])
    train_end, val_end, test_end = get_date_splits(df, args.dec_len)
    static_maps = build_onehot_maps(df, STATIC_COLS)
    static_dims = [len(static_maps[c]) for c in STATIC_COLS]
    static_dim_total = int(np.sum(static_dims))

    # Data loaders
    (
        _,
        _,
        test_loader,
        _static_dims,
        _train_len,
        _val_len,
        _test_len,
    ) = get_data_split(
        args.dec_len, args.enc_len, args.batch_size, args.stride
    )

    # Graph pieces
    indexer = build_node_indexer(df[["store_nbr", "family"]].drop_duplicates())
    X_nodes = build_static_node_features(df, indexer, STATIC_COLS, static_maps)
    A = build_product_graph_adjacency(
        df, indexer, train_end=train_end,
        top_k=args.top_k, min_corr=args.min_corr
    )
    A = A.to(device)
    X_nodes = X_nodes.to(device)
    A_norm = normalize_adjacency(A)

    # Model and checkpoint
    quantiles = [float(x) for x in args.quantiles.split(",")]
    model = HybridGNNTFT(
        static_input_dims=[static_dim_total + args.gnn_embed],
        past_input_dims=[1] * len(ENC_VARS),
        future_input_dims=[1] * len(DEC_VARS),
        tft_d_model=args.d_model,
        tft_hidden_dim=args.hidden_dim,
        tft_heads=args.heads,
        tft_lstm_hidden=args.lstm_hidden,
        tft_lstm_layers=args.lstm_layers,
        tft_dropout=args.dropout,
        tft_num_quantiles=len(quantiles),
        gnn_in_dim=static_dim_total,
        gnn_hidden=args.gnn_hidden,
        gnn_out_dim=args.gnn_embed,
        gnn_dropout=0.1,
    ).to(device)

    best_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "gnn_tft_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best hybrid model from {best_path}")
    model.eval()

    # Collect background samples for LIME
    background = []
    metas_ref = []
    enc_len = args.enc_len
    dec_len = args.dec_len
    # Expected static length = TFT static dims minus GNN embedding
    s_expected = int(model.tft.static_input_dims[0] - model.gnn_out_dim)
    for batch in test_loader:
        past = batch["past_inputs"].to(device)   # [B, Lenc, E]
        future = batch["future_inputs"].to(device)  # [B, Ldec, D]
        static = batch["static_inputs"].to(device)  # [B, S]
        metas = batch.get("meta", [])
        for i in range(past.shape[0]):
            s_vec = static[i]
            if s_vec.shape[-1] > s_expected:
                s_vec = s_vec[:s_expected]
            elif s_vec.shape[-1] < s_expected:
                pad = np.zeros(
                    (s_expected - s_vec.shape[-1],), dtype=s_vec.dtype
                    )
                s_vec = np.concatenate([s_vec, pad], axis=0)
            background.append(_flatten_sample(past[i], future[i], s_vec))
            metas_ref.append(metas[i] if i < len(metas) else None)
            if len(background) >= args.lime_background:
                break
        if len(background) >= args.lime_background:
            break
    background = np.array(background, dtype=np.float32)

    # Feature names
    feature_names = build_feature_names(enc_len, dec_len, s_expected)

    # Pick a single sample to explain (first one by default)
    target_idx = int(args.sample_index)
    # Recreate tensors for node id and meta
    # Use node id from metas_ref[target_idx] if available
    meta = metas_ref[target_idx]
    if meta is None:
        raise RuntimeError("No metadata available to derive node id for sample.")
    node_id = torch.tensor(
        [indexer.id((meta["store_nbr"], meta["family"]))],
        dtype=torch.long, device=device
    )

    n_static = s_expected

    # Predict function for LIME (regression on median quantile at a step)
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    step_to_explain = int(args.dec_step)

    def predict_fn(X: np.ndarray):
        preds = []
        for vec in X:
            past_t, future_t, static_t = _unflatten_sample(
                vec, enc_len, dec_len,
                len(ENC_VARS), len(DEC_VARS),
                n_static, device
            )
            past_t = past_t.unsqueeze(0)    # [1, Lenc, E]
            future_t = future_t.unsqueeze(0)  # [1, Ldec, D]
            static_t = static_t.unsqueeze(0)  # [1, S]
            out = model(
                past_t, future_t, static_t,
                node_ids=node_id, A_norm=A_norm, X_nodes=X_nodes,
                return_attention=False,
            )
            yhat = out["prediction"][0, step_to_explain, median_idx].detach().cpu().item()
            preds.append(yhat)
        return np.array(preds, dtype=np.float32)

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False,
        verbose=False,
    )

    instance = background[target_idx]
    explanation = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=int(args.num_features),
        num_samples=int(args.num_samples),
    )

    contribs = explanation.as_list()
    df_exp = pd.DataFrame(contribs, columns=["feature", "weight"]).sort_values(
        "weight", ascending=False
    )
    out_csv = os.path.join(HYBRID_CHECKPOINTS_PATH, "gnn_tft_lime_explanation.csv")
    df_exp.to_csv(out_csv, index=False)
    print(f"Saved LIME explanation -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    # Data/model settings (align with training defaults)
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--dec-len", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.2)
    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--gnn-embed", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    # LIME settings
    parser.add_argument("--lime-background", type=int, default=200,
                        help="Number of background samples to fit LIME")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index into background array to explain")
    parser.add_argument("--dec-step", type=int, default=27,
                        help="Decoder step (0-based) to explain")
    parser.add_argument("--num-features", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=1000)

    args = parser.parse_args()
    explain_hybrid_with_lime(args)

    # Plot results
    df_lime = load_lime_csv()
    plot_top_features(df_lime, top_n=25)
    plot_aggregated_variables(df_lime)


if __name__ == "__main__":
    main()

import argparse
import os
import random
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import TFT dataset and architecture
from TFT.tft_dataset import TFTWindowDataset, tft_collate
from TFT.architecture.tft import QuantileLoss

# Make local package imports work when running as a script
from hybrid_model import HybridGNNTFT, normalize_adjacency
from utils import (
    build_node_indexer,
    build_static_node_features,
    build_product_graph_adjacency,
)
from plot_hybrid_results import (
    load_forecasts as load_hybrid_forecasts,
    plot_store_family as plot_hybrid_store_family,
    plot_family_all_stores as plot_hybrid_family_all_stores,
    plot_family_aggregate as plot_hybrid_family_aggregate,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum() + 1e-8
    return float(np.abs(y_true - y_pred).sum() / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    return float((2.0 * np.abs(y_true - y_pred) / denom).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--dec-len", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.2)
    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--gnn-embed", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join("HYBRID_GNN_TFT", "checkpoints"), exist_ok=True)

    # Load TFT panel
    panel_path = os.path.join("TFT", "data", "panel.csv")
    assert os.path.exists(panel_path), (
        "Run data preprocessing first: python src/data/preprocess_favorita.py"
    )
    df = pd.read_csv(panel_path, parse_dates=["date"])

    # Variables as in TFT baseline
    enc_vars = [
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
    dec_vars = [
        "onpromotion",
        "dow",
        "month",
        "weekofyear",
        "is_holiday",
        "is_workday",
    ]
    static_cols = ["store_nbr", "family", "state", "cluster"]

    # Define splits
    max_date = df["date"].max()
    test_days = pd.Timedelta(days=args.dec_len)
    val_days = pd.Timedelta(days=args.dec_len)
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days

    # Build one-hot maps for static features
    def build_onehot_maps(_df: pd.DataFrame, cols: List[str]):
        maps = {}
        for c in cols:
            cats = sorted(_df[c].dropna().unique().tolist())
            idx = {v: i for i, v in enumerate(cats)}
            dim = len(cats)
            maps[c] = {}
            eye = np.eye(dim, dtype=np.float32)
            for v in cats:
                maps[c][v] = eye[idx[v]]
        return maps

    static_maps = build_onehot_maps(df, static_cols)
    static_dims = [len(static_maps[c]) for c in static_cols]
    static_dim_total = int(np.sum(static_dims))

    # Datasets / loaders
    split_bounds = (train_end, val_end, test_end)
    train_ds = TFTWindowDataset(
        df,
        args.enc_len,
        args.dec_len,
        enc_vars,
        dec_vars,
        static_cols,
        split_bounds,
        split="train",
        stride=args.stride,
        static_onehot_maps=static_maps,
    )
    val_ds = TFTWindowDataset(
        df,
        args.enc_len,
        args.dec_len,
        enc_vars,
        dec_vars,
        static_cols,
        split_bounds,
        split="val",
        stride=args.stride,
        static_onehot_maps=static_maps,
    )
    test_ds = TFTWindowDataset(
        df,
        args.enc_len,
        args.dec_len,
        enc_vars,
        dec_vars,
        static_cols,
        split_bounds,
        split="test",
        stride=args.stride,
        static_onehot_maps=static_maps,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=tft_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=tft_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=tft_collate,
    )

    print(
        f"Train samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    # Build graph indexer and tensors
    indexer = build_node_indexer(df[["store_nbr", "family"]].drop_duplicates())
    X_nodes = build_static_node_features(df, indexer, static_cols, static_maps)
    A = build_product_graph_adjacency(
        df, indexer, train_end=train_end, top_k=args.top_k, min_corr=args.min_corr
    )
    # Normalize adjacency once
    A = A.to(device)
    X_nodes = X_nodes.to(device)
    A_norm = normalize_adjacency(A)

    # Model
    quantiles = [float(x) for x in args.quantiles.split(",")]
    model = HybridGNNTFT(
        static_input_dims=[static_dim_total + args.gnn_embed],
        past_input_dims=[1] * len(enc_vars),
        future_input_dims=[1] * len(dec_vars),
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

    criterion = QuantileLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = float("inf")
    best_path = os.path.join("HYBRID_GNN_TFT", "checkpoints", "gnn_tft_best.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            past = batch["past_inputs"].to(device)
            future = batch["future_inputs"].to(device)
            static = batch["static_inputs"].to(device)
            metas = batch.get("meta", [])
            node_ids = torch.tensor(
                [indexer.id((m["store_nbr"], m["family"])) for m in metas],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad()
            out = model(
                past,
                future,
                static,
                node_ids=node_ids,
                A_norm=A_norm,
                X_nodes=X_nodes,
                return_attention=False,
            )
            y = batch["target"].to(device)
            loss = criterion(out["prediction"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * past.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss /= max(len(train_ds), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                past = batch["past_inputs"].to(device)
                future = batch["future_inputs"].to(device)
                static = batch["static_inputs"].to(device)
                metas = batch.get("meta", [])
                node_ids = torch.tensor(
                    [indexer.id((m["store_nbr"], m["family"])) for m in metas],
                    dtype=torch.long,
                    device=device,
                )
                out = model(
                    past,
                    future,
                    static,
                    node_ids=node_ids,
                    A_norm=A_norm,
                    X_nodes=X_nodes,
                    return_attention=False,
                )
                y = batch["target"].to(device)
                val_loss += criterion(out["prediction"], y).item() * past.size(0)
        val_loss /= max(len(val_ds), 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": vars(args),
                    "quantiles": quantiles,
                },
                best_path,
            )
            print(f"Saved best model to {best_path}")

    # Load best and evaluate
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (val pinball: {best_val:.5f})")
    model.eval()

    def eval_loader(dloader):
        total_loss = 0.0
        ys, preds = [], []
        with torch.no_grad():
            for batch in dloader:
                past = batch["past_inputs"].to(device)
                future = batch["future_inputs"].to(device)
                static = batch["static_inputs"].to(device)
                metas = batch.get("meta", [])
                node_ids = torch.tensor(
                    [indexer.id((m["store_nbr"], m["family"])) for m in metas],
                    dtype=torch.long,
                    device=device,
                )
                out = model(
                    past,
                    future,
                    static,
                    node_ids=node_ids,
                    A_norm=A_norm,
                    X_nodes=X_nodes,
                    return_attention=False,
                )
                y = batch["target"].to(device)
                total_loss += (
                    QuantileLoss(quantiles=quantiles)(out["prediction"], y).item()
                    * past.size(0)
                )
                median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
                yhat = out["prediction"][..., median_idx]
                ys.append(y.detach().cpu().numpy())
                preds.append(yhat.detach().cpu().numpy())
        pinball = total_loss / max(len(dloader.dataset), 1)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        return pinball, wape(ys, preds), smape(ys, preds)

    val_pinball, val_wape, val_smape = eval_loader(val_loader)
    test_pinball, test_wape, test_smape = eval_loader(test_loader)
    print(
        f"Validation  - Pinball: {val_pinball:.5f} | WAPE: {val_wape:.5f} | sMAPE: {val_smape:.5f}"
    )
    print(
        f"Test        - Pinball: {test_pinball:.5f} | WAPE: {test_wape:.5f} | sMAPE: {test_smape:.5f}"
    )

    # Export test forecasts (median) for plotting
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            past = batch["past_inputs"].to(device)
            future = batch["future_inputs"].to(device)
            static = batch["static_inputs"].to(device)
            metas = batch.get("meta", [])
            node_ids = torch.tensor(
                [indexer.id((m["store_nbr"], m["family"])) for m in metas],
                dtype=torch.long,
                device=device,
            )
            out = model(
                past,
                future,
                static,
                node_ids=node_ids,
                A_norm=A_norm,
                X_nodes=X_nodes,
                return_attention=False,
            )
            preds_med = out["prediction"][..., median_idx].cpu().numpy()
            targets = batch["target"].cpu().numpy()
            for i, meta in enumerate(metas):
                store_nbr = meta["store_nbr"]
                family = meta["family"]
                fut_dates = meta["future_dates"]
                for d_idx, date in enumerate(fut_dates):
                    rows.append(
                        {
                            "date": pd.to_datetime(date),
                            "store_nbr": store_nbr,
                            "family": family,
                            "y_true": float(targets[i, d_idx]),
                            "y_pred": float(preds_med[i, d_idx]),
                        }
                    )
                # Append encoder history (past sales) before forecast horizon
                # Use the 'sales' feature from encoder inputs (assumed raw units)
                sales_idx = enc_vars.index("sales")
                past_dates = meta["past_dates"]
                for d_idx, date in enumerate(past_dates):
                    rows.append(
                        {
                            "date": pd.to_datetime(date),
                            "store_nbr": store_nbr,
                            "family": family,
                            "y_past": float(past[i, d_idx, sales_idx].cpu()),
                        }
                    )
    if rows:
        out_csv = os.path.join(
            "HYBRID_GNN_TFT", "checkpoints", "gnn_tft_test_forecasts.csv"
        )
        df_out = pd.DataFrame(rows).sort_values(["family", "store_nbr", "date"])
        # Enforce TFT-like schema and column order
        for col in ["y_true", "y_pred", "y_past"]:
            if col not in df_out.columns:
                df_out[col] = np.nan
        df_out = df_out[["date", "store_nbr", "family", "y_true", "y_pred", "y_past"]]
        df_out.to_csv(out_csv, index=False)
        print(f"Saved hybrid test forecasts -> {out_csv}")
        # Quick summary counts akin to TFT workflow
        c_past = int(df_out["y_past"].notna().sum())
        c_pred = int(df_out["y_pred"].notna().sum())
        c_true = int(df_out["y_true"].notna().sum())
        print(
            f"Rows -> y_past: {c_past} | y_pred: {c_pred} | y_true: {c_true}"
        )
        # Optional: generate a couple of example plots like TFT
        try:
            forecasts_df = load_hybrid_forecasts(out_csv)
            # Pick a frequent (family, store) for demo
            grp_counts = (
                forecasts_df.groupby(["family", "store_nbr"]).size().sort_values(ascending=False)
            )
            if not grp_counts.empty:
                fam, store = grp_counts.index[0]
                plot_hybrid_store_family(forecasts_df, int(store), str(fam))
                plot_hybrid_family_all_stores(forecasts_df, str(fam))
                plot_hybrid_family_aggregate(forecasts_df, str(fam))
        except Exception as e:
            print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()

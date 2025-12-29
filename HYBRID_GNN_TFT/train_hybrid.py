import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from TFT.architecture.tft import QuantileLoss
from HYBRID_GNN_TFT.hybrid_model import HybridGNNTFT, normalize_adjacency
from HYBRID_GNN_TFT.hybrid_utils import (
    build_node_indexer,
    build_static_node_features,
    build_product_graph_adjacency
)
from config.settings import ENC_VARS, DEC_VARS, STATIC_COLS
from config.settings import TFT_DATA_DIR, HYBRID_CHECKPOINTS_PATH
from utils.utils import compute_metrics
from utils.utils import set_seed, get_date_splits
from utils.utils import build_onehot_maps
from utils.utils import TensorboardConfig
from TFT.train_tft import get_data_split


def train_hybrid_model(args, model, train_loader, val_loader, df, train_end,
                       static_maps):
    quantiles = [float(x) for x in args.quantiles.split(",")]
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    criterion = QuantileLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
        )
    tensorboard_writer = TensorboardConfig(args.tensorboard, args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(HYBRID_CHECKPOINTS_PATH), exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "gnn_tft_best.pt")
    patience = int(getattr(args, "early_stopping_patience", 5))
    no_improve_epochs = 0

    model = model.to(device)

    indexer = build_node_indexer(df[["store_nbr", "family"]].drop_duplicates())
    X_nodes = build_static_node_features(df, indexer, STATIC_COLS, static_maps)
    A = build_product_graph_adjacency(
        df, indexer, train_end=train_end,
        top_k=args.top_k, min_corr=args.min_corr
    )
    # Normalize adjacency once
    A = A.to(device)
    X_nodes = X_nodes.to(device)
    A_norm = normalize_adjacency(A)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_t, train_pred = [], []
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
            loss = criterion(out["prediction"].to(device), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * past.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            train_t.append(y.detach().cpu().numpy())
            yhat = out["prediction"][..., median_idx]
            train_pred.append(yhat.detach().cpu().numpy())
        train_loss /= max(len(train_loader.dataset), 1)
        train_metrics = compute_metrics(train_t, train_pred)
        print(f"Epoch {epoch} Train Metrics: {train_loss:.5f} \
              | {train_metrics}")
        tensorboard_writer.write("loss/train", train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_t, val_pred = [], []
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
                val_loss += (criterion(out["prediction"].to(device), y)
                             .item() * past.size(0))
                val_t.append(y.detach().cpu().numpy())
                yhat = out["prediction"][..., median_idx]
                val_pred.append(yhat.detach().cpu().numpy())
            val_loss /= max(len(val_loader.dataset), 1)
            val_metrics = compute_metrics(val_t, val_pred)
            tensorboard_writer.write("loss/val", val_loss, epoch)
            print(f"Validation Epoch {epoch} loss {val_loss:.5f} \
                  | {val_metrics}")

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
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch})"
                )
                break
    tensorboard_writer.close()


def eval_hybrid_model(model, args, test_loader, df, train_end, static_maps):
    quantiles = [float(x) for x in args.quantiles.split(",")]
    criterion = QuantileLoss(quantiles=quantiles)
    best_path = os.path.join(HYBRID_CHECKPOINTS_PATH, "gnn_tft_best.pt")
    quantiles = [float(x) for x in args.quantiles.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indexer = build_node_indexer(df[["store_nbr", "family"]].drop_duplicates())
    X_nodes = build_static_node_features(df, indexer, STATIC_COLS, static_maps)
    A = build_product_graph_adjacency(
        df, indexer, train_end=train_end,
        top_k=args.top_k, min_corr=args.min_corr
    )
    # Normalize adjacency once
    A = A.to(device)
    X_nodes = X_nodes.to(device)
    A_norm = normalize_adjacency(A)

    # Load best and evaluate
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (val pinball: {best_path})")
    model = model.to(device)
    model.eval()

    # Export test forecasts (median) for plotting
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    rows = []
    with torch.no_grad():
        total_loss = 0.0
        test_t, test_pred = [], []
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
            # preds_med = out["prediction"][..., median_idx]
            targets = batch["target"]
            test_t.append(targets.detach().cpu().numpy())
            yhat = out["prediction"][..., median_idx]
            test_pred.append(yhat.detach().cpu().numpy())
            total_loss += (criterion
                           (out["prediction"].to(device), targets.to(device)
                            ).item() * past.size(0))
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
                            "y_pred": float(yhat[i, d_idx]),
                        }
                    )
                # Append encoder history (past sales) before forecast horizon
                # Use the 'sales' feature from encoder inputs
                sales_idx = ENC_VARS.index("sales")
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
        total_loss /= max(len(test_loader.dataset), 1)
        test_metrics = compute_metrics(test_t, test_pred)
        print(f"Total test loss for export: {total_loss:.5f} \
              | {test_metrics}")
    if rows:
        out_csv = os.path.join(
            "HYBRID_GNN_TFT", "checkpoints", "gnn_tft_test_forecasts.csv"
        )
        df_out = pd.DataFrame(rows).sort_values(
            ["family", "store_nbr", "date"]
            )
        # Enforce TFT-like schema and column order
        for col in ["y_true", "y_pred", "y_past"]:
            if col not in df_out.columns:
                df_out[col] = np.nan
        df_out = df_out[["date", "store_nbr", "family",
                         "y_true", "y_pred", "y_past"]]
        df_out.to_csv(out_csv, index=False)
        print(f"Saved hybrid test forecasts -> {out_csv}")
        # Quick summary counts akin to TFT workflow
        c_past = int(df_out["y_past"].notna().sum())
        c_pred = int(df_out["y_pred"].notna().sum())
        c_true = int(df_out["y_true"].notna().sum())
        print(
            f"Rows -> y_past: {c_past} | y_pred: {c_pred} | y_true: {c_true}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--dec-len", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging"
                        )
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join("HYBRID_GNN_TFT", "logs"),
                        help="Directory to store TensorBoard logs"
                        )
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Stop if no val loss improvement for N epochs"
                        )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(HYBRID_CHECKPOINTS_PATH), exist_ok=True)

    # Load TFT panel
    panel_path = os.path.join(TFT_DATA_DIR, "panel.csv")
    assert os.path.exists(panel_path), (
        "Run data preprocessing first: python src/data/preprocess_favorita.py"
    )
    df = pd.read_csv(panel_path, parse_dates=["date"])

    # Define splits
    train_end, val_end, test_end = get_date_splits(df, args.dec_len)

    static_maps = build_onehot_maps(df, STATIC_COLS)
    static_dims = [len(static_maps[c]) for c in STATIC_COLS]
    static_dim_total = int(np.sum(static_dims))

    # Data loaders
    (
        train_loader,
        val_loader,
        test_loader,
        static_dims,
        train_len,
        val_len,
        test_len,
    ) = get_data_split(
        args.dec_len,
        args.enc_len,
        args.batch_size,
        args.stride
        )

    # Model
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

    train_hybrid_model(args, model, train_loader, val_loader, df, train_end,
                       static_maps)
    eval_hybrid_model(model, args, test_loader, df, train_end, static_maps)


if __name__ == "__main__":
    main()

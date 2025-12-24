import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from TFT.architecture.tft import TemporalFusionTransformer, QuantileLoss
from TFT.tft_dataset import TFTWindowDataset, tft_collate
from utils.utils import set_seed, build_onehot_maps
from utils.utils import compute_metrics, write_metrics_to_tensorboard
from config.settings import ENC_VARS, DEC_VARS, STATIC_COLS, REALS_TO_SCALE
from config.settings import TFT_CHECKPOINTS_DIR, TFT_DATA_DIR, WORKING_DIR
from utils.utils import TensorboardConfig, get_date_splits


def save_results_csv(rows):
    if rows:
        test_forecasts_df = (
            pd.DataFrame(rows)
            .sort_values(["family", "store_nbr", "date"])
        )
        out_csv = os.path.join(
            TFT_CHECKPOINTS_DIR, "tft_test_forecasts.csv"
        )
        test_forecasts_df.to_csv(out_csv, index=False)
        print(f"Saved test forecasts CSV -> {out_csv}")


def get_data_split(dec_len, enc_len, batch_size, stride):
    # Load data
    panel_path = os.path.join(TFT_DATA_DIR, "panel.csv")
    assert os.path.exists(panel_path), (
        "Run data preprocessing first: "
        "python src/data/preprocess_favorita.py"
    )
    df = pd.read_csv(panel_path, parse_dates=["date"])

    # Scale continuous features (fit on train period only)
    train_end, val_end, test_end = get_date_splits(df, dec_len)

    scaler = StandardScaler()
    train_mask = df["date"] <= train_end
    df.loc[train_mask, REALS_TO_SCALE] = scaler.fit_transform(
        df.loc[train_mask, REALS_TO_SCALE]
    )
    df.loc[~train_mask, REALS_TO_SCALE] = scaler.transform(
        df.loc[~train_mask, REALS_TO_SCALE]
    )

    # One-hot maps for static features
    static_maps = build_onehot_maps(df, STATIC_COLS)
    static_dims = [len(static_maps[c]) for c in STATIC_COLS]

    # Dataset and loaders
    split_bounds = (train_end, val_end, test_end)
    train_ds = TFTWindowDataset(
        df, enc_len, dec_len, ENC_VARS, DEC_VARS, STATIC_COLS,
        split_bounds, split="train", stride=stride,
        static_onehot_maps=static_maps,
    )
    val_ds = TFTWindowDataset(
        df, enc_len, dec_len, ENC_VARS, DEC_VARS, STATIC_COLS,
        split_bounds, split="val", stride=stride,
        static_onehot_maps=static_maps,
    )
    test_ds = TFTWindowDataset(
        df, enc_len, dec_len, ENC_VARS, DEC_VARS, STATIC_COLS,
        split_bounds, split="test", stride=stride,
        static_onehot_maps=static_maps,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )
    print(
        f"Train samples: {len(train_ds)} | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return (train_loader, val_loader, test_loader,
            static_dims,
            len(train_ds), len(val_ds), len(test_ds)
            )


def train_model(model, quantiles, args, train_loader, val_loader,
                train_len, val_len
                ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensorboard_writer = TensorboardConfig(
        store_flag=args.tensorboard, log_dir=args.log_dir
    )
    criterion = QuantileLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    best_val = float("inf")
    best_path = os.path.join(TFT_CHECKPOINTS_DIR, "tft_best.pt")
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    # Early stopping state
    patience = int(getattr(args, "early_stopping_patience", 7))
    min_delta = float(getattr(args, "early_stopping_min_delta", 0.0))
    no_improve_epochs = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_ys, train_preds = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            optimizer.zero_grad()

            past = batch["past_inputs"].to(device)     # [B, L_enc, E]
            future = batch["future_inputs"].to(device)  # [B, L_dec, D]
            static = batch["static_inputs"].to(device)  # [B, S]
            y = batch["target"].to(device)             # [B, L_dec]

            out = model(past, future, static)
            loss = criterion(out["prediction"].to(device), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * past.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            yhat = out["prediction"][..., median_idx]
            train_ys.append(y.detach().cpu().numpy())
            train_preds.append(yhat.detach().cpu().numpy())

        train_loss /= max(train_len, 1)

        metrics_train = compute_metrics(train_ys, train_preds)
        write_metrics_to_tensorboard(tensorboard_writer, metrics_train, epoch)
        tensorboard_writer.write("loss_train", train_loss, epoch)
        print(f"Epoch {epoch} Train Loss: {train_loss:.6f}, \
              Train Metrics: {metrics_train}"
              )

        # Validation
        model.eval()
        val_loss = 0.0
        valid_ys, valid_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                past = batch["past_inputs"].to(device)
                future = batch["future_inputs"].to(device)
                static = batch["static_inputs"].to(device)
                y = batch["target"].to(device)
                out = model(past, future, static)
                loss = criterion(out["prediction"].to(device), y)
                val_loss += loss.item() * past.size(0)

                yhat = out["prediction"][..., median_idx]
                valid_ys.append(y.detach().cpu().numpy())
                valid_preds.append(yhat.detach().cpu().numpy())

        val_loss /= max(val_len, 1)
        tensorboard_writer.write("loss_val", val_loss, epoch)
        metrics_val = compute_metrics(valid_ys, valid_preds)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.6f}, \
              Validation Metrics: {metrics_val}"
              )
        write_metrics_to_tensorboard(tensorboard_writer, metrics_val, epoch)

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": vars(args),
                    "quantiles": quantiles,
                },
                best_path,
            )
            print(
                f"Saved TFT model to {best_path} at {epoch} \
                    epochs (val_loss={val_loss:.6f})"
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch} (patience={patience},\
                    min_delta={min_delta})"
                )
                break
    tensorboard_writer.close()


# Evaluation on test set
def eval_loader(model, data_loader, quantiles, test_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    # Load saved TFT model and evaluate on test set
    best_path = os.path.join("TFT", "checkpoints", "tft_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded stored TFT model for evaluation {best_path}")
    model.eval()

    criterion = QuantileLoss(quantiles=quantiles)
    rows = []
    total_loss = 0.0
    test_ys, test_preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            past = batch["past_inputs"].to(device)
            future = batch["future_inputs"].to(device)
            static = batch["static_inputs"].to(device)
            y = batch["target"].to(device)

            out = model(past, future, static)
            preds_med = out["prediction"][..., median_idx]  # [B, L_dec]
            preds = preds_med.cpu().numpy()
            loss = criterion(out["prediction"].to(device), y)
            total_loss += loss.item() * past.size(0)
            yhat = out["prediction"][..., median_idx]
            test_ys.append(y.detach().cpu().numpy())
            test_preds.append(yhat.detach().cpu().numpy())

            metas = batch.get("meta", [])
            for i, meta in enumerate(metas):
                store_nbr = meta["store_nbr"]
                family = meta["family"]
                fut_dates = meta["future_dates"]
                targets = batch["target"].cpu().numpy()   # [B, L_dec]
                for d_idx, date in enumerate(fut_dates):
                    rows.append({
                        "date": pd.to_datetime(date),
                        "store_nbr": store_nbr,
                        "family": family,
                        "y_true": float(targets[i, d_idx]),
                        "y_pred": float(preds[i, d_idx]),
                    })
                # Append encoder history (past sales) before
                #   forecast horizon
                # Use the 'sales' feature from encoder inputs
                sales_idx = ENC_VARS.index("sales")
                past_dates = meta["past_dates"]
                for d_idx, date in enumerate(past_dates):
                    rows.append(
                        {
                            "date": pd.to_datetime(date),
                            "store_nbr": store_nbr,
                            "family": family,
                            "y_past": float(
                                past[i, d_idx, sales_idx].cpu()
                                ),
                        }
                    )
    save_results_csv(rows)
    total_loss /= max(test_len, 1)
    test_metrics = compute_metrics(test_ys, test_preds)
    print(f"Test Loss: {total_loss:.6f}. Test Metrics: {test_metrics}")
    return test_metrics


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
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Stop if no val loss improvement for N epochs")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.5,
                        help="Minimum val loss improvement to reset patience")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(WORKING_DIR, "TFT", "logs"),
                        help="Directory to store TensorBoard logs")
    parser.add_argument("--train-flag", type=bool, default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(TFT_CHECKPOINTS_DIR, exist_ok=True)

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
    past_input_dims = [1] * len(ENC_VARS)
    future_input_dims = [1] * len(DEC_VARS)
    static_input_dims = static_dims
    quantiles = [float(x) for x in args.quantiles.split(",")]

    model = TemporalFusionTransformer(
        static_input_dims=static_input_dims,
        past_input_dims=past_input_dims,
        future_input_dims=future_input_dims,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        n_heads=args.heads,
        lstm_hidden_size=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        num_quantiles=len(quantiles),
    ).to(device)

    # train
    if args.train_flag:
        train_model(model, quantiles, args,
                    train_loader, val_loader,
                    train_len, val_len
                    )

    test_metrics = eval_loader(model, test_loader, quantiles, test_len)
    print(f"Test matrics: {test_metrics}")


if __name__ == "__main__":
    main()

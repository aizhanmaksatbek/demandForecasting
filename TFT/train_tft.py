import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from architecture.tft import TemporalFusionTransformer, QuantileLoss
from tft_dataset import TFTWindowDataset, tft_collate
from utils.utils import set_seed, build_onehot_maps
from utils.utils import calc_wape, calc_smape, calc_mae
from config.settings import enc_vars, dec_vars, static_cols
from utils.utils import TensorboardConfig

# Add src to path
CUR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CUR_DIR, ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--dec-len", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
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
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join("TFT", "logs"),
                        help="Directory to store TensorBoard logs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join("TFT", "checkpoints"), exist_ok=True)

    tensorboard_writer = TensorboardConfig(
        store_flag=args.tensorboard, log_dir=args.log_dir
    )

    # Load panel
    panel_path = os.path.join("TFT", "data", "panel.csv")
    assert os.path.exists(panel_path), (
        "Run data preprocessing first: "
        "python src/data/preprocess_favorita.py"
    )
    df = pd.read_csv(panel_path, parse_dates=["date"])

    # Scale continuous features (fit on train period only)
    # Determine split dates
    max_date = df["date"].max()
    test_days = pd.Timedelta(days=args.dec_len)  # hold-out horizon for test
    val_days = pd.Timedelta(days=args.dec_len)   # validation horizon
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days

    reals_to_scale = ["transactions", "dcoilwtico"]
    scaler = StandardScaler()
    train_mask = df["date"] <= train_end
    df.loc[train_mask, reals_to_scale] = scaler.fit_transform(
        df.loc[train_mask, reals_to_scale]
    )
    df.loc[~train_mask, reals_to_scale] = scaler.transform(
        df.loc[~train_mask, reals_to_scale]
    )

    # One-hot maps for static features
    static_maps = build_onehot_maps(df, static_cols)
    static_dims = [len(static_maps[c]) for c in static_cols]

    # Dataset and loaders
    split_bounds = (train_end, val_end, test_end)
    train_ds = TFTWindowDataset(
        df, args.enc_len, args.dec_len, enc_vars, dec_vars, static_cols,
        split_bounds, split="train", stride=args.stride,
        static_onehot_maps=static_maps,
    )
    val_ds = TFTWindowDataset(
        df, args.enc_len, args.dec_len, enc_vars, dec_vars, static_cols,
        split_bounds, split="val", stride=args.stride,
        static_onehot_maps=static_maps,
    )
    test_ds = TFTWindowDataset(
        df, args.enc_len, args.dec_len, enc_vars, dec_vars, static_cols,
        split_bounds, split="test", stride=args.stride,
        static_onehot_maps=static_maps,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )

    print(
        f"Train samples: {len(train_ds)} | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    # Model
    past_input_dims = [1] * len(enc_vars)
    future_input_dims = [1] * len(dec_vars)
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

    criterion = QuantileLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    best_val = float("inf")
    best_path = os.path.join("TFT", "checkpoints", "tft_best.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            past = batch["past_inputs"].to(device)     # [B, L_enc, E]
            future = batch["future_inputs"].to(device)  # [B, L_dec, D]
            static = batch["static_inputs"].to(device)  # [B, S]
            y = batch["target"].to(device)             # [B, L_dec]

            optimizer.zero_grad()
            out = model(past, future, static, return_attention=False)
            loss = criterion(out["prediction"].to(device), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * past.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss /= max(len(train_ds), 1)

        tensorboard_writer.write("train", train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                past = batch["past_inputs"].to(device)
                future = batch["future_inputs"].to(device)
                static = batch["static_inputs"].to(device)
                y = batch["target"].to(device)
                out = model(past, future, static, return_attention=False)
                loss = criterion(out["prediction"].to(device), y)
                val_loss += loss.item() * past.size(0)
        val_loss /= max(len(val_ds), 1)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f}"
        )

        tensorboard_writer.write("val", val_loss, epoch)

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

    tensorboard_writer.close()

    # Load best and evaluate on test
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (val pinball: {best_val:.5f})")
    model.eval()

    # Evaluation on validation and test with WAPE, sMAPE and pinball
    def eval_loader(data_loader):
        total_loss = 0.0
        ys, preds = [], []
        with torch.no_grad():
            for batch in data_loader:
                past = batch["past_inputs"].to(device)
                future = batch["future_inputs"].to(device)
                static = batch["static_inputs"].to(device)
                y = batch["target"].to(device)
                out = model(past, future, static, return_attention=False)
                total_loss += (
                    criterion(out["prediction"].to(device), y)
                    .item() * past.size(0)
                )
                # take median quantile as point forecast
                # number of quantiles (unused variable)
                # _Q = out["prediction"].size(-1)  # (ignored)
                median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
                yhat = out["prediction"][..., median_idx]
                ys.append(y.detach().cpu().numpy())
                preds.append(yhat.detach().cpu().numpy())
        pinball = total_loss / max(len(data_loader.dataset), 1)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=0)
        wape_ = calc_wape(ys, preds)
        smape_ = calc_smape(ys, preds)
        mae_ = calc_mae(ys, preds)

        return pinball, wape_, smape_, mae_

    val_pinball, val_wape, val_smape, val_mae = eval_loader(val_loader)
    test_pinball, test_wape, test_smape, test_mae = eval_loader(test_loader)
    print(
        f"Validation  - Pinball: {val_pinball:.5f} | "
        f"WAPE: {val_wape:.5f} | sMAPE: {val_smape:.5f} | MAE: {val_mae:.5f}"
    )
    print(
        f"Test        - Pinball: {test_pinball:.5f} | "
        f"WAPE: {test_wape:.5f} | sMAPE: {test_smape:.5f} \
        | MAE: {test_mae:.5f}"
    )

    # Export per-sample test forecasts (median quantile) for plotting
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            past = batch["past_inputs"].to(device)
            future = batch["future_inputs"].to(device)
            static = batch["static_inputs"].to(device)
            out = model(past, future, static, return_attention=False)
            preds_med = out["prediction"][..., median_idx]  # [B, L_dec]
            targets = batch["target"].cpu().numpy()         # [B, L_dec]
            preds = preds_med.cpu().numpy()
            metas = batch.get("meta", [])
            for i, meta in enumerate(metas):
                store_nbr = meta["store_nbr"]
                family = meta["family"]
                fut_dates = meta["future_dates"]
                for d_idx, date in enumerate(fut_dates):
                    rows.append({
                        "date": pd.to_datetime(date),
                        "store_nbr": store_nbr,
                        "family": family,
                        "y_true": float(targets[i, d_idx]),
                        "y_pred": float(preds[i, d_idx]),
                    })
                # Append encoder history (past sales) before forecast horizon
                # Use the 'sales' feature from encoder inputs
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
        test_forecasts_df = (
            pd.DataFrame(rows)
            .sort_values(["family", "store_nbr", "date"])
        )
        out_csv = os.path.join(
            "TFT", "checkpoints", "tft_test_forecasts.csv"
        )
        test_forecasts_df.to_csv(out_csv, index=False)
        print(
            f"Saved test forecasts CSV -> {out_csv} "
            f"(rows={len(test_forecasts_df)})"
        )


if __name__ == "__main__":
    main()

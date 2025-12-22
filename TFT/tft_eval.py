import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from architecture.tft import TemporalFusionTransformer, QuantileLoss
from tft_dataset import TFTWindowDataset, tft_collate
from utils.utils import set_seed, build_onehot_maps
from utils.utils import compute_metrics
from config.settings import enc_vars, dec_vars, static_cols, reals_to_scale
from utils.utils import get_date_splits

# Add src to path
CUR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CUR_DIR, ".."))


def save_results_csv(rows):
    if rows:
        test_forecasts_df = (
            pd.DataFrame(rows)
            .sort_values(["family", "store_nbr", "date"])
        )
        out_csv = os.path.join("TFT", "checkpoints", "tft_test_forecasts.csv")
        test_forecasts_df.to_csv(out_csv, index=False)
        print(f"Saved test forecasts CSV -> {out_csv}")


def get_data_split(dec_len, enc_len, batch_size, stride):
    panel_path = os.path.join("TFT", "data", "panel.csv")
    assert os.path.exists(panel_path), (
        "Run data preprocessing first: python src/data/preprocess_favorita.py"
    )
    df = pd.read_csv(panel_path, parse_dates=["date"])

    train_end, val_end, test_end = get_date_splits(df, dec_len)

    scaler = StandardScaler()
    train_mask = df["date"] <= train_end
    df.loc[train_mask, reals_to_scale] = scaler.fit_transform(
        df.loc[train_mask, reals_to_scale]
    )
    df.loc[~train_mask, reals_to_scale] = scaler.transform(
        df.loc[~train_mask, reals_to_scale]
    )

    static_maps = build_onehot_maps(df, static_cols)
    static_dims = [len(static_maps[c]) for c in static_cols]

    split_bounds = (train_end, val_end, test_end)

    test_ds = TFTWindowDataset(
        df, enc_len, dec_len, enc_vars, dec_vars, static_cols,
        split_bounds, split="test", stride=stride,
        static_onehot_maps=static_maps,
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=tft_collate,
    )
    print(f"Test: {len(test_ds)}")
    return (None, None, test_loader, static_dims, 0, 0, len(test_ds))


def eval_loader(model, data_loader, quantiles, test_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    # Important: do NOT load a checkpoint here; model is already loaded
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
                sales_idx = enc_vars.index("sales")
                past_dates = meta["past_dates"]
                for d_idx, date in enumerate(past_dates):
                    rows.append({
                        "date": pd.to_datetime(date),
                        "store_nbr": store_nbr,
                        "family": family,
                        "y_past": float(past[i, d_idx, sales_idx].cpu()),
                    })

    save_results_csv(rows)
    total_loss /= max(test_len, 1)
    print(f"Test loss: {total_loss:.4f}")
    return compute_metrics(test_ys, test_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--dec-len", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    # CLI defaults only used if checkpoint doesn't contain cfg
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-file-path", type=str,
                        default="tft_best_train_final.pt",
                        help="Filename under TFT/checkpoints to load")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    _, _, test_loader, static_dims, _, _, test_len = get_data_split(
        args.dec_len, args.enc_len, args.batch_size, args.stride
    )

    # Load checkpoint first to read training-time config
    checkpoint_path = os.path.join(
        "TFT", "checkpoints", f"{args.checkpoints_file_path}"
        )
    assert os.path.exists(checkpoint_path), f"Checkpoint not \
        found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Use quantiles from checkpoint to match output head size
    if "quantiles" in ckpt and isinstance(ckpt["quantiles"], (list, tuple)):
        quantiles = [float(x) for x in ckpt["quantiles"]]
    else:
        quantiles = [float(x) for x in args.quantiles.split(",")]

    # Read model hyperparameters from checkpoint cfg
    cfg = ckpt.get("cfg", {})
    d_model = int(cfg.get("d_model", args.d_model))
    hidden_dim = int(cfg.get("hidden_dim", args.hidden_dim))
    n_heads = int(cfg.get("heads", args.heads))
    lstm_hidden_size = int(cfg.get("lstm_hidden", args.lstm_hidden))
    lstm_layers = int(cfg.get("lstm_layers", args.lstm_layers))
    dropout = float(cfg.get("dropout", args.dropout))

    # Inputs
    past_input_dims = [1] * len(enc_vars)
    future_input_dims = [1] * len(dec_vars)
    static_input_dims = static_dims

    # Build model to match checkpoint shapes
    model = TemporalFusionTransformer(
        static_input_dims=static_input_dims,
        past_input_dims=past_input_dims,
        future_input_dims=future_input_dims,
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        num_quantiles=len(quantiles),
    ).to(device)

    # Load weights strictly
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"Loaded stored TFT model for evaluation {checkpoint_path}")
    print(f"Configured d_model={d_model}, hidden_dim={hidden_dim},\
          heads={n_heads}, "
          f"lstm_hidden_size={lstm_hidden_size}, lstm_layers={lstm_layers}, \
            dropout={dropout}, "
          f"num_quantiles={len(quantiles)}")

    # Evaluate
    test_metrics = eval_loader(model, test_loader, quantiles, test_len)
    print(f"Test matrics: {test_metrics}")


if __name__ == "__main__":
    main()

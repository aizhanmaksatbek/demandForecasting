import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from graph_dataset import GraphDemandDataset
from architecture.stgnn import STGNN, QuantileLoss


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
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    panel_csv = os.path.join("GNN", "data", "panel.csv")
    node_index_csv = os.path.join("GNN", "data", "node_index.csv")
    adj_path = os.path.join("GNN", "data", "adjacency.npy")
    assert os.path.exists(panel_csv), (
        "Run preprocessing to create data/processed/panel.csv"
    )
    assert os.path.exists(node_index_csv) and os.path.exists(
        adj_path
    ), "Run scripts/build_graphs_favorita_gnn.py first"

    # Determine split dates based on panel
    df = pd.read_csv(panel_csv, parse_dates=["date"])
    max_date = df["date"].max()
    test_days = pd.Timedelta(days=args.horizon)
    val_days = pd.Timedelta(days=args.horizon)
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days
    split_bounds = (train_end, val_end, test_end)

    # Features (past covariates)
    feature_cols = ["sales", "transactions", "dcoilwtico", "onpromotion",
                    "dow", "month", "weekofyear", "is_holiday", "is_workday"
                    ]

    # Optionally scale continuous features on train period only
    # We scale later inside dataset pivoting by passing scaled panel,
    #   but for simplicity we scale panel in place
    reals = ["transactions", "dcoilwtico"]
    scaler = None
    if len(reals) > 0:
        scaler = {}
        train_mask = df["date"] <= train_end
        for col in reals:
            mu = df.loc[train_mask, col].mean()
            sd = df.loc[train_mask, col].std()
            sd = 1.0 if pd.isna(sd) or sd == 0 else sd
            df[col] = (df[col] - mu) / sd
            scaler[col] = (float(mu), float(sd))
        # write scaled panel temp
        panel_csv_scaled = os.path.join("GNN", "data",
                                        "panel_scaled.csv"
                                        )
        df.to_csv(panel_csv_scaled, index=False)
        panel_csv_use = panel_csv_scaled
    else:
        panel_csv_use = panel_csv

    # Datasets
    train_ds = GraphDemandDataset(panel_csv_use, node_index_csv, args.enc_len,
                                  args.horizon, split_bounds, split="train",
                                  feature_cols=feature_cols)
    val_ds = GraphDemandDataset(panel_csv_use, node_index_csv, args.enc_len,
                                args.horizon, split_bounds, split="val",
                                feature_cols=feature_cols)
    test_ds = GraphDemandDataset(panel_csv_use, node_index_csv, args.enc_len,
                                 args.horizon, split_bounds, split="test",
                                 feature_cols=feature_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    A_hat = torch.from_numpy(np.load(adj_path)).to(device)  # [N, N]
    num_nodes = A_hat.shape[0]
    in_features = len(feature_cols)
    quantiles = [float(x) for x in args.quantiles.split(",")]

    model = STGNN(
        num_nodes=num_nodes,
        in_features=in_features,
        horizon=args.horizon,
        A_hat=A_hat,
        hidden_channels=args.hidden,
        num_blocks=args.blocks,
        kernel_size=args.kernel,
        dropout=args.dropout,
        quantiles=quantiles,
    ).to(device)

    criterion = QuantileLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
        )

    best_val = float("inf")
    best_path = os.path.join("GNN", "checkpoints", "stgnn_best.pt")

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            x = batch["x_hist"].to(device)  # [B, L_enc, N, F]
            y = batch["y_fut"].to(device)   # [B, H, N]
            optimizer.zero_grad()
            yhat = model(x)                 # [B, H, N, Q]
            loss = criterion(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss /= max(len(train_ds), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x_hist"].to(device)
                y = batch["y_fut"].to(device)
                yhat = model(x)
                loss = criterion(yhat, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(len(val_ds), 1)
        print(f"Epoch {epoch}: train_pinball={train_loss:.5f} \
              | val_pinball={val_loss:.5f}"
              )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(),
                 "cfg": vars(args),
                 "quantiles": quantiles},
                best_path,
            )
            print(f"Saved best model to {best_path}")

    # Load best and evaluate
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (val pinball: {best_val:.5f})")

    def evaluate(data_loader):
        total_loss = 0.0
        ys, preds = [], []
        with torch.no_grad():
            for batch in data_loader:
                x = batch["x_hist"].to(device)
                y = batch["y_fut"].to(device)
                yhat = model(x)  # [B,H,N,Q]
                total_loss += criterion(yhat, y).item() * x.size(0)
                # median quantile
                median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
                yhat_med = yhat[..., median_idx]  # [B,H,N]
                ys.append(y.cpu().numpy())
                preds.append(yhat_med.cpu().numpy())
        pinball = total_loss / max(len(data_loader.dataset), 1)
        ys = np.concatenate(ys, axis=0)      # [M,H,N]
        preds = np.concatenate(preds, axis=0)
        # Flatten for metrics
        y_flat = ys.reshape(-1)
        p_flat = preds.reshape(-1)
        return pinball, wape(y_flat, p_flat), smape(y_flat, p_flat)

    val_pin, val_w, val_s = evaluate(val_loader)
    test_pin, test_w, test_s = evaluate(test_loader)
    print(f"Validation - Pinball: {val_pin:.5f} \
           | WAPE: {val_w:.5f} | sMAPE: {val_s:.5f}")
    print(f"Test       - Pinball: {test_pin:.5f} \
          | WAPE: {test_w:.5f} | sMAPE: {test_s:.5f}")


if __name__ == "__main__":
    main()

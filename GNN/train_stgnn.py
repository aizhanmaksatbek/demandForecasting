import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from graph_dataset import GraphDemandDataset
from architecture.stgnn import STGNN, QuantileLoss
from utils.utils import set_seed, compute_metrics
from config.settings import ENC_VARS as feature_cols
from utils.utils import get_date_splits
from config.settings import REALS_TO_SCALE as reals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
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
    os.makedirs(os.path.join("GNN", "checkpoints"), exist_ok=True)

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
    train_end, val_end, test_end = get_date_splits(df, args.horizon)
    split_bounds = (train_end, val_end, test_end)

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
        train_t, train_pred = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            x = batch["x_hist"].to(device)  # [B, L_enc, N, F]
            y = batch["y_fut"].to(device)   # [B, H, N]
            optimizer.zero_grad()
            yhat = model(x)                 # [B, H, N, Q]
            loss = criterion(yhat.to(device), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            train_t.append(y.detach().cpu().numpy())
            train_pred.append(yhat.detach().cpu().numpy())
        train_loss /= max(len(train_ds), 1)
        train_metrics = compute_metrics(train_t, train_pred)
        print(f"Train Epoch {epoch} Loss: {train_loss:.6f} |\
              Metrics: {train_metrics}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_t, val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x_hist"].to(device)
                y = batch["y_fut"].to(device)
                yhat = model(x)
                loss = criterion(yhat.to(device), y)
                val_loss += loss.item() * x.size(0)
                val_t.append(y.detach().cpu().numpy())
                val_pred.append(yhat.detach().cpu().numpy())
        val_loss /= max(len(val_ds), 1)
        val_metrics = compute_metrics(val_t, val_pred)
        print(f"Validation Epoch {epoch}: Loss {val_loss:.6f} \
              | Metrics: {val_metrics:.6f}"
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
        print(f"Loaded best model loss: {best_val:.5f})")

    def evaluate(test_loader):
        test_loss = 0.0
        test_t, test_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x_hist"].to(device)
                y = batch["y_fut"].to(device)
                pred = model(x)  # [B,H,N,Q]
                loss = criterion(pred.to(device), y)
                test_loss += loss.item() * x.size(0)
                test_t.append(y.detach().cpu().numpy())
                test_pred.append(pred.detach().cpu().numpy())
        test_loss = test_loss / max(len(test_loader.dataset), 1)
        test_metrics = compute_metrics(test_t, test_pred)
        print(f"Test Loss: {test_loss:.6f} | Metrics: {test_metrics}")

        return test_metrics["wape"]

    val_wape = evaluate(val_loader)
    test_w = evaluate(test_loader)
    print(f"Validation WAPE: {val_wape:.5f}")
    print(f"Test WAPE: {test_w:.5f}")


if __name__ == "__main__":
    main()

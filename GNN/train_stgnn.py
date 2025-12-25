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
from utils.utils import TensorboardConfig


GNN_CHECKPOINTS_PATH = "GNN/checkpoints"
GNN_DATA_PATH = "GNN/data"
GNN_LOG_DIR = "GNN/logs"


def get_gnn_data_splits(args):
    panel_csv = os.path.join(GNN_DATA_PATH, "panel.csv")
    assert os.path.exists(panel_csv), (
        "Run preprocessing to create data/processed/panel.csv"
    )
    node_index_csv = os.path.join(GNN_DATA_PATH, "node_index.csv")
    assert os.path.exists(node_index_csv), "Run data-preprocessing first"

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
        panel_csv_scaled = os.path.join(GNN_DATA_PATH,
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

    return train_loader, val_loader, test_loader


def train_gnn_model(model, train_loader, val_loader, criterion, optimizer,
                    args, best_path, quantiles):
    # Training
    tensorboardpanel = TensorboardConfig(True, GNN_LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val = float("inf")
    epochs_no_improve = 0
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
            median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
            pred_med = yhat[..., median_idx]  # [B,H,N]
            train_pred.append(pred_med.detach().cpu().numpy())
        train_loss /= max(len(train_loader.dataset), 1)
        train_metrics = compute_metrics(train_t, train_pred)
        tensorboardpanel.write("train_loss", train_loss, epoch)
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
                median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
                pred_med = yhat[..., median_idx]  # [B,H,N]
                val_pred.append(pred_med.detach().cpu().numpy())
        val_loss /= max(len(val_loader.dataset), 1)
        val_metrics = compute_metrics(val_t, val_pred)
        tensorboardpanel.write("val_loss", val_loss, epoch)
        print(f"Validation Epoch {epoch}: Loss {val_loss:.6f} \
              | Metrics: {val_metrics}"
              )

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(),
                 "cfg": vars(args),
                 "quantiles": quantiles},
                best_path,
            )
            print(f"Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    tensorboardpanel.close()


def eval_gnn_model(model, test_loader, criterion, best_path, quantiles):
    # Load best and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model from {best_path}")

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
            median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
            pred_med = pred[..., median_idx]  # [B,H,N]
            test_pred.append(pred_med.detach().cpu().numpy())
    test_loss = test_loss / max(len(test_loader.dataset), 1)
    test_metrics = compute_metrics(test_t, test_pred)
    print(f"Test Loss: {test_loss:.6f} | Metrics: {test_metrics}")

    return test_metrics["wape"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-len", type=int, default=56)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(GNN_CHECKPOINTS_PATH), exist_ok=True)

    train_loader, val_loader, test_loader = get_gnn_data_splits(args)
    adj_path = os.path.join(GNN_DATA_PATH, "adjacency.npy")
    assert os.path.exists(adj_path), "Run data preprocessing first"

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

    best_model_path = os.path.join(GNN_CHECKPOINTS_PATH, "stgnn_best.pt")

    train_gnn_model(model, train_loader, val_loader, criterion, optimizer,
                    args, best_model_path, quantiles
                    )
    quantiles = [float(x) for x in args.quantiles.split(",")]

    eval_gnn_model(model, test_loader, criterion, best_model_path, quantiles)


if __name__ == "__main__":
    main()

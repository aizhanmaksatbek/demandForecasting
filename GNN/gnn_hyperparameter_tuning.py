import optuna
import os
import torch
import numpy as np
from GNN.train_stgnn import (
    get_gnn_data_splits,
    train_gnn_model,
    eval_gnn_model
    )
from architecture.stgnn import STGNN, QuantileLoss
from config.settings import GNN_CHECKPOINTS_PATH, GNN_DATA_PATH
from config.settings import ENC_VARS as feature_cols


def objective(trial):
    # Suggest hyperparameters
    hidden = trial.suggest_categorical("hidden", [32, 64, 128])
    blocks = trial.suggest_int("blocks", 2, 4)
    kernel = trial.suggest_int("kernel", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    patience = 5

    # Build args namespace
    class Args:
        pass
    args = Args()
    args.enc_len = 56
    args.horizon = 28
    args.batch_size = 128
    args.epochs = 1
    args.lr = lr
    args.hidden = hidden
    args.blocks = blocks
    args.kernel = kernel
    args.dropout = dropout
    args.quantiles = "0.1,0.5,0.9"
    args.seed = 42
    args.patience = patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_gnn_data_splits(args)
    adj_path = os.path.join(GNN_DATA_PATH, "adjacency.npy")
    A_hat = torch.from_numpy(np.load(adj_path)).to(device)
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
        model.parameters(), lr=args.lr, weight_decay=1e-5
        )
    best_model_path = os.path.join(
        GNN_CHECKPOINTS_PATH, "stgnn_best_trial.pt"
        )

    train_gnn_model(
        model, train_loader, val_loader, criterion,
        optimizer, args, best_model_path, quantiles
        )
    val_wape = eval_gnn_model(
        model, val_loader, criterion, best_model_path, quantiles
        )
    return val_wape


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2)
    print('Best trial:', study.best_trial)
    print('Best params:', study.best_params)


if __name__ == "__main__":
    main()

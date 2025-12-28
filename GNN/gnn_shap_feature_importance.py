import os
import numpy as np
import torch
import pandas as pd
import shap
from architecture.stgnn import STGNN, QuantileLoss
from torch.utils.data import DataLoader
import argparse
from graph_dataset import GraphDemandDataset
from config.settings import GNN_DATA_PATH, GNN_CHECKPOINTS_PATH
from utils.utils import get_date_splits
from config.settings import ENC_VARS as feature_cols
import matplotlib.pyplot as plt

feature_names = feature_cols


def main():
    parser = argparse.ArgumentParser(description="STGNN explainer")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(
                            GNN_CHECKPOINTS_PATH, "stgnn_best.pt"))
    parser.add_argument("--panel-csv", type=str,
                        default=os.path.join(GNN_DATA_PATH, "panel.csv"))
    parser.add_argument("--node-index-csv", type=str,
                        default=os.path.join(GNN_DATA_PATH, "node_index.csv"))
    parser.add_argument("--adj", type=str,
                        default=os.path.join(GNN_DATA_PATH, "adjacency.npy"))
    parser.add_argument("--target-node", type=int, default=2)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(GNN_CHECKPOINTS_PATH))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K neighbors of target node for explainer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    assert os.path.exists(args.panel_csv), f"{args.panel_csv} not found"
    assert os.path.exists(args.node_index_csv), "node_index.csv not found"
    assert os.path.exists(args.adj), f"adjacency.npy not found: {args.adj}"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    quantiles = ckpt.get("quantiles", [0.1, 0.5, 0.9])
    enc_len = int(cfg.get("enc_len", 56))
    horizon = int(cfg.get("horizon", 28))

    # Split bounds via panel
    df = pd.read_csv(args.panel_csv, parse_dates=["date"])
    train_end, val_end, test_end = get_date_splits(df, horizon)
    split_bounds = (train_end, val_end, test_end)

    test_ds = GraphDemandDataset(
        args.panel_csv, args.node_index_csv, enc_len, horizon,
        split_bounds, split="test", feature_cols=feature_cols,
    )
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset is empty")
    idx = max(0, min(args.batch_index, len(test_ds) - 1))
    sample = test_ds[idx]
    x_hist = sample["x_hist"].unsqueeze(0).to(device)  # [1,L,N,F]

    A_hat = torch.from_numpy(np.load(args.adj)).to(device)
    num_nodes = A_hat.shape[0]
    model = STGNN(
        num_nodes=num_nodes,
        in_features=len(feature_cols),
        horizon=horizon,
        A_hat=A_hat,
        hidden_channels=int(cfg.get("hidden", 64)),
        num_blocks=int(cfg.get("blocks", 3)),
        kernel_size=int(cfg.get("kernel", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        quantiles=quantiles,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.eval()

    bs = int(args.batch_size or cfg.get("batch_size", 32))
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True
        )
    criterion = QuantileLoss(quantiles=quantiles)
    # eval_gnn_model(model, test_loader, criterion, args.ckpt, quantiles)

    # --- Assume you have a trained GNN model and test_loader with batch_size=1
    # model: your trained GNN model
    # test_loader: DataLoader with batch_size=1

    # --- Get background and test samples
    background_samples = []
    test_samples = []
    for i, batch in enumerate(test_loader):
        x = batch["x_hist"] if isinstance(batch, dict) else batch[0]
        if i < 1:
            background_samples.append(x)
        if i < 10:
            test_samples.append(x)
        else:
            break

    background = torch.cat(background_samples, dim=0).cpu().numpy()  # shape: [1, L, N, F]
    X_test = torch.cat(test_samples, dim=0).cpu().numpy()  # shape: [10, L, N, F]

    # # --- Define prediction function for SHAP
    # def prediction_function(x):
    #     # x: [batch, L, N, F]
    #     x_tensor = torch.tensor(x, dtype=torch.float32)
    #     with torch.no_grad():
    #         y = model(x_tensor)  # shape: [batch, H, N, ...]
    #         # Example: mean over horizon and nodes, adjust as needed
    #         return y.mean(dim=(1, 2)).cpu().numpy()

    # # --- SHAP KernelExplainer
    # explainer = shap.KernelExplainer(prediction_function, background)

    # # --- Calculate SHAP values
    # shap_values = explainer.shap_values(X_test)
    target_node = args.target_node  # or any node index
    timestep = 0  # or any timestep

    X_test_node = X_test[:, timestep, target_node, :]  # shape: (10, 9)
    background_node = background[:, timestep, target_node, :]  # shape: (1, 9)

    def prediction_function(x):
        # x: [batch, features]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_full = torch.zeros((x.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        x_full[:, timestep, target_node, :] = x_tensor
        with torch.no_grad():
            y = model(x_full)
            return y[..., 1, target_node].cpu().numpy()  # adjust for your output

    explainer = shap.KernelExplainer(prediction_function, background_node)
    shap_values = explainer.shap_values(X_test_node)

    # --- Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv("shap_values.csv", index=False)
    print("Saved SHAP values to shap_values.csv")

    # --- Visualize local explanation (force plot for first test item)
    shap.initjs()
    force = shap.force_plot(
        explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        shap_values[0],           # shape: (num_features,)
        X_test_node[0],           # shape: (num_features,)
        feature_names=feature_names
    )
    shap.save_html("force_plot.html", force)

    # --- Visualize global feature importance (summary plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test_node, feature_names=feature_names, show=False)
    plt.savefig("summary_plot.png", dpi=300)
    plt.close()


def plot_shap_from_file(csv_path, feature_names=None, out_path="summary_plot_from_file.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    shap_df = pd.read_csv(csv_path)
    if feature_names is None:
        feature_names = list(shap_df.columns)
    plt.figure()
    shap_vals = shap_df.values
    # Plot mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    plt.bar(range(len(feature_names)), mean_abs_shap)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.ylabel("Mean |SHAP value|")
    plt.title("Global SHAP Feature Importance (from file)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot from file to {out_path}")
    return out_path

# Example usage:
# plot_shap_from_file("shap_values.csv", feature_names=feature_names)


if __name__ == "__main__":
    main()

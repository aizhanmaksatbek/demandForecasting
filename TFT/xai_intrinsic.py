import torch
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tft_dataset import TFTWindowDataset, tft_collate
from architecture.tft import TemporalFusionTransformer
from utils.utils import build_onehot_maps
from config.settings import ENC_VARS, DEC_VARS, STATIC_COLS
from utils.utils import get_date_splits
from config.settings import TFT_CHECKPOINTS_DIR, TFT_DATA_DIR


out_dir = TFT_CHECKPOINTS_DIR


def extract_tft_intrinsic(model, batch, device):
    """Retrieves the variable selection weights from TFT.
    TFT implementation provides variable selection weights under these keys:
    - 'encoder_variable_importance': [B, L_enc, V_enc]
    - 'decoder_variable_importance': [B, L_dec, V_dec]
    - 'static_variable_importance':  [B, V_static]
    """
    model.eval()
    with torch.no_grad():
        out = model(
            batch["past_inputs"].to(device),
            batch["future_inputs"].to(device),
            batch["static_inputs"].to(device),
            return_attention=True,
        )
    return {
        "vsn_enc": out.get("encoder_variable_importance"),
        "vsn_dec": out.get("decoder_variable_importance"),
        "vsn_static": out.get("static_variable_importance")
    }


def summarize_vsn(
        vsn_enc: torch.Tensor,
        vsn_dec: torch.Tensor,
        vsn_static: torch.Tensor
        ):
    """Computes mean importance across batch/time for encoder/decoder/static.
    """
    enc_imp = (vsn_enc.mean(dim=(0, 1)).detach().cpu().numpy()
               if vsn_enc is not None else None)
    dec_imp = (vsn_dec.mean(dim=(0, 1)).detach().cpu().numpy()
               if vsn_dec is not None else None)
    stat_imp = (vsn_static.mean(dim=0).detach().cpu().numpy()
                if vsn_static is not None else None)
    return enc_imp, dec_imp, stat_imp


def plot_intrinsic_importance(intr, plot_path):
    # Variable Importances
    enc_imp, dec_imp, stat_imp = summarize_vsn(
        intr["vsn_enc"],
        intr["vsn_dec"],
        intr["vsn_static"]
        )
    rows = 3
    fig, axes = plt.subplots(
        rows, 1, figsize=(12, max(6, rows * 1.6)), sharex=False
        )
    for j in range(rows):
        ax = axes[j]
        if j == 0 and enc_imp is not None:
            x = np.arange(len(ENC_VARS))
            ax.bar(x, enc_imp)
            ax.set_title("Encoder Variable Importance (avg)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                ENC_VARS, rotation=45, ha="right"
                )
            ax.set_ylabel("Importance")
            ax.grid(True, alpha=0.2)
        elif j == 1 and dec_imp is not None:
            x = np.arange(len(DEC_VARS))
            ax.bar(x, dec_imp)
            ax.set_title("Decoder Variable Importance (avg)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                DEC_VARS, rotation=45, ha="right"
                )
            ax.set_ylabel("Importance")
            ax.grid(True, alpha=0.2)
        elif j == 2 and stat_imp is not None:
            x = np.arange(len(stat_imp))
            ax.bar(x, stat_imp)
            ax.set_title("Static Variable Importance (avg)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                STATIC_COLS,
                rotation=45, ha="right"
                )
            ax.set_ylabel("Importance")
            ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path)
    plt.close(fig)


def plot_input_set(
        model, batch, batch_size, enc_vars, dec_vars, device
        ):
    """
    Plots one sample input and output from a batch data:
    - past_inputs: [B, L_enc, E]
    - future_inputs: [B, L_dec, D]
    - target: [B, L_dec]
    - meta: list of dicts with 'past_dates' and 'future_dates'
    - prediction (from model)   : [B, L_dec, Q] or [B, L_dec]
    Saves plot to out_path.
    """
    try:
        sample_idx = random.randint(0, batch_size - 1)
        past = batch["past_inputs"][sample_idx].detach().cpu().numpy()
        future = batch["future_inputs"][sample_idx].detach().cpu().numpy()
        target = batch.get("target", None)
        target = (None if target is None
                  else target[sample_idx].detach().cpu().numpy()
                  )
        # Compute model forecast (median quantile) for this sample
        yhat = None
        try:
            model.eval()
            with torch.no_grad():
                out = model(
                    batch["past_inputs"].to(device),
                    batch["future_inputs"].to(device),
                    batch["static_inputs"].to(device),
                )
            preds = out.get("prediction")  # [B, L_dec, Q] or [B, L_dec]
            if preds is not None:
                preds = preds.detach().cpu().numpy()
                if preds.ndim == 3:
                    median_idx = preds.shape[-1] // 2
                    yhat = preds[sample_idx, :, median_idx]
                elif preds.ndim == 2:
                    yhat = preds[sample_idx, :]
        except Exception:
            yhat = None
        meta_list = batch.get("meta", [])
        meta = meta_list[sample_idx] if meta_list else {}
        past_dates = meta.get("past_dates", list(range(past.shape[0])))
        future_dates = meta.get("future_dates", list(range(future.shape[0])))
        prod_fam = meta.get("family", "N/A")
        store_nbr = meta.get("store_nbr", "N/A")

        # Converts dates to pandas datetime for nicer x-axis, if available
        try:
            past_x = pd.to_datetime(past_dates)
        except Exception:
            past_x = list(range(past.shape[0]))
        try:
            future_x = pd.to_datetime(future_dates)
        except Exception:
            future_x = list(range(future.shape[0]))

        E = past.shape[1]
        D = future.shape[1]
        rows = max(E, D)
        fig, axes = plt.subplots(
            rows, 2, figsize=(12, max(6, rows * 1.6)), sharex=False
            )
        if rows == 1:
            # Ensure axes is 2D array-like
            axes = np.array([axes])

        # Plot encoder covariates
        for i in range(E):
            ax = axes[i, 0]
            ax.plot(past_x, past[:, i], lw=1.5)
            ax.set_title(
                f"Past (Encoder): {enc_vars[i] if i < len(enc_vars) else i}"
                )
            ax.grid(True, alpha=0.2)

        axes[0, 0].set_ylabel("Value")
        axes[min(E, rows) - 1, 0].set_xlabel("Past timeline")

        # Plot decoder covariates and optional target overlay in bottom panel
        for j in range(D):
            ax = axes[j, 1]
            ax.plot(future_x, future[:, j], lw=1.5, color="tab:blue")
            ax.set_title(
                f"Future (Decoder): {dec_vars[j] if j < len(dec_vars) else j}"
                )
            ax.grid(True, alpha=0.2)

        # If target/prediction exist, overlay them on the last decoder panel
        ax_t = axes[min(D, rows) - 0, 1]
        if target is not None:
            ax_t.plot(
                future_x, target, lw=2.0, color="tab:red", label="target"
                )
        if yhat is not None:
            ax_t.plot(
                future_x, yhat, lw=2.0, color="tab:green", label="prediction"
                )
        if target is not None or yhat is not None:
            ax_t.legend(loc="upper right")

        axes[0, 1].set_ylabel("Value")
        axes[min(D, rows) - 0, 1].set_xlabel("Future horizon")

        fig.suptitle(
            f"TFT XAI for product family {prod_fam} - Store {store_nbr}"
            )
        plot_path = os.path.join(out_dir, f"xai_{prod_fam}_{store_nbr}.png")
        print(f"Saved input set plot -> {plot_path}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot input set: {e}")


def save_variable_imp():
    # Plot encoder/decoder/static VSN importance if saved
    enc_path = os.path.join(TFT_CHECKPOINTS_DIR, "xai_vsn_enc.npy")
    dec_path = os.path.join(TFT_CHECKPOINTS_DIR, "xai_vsn_dec.npy")
    stat_path = os.path.join(TFT_CHECKPOINTS_DIR, "xai_vsn_static.npy")
    plots = []
    if os.path.exists(enc_path):
        enc_imp = np.load(enc_path)  # [V_enc]
        plots.append(("Encoder Variable Importance (avg)", ENC_VARS, enc_imp))
    if os.path.exists(dec_path):
        dec_imp = np.load(dec_path)  # [V_dec]
        plots.append(("Decoder Variable Importance (avg)", DEC_VARS, dec_imp))
    if os.path.exists(stat_path):
        stat_imp = np.load(stat_path)  # [V_static]
        plots.append(
            ("Static Variable Importance (avg)", STATIC_COLS, stat_imp)
        )

    if len(plots) > 0:
        fig, axes = plt.subplots(
            len(plots), 1, figsize=(12, 3.5 * len(plots)), sharex=False
        )
        if len(plots) == 1:
            axes = [axes]
        for idx, (title, labels, vals) in enumerate(plots):
            ax = axes[idx]
            x = np.arange(len(labels))
            ax.bar(x, vals)
            ax.set_title(title)
            ax.set_ylabel("Importance")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        out_png = os.path.join("TFT", "checkpoints", "xai_vsn_bars.png")
        plt.savefig(out_png)
        plt.close(fig)
        print(f"Saved combined importance bars -> {out_png}")
    else:
        print("No VSN importance arrays found to plot.")


def run_tft_intrinsic_once(enc_len=56, dec_len=28, stride=1,
                           d_model=64, hidden_dim=128,
                           heads=4, lstm_hidden=64, lstm_layers=1, dropout=0.1,
                           checkpoint_path=os.path.join(
                               "TFT", "checkpoints", "tft_best_train_final.pt"
                               ), device: torch.device | None = None):
    """Loads a panel.csv, model and checkpoint, take one test batch,
    and dumps intrinsic XAI arrays from TFT model architecture.

    Saves: xai_vsn_enc.npy, xai_vsn_dec.npy, xai_vsn_static.npy under out_dir.
    """
    # out_dir=os.path.join("TFT", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # Load panel
    panel_path = os.path.join(TFT_DATA_DIR, "panel.csv")
    print(f"Loading panel data from {panel_path}...")
    assert os.path.exists(panel_path), "Run data preprocessing first."
    df = pd.read_csv(panel_path, parse_dates=["date"])

    train_end, val_end, test_end = get_date_splits(df, dec_len)
    split_bounds = (train_end, val_end, test_end)

    static_maps = build_onehot_maps(df, STATIC_COLS)

    test_ds = TFTWindowDataset(
        df, enc_len, dec_len, ENC_VARS, DEC_VARS, STATIC_COLS,
        split_bounds, split="test", stride=stride,
        static_onehot_maps=static_maps,
    )
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=tft_collate
                             )

    # Model definition must match training config for shapes;
    # we only need forward for XAI
    past_input_dims = [1] * len(ENC_VARS)
    future_input_dims = [1] * len(DEC_VARS)
    static_input_dims = [len(static_maps[c]) for c in STATIC_COLS]
    model = TemporalFusionTransformer(
        static_input_dims=static_input_dims,
        past_input_dims=past_input_dims,
        future_input_dims=future_input_dims,
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_heads=heads,
        lstm_hidden_size=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
        num_quantiles=3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint if present
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    # Plot the input set used for inference (with forecast overlay)
    try:
        for batch in test_loader:
            plot_input_set(model, batch, 1, ENC_VARS, DEC_VARS, device)
            intr = extract_tft_intrinsic(model, batch, device=device)
            plot_intrinsic_importance(intr)

    except Exception as e:
        print(f"Could not save input set plot: {e}")


if __name__ == "__main__":
    # Prefer GPU when available; allow override via environment
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes = run_tft_intrinsic_once(device=dev)
    print("Saved TFT XAI arrays. Shapes:", shapes)

    save_variable_imp()

import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from tft_dataset import TFTWindowDataset, tft_collate
from architecture.tft import TemporalFusionTransformer
import matplotlib.pyplot as plt
from utils.utils import build_onehot_maps
from config.settings import enc_vars, dec_vars, static_cols


def extract_tft_intrinsic(model, batch, device=None):
    """Return attention and variable selection weights from TFT.

    Expects TFT to support return_attention and return_vsn in forward.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            batch["past_inputs"].to(device),
            batch["future_inputs"].to(device),
            batch["static_inputs"].to(device),
            return_attention=True,
        )
    # Safely pick keys if present
    attn = out.get("attn_weights")  # [B, L_dec, L_enc]
    # TFT implementation provides variable selection weights under these keys
    # [B, L_enc, V_enc] or None
    vsn_enc = out.get("encoder_variable_importance")
    # [B, L_dec, V_dec] or None
    vsn_dec = out.get("decoder_variable_importance")
    # [B, V_static] or None
    vsn_static = out.get("static_variable_importance")
    return {
        "attn_weights": attn,
        "vsn_enc": vsn_enc,
        "vsn_dec": vsn_dec,
        "vsn_static": vsn_static,
    }


def summarize_vsn(
        vsn_enc: torch.Tensor,
        vsn_dec: torch.Tensor,
        vsn_static: torch.Tensor
        ):
    """Compute mean importance across batch/time for encoder/decoder/static."""
    enc_imp = (vsn_enc.mean(dim=(0, 1)).detach().cpu().numpy()
               if vsn_enc is not None else None)
    dec_imp = (vsn_dec.mean(dim=(0, 1)).detach().cpu().numpy()
               if vsn_dec is not None else None)
    stat_imp = (vsn_static.mean(dim=0).detach().cpu().numpy()
                if vsn_static is not None else None)
    return enc_imp, dec_imp, stat_imp


def summarize_attention(attn_weights: torch.Tensor):
    """Average decoder attention over batch to get [L_dec, L_enc]."""
    if attn_weights is None:
        return None
    # Handle potential extra head/channel dim: [B, L_dec, L_enc]
    # or [B, L_dec, L_enc, K]
    aw = attn_weights
    if aw.ndim == 4:
        # average over batch and head/channel
        attn = aw.mean(dim=(0, 3)).detach().cpu().numpy()  # [L_dec, L_enc]
    elif aw.ndim == 3:
        attn = aw.mean(dim=0).detach().cpu().numpy()       # [L_dec, L_enc]
    elif aw.ndim == 2:
        attn = aw.detach().cpu().numpy()                   # [L_dec, L_enc]
    else:
        # Fallback: collapse all but last two dims
        while aw.ndim > 2:
            aw = aw.mean(dim=0)
        attn = aw.detach().cpu().numpy()
    return attn


def plot_input_set(model, batch, enc_vars, dec_vars, out_path, device=None):
    """Plot past and future covariates (and target) for a single batch sample.

    Saves a PNG to out_path. Expects `batch` to include keys:
    - past_inputs: [B, L_enc, E]
    - future_inputs: [B, L_dec, D]
    - target: [B, L_dec]
    - meta: list of dicts with 'past_dates' and 'future_dates'
    """
    try:
        sample_idx = 0
        past = batch["past_inputs"][sample_idx].detach().cpu().numpy()
        future = batch["future_inputs"][sample_idx].detach().cpu().numpy()
        target = batch.get("target", None)
        target = (None if target is None else target[sample_idx].detach().cpu().numpy())
        # Compute model forecast (median quantile) for this sample
        yhat = None
        try:
            if device is None:
                device = next(model.parameters()).device
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

        import matplotlib.pyplot as plt
        import pandas as pd

        # Convert dates to pandas datetime for nicer x-axis, if available
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
        fig, axes = plt.subplots(rows, 2, figsize=(12, max(6, rows * 1.6)), sharex=False)
        if rows == 1:
            # Ensure axes is 2D array-like
            axes = np.array([axes])

        # Plot encoder covariates
        for i in range(E):
            ax = axes[i, 0]
            ax.plot(past_x, past[:, i], lw=1.5)
            ax.set_title(f"Enc: {enc_vars[i] if i < len(enc_vars) else i}")
            ax.grid(True, alpha=0.2)

        axes[0, 0].set_ylabel("Value")
        axes[min(E, rows) - 1, 0].set_xlabel("Past timeline")

        # Plot decoder covariates and optional target overlay in bottom panel
        for j in range(D):
            ax = axes[j, 1]
            ax.plot(future_x, future[:, j], lw=1.5, color="tab:blue")
            ax.set_title(f"Dec: {dec_vars[j] if j < len(dec_vars) else j}")
            ax.grid(True, alpha=0.2)

        # If target/prediction exist, overlay them on the last decoder panel
        ax_t = axes[min(D, rows) - 1, 1]
        if target is not None:
            ax_t.plot(future_x, target, lw=2.0, color="tab:red", label="target")
        if yhat is not None:
            ax_t.plot(future_x, yhat, lw=2.0, color="tab:green", label="prediction")
        if target is not None or yhat is not None:
            ax_t.legend(loc="upper right")

        axes[0, 1].set_ylabel("Value")
        axes[min(D, rows) - 1, 1].set_xlabel("Future horizon")

        fig.suptitle("TFT Input Set (Sample 0): Past & Future Covariates")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot input set: {e}")


def run_tft_intrinsic_once(enc_len=56, dec_len=28, stride=1,
                           d_model=64, hidden_dim=128,
                           heads=4, lstm_hidden=64, lstm_layers=1, dropout=0.1,
                           checkpoint_path=os.path.join(
                               "TFT", "checkpoints", "tft_best_train_final.pt"
                               ),
                           out_dir=os.path.join("TFT", "checkpoints"),
                           device: torch.device | None = None):
    """Load panel, model + checkpoint, take one val batch,
    and dump intrinsic XAI arrays.

    Saves: xai_attn.npy, xai_vsn_enc.npy, xai_vsn_dec.npy,
    xai_vsn_static.npy under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load panel
    panel_path = os.path.join("TFT", "data", "panel.csv")
    assert os.path.exists(panel_path), "Run data preprocessing first."
    df = pd.read_csv(panel_path, parse_dates=["date"])

    # Splits identical to train script
    max_date = df["date"].max()
    test_days = pd.Timedelta(days=dec_len)
    val_days = pd.Timedelta(days=dec_len)
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days
    split_bounds = (train_end, val_end, test_end)

    static_maps = build_onehot_maps(df, static_cols)

    test_ds = TFTWindowDataset(
        df, enc_len, dec_len, enc_vars, dec_vars, static_cols,
        split_bounds, split="test", stride=stride,
        static_onehot_maps=static_maps,
    )
    test_loader = DataLoader(test_ds, batch_size=64,
                             shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=tft_collate
                             )

    # Model definition must match training config for shapes;
    # we only need forward for XAI
    past_input_dims = [1] * len(enc_vars)
    future_input_dims = [1] * len(dec_vars)
    static_input_dims = [len(static_maps[c]) for c in static_cols]
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint if present
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    # One batch
    batch = next(iter(test_loader))
    # Plot the input set used for inference (with forecast overlay)
    try:
        plot_path = os.path.join(out_dir, "xai_input_set.png")
        plot_input_set(model, batch, enc_vars, dec_vars, plot_path, device=device)
        print(f"Saved input set plot -> {plot_path}")
    except Exception as e:
        print(f"Could not save input set plot: {e}")
    intr = extract_tft_intrinsic(model, batch, device=device)
    enc_imp, dec_imp, stat_imp = summarize_vsn(
        intr["vsn_enc"],
        intr["vsn_dec"],
        intr["vsn_static"]
        )
    attn = summarize_attention(intr["attn_weights"])

    # Save arrays if available
    if attn is not None:
        np.save(os.path.join(out_dir, "xai_attn.npy"), attn)
    if enc_imp is not None:
        np.save(os.path.join(out_dir, "xai_vsn_enc.npy"), enc_imp)
    if dec_imp is not None:
        np.save(os.path.join(out_dir, "xai_vsn_dec.npy"), dec_imp)
    if stat_imp is not None:
        np.save(os.path.join(out_dir, "xai_vsn_static.npy"), stat_imp)
    return {
        "attn_shape": None if attn is None else attn.shape,
        "vsn_enc_shape": None if enc_imp is None else enc_imp.shape,
        "vsn_dec_shape": None if dec_imp is None else dec_imp.shape,
        "vsn_static_shape": None if stat_imp is None else stat_imp.shape,
    }


if __name__ == "__main__":
    # Prefer GPU when available; allow override via environment
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes = run_tft_intrinsic_once(device=dev)
    print("Saved TFT XAI arrays. Shapes:", shapes)
    # Plot attention heatmap if available
    attn_path = os.path.join("TFT", "checkpoints", "xai_attn.npy")
    if os.path.exists(attn_path):
        attn = np.load(attn_path)
        # Ensure 2D matrix for imshow
        if attn.ndim > 2:
            # average all leading dims to get [L_dec, L_enc]
            while attn.ndim > 2:
                attn = attn.mean(axis=0)

        A_enc = attn[:, :56]
        plt.figure(figsize=(8, 6))
        im = plt.imshow(A_enc, aspect='auto', cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Encoder timestep (past)")
        plt.ylabel("Decoder timestep (forecast horizon)")
        plt.title("TFT Decoder-to-Encoder Attention")
        out_png = os.path.join("TFT", "checkpoints", "xai_attn_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Saved attention heatmap -> {out_png}")

    # Plot encoder/decoder/static VSN importance if saved
    enc_path = os.path.join("TFT", "checkpoints", "xai_vsn_enc.npy")
    dec_path = os.path.join("TFT", "checkpoints", "xai_vsn_dec.npy")
    stat_path = os.path.join("TFT", "checkpoints", "xai_vsn_static.npy")

    if os.path.exists(enc_path):
        enc_imp = np.load(enc_path)  # [V_enc]
        plt.figure(figsize=(10, 3))
        plt.bar(enc_vars, enc_imp)
        plt.title("Encoder Variable Importance (avg)")
        plt.xlabel("Encoder variables index")
        plt.ylabel("Importance")
        out_png = os.path.join("TFT", "checkpoints", "xai_vsn_enc_bar.png")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Saved encoder importance bar -> {out_png}")

    if os.path.exists(dec_path):
        dec_imp = np.load(dec_path)  # [V_dec]
        plt.figure(figsize=(10, 3))
        plt.bar(dec_vars, dec_imp)
        plt.title("Decoder Variable Importance (avg)")
        plt.xlabel("Decoder variables index")
        plt.ylabel("Importance")
        out_png = os.path.join("TFT", "checkpoints", "xai_vsn_dec_bar.png")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Saved decoder importance bar -> {out_png}")

    if os.path.exists(stat_path):
        stat_imp = np.load(stat_path)  # [V_static]
        plt.figure(figsize=(10, 3))
        plt.bar(static_cols, stat_imp)
        plt.title("Static Variable Importance (avg)")
        plt.xlabel("Static variables index")
        plt.ylabel("Importance")
        out_png = os.path.join("TFT", "checkpoints", "xai_vsn_static_bar.png")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Saved static importance bar -> {out_png}")

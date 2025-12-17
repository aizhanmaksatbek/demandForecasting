import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from architecture.tft import TemporalFusionTransformer
from tft_dataset import TFTWindowDataset, tft_collate
from config.settings import enc_vars, dec_vars, static_cols


def wape_loader(model, loader, device, quantiles=(0.1, 0.5, 0.9)):
    model.eval()
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
    with torch.no_grad():
        num = 0.0
        den = 0.0
        for batch in loader:
            past = batch["past_inputs"].to(device)
            future = batch["future_inputs"].to(device)
            static = batch["static_inputs"].to(device)
            y = batch["target"].to(device)
            out = model(past, future, static, return_attention=False)
            yhat = out["prediction"][..., median_idx]
            num += torch.abs(yhat - y).sum().item()
            den += torch.abs(y).sum().item() + 1e-8
    return num / den


def perm_importance(model, loader, device, feature_space: str, var_idx: int,
                    repeats: int = 3, quantiles=(0.1, 0.5, 0.9)):
    """Permutation importance for a single variable index.

    feature_space: 'encoder' | 'decoder' | 'static'
    var_idx: index of variable within that space
    """
    start = time.time()
    base = wape_loader(model, loader, device, quantiles)
    deltas = []
    for _ in range(repeats):
        model.eval()
        median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
        with torch.no_grad():
            num = 0.0
            den = 0.0
            for batch in loader:
                past = batch["past_inputs"].clone().to(device)
                future = batch["future_inputs"].clone().to(device)
                static = batch["static_inputs"].clone().to(device)
                y = batch["target"].to(device)
                if feature_space == "encoder":
                    x = past[..., var_idx]  # [B, L_enc]
                    flat = x.reshape(-1)
                    perm = flat[torch.randperm(flat.numel())].reshape_as(x)
                    past[..., var_idx] = perm
                elif feature_space == "decoder":
                    x = future[..., var_idx]  # [B, L_dec]
                    flat = x.reshape(-1)
                    perm = flat[torch.randperm(flat.numel())].reshape_as(x)
                    future[..., var_idx] = perm
                elif feature_space == "static":
                    s = static[..., var_idx]  # [B]
                    s_perm = s[torch.randperm(s.numel())]
                    static[..., var_idx] = s_perm
                else:
                    raise ValueError(
                        "feature_space must be encoder|decoder|static"
                        )

                out = model(past, future, static, return_attention=False)
                yhat = out["prediction"][..., median_idx]
                yhat = yhat.to(device)
                num += torch.abs(yhat - y).sum().item()
                den += torch.abs(y).sum().item() + 1e-8
        deltas.append(num / den)
    elapsed = time.time() - start
    return np.mean(deltas) - base, elapsed


def build_test_loader(enc_len=56, dec_len=28, stride=1):
    panel_path = os.path.join("TFT", "data", "panel.csv")
    assert os.path.exists(panel_path), "Missing TFT/data/panel.csv"
    df = pd.read_csv(panel_path, parse_dates=["date"])
    max_date = df["date"].max()
    test_days = pd.Timedelta(days=dec_len)
    val_days = pd.Timedelta(days=dec_len)
    test_end = max_date
    val_end = test_end - test_days
    train_end = val_end - val_days
    split_bounds = (train_end, val_end, test_end)

    # one-hot maps for static and dims per static var
    maps = {}
    static_dims = []
    for c in static_cols:
        cats = sorted(df[c].dropna().unique().tolist())
        idx = {v: i for i, v in enumerate(cats)}
        dim = len(cats)
        static_dims.append(dim)
        maps[c] = {}
        eye = np.eye(dim, dtype=np.float32)
        for v in cats:
            maps[c][v] = eye[idx[v]]

    test_ds = TFTWindowDataset(
        df, enc_len, dec_len, enc_vars, dec_vars, static_cols,
        split_bounds, split="test", stride=stride, static_onehot_maps=maps,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=True,
                             collate_fn=tft_collate
                             )
    return test_loader, enc_vars, dec_vars, static_cols, static_dims


def build_model(enc_vars, dec_vars, static_cols, static_dims,
                d_model=64, hidden_dim=128, heads=4,
                lstm_hidden=64, lstm_layers=1, dropout=0.1,
                quantiles=(0.1, 0.5, 0.9), checkpoint_path=None):
    past_input_dims = [1] * len(enc_vars)
    future_input_dims = [1] * len(dec_vars)
    # compute dims for static from maps length is unknown here;
    # approximate by len categories captured in dataset runtime
    # For black-box importance, we only need shapes matching dataset tensors,
    # so static dims will be inferred from batch.
    model = TemporalFusionTransformer(
        static_input_dims=static_dims,
        past_input_dims=past_input_dims,
        future_input_dims=future_input_dims,
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_heads=heads,
        lstm_hidden_size=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
        num_quantiles=len(quantiles),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    return model, device


def run_permutation_suite(device: torch.device | None = None):
    out_dir = os.path.join("TFT", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    test_loader, enc_vars, dec_vars, static_cols, static_dims = (
        build_test_loader()
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = build_model(
        enc_vars,
        dec_vars,
        static_cols,
        static_dims,
        checkpoint_path=os.path.join("TFT", "checkpoints", "tft_best_train_final.pt")
        )

    print("[XAI] Starting permutation importance...")
    # Compute importance for decoder variables (commonly actionable)
    results = {}
    for i, name in enumerate(dec_vars):
        print(f"[XAI] Decoder var {i+1}/{len(dec_vars)}: {name}...")
        delta, secs = perm_importance(
            model, test_loader, device, "decoder", i, repeats=3
            )
        results[f"decoder::{name}"] = float(delta)
        print(f"[XAI] Done {name}: ΔWAPE={delta:.5f} in {secs:.1f}s")

    # Encoder variables
    for i, name in enumerate(enc_vars):
        print(f"[XAI] Encoder var {i+1}/{len(enc_vars)}: {name}...")
        delta, secs = perm_importance(
            model, test_loader, device, "encoder", i, repeats=3
            )
        results[f"encoder::{name}"] = float(delta)
        print(f"[XAI] Done {name}: ΔWAPE={delta:.5f} in {secs:.1f}s")

    # Static variables (permutation across batch)
    for i, name in enumerate(static_cols):
        print(f"[XAI] Static var {i+1}/{len(static_cols)}: {name}...")
        delta, secs = perm_importance(
            model, test_loader, device, "static", i, repeats=3
            )
        results[f"static::{name}"] = float(delta)
        print(f"[XAI] Done {name}: ΔWAPE={delta:.5f} in {secs:.1f}s")

    # Save JSON-like CSV
    rows = [
        (k.split("::")[0], k.split("::")[1], v)
        for k, v in results.items()
        ]
    df = pd.DataFrame(rows, columns=["space", "variable", "wape_delta"])
    df.sort_values(["space", "wape_delta"], ascending=[True, False]).to_csv(
        os.path.join(out_dir, "xai_permutation_importance.csv"), index=False
    )
    print("[XAI] Saved permutation importance ->",
          os.path.join(out_dir, "xai_permutation_importance.csv"))
    print("[XAI] Completed permutation importance for",
          f"{len(enc_vars)} encoder, {len(dec_vars)} decoder,",
          f"{len(static_cols)} static variables."
          )


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_permutation_suite(device=dev)

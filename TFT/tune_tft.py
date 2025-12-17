import os
import sys
import subprocess
import argparse
import re
import json
import optuna
from datetime import datetime
try:
    # tqdm progress bar for Optuna
    from optuna.integration import TQDMCallback
except Exception:
    TQDMCallback = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_train_cli(hparams, fixed_epochs):
    """
    Launch TFT/train_tft.py as a module with mapped CLI args.
    Parse WAPE from stdout; prefer Validation line, fallback to Test metrics.
    """
    cmd = [
        sys.executable, "-m", "TFT.train_tft",
        "--enc-len", str(hparams["enc_len"]),
        "--dec-len", str(hparams["dec_len"]),
        "--batch-size", str(hparams["batch_size"]),
        "--epochs", str(fixed_epochs),
        "--lr", str(hparams["lr"]),
        "--hidden-dim", str(hparams["hidden_dim"]),
        "--d-model", str(hparams["d_model"]),
        "--heads", str(hparams["heads"]),
        "--dropout", str(hparams["dropout"]),
        "--stride", "1",
        "--seed", "42",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env.get("PYTHONPATH", ""))
    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Training failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
            )

    wape = None
    # Parse validation print (dict) or test metrics line
    for line in proc.stdout.splitlines():
        if "Validation" in line and "WAPE:" in line:
            try:
                wape = float(line.split("WAPE:")[1].split("|")[0].strip())
                break
            except Exception:
                pass
        if "Test matrics:" in line and "'wape':" in line:
            m = re.search(r"'wape':\s*([0-9\.eE+-]+)", line)
            if m:
                wape = float(m.group(1))
    if wape is None:
        # Last resort: scan any dict-looking line for wape
        for line in proc.stdout.splitlines():
            m = re.search(r"'wape':\s*([0-9\.eE+-]+)", line)
            if m:
                wape = float(m.group(1))
                break
    if wape is None:
        raise RuntimeError(
            "Could not parse WAPE from train output.\n" + proc.stdout
            )
    return wape


def objective(trial: optuna.Trial):
    # Simple search space (no epoch search)
    hparams = {
        "enc_len": trial.suggest_categorical("enc_len", [56, 90]),
        "dec_len": trial.suggest_categorical("dec_len", [15, 28, 60, 90]),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048]),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "heads": trial.suggest_categorical("heads", [4, 8, 32, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
    }
    fixed_epochs = 20
    wape = run_train_cli(hparams, fixed_epochs=fixed_epochs)
    return wape  # minimize


def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _trial_csv_logger(log_path):
    """Optuna callback: append each trial's params and value to CSV."""
    header_written = os.path.exists(log_path)

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        nonlocal header_written
        _ensure_dir(log_path)
        # Compose row
        row = {
            "datetime": datetime.utcnow().isoformat(),
            "trial_number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
        }
        # Flatten params
        for k, v in trial.params.items():
            row[f"param_{k}"] = v
        # Write CSV (append)
        import csv
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)

    return _cb


def _trial_text_logger(log_path):
    """Optuna callback: append a human-readable line per trial."""
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        _ensure_dir(log_path)
        line = (
            f"{datetime.utcnow().isoformat()} | trial={trial.number} "
            f"state={trial.state} value={trial.value} best={getattr(study, 'best_value', None)} "
            f"params={trial.params}\n"
        )
        with open(log_path, "a") as f:
            f.write(line)
    return _cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--study_name", type=str, default="tft_tuning")
    ap.add_argument("--direction", type=str, default="minimize")
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage URL, e.g. sqlite:///TFT/tft_optuna.db")
    ap.add_argument("--log_csv", type=str, default=os.path.join("TFT", "tuning_results.csv"),
                    help="CSV file to append per-trial results")
    ap.add_argument("--log_text", type=str, default=os.path.join("TFT", "tuning_progress.log"),
                    help="Text log capturing per-trial progress")
    args = ap.parse_args()

    # Create study with optional storage for persistent tracking
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name, direction=args.direction
        )

    # Set per-trial loggers
    trial_cb = _trial_csv_logger(os.path.join(PROJECT_ROOT, args.log_csv))
    text_cb = _trial_text_logger(os.path.join(PROJECT_ROOT, args.log_text))

    callbacks = [trial_cb, text_cb]
    if TQDMCallback is not None:
        callbacks.append(TQDMCallback(n_trials=args.trials, metric_name="WAPE"))

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
        callbacks=callbacks,
    )

    print("Best WAPE:", study.best_value)
    print("Best params:", study.best_params)

    out_json = os.path.join(PROJECT_ROOT, "TFT", "tft_best_params.json")
    with open(out_json, "w") as f:
        json.dump(
            {"best_value": study.best_value, "best_params": study.best_params},
            f,
            indent=2
            )
    print(f"Saved best params -> {out_json}")

    # Also export full trials dataframe if pandas is available
    try:
        import pandas as pd
        df = study.trials_dataframe()
        out_csv = os.path.join(PROJECT_ROOT, "TFT", "tuning_trials_full.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved all trial results -> {out_csv}")
    except Exception as e:
        print(f"Could not write full trials CSV: {e}")


if __name__ == "__main__":
    main()

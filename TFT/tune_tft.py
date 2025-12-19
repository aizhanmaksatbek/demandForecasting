import os
import sys
import subprocess
import argparse
import re
import json
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


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
        if "Test Metrics:" in line and "'wape':" in line:
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
        "batch_size": trial.suggest_categorical(
            "batch_size", [256, 512, 1024]
            ),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "heads": trial.suggest_categorical("heads", [4, 8, 32, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
    }
    fixed_epochs = 1
    wape = run_train_cli(hparams, fixed_epochs=fixed_epochs)
    return wape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--study_name", type=str, default="tft_tuning")
    ap.add_argument("--direction", type=str, default="minimize")
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage log file name")
    args = ap.parse_args()

    # Create study with optional storage for persistent tracking
    # create a file if doesn't exist optuna_journal.log
    if not os.path.exists("TFT/checkpoints/optuna_journal.log"):
        open("TFT/checkpoints/optuna_journal.log", 'a').close()

    storage = JournalStorage(
        JournalFileBackend("TFT/checkpoints/optuna_journal.log")
        )
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True
    )

    print("Best WAPE:", study.best_value)
    print("Best params:", study.best_params)

    out_json = os.path.join(
        PROJECT_ROOT, "checkpoints", "tft_best_params.json"
        )
    with open(out_json, "w") as f:
        json.dump(
            {"best_value": study.best_value, "best_params": study.best_params},
            f,
            indent=2
            )
    print(f"Saved best params -> {out_json}")

    # Also export full trials dataframe if pandas is available
    try:
        df = study.trials_dataframe()
        out_csv = os.path.join(
            PROJECT_ROOT, "checkpoints", "tuning_trials_full.csv"
            )
        df.to_csv(out_csv, index=False)
        print(f"Saved all trial results -> {out_csv}")
    except Exception as e:
        print(f"Could not write full trials CSV: {e}")


if __name__ == "__main__":
    main()

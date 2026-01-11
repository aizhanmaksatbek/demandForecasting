import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_mae_comparison(models: List[str], mae_data: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot grouped bar chart comparing MAE across Train, Validation, and Test for multiple models.

    Args:
        models: List of model names (e.g., ["Model 1", "Model 2", ...]).
        mae_data: Dict with keys "Train", "Validation", "Test" mapping to lists of MAE values
                  corresponding to models order.
        save_path: Path to save the generated PNG.
    """
    categories = ["Train", "Validation", "Test"]

    # Validate inputs
    num_models = len(models)
    for cat in categories:
        if cat not in mae_data:
            raise ValueError(f"Missing category '{cat}' in mae_data")
        if len(mae_data[cat]) != num_models:
            raise ValueError(f"Length mismatch for category '{cat}': expected {num_models}, got {len(mae_data[cat])}")

    x = np.arange(len(categories))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose a color palette
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    # Plot bars for each model
    for i, model in enumerate(models):
        offsets = (i - (num_models - 1) / 2) * width
        values = [mae_data[cat][i] for cat in categories]
        bars = ax.bar(x + offsets, values, width, label=model, color=colors[i % len(colors)])

        # Add value labels on top of bars
        ax.bar_label(bars, fmt="{:.2f}", padding=3, fontsize=8)

    ax.set_ylabel("MAE")
    ax.set_title("TFT Models: MAE Comparison Across Train / Validation / Test")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(title="Models", ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Saved MAE comparison plot to: {save_path}")


def main():
    # Model names (feel free to rename to actual model names)
    models = [
        "Model 1",
        "Model 2",
        "Model 3",
        "Model 4",
    ]

    # MAE values extracted from the provided logs
    mae_data = {
        "Train": [
            42.6364,  # Early stopping 81/200
            54.9210,  # Early stopping 14/100
            63.4196,  # Early stopping 27/100
            43.8038,  # Early stopping 19/100 (MAE')
        ],
        "Validation": [
            62.6490,
            71.0156,
            74.1687,
            63.7361,
        ],
        "Test": [
            61.3641,
            70.6296,
            71.5808,
            64.9195,
        ],
    }

    save_path = os.path.join(os.path.dirname(__file__), "..", "improvements", "mae_comparison.png")
    plot_mae_comparison(models, mae_data, save_path)


if __name__ == "__main__":
    main()

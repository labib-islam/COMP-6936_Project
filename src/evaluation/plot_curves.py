# Plots training curves from the log.csv of every run (no test data is used).
# One row per model, three seeds per panel: loss (train vs val) and val macro-F1.
# The dotted line marks the start of stage 2 (full fine-tuning).
# Saves results/figures/learning_curves.png
# Run from the project root:
#   uv run python -m src.evaluation.plot_curves

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.training.models import model_names

runs_dir = Path("runs")
fig_dir = Path("results/figures")

if __name__ == "__main__":
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(model_names), 2, figsize=(11, 3.2 * len(model_names)), squeeze=False)

    for row, model in enumerate(model_names):
        ax_loss, ax_f1 = axes[row]
        for log_file in sorted(runs_dir.glob(f"{model}_*/log.csv")):
            with open(log_file.parent / "summary.json") as f:
                seed = json.load(f)["seed"]
            log = pd.read_csv(log_file)

            line = ax_loss.plot(log["epoch"], log["train_loss"], label=f"train, seed {seed}")[0]
            ax_loss.plot(log["epoch"], log["val_loss"], "--", color=line.get_color(),
                         label=f"val, seed {seed}")
            ax_f1.plot(log["epoch"], log["val_f1"], color=line.get_color(), label=f"seed {seed}")

            stage2_start = log[log["stage"] == 2]["epoch"].min()

        for ax in (ax_loss, ax_f1):
            ax.axvline(stage2_start - 0.5, color="gray", linestyle=":")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        ax_loss.set_ylabel("loss")
        ax_loss.set_title(f"{model}: loss")
        ax_f1.set_ylabel("val macro-F1")
        ax_f1.set_title(f"{model}: val macro-F1")
        ax_f1.set_ylim(0.9, 1.0)
        ax_loss.legend(fontsize=7)
        ax_f1.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=120)
    plt.close(fig)
    print(f"Saved {fig_dir / 'learning_curves.png'}")

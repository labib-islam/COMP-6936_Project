# Quick sanity check of the data pipeline before training.
# Prints dataset sizes and batch info, and saves two figures:
#   results/figures/check_letterbox.png     non-square images before and after letterboxing
#   results/figures/check_augmentation.png  one image with 8 random train augmentations
# Run from the project root:
#   uv run python -m src.preprocessing.check_pipeline

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.preprocessing.dataset import BriscDataset
from src.preprocessing.transforms import eval_transform, mean, std, train_transform

fig_dir = Path("results/figures")


def to_display(tensor):
    # undo the normalization so the image can be shown
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = tensor * std_t + mean_t
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


if __name__ == "__main__":
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. dataset sizes
    train_ds = BriscDataset("train", train_transform)
    val_ds = BriscDataset("val", eval_transform)
    test_ds = BriscDataset("test", eval_transform)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # 2. one batch from each loader
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
        images, labels, paths = next(iter(loader))
        print(f"{name} batch: {tuple(images.shape)}  {images.dtype}  "
              f"min {images.min():.2f}  max {images.max():.2f}  labels {labels.tolist()}")

    # 3. the most non-square images: original next to letterboxed
    df = pd.read_csv("data/splits.csv")
    df["ratio"] = df["width"] / df["height"]
    tall = df.nsmallest(2, "ratio")
    wide = df.nlargest(2, "ratio")
    examples = pd.concat([tall, wide])

    fig, axes = plt.subplots(2, len(examples), figsize=(3 * len(examples), 6))
    for col, (_, row) in enumerate(examples.iterrows()):
        with Image.open(row["path"]) as img:
            original = img.convert("RGB")
            processed = eval_transform(img)

        axes[0, col].imshow(original)
        axes[0, col].set_title(f"original {row['width']}x{row['height']}", fontsize=9)
        axes[1, col].imshow(to_display(processed))
        axes[1, col].set_title("letterboxed 384x384", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "check_letterbox.png", dpi=100)
    plt.close(fig)

    # 4. one train image with 8 random augmentations
    path = df[df["split"] == "train"].iloc[0]["path"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    with Image.open(path) as img:
        axes[0, 0].imshow(to_display(eval_transform(img)))
        axes[0, 0].set_title("no augmentation", fontsize=9)
        for k in range(1, 9):
            ax = axes.flat[k]
            ax.imshow(to_display(train_transform(img)))
            ax.set_title(f"augmented {k}", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "check_augmentation.png", dpi=100)
    plt.close(fig)

    print(f"Saved figures to {fig_dir}")

# Tumor masks in the same 384x384 letterboxed frame as the model input, so they can be
# compared pixel by pixel with the Grad-CAM and IG heatmaps.
# Import with:
#   from src.xai.masks import mask_path, load_mask, valid_region
# Run directly to check that the masks line up with the letterboxed images and that
# small tumors survive the resizing:
#   uv run python -m src.xai.masks
# Saves results/figures/check_masks.png

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing.transforms import image_size, letterbox

seg_dir = Path("dataset/brisc2025/segmentation_task")
fig_dir = Path("results/figures")


def mask_path(image_path):
    # .../classification_task/<train or test>/<label>/X.jpg  ->  segmentation_task/<train or test>/masks/X.png
    image_path = Path(image_path)
    orig_split = image_path.parts[-3]
    return seg_dir / orig_split / "masks" / (image_path.stem + ".png")


def letterbox_geometry(width, height, size=image_size):
    # same arithmetic as letterbox() in transforms.py
    scale = size / max(width, height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    return new_width, new_height, left, top


def load_mask(path, size=image_size):
    # returns a size x size boolean array, True inside the tumor
    with Image.open(path) as img:
        mask = np.array(img.convert("L")) > 127   # remove compression noise
    height, width = mask.shape
    new_width, new_height, left, top = letterbox_geometry(width, height, size)

    # nearest-neighbour resizing keeps the mask strictly 0 or 1
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((new_width, new_height), Image.NEAREST)
    canvas = np.zeros((size, size), dtype=bool)
    canvas[top:top + new_height, left:left + new_width] = np.array(small) > 127
    return canvas


def valid_region(width, height, size=image_size):
    # size x size boolean array, True where the real image is and False on the black padding
    new_width, new_height, left, top = letterbox_geometry(width, height, size)
    region = np.zeros((size, size), dtype=bool)
    region[top:top + new_height, left:left + new_width] = True
    return region


def show(ax, row, title):
    with Image.open(row["path"]) as img:
        ax.imshow(letterbox(img))
    mask = load_mask(mask_path(row["path"]))
    region = valid_region(row["width"], row["height"])
    ax.contour(mask, levels=[0.5], colors="red", linewidths=1)
    ax.contour(region, levels=[0.5], colors="yellow", linewidths=0.8, linestyles="dashed")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


if __name__ == "__main__":
    df = pd.read_csv("data/splits.csv")
    tumors = df[df["label"] != "no_tumor"].copy()

    # 1. compare tumor area before and after letterboxing, for every val and test mask
    rows = []
    for row in tumors[tumors["split"] != "train"].itertuples():
        with Image.open(mask_path(row.path)) as img:
            original = np.array(img.convert("L")) > 127
        boxed = load_mask(mask_path(row.path))
        region = valid_region(row.width, row.height)
        rows.append({"path": row.path, "split": row.split,
                     "area_original": original.mean(),
                     "area_boxed": boxed.sum() / region.sum(),
                     "pixels_boxed": int(boxed.sum())})
    areas = pd.DataFrame(rows)
    areas["change"] = areas["area_boxed"] / areas["area_original"] - 1

    print(f"Checked {len(areas)} masks (val {sum(areas['split'] == 'val')}, "
          f"test {sum(areas['split'] == 'test')})")
    print(f"Empty after letterboxing: {(areas['pixels_boxed'] == 0).sum()}")
    print(f"Smallest mask after letterboxing: {areas['pixels_boxed'].min()} pixels")
    print(f"Tumor area change (share of the real image): median {areas['change'].median():+.2%}, "
          f"worst {areas['change'].abs().max():.2%}")

    # 2. figure: one random image per class and plane, plus the three smallest tumors
    fig, axes = plt.subplots(4, 3, figsize=(10, 13.5))
    examples = tumors[tumors["split"] == "test"]
    for r, label in enumerate(["glioma", "meningioma", "pituitary"]):
        for c, plane in enumerate(["ax", "co", "sa"]):
            row = examples[(examples["label"] == label) & (examples["plane"] == plane)].sample(1, random_state=0).iloc[0]
            show(axes[r, c], row, f"{label} {plane} ({row['width']}x{row['height']})\n{Path(row['path']).name}")

    smallest = areas.nsmallest(3, "pixels_boxed")
    for c, small in enumerate(smallest.itertuples()):
        row = df[df["path"] == small.path].iloc[0]
        show(axes[3, c], row, f"smallest: {small.pixels_boxed} pixels\n{Path(row['path']).name}")

    fig.suptitle("Letterboxed images with mask (red) and real image area (yellow)")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "check_masks.png", dpi=100)
    plt.close(fig)
    print(f"Saved {fig_dir / 'check_masks.png'}")

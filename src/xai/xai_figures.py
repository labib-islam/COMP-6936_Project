# Heatmap figures for the report, made from the maps saved by explain.py
# (results/xai/<run>/maps/) and the per-image scores. No model is run.
#
#   results/figures/xai_examples.png  selected model: best and worst example per tumor class
#   results/figures/xai_models.png    one typical image per class, Grad-CAM and IG for all 4 models
#   results/figures/xai_errors.png    selected model's wrong predictions: maps for the predicted
#                                     and for the true class
#
# In every panel: white outline = tumor mask, white + = hottest pixel (the pointing-game point).
# IG maps are speckled, so for display only their top 0.5% of pixels are saturated.
# Uses the representative seed of each model.
# Run from the project root:
#   uv run python -m src.xai.xai_figures

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing.transforms import letterbox
from src.xai.masks import load_mask, mask_path, valid_region

xai_dir = Path("results/xai")
results_dir = Path("results")
fig_dir = results_dir / "figures"
tumor_classes = ["glioma", "meningioma", "pituitary"]
method_names = {"gradcam": "Grad-CAM", "ig_smooth": "IG"}


# ---------- loading ----------

def representative_runs():
    # (model, run folder) for each model, best mean val F1 first (= the selected model)
    summary = pd.read_csv(results_dir / "test_summary.csv").sort_values("val_f1_mean", ascending=False)
    runs = []
    for row in summary.itertuples():
        run_dir = next(xai_dir.glob(f"{row.model}_lr*_seed{row.rep_seed}"))
        runs.append((row.model, run_dir))
    return runs


def load_scores(run_dir):
    # scores of the predicted-class maps, one row per image and method
    frames = [pd.read_csv(f) for f in run_dir.glob("*_scores.csv")]
    scores = pd.concat(frames, ignore_index=True)
    return scores[scores["target"] == "pred"]


def load_map(run_dir, path, method, target="pred"):
    file_kind = "gradcam" if method == "gradcam" else "ig"
    maps = np.load(run_dir / "maps" / f"{Path(path).stem}_{file_kind}.npz")
    return maps[f"{method}_{target}"].astype(np.float32)


def saved_paths(run_dir):
    stems = {f.name.replace("_gradcam.npz", "") for f in (run_dir / "maps").glob("*_gradcam.npz")}
    return stems


@lru_cache(maxsize=None)
def load_view(path):
    # the letterboxed image, its mask and the real-image area, all 384x384
    df = load_splits()
    width, height = df.loc[path, ["width", "height"]]
    with Image.open(path) as img:
        image = np.array(letterbox(img))
    return image, load_mask(mask_path(path)), valid_region(width, height)


@lru_cache(maxsize=None)
def load_splits():
    return pd.read_csv("data/splits.csv").set_index("path")


def score_of(scores, path, method):
    return scores[(scores["path"] == path) & (scores["method"] == method)].iloc[0]


# ---------- drawing one panel ----------

def show_image(ax, path, title):
    image, mask, _ = load_view(path)
    ax.imshow(image)
    ax.contour(mask, levels=[0.5], colors="red", linewidths=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_heatmap(ax, path, heatmap, method, title):
    image, mask, region = load_view(path)
    if method != "gradcam":
        heatmap = np.clip(heatmap / (np.percentile(heatmap, 99.5) + 1e-8), 0, 1)   # display only
    peak = np.unravel_index(np.argmax(np.where(region, heatmap, -np.inf)), heatmap.shape)

    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors="white", linewidths=0.8)
    ax.plot(peak[1], peak[0], "w+", markersize=12, markeredgewidth=2)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def score_title(name, score):
    hit = "hit" if score["pointing_hit"] else "miss"
    return f"{name}: {hit}, energy x{score['energy_ratio']:.1f}"


def rank_examples(scores, fixed_paths):
    # orders the fixed example images from worst to best: first by how many maps hit the
    # tumor (pointing game), then by mean energy inside the tumor
    both = scores[scores["path"].isin(fixed_paths) & scores["method"].isin(method_names)]
    ranked = both.groupby(["path", "label"]).agg(hits=("pointing_hit", "sum"),
                                                 energy=("energy", "mean")).reset_index()
    return ranked.sort_values(["hits", "energy"])


# ---------- figure 1: best and worst examples for the selected model ----------

def figure_examples(model, run_dir, scores, fixed_paths):
    # per class: the best and the worst of the fixed example images
    ranked = rank_examples(scores, fixed_paths)

    rows = []
    for label in tumor_classes:
        part = ranked[ranked["label"] == label]
        rows.append(("best example", part.iloc[-1]["path"]))
        rows.append(("worst example", part.iloc[0]["path"]))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)))
    for r, (kind, path) in enumerate(rows):
        label = load_splits().loc[path, "label"]
        show_image(axes[r, 0], path, f"{label}, {kind}\n{Path(path).name}")
        for c, method in enumerate(method_names, start=1):
            title = score_title(method_names[method], score_of(scores, path, method))
            show_heatmap(axes[r, c], path, load_map(run_dir, path, method), method, title)

    fig.suptitle(f"{model} ({run_dir.name.split('_')[-1]}): best and worst example per class (of 5 fixed images)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the figure title
    fig.savefig(fig_dir / "xai_examples.png", dpi=110)
    plt.close(fig)


# ---------- figure 2: the same images across all models ----------

def figure_models(runs, all_scores, fixed_paths):
    # per class, the most typical image (the middle one when ranked over all models and methods)
    ranked = rank_examples(pd.concat(all_scores.values()), fixed_paths)
    paths = []
    for label in tumor_classes:
        part = ranked[ranked["label"] == label]
        paths.append(part.iloc[len(part) // 2]["path"])

    n_rows = len(paths) * len(method_names)
    fig, axes = plt.subplots(n_rows, len(runs) + 1, figsize=(2.8 * (len(runs) + 1), 2.9 * n_rows))
    for i, path in enumerate(paths):
        for j, method in enumerate(method_names):
            r = i * len(method_names) + j
            if j == 0:
                show_image(axes[r, 0], path, f"{load_splits().loc[path, 'label']}\n{Path(path).name}")
            else:
                axes[r, 0].axis("off")
            axes[r, 0].text(-0.05, 0.5, method_names[method], fontsize=11, rotation=90,
                            transform=axes[r, 0].transAxes, ha="right", va="center")
            for c, (model, run_dir) in enumerate(runs, start=1):
                title = f"{model}\n" + score_title(method_names[method], score_of(all_scores[model], path, method))
                show_heatmap(axes[r, c], path, load_map(run_dir, path, method), method, title)

    fig.suptitle("The same images explained by each model (representative seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the figure title
    fig.savefig(fig_dir / "xai_models.png", dpi=100)
    plt.close(fig)


# ---------- figure 3: the selected model's mistakes ----------

def figure_errors(model, run_dir, scores):
    wrong = scores[(~scores["correct"]) & (scores["method"] == "gradcam")]
    if len(wrong) == 0:
        print("No wrong predictions, skipping xai_errors.png")
        return

    columns = [("gradcam", "pred"), ("gradcam", "true"), ("ig_smooth", "pred"), ("ig_smooth", "true")]
    fig, axes = plt.subplots(len(wrong), len(columns) + 1, figsize=(2.9 * (len(columns) + 1), 3.1 * len(wrong)),
                             squeeze=False)
    for r, row in enumerate(wrong.itertuples()):
        show_image(axes[r, 0], row.path, f"{Path(row.path).name}\ntrue {row.label}, "
                                         f"pred {row.pred} ({row.confidence:.2f})")
        for c, (method, target) in enumerate(columns, start=1):
            class_name = row.pred if target == "pred" else row.label
            title = f"{method_names[method]} for '{class_name}'\n({'predicted' if target == 'pred' else 'true'} class)"
            show_heatmap(axes[r, c], row.path, load_map(run_dir, row.path, method, target), method, title)

    fig.suptitle(f"{model}: wrong predictions, explained for the predicted and the true class", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))   # leave room for the figure title
    fig.savefig(fig_dir / "xai_errors.png", dpi=100)
    plt.close(fig)


# ---------- main ----------

if __name__ == "__main__":
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = representative_runs()
    all_scores = {model: load_scores(run_dir) for model, run_dir in runs}

    # the fixed example images: saved for every model (wrong predictions differ per model)
    common = set.intersection(*(saved_paths(run_dir) for _, run_dir in runs))
    fixed_paths = [p for p in load_splits().index if Path(p).stem in common]
    print(f"{len(fixed_paths)} example images have maps for all 4 models")

    selected_model, selected_dir = runs[0]
    figure_examples(selected_model, selected_dir, all_scores[selected_model], fixed_paths)
    figure_models(runs, all_scores, fixed_paths)
    figure_errors(selected_model, selected_dir, all_scores[selected_model])
    print(f"Saved xai_examples.png, xai_models.png and xai_errors.png to {fig_dir}")

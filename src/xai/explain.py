# Explains one trained model on the 860 test tumor images and scores every heatmap
# against the tumor mask.
#
# For each image:
#   1. predict the class
#   2. make a heatmap for the predicted class (and for the true class if the prediction is wrong)
#   3. score it: pointing game and energy inside the mask (see scores.py)
#
# Each method has its own CSV, with one row per image and target:
#   <out>/<run>/gradcam_scores.csv   method "gradcam"
#   <out>/<run>/ig_scores.csv        methods "ig_raw" (no smoothing) and "ig_smooth" (Gaussian, --ig-sigma)
#   <out>/<run>/maps/<image>_<method>.npz   heatmaps of a few example images, for figures
# Images already in a CSV are skipped, so a stopped run continues where it left off.
#
# Run from the project root, for example:
#   uv run python -m src.xai.explain --model densenet121 --seed 42
#   uv run python -m src.xai.explain --model resnet50 --seed 42 --methods gradcam
# Quick test on a few images:
#   uv run python -m src.xai.explain --model densenet121 --seed 42 --limit 5 --out runs_test/xai

import os

# same memory safeguard for Macs as in train.py (must be set before torch is imported)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.preprocessing.transforms import eval_transform
from src.training.models import get_target_layer, model_names
from src.training.train import get_device
from src.xai.masks import load_mask, mask_path, valid_region
from src.xai.methods import (black_baseline, gradcam, gradcam_heatmap, ig_heatmap,
                             integrated_gradients, load_trained_model)
from src.xai.scores import area_share, score_heatmap

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=model_names)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", default="runs", help="folder with the trained runs (best.pt)")
    parser.add_argument("--out", default="results/xai")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "ig"], choices=["gradcam", "ig"])
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--ig-retry-steps", type=int, default=200, help="steps used when the delta is too large")
    parser.add_argument("--ig-max-delta", type=float, default=0.05, help="largest accepted delta (share of score change)")
    parser.add_argument("--ig-batch", type=int, default=10, help="IG steps per forward pass (memory)")
    parser.add_argument("--ig-sigma", type=float, default=4, help="Gaussian smoothing for ig_smooth (pixels)")
    parser.add_argument("--examples-per-class", type=int, default=5, help="images per class whose maps are saved")
    parser.add_argument("--limit", type=int, default=0, help="use only this many images (for testing)")
    return parser.parse_args()


# ---------- which images ----------

def find_run_dir(runs, model, seed):
    matches = sorted(Path(runs).glob(f"{model}_lr*_seed{seed}"))
    if len(matches) != 1 or not (matches[0] / "best.pt").exists():
        raise SystemExit(f"Expected one run folder with best.pt for {model} seed {seed} in {runs}, found {matches}")
    return matches[0]


def test_tumor_images():
    df = pd.read_csv("data/splits.csv")
    return df[(df["split"] == "test") & (df["label"] != "no_tumor")].reset_index(drop=True)


def example_paths(images, per_class):
    # the same random images for every model (fixed seed), so figures can compare models
    sample = images.groupby("label").sample(per_class, random_state=0)
    return set(sample["path"])


# ---------- saving ----------

def done_paths(csv_file):
    if not csv_file.exists():
        return set()
    return set(pd.read_csv(csv_file)["path"])


def append_rows(csv_file, rows):
    if rows:
        pd.DataFrame(rows).to_csv(csv_file, mode="a", header=not csv_file.exists(), index=False)


def save_maps(maps_dir, path, method, maps):
    maps_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: heatmap.astype(np.float16) for name, heatmap in maps.items()}
    np.savez_compressed(maps_dir / f"{Path(path).stem}_{method}.npz", **arrays)


# ---------- explaining one image ----------

def load_inputs(row, device):
    with Image.open(row["path"]) as img:
        image = eval_transform(img).to(device)
    mask = load_mask(mask_path(row["path"]))
    region = valid_region(row["width"], row["height"])
    return image, mask, region


def predict(model, image):
    with torch.no_grad():
        probs = torch.softmax(model(image.unsqueeze(0)), dim=1)[0]
    pred = int(probs.argmax())
    return pred, float(probs[pred])


def targets_to_explain(pred, label_idx):
    # always the predicted class; also the true class when the prediction is wrong
    targets = [("pred", pred)]
    if pred != label_idx:
        targets.append(("true", label_idx))
    return targets


def explain_gradcam(model, target_layer, image, target_class, args):
    cam = gradcam(model, target_layer, image, target_class)
    return {"gradcam": gradcam_heatmap(cam)}, {}


def explain_ig(model, image, target_class, args):
    # if the convergence delta is too large, redo the image with more steps
    for steps in [args.ig_steps, args.ig_retry_steps]:
        attributions, delta, score_change = integrated_gradients(
            model, image, black_baseline(), target_class, steps=steps, batch_size=args.ig_batch)
        delta_rel = abs(delta) / max(abs(score_change), 1e-8)
        if delta_rel <= args.ig_max_delta:
            break

    heatmaps = {"ig_raw": ig_heatmap(attributions, sigma=0),
                "ig_smooth": ig_heatmap(attributions, sigma=args.ig_sigma)}
    extra = {"ig_steps": steps, "ig_delta": delta, "ig_score_change": score_change,
             "ig_delta_rel": delta_rel}
    return heatmaps, extra


def explain_image(model, target_layer, row, method, args, device):
    # returns the CSV rows for this image and the heatmaps (for saving examples)
    image, mask, region = load_inputs(row, device)
    pred, confidence = predict(model, image)
    info = {
        "model": args.model, "seed": args.seed, "path": row["path"],
        "label": row["label"], "plane": row["plane"], "tumor_share": area_share(mask, region),
        "pred": class_names[pred], "correct": pred == row["label_idx"], "confidence": confidence,
    }

    rows = []
    all_maps = {}
    for target_name, target_class in targets_to_explain(pred, row["label_idx"]):
        if method == "gradcam":
            heatmaps, extra = explain_gradcam(model, target_layer, image, target_class, args)
        else:
            heatmaps, extra = explain_ig(model, image, target_class, args)

        for method_name, heatmap in heatmaps.items():
            rows.append({**info, "target": target_name, "target_class": class_names[target_class],
                         "method": method_name, **score_heatmap(heatmap, mask, region), **extra})
            all_maps[f"{method_name}_{target_name}"] = heatmap
    return rows, all_maps


# ---------- main loop ----------

def run_method(model, target_layer, images, examples, method, run_out, args, device):
    csv_file = run_out / f"{method}_scores.csv"
    done = done_paths(csv_file)
    todo = images[~images["path"].isin(done)]
    print(f"\n{method}: {len(done)} images already done, {len(todo)} to go -> {csv_file}")

    buffer = []
    start = time.time()
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        rows, maps = explain_image(model, target_layer, row, method, args, device)
        buffer += rows

        wrong = not rows[0]["correct"]
        if row["path"] in examples or wrong:
            save_maps(run_out / "maps", row["path"], method, maps)

        if i % 25 == 0 or i == len(todo):
            append_rows(csv_file, buffer)   # save regularly, in case the session stops
            buffer = []
            per_image = (time.time() - start) / i
            print(f"  {i}/{len(todo)} images, {per_image:.2f}s per image, "
                  f"about {per_image * (len(todo) - i) / 60:.0f} min left")


def print_summary(run_out, methods):
    print("\nSummary (predicted class, all images so far):")
    for method in methods:
        csv_file = run_out / f"{method}_scores.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        df = df[df["target"] == "pred"]
        for name, part in df.groupby("method"):
            print(f"  {name:10s} n {len(part):4d} | pointing {part['pointing_hit'].mean():.3f} | "
                  f"energy {part['energy'].mean():.3f} | energy ratio {part['energy_ratio'].mean():.1f}")
        if "ig_delta_rel" in df.columns:
            ig = df[df["method"] == "ig_raw"]
            print(f"  IG convergence: median delta {ig['ig_delta_rel'].median():.1%} of the score change, "
                  f"{(ig['ig_steps'] > ig['ig_steps'].min()).sum()} images needed more steps, "
                  f"{(ig['ig_delta_rel'] > 0.05).sum()} still above 5%")
    print("  chance level: pointing = tumor area share (about 0.02), energy ratio = 1")


if __name__ == "__main__":
    args = get_args()
    device = get_device()

    run_dir = find_run_dir(args.runs, args.model, args.seed)
    run_out = Path(args.out) / run_dir.name
    run_out.mkdir(parents=True, exist_ok=True)
    print(f"Explaining {run_dir.name} on {device}, methods {args.methods}")

    model = load_trained_model(args.model, run_dir / "best.pt", device)
    for p in model.parameters():
        p.requires_grad_(False)   # we only need gradients for the input, not the weights
    target_layer = get_target_layer(model, args.model)

    images = test_tumor_images()
    if args.limit > 0:
        images = images.sample(args.limit, random_state=0)
    examples = example_paths(test_tumor_images(), args.examples_per_class)

    for method in args.methods:
        run_method(model, target_layer, images, examples, method, run_out, args, device)

    print_summary(run_out, args.methods)

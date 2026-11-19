# Runs every trained model (runs/*/best.pt) on the test set and saves its predictions.
# Saves <run>/test_predictions.csv with one row per test image:
#   path, label, label_idx, plane, pred_idx, pred, prob_<class> for each class
# Runs that already have test_predictions.csv are skipped (use --force to redo them).
# Run from the project root:
#   uv run python -m src.evaluation.predict

import os

# same memory safeguard for Macs as in train.py (must be set before torch is imported)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.preprocessing.dataset import BriscDataset
from src.preprocessing.transforms import eval_transform
from src.training.models import build_model
from src.training.train import get_device


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs", help="folder with one subfolder per run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="redo runs that already have predictions")
    return parser.parse_args()


def predict(model, loader, device):
    model.eval()
    all_paths = []
    all_probs = []

    with torch.no_grad():
        for images, _, paths in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_probs += probs.cpu().tolist()
            all_paths += list(paths)

    return all_paths, all_probs


if __name__ == "__main__":
    args = get_args()
    device = get_device()

    test_ds = BriscDataset("test", eval_transform)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # class names in label_idx order: glioma, meningioma, no_tumor, pituitary
    class_names = test_ds.df.sort_values("label_idx")["label"].unique().tolist()
    print(f"{len(test_ds)} test images on {device}, classes {class_names}")

    run_dirs = sorted(p.parent for p in Path(args.runs).glob("*/best.pt"))
    for run_dir in run_dirs:
        out_file = run_dir / "test_predictions.csv"
        if out_file.exists() and not args.force:
            print(f"{run_dir.name}: already done, skipping")
            continue

        with open(run_dir / "summary.json") as f:
            model_name = json.load(f)["model"]

        # build the architecture without ImageNet weights, then load our trained weights
        model = build_model(model_name, pretrained=False)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
        model = model.to(device)

        t0 = time.time()
        paths, probs = predict(model, loader, device)

        preds = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_names])
        preds.insert(0, "path", paths)
        preds["pred_idx"] = preds[[f"prob_{c}" for c in class_names]].values.argmax(axis=1)
        preds["pred"] = [class_names[i] for i in preds["pred_idx"]]

        # add the true label and plane from the CSV, matched by path
        info = test_ds.df[["path", "label", "label_idx", "plane"]]
        preds = info.merge(preds, on="path")
        preds.to_csv(out_file, index=False)

        acc = (preds["pred_idx"] == preds["label_idx"]).mean()
        print(f"{run_dir.name}: test acc {acc:.4f} ({time.time() - t0:.0f}s)")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print("Done")

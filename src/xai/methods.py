# Grad-CAM and Integrated Gradients (IG), written out by hand.
# Import with:
#   from src.xai.methods import load_trained_model, gradcam, gradcam_heatmap,
#       integrated_gradients, ig_heatmap, black_baseline, blurred_baseline
# All functions expect the model in eval mode and one image tensor [3, 384, 384]
# that went through eval_transform (letterboxed and normalized).
# Run directly for a quick self-test on a few test images with DenseNet121 seed 42:
#   uv run python -m src.xai.methods
# Saves results/figures/check_methods.png

import os

# same memory safeguard for Macs as in train.py (must be set before torch is imported)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from torchvision.transforms.v2 import functional as TF

from src.preprocessing.check_pipeline import to_display
from src.preprocessing.transforms import eval_transform, image_size, mean, std
from src.training.models import build_model, get_target_layer
from src.training.train import get_device
from src.xai.masks import load_mask, mask_path


def load_trained_model(model_name, checkpoint, device):
    # build the architecture without ImageNet weights, then load our trained weights
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def normalize(heatmap):
    # scale to 0..1 by dividing by the largest value
    top = heatmap.max()
    return heatmap / top if top > 0 else np.zeros_like(heatmap)


# ---------- Grad-CAM ----------

def gradcam(model, target_layer, image, target_class):
    # Returns the raw Grad-CAM map at the target layer's resolution (12x12 for a 384 input).
    saved = {}

    def forward_hook(module, inputs, output):
        saved["activations"] = output
        output.register_hook(lambda grad: saved.update(gradients=grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        # the input needs gradients so the backward pass reaches the target layer,
        # even when the model's own weights are frozen
        x = image.unsqueeze(0).clone().requires_grad_(True)
        output = model(x)
        model.zero_grad()
        output[0, target_class].backward()   # gradient of the class score (logit)
    finally:
        handle.remove()

    activations = saved["activations"][0]   # [channels, 12, 12]
    gradients = saved["gradients"][0]       # [channels, 12, 12]
    weights = gradients.mean(dim=(1, 2))    # one weight per channel: its average gradient
    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
    model.zero_grad(set_to_none=True)
    return cam.detach().cpu().numpy()


def gradcam_heatmap(cam, size=image_size):
    # upsample the 12x12 map to the image size (bilinear) and scale to 0..1
    cam = torch.tensor(cam)[None, None]
    big = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
    return normalize(big[0, 0].numpy())


# ---------- Integrated Gradients ----------

def black_baseline(size=image_size):
    # a black image AFTER normalization: (0 - mean) / std per channel.
    # A tensor of zeros would be mid-gray, not black.
    values = [(0 - m) / s for m, s in zip(mean, std)]
    return torch.tensor(values, dtype=torch.float32).view(3, 1, 1).expand(3, size, size).clone()


def blurred_baseline(image, sigma=20):
    # a heavily blurred copy of the image: keeps the overall brightness, removes the details
    kernel = 2 * round(3 * sigma) + 1
    return TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def integrated_gradients(model, image, baseline, target_class, steps=50, batch_size=10):
    # Walks from the baseline to the image, averages the gradients along the way,
    # and multiplies by (image - baseline).
    # The points on the path and their weights come from Gauss-Legendre quadrature, which
    # gives a much more accurate average than equal steps for the same number of steps
    # (it is also Captum's default).
    # Returns the attributions [3, H, W] and the convergence delta: how far the sum of the
    # attributions is from the real change in the class score (should be close to 0).
    image = image.detach()
    baseline = baseline.to(image.device)
    nodes, weights = np.polynomial.legendre.leggauss(steps)   # points and weights on [-1, 1]
    alphas = torch.tensor((nodes + 1) / 2, dtype=torch.float32, device=image.device)   # moved to [0, 1]
    weights = torch.tensor(weights / 2, dtype=torch.float32, device=image.device)      # they sum to 1
    total_gradient = torch.zeros_like(image)

    for start in range(0, steps, batch_size):
        a = alphas[start:start + batch_size].view(-1, 1, 1, 1)
        w = weights[start:start + batch_size].view(-1, 1, 1, 1)
        points = (baseline + a * (image - baseline)).requires_grad_(True)
        scores = model(points)[:, target_class]
        gradients, = torch.autograd.grad(scores.sum(), points)
        total_gradient += (gradients * w).sum(dim=0)   # weighted average of the gradients

    attributions = (image - baseline) * total_gradient

    with torch.no_grad():
        scores = model(torch.stack([image, baseline]))[:, target_class]
    score_change = (scores[0] - scores[1]).item()
    delta = attributions.sum().item() - score_change
    return attributions.cpu(), delta, score_change


def ig_heatmap(attributions, sigma=0):
    # one value per pixel: sum of absolute attributions over the 3 channels,
    # optionally smoothed with a Gaussian (sigma in pixels), then scaled to 0..1
    heatmap = attributions.abs().sum(dim=0).numpy()
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    return normalize(heatmap)


# ---------- self-test ----------

def overlay(ax, image, heatmap, mask, title, clip=False):
    if clip:
        # IG maps are speckled; for display only, saturate the top 0.5% of pixels
        heatmap = np.clip(heatmap / (np.percentile(heatmap, 99.5) + 1e-8), 0, 1)
    ax.imshow(to_display(image))
    ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors="white", linewidths=0.8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


if __name__ == "__main__":
    device = get_device()
    run_dir = Path("runs/densenet121_lr0.0001_seed42")
    model = load_trained_model("densenet121", run_dir / "best.pt", device)
    target_layer = get_target_layer(model, "densenet121")

    preds = pd.read_csv(run_dir / "test_predictions.csv")
    class_names = preds.sort_values("label_idx")["label"].unique().tolist()
    prob_cols = [f"prob_{c}" for c in class_names]

    # one correct image per tumor class, plus two of the model's mistakes
    tumors = preds[preds["label"] != "no_tumor"]
    correct = tumors[tumors["label_idx"] == tumors["pred_idx"]]
    examples = [correct[correct["label"] == c].sample(1, random_state=0).iloc[0]
                for c in ["glioma", "meningioma", "pituitary"]]
    for name in ["brisc2025_test_00281_me_ax_t1.jpg", "brisc2025_test_00365_me_ax_t1.jpg"]:
        examples.append(preds[preds["path"].str.endswith(name)].iloc[0])

    columns = ["image + mask", "Grad-CAM, predicted", "Grad-CAM, other class",
               "IG black, raw", "IG black, smoothed (sigma 4)", "IG blurred baseline"]
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(3 * len(columns), 3.2 * len(examples)))
    baseline = black_baseline().to(device)

    for r, row in enumerate(examples):
        with Image.open(row["path"]) as img:
            image = eval_transform(img).to(device)
        mask = load_mask(mask_path(row["path"]))
        pred = int(row["pred_idx"])
        # "other class": the true class if the model was wrong, otherwise the second most likely
        if pred != row["label_idx"]:
            other = int(row["label_idx"])
        else:
            other = int(np.argsort(row[prob_cols].values.astype(float))[-2])

        cam_pred = gradcam(model, target_layer, image, pred)
        cam_other = gradcam(model, target_layer, image, other)
        rho = spearmanr(cam_pred.flatten(), cam_other.flatten())[0]

        t0 = time.time()
        attr, delta, change = integrated_gradients(model, image, baseline, pred)
        ig_seconds = time.time() - t0
        attr_blur, delta_blur, change_blur = integrated_gradients(model, image, blurred_baseline(image), pred)

        name = Path(row["path"]).name
        print(f"{name}: true {row['label']}, predicted {class_names[pred]}")
        print(f"  Grad-CAM {class_names[pred]} vs {class_names[other]}: Spearman {rho:.2f} (should be well below 1)")
        print(f"  IG black:   score change {change:7.2f}, delta {delta:+.3f} "
              f"({abs(delta) / abs(change):.1%}), {ig_seconds:.1f}s")
        print(f"  IG blurred: score change {change_blur:7.2f}, delta {delta_blur:+.3f} "
              f"({abs(delta_blur) / abs(change_blur):.1%})")

        axes[r, 0].imshow(to_display(image.cpu()))
        axes[r, 0].contour(mask, levels=[0.5], colors="red", linewidths=0.8)
        axes[r, 0].set_title(f"{name}\ntrue {row['label']}, pred {class_names[pred]}", fontsize=8)
        axes[r, 0].axis("off")
        image = image.cpu()
        overlay(axes[r, 1], image, gradcam_heatmap(cam_pred), mask, f"Grad-CAM {class_names[pred]}")
        overlay(axes[r, 2], image, gradcam_heatmap(cam_other), mask, f"Grad-CAM {class_names[other]}")
        overlay(axes[r, 3], image, ig_heatmap(attr), mask, "IG black, raw", clip=True)
        overlay(axes[r, 4], image, ig_heatmap(attr, sigma=4), mask, "IG black, sigma 4", clip=True)
        overlay(axes[r, 5], image, ig_heatmap(attr_blur, sigma=4), mask, "IG blurred, sigma 4", clip=True)

    # does the delta shrink with more steps? (first image only)
    row = examples[0]
    with Image.open(row["path"]) as img:
        image = eval_transform(img).to(device)
    print(f"\nIG convergence on {Path(row['path']).name}:")
    for steps in [25, 50, 100]:
        _, delta, change = integrated_gradients(model, image, baseline, int(row["pred_idx"]), steps=steps)
        print(f"  {steps:3d} steps: delta {delta:+.4f} ({abs(delta) / abs(change):.2%} of the score change)")

    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/check_methods.png", dpi=90)
    plt.close(fig)
    print("\nSaved results/figures/check_methods.png")

# Trains one model with one learning rate and one seed.
#   Stage 1: backbone frozen, only the new output layer is trained.
#   Stage 2: everything is fine-tuned, with early stopping on val macro-F1.
# Saves to <out>/<model>_lr<lr>_seed<seed>/:
#   best.pt       weights of the epoch with the best val macro-F1
#   log.csv       one row per epoch
#   summary.json  settings and final results
# Run from the project root, for example:
#   uv run python -m src.training.train --model resnet50 --lr 1e-4 --seed 42
# Quick test on a few images:
#   uv run python -m src.training.train --model resnet50 --lr 1e-4 --seed 42 --limit 64 --epochs 2

import os

# Memory safeguard for Macs (MPS). The GPU shares RAM with the system, so without a
# limit a large model can use all memory and freeze the computer. These limit PyTorch to
# about 60% of the GPU's recommended memory, so it stops with an "out of memory" error
# instead. They must be set before torch is imported, and are ignored on CUDA and CPU.
# To change the limit, set the variables in the terminal before running.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.preprocessing.dataset import BriscDataset
from src.preprocessing.transforms import eval_transform, train_transform
from src.training.models import build_model, get_head, model_names


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=model_names)
    parser.add_argument("--lr", type=float, required=True, help="learning rate for stage 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=15, help="max epochs for stage 2")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience for stage 2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="use only this many images (for testing)")
    parser.add_argument("--amp", action="store_true", help="use mixed precision (float16)")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def memory_used_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    return 0.0


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # with --amp, the forward pass runs in float16 where it is safe
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()  # scale the loss up so small gradients don't become 0
        scaler.step(optimizer)         # scale the gradients back down, then update the weights
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        count += len(labels)

    return total_loss / count, correct / count


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * len(labels)
            all_preds += outputs.argmax(1).tolist()
            all_labels += labels.tolist()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return loss, acc, macro_f1


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    device = get_device()

    run_name = f"{args.model}_lr{args.lr:g}_seed{args.seed}"
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run {run_name} on {device}")

    # data
    train_ds = BriscDataset("train", train_transform)
    val_ds = BriscDataset("val", eval_transform)

    # class weights from the full train split: n_total / (n_classes * n_class)
    counts = train_ds.df["label_idx"].value_counts().sort_index().values
    class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)

    if args.limit > 0:
        # random images (fixed seed), so the small test set has every class.
        # The CSV is sorted by path, so the first rows would all be glioma.
        rng = np.random.default_rng(0)
        train_ds = Subset(train_ds, rng.permutation(len(train_ds))[:args.limit].tolist())
        val_ds = Subset(val_ds, rng.permutation(len(val_ds))[:args.limit].tolist())

    use_workers = args.workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=use_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=use_workers)

    # model and loss
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)  # does nothing without --amp

    log = []
    best_f1 = -1.0
    best_epoch = 0
    peak_memory = 0.0
    start = time.time()

    def run_epoch(stage, epoch, optimizer):
        global best_f1, best_epoch, peak_memory
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device,
                                                scaler, args.amp)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)
        seconds = time.time() - t0
        peak_memory = max(peak_memory, memory_used_mb(device))
        if device.type == "mps":
            torch.mps.empty_cache()  # give unused cached memory back to the system

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_epoch = len(log) + 1
            torch.save(model.state_dict(), run_dir / "best.pt")

        lr = optimizer.param_groups[0]["lr"]
        log.append({
            "epoch": len(log) + 1, "stage": stage, "stage_epoch": epoch, "lr": lr,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "seconds": seconds,
        })
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)

        mark = " *" if improved else ""
        print(f"stage {stage} epoch {epoch:2d} | lr {lr:.1e} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | {seconds:.0f}s{mark}")
        return improved

    # stage 1: train only the new output layer
    for p in model.parameters():
        p.requires_grad = False
    for p in get_head(model, args.model).parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(get_head(model, args.model).parameters(),
                                  lr=args.head_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.head_epochs + 1):
        run_epoch(1, epoch, optimizer)

    # stage 2: fine-tune everything
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        improved = run_epoch(2, epoch, optimizer)
        scheduler.step()

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    total_minutes = (time.time() - start) / 60
    summary = {
        "model": args.model,
        "lr": args.lr,
        "seed": args.seed,
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_run": len(log),
        "total_minutes": total_minutes,
        "peak_memory_mb": peak_memory,
        "device": str(device),
        "torch_version": torch.__version__,
        "settings": vars(args),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best val macro-F1 {best_f1:.4f} at epoch {best_epoch}, {total_minutes:.1f} min total")
    print(f"Saved to {run_dir}")

# Computes test-set results from the saved predictions (run predict.py first).
# Does not need a GPU and does not run any model, so it can be rerun at any time.
# Saves to results/:
#   test_runs.csv        accuracy and macro-F1 of every run (model x seed)
#   test_summary.csv     per model: mean +- SD over seeds, representative seed, bootstrap 95% CI
#   test_per_class.csv   precision, recall, F1 per class for the selected model
#   test_mcnemar.csv     McNemar test between every pair of models, with Holm correction
#   test_per_plane.csv   accuracy per plane (ax, co, sa) for every model
#   figures/test_confusion_matrix.png   for the selected model
# Run from the project root:
#   uv run python -m src.evaluation.evaluate

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

runs_dir = Path("runs")
results_dir = Path("results")
fig_dir = results_dir / "figures"
num_bootstrap = 2000


def macro_f1(df):
    return f1_score(df["label_idx"], df["pred_idx"], average="macro")


def accuracy(df):
    return (df["label_idx"] == df["pred_idx"]).mean()


def bootstrap_ci(df, rng):
    # resample the test images with replacement and recompute macro-F1 each time
    labels = df["label_idx"].values
    preds = df["pred_idx"].values
    scores = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, len(df), len(df))
        scores.append(f1_score(labels[idx], preds[idx], average="macro"))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def mcnemar(df_a, df_b):
    # exact McNemar test: only the images where exactly one of the two models is right count
    right_a = (df_a["label_idx"] == df_a["pred_idx"]).values
    right_b = (df_b["label_idx"] == df_b["pred_idx"]).values
    only_a = int((right_a & ~right_b).sum())
    only_b = int((~right_a & right_b).sum())
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue
    return only_a, only_b, p


def holm(p_values):
    # Holm correction: multiply the smallest p by m, the next by m-1, ... and keep them in order
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


if __name__ == "__main__":
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. load every run that has test predictions
    runs = []
    preds = {}
    for pred_file in sorted(runs_dir.glob("*/test_predictions.csv")):
        with open(pred_file.parent / "summary.json") as f:
            summary = json.load(f)
        df = pd.read_csv(pred_file)
        key = (summary["model"], summary["seed"])
        preds[key] = df
        runs.append({
            "model": summary["model"], "seed": summary["seed"],
            "val_f1": summary["best_val_f1"],
            "test_acc": accuracy(df), "test_f1": macro_f1(df),
        })
    runs = pd.DataFrame(runs)
    if len(runs) == 0:
        raise SystemExit("No test_predictions.csv found. Run src.evaluation.predict first.")
    runs.to_csv(results_dir / "test_runs.csv", index=False)
    print(f"Loaded {len(runs)} runs\n")
    print(runs.round(4).to_string(index=False))

    # 2. per model: mean +- SD over seeds, and the representative seed
    #    (the seed whose val F1 is closest to the model's mean val F1, lowest seed on a tie)
    rng = np.random.default_rng(0)
    summary_rows = []
    for model, group in runs.groupby("model"):
        mean_val = group["val_f1"].mean()
        group = group.assign(gap=(group["val_f1"] - mean_val).abs())
        rep_seed = int(group.sort_values(["gap", "seed"]).iloc[0]["seed"])
        ci_low, ci_high = bootstrap_ci(preds[(model, rep_seed)], rng)
        summary_rows.append({
            "model": model,
            "val_f1_mean": mean_val, "val_f1_sd": group["val_f1"].std(),
            "test_acc_mean": group["test_acc"].mean(), "test_acc_sd": group["test_acc"].std(),
            "test_f1_mean": group["test_f1"].mean(), "test_f1_sd": group["test_f1"].std(),
            "rep_seed": rep_seed,
            "rep_test_f1": macro_f1(preds[(model, rep_seed)]),
            "rep_f1_ci_low": ci_low, "rep_f1_ci_high": ci_high,
        })
    summary = pd.DataFrame(summary_rows).sort_values("val_f1_mean", ascending=False)
    summary.to_csv(results_dir / "test_summary.csv", index=False)

    print("\nPer model (mean +- SD over seeds; CI is for the representative seed):")
    for row in summary.itertuples():
        print(f"  {row.model:18s} val F1 {row.val_f1_mean:.4f} +- {row.val_f1_sd:.4f} | "
              f"test acc {row.test_acc_mean:.4f} +- {row.test_acc_sd:.4f} | "
              f"test F1 {row.test_f1_mean:.4f} +- {row.test_f1_sd:.4f} | "
              f"seed {row.rep_seed}: F1 {row.rep_test_f1:.4f} [{row.rep_f1_ci_low:.4f}, {row.rep_f1_ci_high:.4f}]")

    # 3. the selected model (best mean val F1, chosen before looking at the test set)
    selected = summary.iloc[0]
    sel_model, sel_seed = selected["model"], int(selected["rep_seed"])
    sel_df = preds[(sel_model, sel_seed)]
    class_names = sel_df.sort_values("label_idx")["label"].unique().tolist()
    print(f"\nSelected model: {sel_model} seed {sel_seed}")

    report = classification_report(sel_df["label_idx"], sel_df["pred_idx"],
                                   target_names=class_names, digits=4, output_dict=True)
    per_class = pd.DataFrame(report).T
    per_class.to_csv(results_dir / "test_per_class.csv", index_label="class")
    print(per_class.round(4).to_string())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(sel_df["label"], sel_df["pred"], labels=class_names,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{sel_model} (seed {sel_seed}), test set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 4. McNemar between every pair of models (representative seeds), Holm-corrected
    rep = dict(zip(summary["model"], summary["rep_seed"]))
    tests = []
    for model_a, model_b in itertools.combinations(summary["model"], 2):
        only_a, only_b, p = mcnemar(preds[(model_a, rep[model_a])], preds[(model_b, rep[model_b])])
        tests.append({"model_a": model_a, "model_b": model_b,
                      "only_a_right": only_a, "only_b_right": only_b, "p": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].values)
    tests["significant"] = tests["p_holm"] < 0.05
    tests.to_csv(results_dir / "test_mcnemar.csv", index=False)
    print("\nMcNemar tests (representative seeds):")
    print(tests.round(4).to_string(index=False))

    # 5. accuracy per plane, averaged over seeds
    plane_rows = []
    for (model, seed), df in preds.items():
        for plane, part in df.groupby("plane"):
            plane_rows.append({"model": model, "seed": seed, "plane": plane,
                               "n": len(part), "acc": accuracy(part)})
    planes = pd.DataFrame(plane_rows)
    per_plane = planes.groupby(["model", "plane"]).agg(n=("n", "first"), acc_mean=("acc", "mean"),
                                                       acc_sd=("acc", "std")).reset_index()
    per_plane.to_csv(results_dir / "test_per_plane.csv", index=False)
    print("\nAccuracy per plane (mean over seeds):")
    print(per_plane.pivot(index="model", columns="plane", values="acc_mean").round(4))

    print(f"\nSaved tables to {results_dir} and the confusion matrix to {fig_dir}")

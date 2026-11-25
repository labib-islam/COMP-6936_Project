# Turns the per-image XAI scores (results/xai/<run>/*_scores.csv, from explain.py) into the
# tables, tests and charts that answer the research questions.
# Only the explanation of the PREDICTED class is used here.
#
#   RQ2  How well do Grad-CAM and IG align with the tumor masks, per model?
#        -> results/xai_rq2.csv          scores per model x method, with 95% CIs and chance levels
#        -> results/xai_rq2_tests.csv    Grad-CAM vs IG, and each method vs the chance prior
#        -> results/xai_ig_converged.csv IG scores again, using only images whose IG converged
#   RQ3  How does alignment vary by class, plane, tumor size and correctness?
#        -> results/xai_rq3.csv, results/figures/xai_rq3.png
#   RQ4  Is classification performance related to alignment across models?
#        -> results/xai_rq4.csv, results/figures/xai_rq4.png
#
# RQ2 and RQ3 use the representative seed of each model (Grad-CAM and IG);
# RQ4 uses Grad-CAM on all 12 runs.
# Run from the project root (no GPU needed; about 1 min the first time, to build the chance prior):
#   uv run python -m src.xai.summarize_xai

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, spearmanr, wilcoxon

from src.evaluation.evaluate import holm
from src.preprocessing.transforms import image_size
from src.training.models import model_names
from src.xai.masks import load_mask, mask_path, valid_region
from src.xai.scores import score_heatmap

xai_dir = Path("results/xai")
results_dir = Path("results")
fig_dir = results_dir / "figures"
prior_file = xai_dir / "prior_scores.csv"

methods = ["gradcam", "ig_smooth"]        # the main comparison; ig_raw is also shown in the RQ2 table
num_bootstrap = 2000
max_ig_delta = 0.05                        # IG counts as converged below this delta
rng = np.random.default_rng(0)


# ---------- loading ----------

def load_scores():
    frames = [pd.read_csv(f) for f in sorted(xai_dir.glob("*/*_scores.csv"))]
    scores = pd.concat(frames, ignore_index=True)
    return scores[scores["target"] == "pred"].reset_index(drop=True)


def representative_seeds():
    summary = pd.read_csv(results_dir / "test_summary.csv")
    return dict(zip(summary["model"], summary["rep_seed"]))


def only_rep_seeds(scores, rep_seeds):
    return scores[scores["seed"] == scores["model"].map(rep_seeds)]


# ---------- chance levels ----------

def uniform_scores(images):
    # a heatmap with the same value everywhere: the peak is a random pixel, so the chance
    # of a hit equals the tumor's area share, and so does the energy inside the tumor
    return images.assign(method="uniform", pointing_hit=images["tumor_share"],
                         energy=images["tumor_share"], energy_ratio=1.0)


def prior_scores(images):
    # a stronger chance level: the average TRAINING tumor mask used as a heatmap.
    # It knows where tumors usually are, but nothing about the image itself.
    if prior_file.exists():
        return pd.read_csv(prior_file)

    df = pd.read_csv("data/splits.csv")
    train = df[(df["split"] == "train") & (df["label"] != "no_tumor")]
    print(f"Building the average mask from {len(train)} training masks...")
    prior = np.zeros((image_size, image_size))
    for path in train["path"]:
        prior += load_mask(mask_path(path))
    prior /= len(train)

    sizes = df.set_index("path")[["width", "height"]]
    rows = []
    for row in images.itertuples():
        width, height = sizes.loc[row.path]
        scores = score_heatmap(prior, load_mask(mask_path(row.path)), valid_region(width, height))
        rows.append({"path": row.path, "label": row.label, "plane": row.plane,
                     "tumor_share": row.tumor_share, "method": "prior", **scores})
    result = pd.DataFrame(rows)
    result.to_csv(prior_file, index=False)
    return result


# ---------- statistics helpers ----------

def bootstrap_ci(values):
    # 95% CI of the mean, by resampling the images
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), (num_bootstrap, len(values)))
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


def describe(part):
    pointing_ci = bootstrap_ci(part["pointing_hit"])
    energy_ci = bootstrap_ci(part["energy"])
    return {
        "n": len(part),
        "pointing": part["pointing_hit"].mean(),
        "pointing_ci_low": pointing_ci[0], "pointing_ci_high": pointing_ci[1],
        "energy": part["energy"].mean(),
        "energy_ci_low": energy_ci[0], "energy_ci_high": energy_ci[1],
        "energy_ratio_median": part["energy_ratio"].median(),
    }


def paired(part_a, part_b, column):
    # the same images in both, in the same order
    both = part_a[["path", column]].merge(part_b[["path", column]], on="path", suffixes=("_a", "_b"))
    return both[f"{column}_a"].values, both[f"{column}_b"].values


def mcnemar_hits(hits_a, hits_b):
    # exact McNemar test on two paired hit/miss lists
    hits_a = hits_a.astype(bool)
    hits_b = hits_b.astype(bool)
    only_a = int((hits_a & ~hits_b).sum())
    only_b = int((~hits_a & hits_b).sum())
    if only_a + only_b == 0:
        return 1.0
    return binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue


# ---------- RQ2: alignment with the masks ----------

def rq2_table(main, chance):
    rows = []
    for model in model_names:
        for method in ["gradcam", "ig_raw", "ig_smooth"]:
            part = main[(main["model"] == model) & (main["method"] == method)]
            rows.append({"model": model, "method": method, **describe(part)})
    for method, part in chance.groupby("method"):
        rows.append({"model": "chance", "method": method, **describe(part)})
    return pd.DataFrame(rows)


def rq2_tests(main, chance):
    # paired tests on the same images: pointing with McNemar, energy with Wilcoxon signed-rank
    prior = chance[chance["method"] == "prior"]
    rows = []
    for model in model_names:
        own = main[main["model"] == model]
        gradcam = own[own["method"] == "gradcam"]
        ig = own[own["method"] == "ig_smooth"]
        comparisons = [("gradcam vs ig_smooth", gradcam, ig),
                       ("gradcam vs prior", gradcam, prior),
                       ("ig_smooth vs prior", ig, prior)]
        for name, part_a, part_b in comparisons:
            hits_a, hits_b = paired(part_a, part_b, "pointing_hit")
            energy_a, energy_b = paired(part_a, part_b, "energy")
            rows.append({"model": model, "comparison": name, "metric": "pointing",
                         "a": hits_a.mean(), "b": hits_b.mean(), "n": len(hits_a),
                         "p": mcnemar_hits(hits_a, hits_b)})
            rows.append({"model": model, "comparison": name, "metric": "energy",
                         "a": energy_a.mean(), "b": energy_b.mean(), "n": len(energy_a),
                         "p": wilcoxon(energy_a, energy_b).pvalue})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm(tests["p"].values)
    tests["significant"] = tests["p_holm"] < 0.05
    return tests


def ig_convergence_check(main):
    # do the IG results change if we drop the images where IG did not converge?
    rows = []
    for model in model_names:
        ig = main[(main["model"] == model) & (main["method"] == "ig_smooth")]
        converged = ig[ig["ig_delta_rel"] <= max_ig_delta]
        rows.append({"model": model, "n_all": len(ig), "n_converged": len(converged),
                     "pointing_all": ig["pointing_hit"].mean(),
                     "pointing_converged": converged["pointing_hit"].mean(),
                     "energy_all": ig["energy"].mean(),
                     "energy_converged": converged["energy"].mean()})
    return pd.DataFrame(rows)


# ---------- RQ3: breakdowns ----------

def add_size_groups(df, images):
    # tumor size tertiles, computed once over the 860 test images so every model uses the same groups
    tertiles = pd.qcut(images.set_index("path")["tumor_share"], 3, labels=["small", "medium", "large"])
    return df.assign(size=df["path"].map(tertiles))


def rq3_table(main):
    factors = {"class": "label", "plane": "plane", "size": "size", "correct": "correct"}
    rows = []
    for model in model_names:
        for method in methods:
            part = main[(main["model"] == model) & (main["method"] == method)]
            for factor, column in factors.items():
                groups = {level: g for level, g in part.groupby(column, observed=True)}
                # Kruskal-Wallis: does energy differ between the groups?
                testable = [g["energy"].values for g in groups.values() if len(g) >= 2]
                p = kruskal(*testable).pvalue if len(testable) >= 2 else np.nan
                for level, g in groups.items():
                    rows.append({"model": model, "method": method, "factor": factor, "level": str(level),
                                 "n": len(g), "pointing": g["pointing_hit"].mean(),
                                 "energy": g["energy"].mean(),
                                 "energy_ratio_median": g["energy_ratio"].median(),
                                 "kruskal_p_energy": p})
    return pd.DataFrame(rows)


def rq3_figure(rq3):
    factors = ["class", "plane", "size"]
    fig, axes = plt.subplots(len(methods), len(factors), figsize=(13, 7), sharey=True, squeeze=False)
    width = 0.8 / len(model_names)
    for r, method in enumerate(methods):
        for c, factor in enumerate(factors):
            ax = axes[r, c]
            part = rq3[(rq3["method"] == method) & (rq3["factor"] == factor)]
            levels = list(dict.fromkeys(part["level"]))
            x = np.arange(len(levels))
            for k, model in enumerate(model_names):
                values = part[part["model"] == model].set_index("level").loc[levels, "pointing"]
                ax.bar(x + k * width - 0.4 + width / 2, values, width, label=model)
            ax.set_xticks(x)
            ax.set_xticklabels(levels)
            ax.set_title(f"{method}: by {factor}", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            if c == 0:
                ax.set_ylabel("pointing game (hit rate)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "xai_rq3.png", dpi=120)
    plt.close(fig)


# ---------- RQ4: classification vs localization ----------

def rq4_table(scores):
    # Grad-CAM on every run (12 points), next to each run's test F1
    gradcam = scores[scores["method"] == "gradcam"]
    per_run = gradcam.groupby(["model", "seed"]).agg(pointing=("pointing_hit", "mean"),
                                                     energy=("energy", "mean")).reset_index()
    test = pd.read_csv(results_dir / "test_runs.csv")[["model", "seed", "test_f1"]]
    return per_run.merge(test, on=["model", "seed"])


def rq4_correlations(rq4, main):
    rows = []
    for metric in ["pointing", "energy"]:
        rho, p = spearmanr(rq4["test_f1"], rq4[metric])
        rows.append({"level": "12 runs, Grad-CAM", "metric": metric, "n": len(rq4), "spearman": rho, "p": p})

    model_means = rq4.groupby("model")[["test_f1", "pointing", "energy"]].mean()
    for metric in ["pointing", "energy"]:
        rho, p = spearmanr(model_means["test_f1"], model_means[metric])
        rows.append({"level": "4 models, Grad-CAM", "metric": metric, "n": 4, "spearman": rho, "p": p})

    ig = main[main["method"] == "ig_smooth"].groupby("model").agg(
        pointing=("pointing_hit", "mean"), energy=("energy", "mean"))
    ig = ig.join(model_means["test_f1"])
    for metric in ["pointing", "energy"]:
        rho, p = spearmanr(ig["test_f1"], ig[metric])
        rows.append({"level": "4 models, IG (rep. seed)", "metric": metric, "n": 4, "spearman": rho, "p": p})
    return pd.DataFrame(rows)


def rq4_figure(rq4):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ["pointing", "energy"]):
        for model in model_names:
            part = rq4[rq4["model"] == model]
            ax.scatter(part["test_f1"], part[metric], label=model, s=50)
        ax.set_xlabel("test macro-F1")
        ax.set_ylabel(f"Grad-CAM {metric}")
        ax.set_title(f"Test F1 vs Grad-CAM {metric} (12 runs)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "xai_rq4.png", dpi=120)
    plt.close(fig)


# ---------- main ----------

if __name__ == "__main__":
    fig_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    scores = load_scores()
    rep_seeds = representative_seeds()
    main = only_rep_seeds(scores, rep_seeds)
    print(f"Loaded {len(scores)} rows from {scores[['model', 'seed']].drop_duplicates().shape[0]} runs; "
          f"representative seeds {rep_seeds}")

    # the 860 test tumor images (the same for every run)
    images = main[["path", "label", "plane", "tumor_share"]].drop_duplicates("path").reset_index(drop=True)
    chance = pd.concat([uniform_scores(images), prior_scores(images)], ignore_index=True)

    # RQ2
    rq2 = rq2_table(main, chance)
    tests = rq2_tests(main, chance)
    converged = ig_convergence_check(main)
    rq2.to_csv(results_dir / "xai_rq2.csv", index=False)
    tests.to_csv(results_dir / "xai_rq2_tests.csv", index=False)
    converged.to_csv(results_dir / "xai_ig_converged.csv", index=False)
    print("\nRQ2: alignment with the tumor masks (predicted class, representative seeds)")
    print(rq2[["model", "method", "n", "pointing", "pointing_ci_low", "pointing_ci_high",
               "energy", "energy_ratio_median"]].round(3).to_string(index=False))
    print("\nRQ2 tests (Holm-corrected over all rows):")
    print(tests.round(4).to_string(index=False))
    print("\nIG with only converged images (delta <= 5%):")
    print(converged.round(3).to_string(index=False))

    # RQ3
    rq3 = rq3_table(add_size_groups(main, images))
    rq3.to_csv(results_dir / "xai_rq3.csv", index=False)
    rq3_figure(rq3)
    print("\nRQ3: pointing game by group (rows: model / method; Kruskal-Wallis p on energy per factor)")
    for factor in ["class", "plane", "size", "correct"]:
        part = rq3[rq3["factor"] == factor]
        table = part.pivot_table(index=["model", "method"], columns="level", values="pointing")
        table["kruskal_p"] = part.groupby(["model", "method"])["kruskal_p_energy"].first()
        print(f"\nby {factor}:")
        print(table.round(3).to_string())
    print("\nNumber of images per group (wrong predictions are few, so read 'correct = False' with care):")
    counts = rq3[(rq3["method"] == "gradcam")].pivot_table(index="model", columns=["factor", "level"], values="n")
    print(counts.to_string())

    # RQ4
    rq4 = rq4_table(scores)
    correlations = rq4_correlations(rq4, main)
    rq4.to_csv(results_dir / "xai_rq4.csv", index=False)
    correlations.to_csv(results_dir / "xai_rq4_correlations.csv", index=False)
    rq4_figure(rq4)
    print("\nRQ4: Grad-CAM per run")
    print(rq4.round(4).to_string(index=False))
    print("\nGrad-CAM pointing per model, mean +- SD over 3 seeds:")
    print(rq4.groupby("model")["pointing"].agg(["mean", "std"]).round(3).to_string())
    print("\nSpearman correlation between test F1 and localization:")
    print(correlations.round(3).to_string(index=False))

    print(f"\nSaved tables to {results_dir} (xai_*.csv) and figures to {fig_dir}")

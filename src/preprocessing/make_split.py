# Splits the original train set into train and val, and saves data/splits.csv.
# The test set is kept as it is.
# Run from the project root:
#   uv run python -m src.preprocessing.make_split

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

manifest_file = Path("data/manifest.csv")
out_file = Path("data/splits.csv")

val_size = 0.15
seed = 42

df = pd.read_csv(manifest_file)

train_df = df[df["orig_split"] == "train"]

# stratify on class and plane together ("glioma_ax")
strat = train_df["label"] + "_" + train_df["plane"]

train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=val_size,
    stratify=strat,
    random_state=seed,
)

df["split"] = "test"
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"

df.to_csv(out_file, index=False)

print(f"Saved {len(df)} rows to {out_file}")
print("--------")
print(pd.crosstab(df["split"], df["label"], margins=True))
print("--------")
print(pd.crosstab(df["split"], df["plane"], margins=True))

# class weights for the loss, from the train split only
train_counts = df[df["split"] == "train"]["label"].value_counts().sort_index()
n_total = train_counts.sum()
n_classes = len(train_counts)
class_weights = n_total / (n_classes * train_counts)

print("--------")
print("Class weights (train split):")
print(class_weights.round(4))

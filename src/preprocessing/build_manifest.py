# Builds data/manifest.csv with one row per image in the classification task.
# Run from the project root:
#   uv run python -m src.preprocessing.build_manifest

from pathlib import Path

import pandas as pd
from PIL import Image

data_dir = Path("dataset/brisc2025/classification_task")
out_file = Path("data/manifest.csv")

labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
codes = {"glioma": "gl", "meningioma": "me", "no_tumor": "no", "pituitary": "pi"}
planes = ["ax", "co", "sa"]

rows = []
for split in ["train", "test"]:
    for label in labels:
        folder = data_dir / split / label
        for img_path in sorted(folder.glob("*.jpg")):
            # e.g. brisc2025_train_00516_gl_co_t1
            parts = img_path.stem.split("_")
            code = parts[3]
            plane = parts[4]

            if code != codes[label]:
                raise ValueError(f"{img_path} is in {label} folder but has code {code}")
            if plane not in planes:
                raise ValueError(f"{img_path} has unknown plane {plane}")

            with Image.open(img_path) as img:
                width, height = img.size

            rows.append({
                "path": img_path.as_posix(),
                "label": label,
                "label_idx": labels.index(label),
                "plane": plane,
                "orig_split": split,
                "width": width,
                "height": height,
            })

df = pd.DataFrame(rows)
df = df.sort_values("path").reset_index(drop=True)

out_file.parent.mkdir(exist_ok=True)
df.to_csv(out_file, index=False)

print(f"Saved {len(df)} rows to {out_file}")
print("--------")
print(df.groupby(["orig_split", "label"]).size().unstack())
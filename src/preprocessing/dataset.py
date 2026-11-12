# PyTorch Dataset that reads images listed in data/splits.csv.
# Import with:
#   from src.preprocessing.dataset import BriscDataset

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BriscDataset(Dataset):
    def __init__(self, split, transform, csv_file="data/splits.csv"):
        df = pd.read_csv(csv_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

        if len(self.df) == 0:
            raise ValueError(f"No images found for split '{split}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        with Image.open(row["path"]) as img:
            image = self.transform(img)

        return image, int(row["label_idx"]), row["path"]

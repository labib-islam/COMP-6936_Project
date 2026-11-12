# Image transforms used for training and evaluation.
# Import with:
#   from src.preprocessing.transforms import train_transform, eval_transform

import torch
from PIL import Image
from torchvision.transforms import v2

image_size = 384

# ImageNet mean and std, used by all four pretrained models
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def letterbox(img, size=image_size):
    # Resize so the longer side is `size`, then pad with black to a square.
    # This keeps the aspect ratio, so the brain and tumor are not stretched.
    img = img.convert("RGB")

    width, height = img.size
    scale = size / max(width, height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    img = img.resize((new_width, new_height), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas.paste(img, (left, top))
    return canvas


train_transform = v2.Compose([
    letterbox,
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

eval_transform = v2.Compose([
    letterbox,
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std),
])

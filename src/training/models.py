# Builds the four pretrained models with a new 4-class output layer.
# Import with:
#   from src.training.models import build_model, get_head, get_target_layer

from torch import nn
from torchvision import models

num_classes = 4
model_names = ["resnet50", "densenet121", "efficientnet_v2_s", "convnext_tiny"]


def build_model(name, pretrained=True):
    # pretrained=False gives random weights (used for quick tests and the sanity check)
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == "efficientnet_v2_s":
        # classifier is [Dropout, Linear]
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "convnext_tiny":
        # classifier is [LayerNorm, Flatten, Linear]
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model {name}, choose from {model_names}")

    return model


def get_head(model, name):
    # The new output layer. In stage 1 of training only this part is trained.
    if name == "resnet50":
        return model.fc
    if name == "densenet121":
        return model.classifier
    if name in ["efficientnet_v2_s", "convnext_tiny"]:
        return model.classifier
    raise ValueError(f"Unknown model {name}")


def get_target_layer(model, name):
    # Last convolutional block, used by Grad-CAM
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet121":
        return model.features.denseblock4
    if name in ["efficientnet_v2_s", "convnext_tiny"]:
        return model.features[-1]
    raise ValueError(f"Unknown model {name}")

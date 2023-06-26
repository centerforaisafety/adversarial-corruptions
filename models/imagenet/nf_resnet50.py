"""Adapted from https://github.com/amazon-science/normalizer-free-robust-training/blob/main/test.py."""

import os
import torch
from timm.models import create_model


def get_model(args):
    model = create_model("nf_resnet50", pretrained=False, num_classes=1000)
    if os.path.basename(args.weights) not in [
        "NoFrost_ResNet50.pth",
        "NoFrost_star_ResNet50.pth",
    ]:
        raise ValueError(f"Weights not supported {args.weights}")

    state = torch.load(args.weights, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    return model

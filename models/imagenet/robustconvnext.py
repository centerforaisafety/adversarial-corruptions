"""Adapted from https://github.com/nmndeep/revisiting-at."""

import sys

sys.path.append("third_party/revisiting-at")

import torch
from utils_architecture import get_new_model


def get_model(args):
    # See https://github.com/nmndeep/revisiting-at/blob/main/AA_eval.py
    if args.weights.endswith("convnext_b_cvst_robust.pt"):  # 'ConvNexT-B-CvSt'
        model = get_new_model(
            "convnext_base", pretrained=False, not_original=1, updated=0
        )
    elif args.weights.endswith("convnext_iso_cvst_robust.pt"):  # 'ConvNexT-iso-CvSt'
        model = get_new_model(
            "convnext_iso", pretrained=False, not_original=1, updated=0
        )
    elif args.weights.endswith("convnext_s_cvst_robust.pt"):  # 'ConvNexT-S-CvSt'
        model = get_new_model(
            "convnext_small", pretrained=False, not_original=1, updated=0
        )
    elif args.weights.endswith("convnext_tiny_cvst_robust.pt"):  # 'ConvNext-T-CvSt'
        model = get_new_model(
            "convnext_tiny", pretrained=False, not_original=1, updated=0
        )
    else:
        raise ValueError(f"Weights not supported {args.weights}")

    ckpt = torch.load(args.weights, map_location="cpu")
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    ckpt = {k.replace("base_model.model.", ""): v for k, v in ckpt.items()}
    ckpt = {k.replace("base_model.", ""): v for k, v in ckpt.items()}
    ckpt = {k.replace("se_", "se_module."): v for k, v in ckpt.items()}
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("normalize.")}
    model.load_state_dict(ckpt)

    return model

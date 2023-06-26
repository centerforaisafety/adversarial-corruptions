"""Adapted from https://github.com/nmndeep/revisiting-at."""

import sys

sys.path.append("third_party/revisiting-at")


import torch
from utils_architecture import get_new_model


def get_model(args):
    # See https://github.com/nmndeep/revisiting-at/blob/main/AA_eval.py
    if args.weights.endswith("vit_b_cvst_robust.pt"):  # ViT-B-CvSt
        model = get_new_model("vit_b", pretrained=False, not_original=1, updated=0)
    elif args.weights.endswith("vit_b_cvst_clean.pt"):  # ViT-B-CvSt
        model = get_new_model("vit_b", pretrained=False, not_original=1, updated=0)
    elif args.weights.endswith("vit_m_cvst_robust.pt"):  # ViT-M-CvSt
        model = get_new_model("vit_m", pretrained=False, not_original=1, updated=0)
    elif args.weights.endswith("vit_s_cvst_robust.pt"):  # ViT-S-CvSt
        model = get_new_model("deit_s", pretrained=False, not_original=1, updated=0)
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

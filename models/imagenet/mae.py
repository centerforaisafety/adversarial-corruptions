"""Adapted from https://github.com/facebookresearch/mae/blob/main/FINETUNE.md"""

import sys

sys.path.append("third_party/mae")

import torch
import config
import timm

# Note that MAE only works with timm 0.3.2,
# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_finetune.py#LL26C1-L27C1
assert timm.__version__ == "0.3.2"  # version check
import models_vit


def get_model(args):
    weights = {
        "vit_base_patch16": "weights/imagenet/mae_finetuned_vit_base.pth",
        "vit_large_patch16": "weights/imagenet/mae_finetuned_vit_large.pth",
        "vit_huge_patch14": "weights/imagenet/mae_finetuned_vit_huge.pth",
    }
    for arch, ckpt in weights.items():
        if args.weights.endswith(arch):
            model = models_vit.__dict__[arch](
                num_classes=1000, drop_path_rate=0.1, global_pool=True
            )
            checkpoint = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.eval()
            return model.to(config.device)
    raise ValueError(f"Weights not supported {args.weights}")

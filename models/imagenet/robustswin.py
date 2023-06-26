"""Adapted from git@github.com:alibaba/easyrobust.git. """


import torch
from timm.models import create_model


def get_model(args):
    # See https://github.com/alibaba/easyrobust/blob/main/benchmarks/adv_robust_bench.sh and
    # https://github.com/alibaba/easyrobust/blob/main/benchmarks/benchmark.py
    if args.weights.endswith("advtrain_swin_small_patch4_window7_224_ep4.pth"):
        model = create_model(
            "swin_small_patch4_window7_224", pretrained=False, num_classes=1000
        )
    elif args.weights.endswith("advtrain_swin_base_patch4_window7_224_ep4.pth"):
        model = create_model(
            "swin_base_patch4_window7_224", pretrained=False, num_classes=1000
        )
    else:
        raise ValueError(f"Weights not supported {args.weights}")

    state = torch.load(args.weights)
    if "state_dict" in state.keys():
        state_dict = state["state_dict"]
    else:
        state_dict = state

    state = {k: v for k, v in state.items() if "0.mean" not in k}
    state = {k: v for k, v in state.items() if "0.std" not in k}

    model.load_state_dict(state_dict)

    return model

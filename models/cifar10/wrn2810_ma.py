"""https://github.com/divyam3897/MNG_AC/blob/40c25a0c162c32cd858803f6bf1790180c727d06/evaluate.py"""

import sys

sys.path.append("third_party/MNG_AC")

import torch
from wideresnet import WideResNet
from torchvision import datasets, transforms
import config


def get_model(args):
    model = WideResNet(28, 10, widen_factor=10, dropRate=0.0)

    sd = torch.load(args.weights, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    return model

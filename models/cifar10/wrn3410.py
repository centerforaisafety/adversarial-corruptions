import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import config

from models.cifar10.wrn import WideResNet


def get_model(args):
    model = WideResNet(34, 10, widen_factor=10, dropRate=0.0)
    sd = torch.load(args.weights, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    return model

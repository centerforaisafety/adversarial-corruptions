import torch
import torch.nn as nn
import torchvision
from robustness.tools.custom_modules import FakeReLU, SequentialWithArgs

import config
from models.imagenet.resnet50 import ResNet
from models.imagenet.resnet50 import Bottleneck


def get_model(args):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=config.device))
    except RuntimeError as e:
        state = torch.load(args.weights, map_location=config.device)["model"]
        if "module.model." in next(iter(state.keys())):
            state = {
                k.replace("module.model.", ""): v
                for k, v in state.items()
                if "module.model." in k
            }
        model.load_state_dict(state)
    return model

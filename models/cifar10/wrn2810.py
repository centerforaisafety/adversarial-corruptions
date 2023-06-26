import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import config

from models.cifar10.wrn import WideResNet


def get_model(args):
    model = WideResNet(28, 10, widen_factor=10, dropRate=0.0)
    state = torch.load(args.weights, map_location=config.device)
    try:
        model.load_state_dict(state)
    except:
        if "model" in state.keys():
            state = state["model"]
        elif "state_dict" in state.keys():
            state = state["state_dict"]
        elif "model_state_dict" in state.keys():
            state = state["model_state_dict"]

        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

    model.eval()

    return model

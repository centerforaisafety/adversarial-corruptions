"""Adapted from https://github.com/facebookresearch/dinov2/issues/64"""

import sys

sys.path.append("third_party/dinov2")

import torch
import torch.nn as nn
from functools import partial
import config
from dinov2.eval.linear import create_linear_input
from dinov2.eval.linear import LinearClassifier


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with self.autocast_ctx():
            features = self.feature_model.get_intermediate_layers(
                images, self.n_last_blocks, return_class_token=True
            )
        return features


class Dino(nn.Module):
    def __init__(self, type):
        super().__init__()
        # get feature model
        model = torch.hub.load("facebookresearch/dinov2", type, pretrained=True).to(
            config.device
        )
        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float16
        )
        self.feature_model = ModelWithIntermediateLayers(
            model, n_last_blocks=1, autocast_ctx=autocast_ctx
        ).to(config.device)

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224).to(config.device)
            sample_output = self.feature_model(sample_input)

        # get linear readout
        out_dim = create_linear_input(
            sample_output, use_n_blocks=1, use_avgpool=True
        ).shape[1]
        self.classifier = LinearClassifier(
            out_dim, use_n_blocks=1, use_avgpool=True
        ).to(config.device)
        vits_linear = torch.load(f"./weights/imagenet/{type}_linear_head.pth")
        self.classifier.linear.load_state_dict(vits_linear)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.classifier(x)
        return x


def get_model(args):
    for type in ["dinov2_vitb14", "dinov2_vitl14"]:
        if args.weights.endswith(type):
            return Dino(type)
    raise ValueError(f"Weights not supported {args.weights}")

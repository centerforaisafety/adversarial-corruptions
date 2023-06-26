import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import models.imagenet100.imagenet100_config as imagenet100_config


def get_test_dataset(args):
    # Currently only resnet50 is supported
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    if args.save == "humanstudy":
        size = 256 if args.attack in ["pixel", "wood"] else (256, 256)
        test_transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        )

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(imagenet100_config.imagenet100_location, "val"),
        transform=test_transform,
    )

    return test_dataset

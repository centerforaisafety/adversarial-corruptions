import importlib

import torch
import torchvision
import torchvision.transforms as transforms

import config
import models.cifar10.cifar10_config as cifar10_config


def get_test_dataset(args):
    test_transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(
        cifar10_config.cifar10_location,
        train=False,
        download=True,
        transform=test_transform,
    )
    return test_dataset


def get_train_dataset(args):
    if args.architecture == "wrn":
        mean, std = cifar10_config.mean_wrn, cifar10_config.std_wrn
    else:
        mean, std = cifar10_config.mean, cifar10_config.std
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        cifar10_config.cifar10_location,
        train=True,
        download=True,
        transform=train_transform,
    )
    return train_dataset

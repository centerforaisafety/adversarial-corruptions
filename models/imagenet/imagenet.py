import importlib
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import timm

import models.imagenet.imagenet_config as imagenet_config


def get_test_dataset(args):
    model_module = importlib.import_module(f".{args.architecture}", __package__)
    model = model_module.get_model(args)

    if args.architecture == "clip":
        test_transform = transforms.Compose(model.preprocess.transforms[:-1])
        assert isinstance(test_transform.transforms[-1], transforms.ToTensor)
    else:
        try:
            transform = timm.data.create_transform(
                **timm.data.resolve_data_config(model.pretrained_cfg)
            )
            assert isinstance(transform.transforms[-2], transforms.ToTensor)
            assert isinstance(transform.transforms[-1], transforms.Normalize)
            test_transform = transforms.Compose(transform.transforms[:-1])
        except:
            if args.architecture == "resnet50":
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
            else:
                test_transform = transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                    ]
                )

    test_dataset = torchvision.datasets.ImageNet(
        imagenet_config.imagenet_location, split="val", transform=test_transform
    )

    return test_dataset


def get_train_dataset(args):
    test_dataset = torchvision.datasets.ImageNet(
        imagenet_config.imagenet_location, split="train", transform=test_transform
    )
    return test_dataset

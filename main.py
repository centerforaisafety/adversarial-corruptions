import config
import configparser
import evaluate
import importlib
import jsonlines
import numpy as np
import os
import random
import timm
import torch
import torch.nn as nn
import warnings

from parse_args import get_parser
from models.model_normalization import (
    Cifar10Wrapper,
    ImageNetWrapper,
    Cifar10WrapperWRN,
    NormalizationWrapper,
    CLIPWrapper,
    IdentityWrapper,
)


def run_experiment_file(file, experiment_num=-1):
    """
    Runs a list of experiment from a file, with the file in the jsonlines format, Can either run all experiments in the file
    in sequence, or some specific experiment number within the file/

    Parameters
    ---
    file: str
    jsonlines file, with each line correposnding to ajson object holding the hyperparameters for an experiment

    experiment_num: int
    default -1, which runs all experiments in the file. Otherwise, runs the experiment with the given number.
    """

    experiment_list = []
    with open(file, mode="r") as f:
        reader = jsonlines.Reader(f)

        for experiment in iter(reader):
            parser = get_parser()
            argument_list = []
            for argument, value in experiment.items():
                argument_list.append("--" + str(argument))
                argument_list.append(str(value))

            args = parser.parse_args(argument_list)

            for k, v in vars(args).items():
                if v == "None":
                    vars(args)[k] = None

            experiment_list.append(args)

    if experiment_num >= 0:
        experiment_list = [experiment_list[experiment_num]]

    for experiment in iter(experiment_list):
        main(experiment)


def main(parser):
    """
    This function fetches the relevant objects which we need for the evaluation loop, and runs the evaluation loop.
    Throught this function, call by name is used to fetch the objects needed for training based on the input argument strings. In particular:

    1) The dataset is fetched based on the value of the args.dataset argument i.e. models.<args.dataset>.<args.dataset>.get_test_dataloader() is called
    2) The model architecture is fetched based on the architecture parameter and the data parameter i.e. <args.dataset>.<args.model>.get_model() is called
    3) The attack is fetched based on the attack parameter, i.e. attacks.<args.attack>.get_attack() is called

    """
    args = init_args(parser)  # Do initalisation based on arguments

    test_dataset = get_test_dataset(args.dataset, args)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True if config.device != "cpu" else False,
        generator=torch.Generator().manual_seed(args.seed) if args.seed else None,
    )

    model = get_model(args.dataset, args.architecture, args.weights, args)
    attack = None
    if args.attack != "none":
        attack = get_attack(args.attack, model, args)

    model.eval()

    results = evaluate.evaluate(model, test_dataloader, attack, args)
    evaluate.log_jsonlines(results, args.log, args)
    print(args.epsilon, args.step_size, args.num_steps, results["accuracy"])

    return results


def init_args(parser):
    """Does some initialization based on arguments"""
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

    config.device = args.device
    if config.project_path is None:
        config.project_path = os.path.abspath(".")

    if args.cuda_autotune:
        torch.backends.cudnn.benchmark = True

    if args.attack == "none":
        return args

    vars(args)[
        "distortion_level"
    ] = None  # Sets a "distortion level" argument, which is used when saving outputs

    if args.num_workers is None:
        args.num_workers = 4 * torch.cuda.device_count() if args.device == "cuda" else 4

    epsilon_vals = ["low", "medium", "high"]
    if args.epsilon in epsilon_vals:
        dataset_config = importlib.import_module(
            "models." + args.dataset + "." + f"{args.dataset}_config", package="."
        )
        args.epsilon = dataset_config.default_epsilons[args.attack][
            epsilon_vals.index(args.epsilon)
        ]
    else:
        try:
            args.epsilon = float(args.epsilon)
        except ValueError:
            raise ValueError(
                "--epsilon must be given a float or one of the following strings: low, medium, high"
            )
    if args.attack_config:
        attack_config = configparser.ConfigParser()
        attack_config.read(args.attack_config)
        for action in parser._actions:
            arg_name, arg_type = action.dest, action.type
            if arg_name in attack_config[args.attack]:
                arg_value = arg_type(attack_config.get(args.attack, arg_name))
                # Load arg value from the config file when arg value is None
                if getattr(args, arg_name) == None:
                    setattr(args, arg_name, arg_value)
                # When the arg value is not None make sure
                # its value is consistent with the value from the config file
                elif getattr(args, arg_name) != arg_value:
                    warnings.warn(
                        f"Conflicting value for {arg_name} in {args.attack_config} and the supplied value"
                    )

    return args


def get_model(dataset, architecture, weights, args):
    """Fetches and returns the relevant architecture, based on both the dataset and the architecture name.

    Parameters
    ----
    dataset: string
        dataset from which the architecture is to be fetched
    architecture: string
        name of architecture to be fetched"""
    model_module = importlib.import_module(
        "models." + dataset + "." + architecture, package="."
    )
    model = model_module.get_model(args)

    if args.dataset == "cifar10":
        if args.architecture == "wrn":
            model = Cifar10WrapperWRN(model)
        elif args.architecture in ["wrn2810_ma", "preactresnet18", "wideresnet_trades"]:
            model = IdentityWrapper(model)
        else:
            model = Cifar10Wrapper(model)
    elif args.dataset in ["imagenet100", "imagenetr"]:
        model = ImageNetWrapper(model)
    elif args.dataset == "imagenet":
        if args.architecture in ["robustvit", "robustconvnext"]:
            if args.weights.endswith(
                "convnext_iso_cvst_robust.pt"
            ) or args.weights.endswith("vit_s_cvst_robust.pt"):
                model = ImageNetWrapper(model)
        elif args.architecture == "clip":
            model = CLIPWrapper(model)
        else:
            try:
                transform = timm.data.create_transform(
                    **timm.data.resolve_data_config(model.pretrained_cfg)
                )
                model = NormalizationWrapper(
                    model,
                    mean=transform.transforms[-1].mean,
                    std=transform.transforms[-1].std,
                )
            except:
                model = ImageNetWrapper(model)
    else:
        raise ValueError("Dataset not supported")

    model.to(args.device)
    model = nn.DataParallel(model)
    return model


def get_test_dataset(dataset, args):
    """Fetches the dataloader relevant for the dataset

    Parameters
    ---
    dataset: string
        the name of the dataset from which the dataloader is to be fetched"""
    dataset_module = importlib.import_module(
        "models." + dataset + "." + dataset, package="."
    )
    test_dataset = dataset_module.get_test_dataset(args)
    return test_dataset


def get_train_dataset(dataset, args):
    """Fetches the dataloader relevant for the dataset

    Parameters
    ---
    dataset: string
        the name of the dataset from which the dataloader is to be fetched"""
    dataset_module = importlib.import_module(
        "models." + dataset + "." + dataset, package="."
    )
    train_dataset = dataset_module.get_train_dataset(args)
    return train_dataset


def get_attack(name, model, args):
    """Fetches the attack which needs to be used

    Parameters
    ---
    name: string
       name of the attack which is bein
    model: nn.Module
       model for which the attack is being optimised for, used when constructing the AttackInstance object
    """
    attack_module = importlib.import_module("attacks." + name, package=".")
    attack = attack_module.get_attack(model, args)

    return attack


if __name__ == "__main__":
    parser = get_parser()
    main(parser)

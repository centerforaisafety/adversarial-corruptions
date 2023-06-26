import os
import sys
import argparse
from pathlib import Path
import warnings

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from models.model_normalization import ImageNetWrapper
from models.imagenet.resnet50 import get_model as get_model_resnet50
from models.imagenet100.resnet50 import get_model as get_model_resnet50_classes100
from models.imagenet.dinov2 import get_model as get_model_dino
from models.imagenet100.dinov2 import get_model as get_model_dino_classes100

sys.path.insert(0, "third_party/auto-attack")
# load attack
from autoattack import AutoAttack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--n_ex", type=int, default=1000)
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="./log_file.txt")
    parser.add_argument("--version", type=str, default="standard")
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument(
        "--model", type=str, required=True, choices=["resnet50", "dinov2"]
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if args.model == "resnet50" and args.num_classes == 1000:
        model = get_model_resnet50(args)
    elif args.model == "resnet50" and args.num_classes == 100:
        model = get_model_resnet50_classes100(args)
    elif args.model == "dinov2" and args.num_classes == 1000:
        model = get_model_dino(args)
    elif args.model == "dinov2" and args.num_classes == 100:
        model = get_model_dino_classes100(args)

    model = ImageNetWrapper(model)
    model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_dir, "val"), transform=test_transform
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(123),
    )

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    adversary = AutoAttack(
        model,
        norm=args.norm,
        eps=args.epsilon,
        log_path=args.log_path,
        version=args.version,
    )

    x_test, y_test = [], []
    for x, y in tqdm(test_loader):
        x_test.append(x)
        y_test.append(y)
    x_test = torch.cat(x_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    args.n_ex = x_test.size(0)
    with torch.no_grad():
        out = model(x_test[:100].to(device))
        _, predicted = torch.max(out.to("cpu"), 1)
        correct = (predicted == y_test[:100]).sum().item()
        accuracy = correct / 100
        print("accuracy for first 100 samples:", accuracy)

    # example of custom version
    if args.version == "custom":
        adversary.attacks_to_run = ["apgd-ce", "fab"]
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(
                x_test[: args.n_ex],
                y_test[: args.n_ex],
                bs=args.batch_size,
                state_path=args.state_path,
            )

            torch.save(
                {"adv_complete": adv_complete},
                "{}/{}_{}_1_{}_eps_{:.5f}.pth".format(
                    args.save_dir,
                    "aa",
                    args.version,
                    adv_complete.shape[0],
                    args.epsilon,
                ),
            )

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(
                x_test[: args.n_ex], y_test[: args.n_ex], bs=args.batch_size
            )

            torch.save(
                adv_complete,
                "{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth".format(
                    args.save_dir, "aa", args.version, args.n_ex, args.epsilon
                ),
            )

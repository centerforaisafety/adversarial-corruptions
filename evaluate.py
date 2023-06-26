import os
import time
import sys

import jsonlines
import torch
import torch.nn.functional as F

import utils
import fcntl
import torchvision.utils as vutils
import torchvision.transforms as transforms


def evaluate(model, dataloader, attack, args):
    """Runs the evaluation loop, calculating and logging adversarial loss and accuracy. The dataloader here is typically an attacks.attacks.AdversarialDataloader object,
    which generates the relevant adversarial examples."""

    model.eval()
    start_time = time.time()

    total_correct = 0
    total_samples = 0
    total_loss = 0

    if args.check_loss_reduction:
        total_loss_reduction = 0

    save_i = 0
    delta = []
    attacked = []
    for i, (xs, ys) in enumerate(dataloader):
        xs, ys = xs.to(args.device), ys.to(args.device)

        if attack is not None:
            adv_xs = attack.generate_attack((xs, ys))
        else:
            adv_xs = xs

        with torch.no_grad():
            output = model(adv_xs)
        adv_loss = F.cross_entropy(output, ys, reduction="none")
        total_loss += adv_loss.sum().item()

        total_correct += torch.sum(torch.argmax(output, dim=1) == ys)
        total_samples += len(ys)
        if (
            args.check_loss_reduction
        ):  # Checks if loss has reduced for any of the examples in the batch
            with torch.no_grad():
                stand_loss = F.cross_entropy(model(xs), ys, reduction="none")
                total_loss_reduction += (adv_loss < stand_loss).sum().item()

        if args.save == "attacked_samples":
            save_path = (
                f"experiments/results/attacked_samples/{args.dataset}/"
                f"{args.attack}/{os.path.basename(args.weights)}-"
                f"{args.distortion_level or args.epsilon}.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            grid_tensor = vutils.make_grid(adv_xs, nrow=nrow)
            vutils.save_image(grid_tensor, save_path)
        if args.save == "humanstudy":
            # For human study
            y_pred = torch.argmax(output, dim=1)
            to_pil = transforms.ToPILImage()
            for j in range(adv_xs.shape[0]):
                if y_pred[j] != ys[j]:
                    save_path = (
                        "experiments/results/humanstudy/"
                        f"{args.attack}/{os.path.basename(args.weights)}/"
                        f"{save_i+1:03d}_{ys[j]}_{y_pred[j]}.png"
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    to_pil(adv_xs[j]).save(save_path)
                    save_i += 1
                    if save_i == 500:
                        sys.exit(0)

        if args.save == "fourier_analysis":
            delta.append((adv_xs - xs).cpu().detach())

        if i + 1 == args.num_batches:  # Early stopping of training
            break

    if args.save == "fourier_analysis":
        # For Fourier analysis
        save_path = (
            "experiments/results/fourier_analysis/"
            f"{args.attack}/{os.path.basename(args.weights)}.pth"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        delta = torch.cat(delta)
        torch.save(delta, save_path)

    end_time = time.time()
    total_correct = total_correct.item()

    accuracy = total_correct / total_samples
    loss = total_loss / total_samples
    metrics = {"accuracy": accuracy, "avg_loss": loss, "time": (end_time - start_time)}

    if args.check_loss_reduction:
        metrics["loss_reduction"] = total_loss_reduction / total_samples

    return metrics


def log_jsonlines(metrics, filename, args=None):
    if args is not None:
        args = vars(args)  # Converts the args Namespace to a dictionary
    else:
        args = {}

    dictonary = dict(metrics, **args)  # Merge the two dictionaries

    with open(filename, mode="a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = jsonlines.Writer(f)
        writer.write(dictonary)
        fcntl.flock(f, fcntl.LOCK_UN)
        print("Saved ", filename)

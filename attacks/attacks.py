import abc

import torch


class AttackInstance:
    """This class is subclassed by each AttackInstance, abstracing away the underlying attack.

    Parameters
    ----

    model: nn.Module
       The model for which the adversarial examples are optimised

    args: argparse.Namespace
        The arguments passed to the program, which are used to determine the attack parameters
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args

    @abc.abstractmethod
    def generate_attack(self, batch):
        """An abstract function, which when implemented takes in a batch, and applies an adversarial perturbation to the datapoints

        Parameters
        ----
        batch: (Tensor,Tensor)
           Takes in a typical tuple (xs,ys) of datapoints and their labels

        Returns
        ---
        perturbed batch: (Tensor,Tensor)
           This returns (adv(xs),ys) where adv denotes our adversarial examples"""
        return None


def tensor_clamp(x, a_min, a_max):
    """
    like torch.clamp, except with bounds defined as tensors
    """
    out = torch.clamp(x - a_max, max=0) + a_max
    out = torch.clamp(out - a_min, min=0) + a_min
    return out


def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)

    for i in range(1, len(x.shape)):
        norm = norm.unsqueeze(-1)

    return x / (norm + 1e-8)


def tensor_clamp_l2(x, center, radius):
    """batched clamp of x into l2 ball around center of given radius"""
    x = x.data
    diff = x - center
    diff_norm = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)
    project_select = diff_norm > radius
    if project_select.any():
        for _ in range(0, len(x.shape) - 1):
            diff_norm = diff_norm.unsqueeze(-1)

        new_x = x
        new_x[project_select] = (center + (diff / diff_norm) * radius)[project_select]
        return new_x
    else:
        return x


def cwloss(outputs, targets, k=None):
    # This is the CW loss function, which is a maximisation of the margin between the correct class and the incorrect class
    # This is the loss function used in the paper
    # https://arxiv.org/abs/1608.04644
    #

    outputs_sorted = torch.argsort(outputs, dim=1)
    predictions = outputs_sorted[:, -1]
    correct_predictions = targets == predictions
    margin_targets = predictions.clone()
    margin_targets[correct_predictions] = outputs_sorted[:, -2][correct_predictions]

    loss = outputs[:, targets] - outputs[:, margin_targets]

    if k is not None:
        loss = torch.max(loss, -k)

    return torch.mean(loss)

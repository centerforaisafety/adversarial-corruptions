import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from attacks.attacks import AttackInstance


def get_bar_variables(image, num_bars, bar_width, grey):
    batch_size, num_channels, height, width = image.shape

    bar_locations = torch.arange(
        math.floor(width / num_bars) // 2,
        width - math.floor(width / num_bars) // 2,
        width // num_bars,
    )

    xx, yy = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )

    mask = torch.zeros((height, width), dtype=torch.bool).to(config.device)

    for b in bar_locations:
        mask += torch.logical_and(
            ((b - bar_width // 2) <= xx), (xx <= (b + bar_width // 2))
        ).to(config.device)

    bar_variables = torch.zeros(
        image.size(), device=config.device, requires_grad=True
    ).to(config.device)

    return bar_variables, mask


def add_bars(inputs, mask, grey):
    inputs = inputs.clone().detach()

    inputs[:, :, mask] = grey.view(1, 3, 1)

    return inputs


def apply_bar_variables(inputs, bar_variables, mask):
    return inputs + bar_variables * mask


class PrisonAdversary(nn.Module):
    """Implementation of the Prison attack, which adds "bars" to the image, who's colour values
    can then be changed arbitrarily.

    Parameters
    ---
    epsilon (float):
        epsilon used in attack

    num_steps (int):
        number of steps used in the opitimsation loop

    step_size (flaot):
        step size used in the optimisation loop

    num_bars (int):
        number of bars used in the attack

    bar_width (int):
        width of the bars used in the attack
    """

    def __init__(self, epsilon, num_steps, step_size, num_bars, bar_width):
        super().__init__()

        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_bars = num_bars
        self.bar_width = bar_width
        self.grey = torch.tensor([0.5, 0.5, 0.5]).to(config.device)

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        """

        The attack works by taking an image an blurring it, and then optimally interpolating pixel-wise between the blurred image
        and the original.

        model: the model to be attacked.
        inputs: batch of unmodified images.
        targets: true labels.

        returns: adversarially perturbed images.
        """

        inputs, targets = inputs.clone().to(config.device), targets.to(config.device)

        # We have a mask to control what is optimised, and varialbes which control the colour of the bars
        bar_variables, mask = get_bar_variables(
            inputs, self.num_bars, self.bar_width, self.grey
        )
        inputs = add_bars(inputs, mask, self.grey)

        # The inner loop
        for i in range(self.num_steps):
            adv_inputs = apply_bar_variables(inputs, bar_variables, mask)

            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)

            # Typical PGD implementation
            grad = torch.autograd.grad(loss, bar_variables, only_inputs=True)[0]
            grad = torch.sign(grad)

            bar_variables = bar_variables + self.step_size * grad
            bar_variables = torch.clamp(bar_variables, 0, self.epsilon)

            bar_variables = bar_variables.detach()
            bar_variables.requires_grad = True

        adv_inputs = apply_bar_variables(inputs, bar_variables, mask)

        return adv_inputs


class PrisonAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = PrisonAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            num_bars=args.prison_num_bars,
            bar_width=args.prison_bar_width,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return PrisonAttack(model, args)

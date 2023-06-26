import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import config
import attacks.attacks
from attacks.attacks import AttackInstance


def apply_strengths(
    inputs,
    centers,
    variables,
    colour_images,
    distance_scaling,
    image_threshold,
    distance_normaliser,
):
    """
    Given some variables describing centers, return the "strength" of each center for each input.
    """
    batch_size, channels, height, width = inputs.size()

    xx, yy = torch.meshgrid(
        torch.arange(0, height), torch.arange(0, width), indexing="xy"
    )
    xx, yy = xx / width, yy / height
    xx, yy = xx.to(config.device), yy.to(config.device)
    centers = centers.unsqueeze(0).unsqueeze(0)

    distances = (
        1
        - torch.sqrt(
            (xx.unsqueeze(-1) - centers[..., 0]) ** 2
            + (yy.unsqueeze(-1) - centers[..., 1]) ** 2
        )
    ) ** distance_normaliser
    distances_scaled = distances.unsqueeze(0) * variables.unsqueeze(1).unsqueeze(1)
    distances_scaled = (
        torch.concat(
            [
                torch.ones_like(distances_scaled[..., 0:1], device=config.device)
                * image_threshold,
                distances_scaled,
            ],
            dim=-1,
        )
        * distance_scaling
    )
    distances_softmax = torch.softmax(distances_scaled, dim=-1)

    interpolation_images = torch.concat([inputs.unsqueeze(-1), colour_images], dim=-1)
    return_images = interpolation_images * distances_softmax.unsqueeze(1)

    return torch.sum(return_images, dim=-1)


class PolkadotAdversary(nn.Module):
    """Implemetnation of the polkadot adversary, which works by adding polkadots to an image,
    and then optimisng their size."""

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        distance_metric,
        num_polkadots,
        distance_scaling,
        image_threshold,
        distance_normaliser,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.distance_metric = distance_metric
        self.num_polkadots = num_polkadots
        self.image_threshold = image_threshold
        self.distance_scaling = distance_scaling
        self.distance_normaliser = distance_normaliser

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: attacks.attacks.normalize_l2(x)
            self.project_tensor = lambda x, epsilon: torch.abs(
                attacks.attacks.tensor_clamp_l2(x, 0, epsilon)
            )
        elif distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.abs(
                torch.clamp(x, -epsilon, epsilon)
            )
            self.normalise_tensor = lambda x: torch.sign(x)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        batch_size, num_channels, height, width = inputs.size()

        # We initalise the interpolation matrix, on which the inner loop performs PGD.

        strength_vars = torch.rand(
            batch_size, self.num_polkadots, requires_grad=True, device=config.device
        )
        centers = torch.rand(self.num_polkadots, 2, device=config.device)
        colours = torch.rand(
            (batch_size, num_channels, 1, 1, self.num_polkadots), device=config.device
        ).repeat(1, 1, height, width, 1)

        # The inner loop
        for _ in range(self.num_steps):
            adv_inputs = apply_strengths(
                inputs,
                centers,
                strength_vars,
                colours,
                self.distance_scaling,
                self.image_threshold,
                self.distance_normaliser,
            )
            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)

            # Typical PGD implementation
            grad = torch.autograd.grad(loss, strength_vars, only_inputs=True)[0]
            grad = self.normalise_tensor(grad)

            strength_vars = strength_vars + self.step_size * grad
            strength_vars = self.project_tensor(strength_vars, self.epsilon)

            strength_vars = strength_vars.detach()
            strength_vars.requires_grad = True

        adv_inputs = apply_strengths(
            inputs,
            centers,
            strength_vars,
            colours,
            self.distance_scaling,
            self.image_threshold,
            self.distance_normaliser,
        )

        return adv_inputs


class PolkadotAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = PolkadotAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            num_polkadots=args.polkadot_num_polkadots,
            distance_scaling=args.polkadot_distance_scaling,
            image_threshold=args.polkadot_image_threshold,
            distance_normaliser=args.polkadot_distance_normaliser,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return PolkadotAttack(model, args)

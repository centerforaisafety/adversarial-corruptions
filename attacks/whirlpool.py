import torch
import math

from attacks.attacks import AttackInstance
import attacks.attacks
import random
import config
import torch.nn.functional as F
import torch.nn as nn


def swirl_mapping(coords, centers, strength, radius, min_strength):
    """
    Uses torch.grid interpolate to create a whirlpool effect.

    Arguments
    ----

    coords: torch.Tensor
        An x,y grid describing the starting coordinates of each pixel, used for interpolation

    centers: torch.Tensor
        A batch of x,y coordinates describing the centers of each whirlpool

    strength: torch.Tensor
        A batch of strength values for each whirlpool

    radius: torch.Tensor
        A batch of radius values for each whirlpool

    min_strength: torch.Tensor
        A batch of minimum strength values for each whirlpool, as we don't want gradients to die by some values being zero.
    """

    coords = coords.unsqueeze(-2)
    xx = coords[..., 0]
    yy = coords[..., 1]

    centers = centers.unsqueeze(1).unsqueeze(1)

    # We calculate the distance between each pixel and the center of the whirlpool
    dx = xx - centers[..., 0]
    dy = yy - centers[..., 1]

    r = torch.sqrt(dx**2 + dy**2)

    strength = strength.unsqueeze(1).unsqueeze(1)

    radius = radius / 5 * math.log(2)

    # The distance from each whirlpool is used to calculate the strength of the whirlpool at that point. See
    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_swirl.html for the formula
    theta = (strength + min_strength) * torch.exp(-r / radius) + torch.arctan2(dy, dx)

    # Calculate the whirlpool effect
    xx_new = centers[..., 0] + r * torch.cos(theta)
    yy_new = centers[..., 1] + r * torch.sin(theta)

    # Translate back to orignal (non-whirlpool centric) coordinates
    d_xnew = xx_new - xx
    d_ynew = yy_new - yy

    dx_ret = torch.max(d_xnew, dim=-1)[0]
    dy_ret = torch.max(d_ynew, dim=-1)[0]

    coords_ret = torch.stack([dx_ret, dy_ret], dim=-1) + coords.squeeze(-2)

    return coords_ret


def get_coordinates(image):
    batch_size, channel, height, width = image.shape
    # We use torch.meshgrid to create a grid of coordinates corresponding to the image pixels
    xx, yy = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )
    # We then normalise the coridnates between -1 and 1, as this is the range of the grid_sample function
    coords = torch.stack(
        [((xx / width) - 0.5) * 2, ((yy / height) - 0.5) * 2], dim=-1
    ).repeat(batch_size, 1, 1, 1)
    return coords.to(config.device)


def swirl_image(images, centers, rotation_strengths, radii, min_strengths):
    batch_size, channel, height, width = images.shape

    coordinates = get_coordinates(images)

    swirl_locations = swirl_mapping(
        coordinates, centers, rotation_strengths, radii, min_strengths
    )

    return torch.nn.functional.grid_sample(
        images,
        swirl_locations,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=None,
    )


class WhirlpoolAdversary(nn.Module):
    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        distance_metric,
        num_whirlpools,
        whirlpool_radius,
        whirlpool_min_strength,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.distance_metric = distance_metric
        self.num_whirlpools = num_whirlpools
        self.whirlpool_radius = whirlpool_radius
        self.whirlpool_min_strength = whirlpool_min_strength

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: attacks.attacks.normalize_l2(x)
            self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )
        elif distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.clamp(x, 0, epsilon)
            self.normalise_tensor = lambda x: torch.sign(x)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        """
        Implements the pixel attack, which works by pixelating the image in a differentiable manner.
        """
        batch_size, num_channels, height, width = inputs.shape

        if self.distance_metric == "l2":
            whirlpool_strength_vars = (
                self.epsilon
                * self.normalise_tensor(
                    torch.rand(batch_size, self.num_whirlpools, device=config.device)
                )
                * random.random()
            )
        if self.distance_metric == "linf":
            whirlpool_strength_vars = self.epsilon * torch.rand(
                batch_size, self.num_whirlpools, device=config.device
            )

        whirlpool_strength_vars.requires_grad = True

        whirlpool_centers = (
            torch.rand(batch_size, self.num_whirlpools, 2, device=config.device) - 0.5
        ) * 2

        for _ in range(0, self.num_steps):
            adv_inputs = swirl_image(
                inputs,
                whirlpool_centers,
                torch.abs(whirlpool_strength_vars),
                self.whirlpool_radius,
                self.whirlpool_min_strength,
            )
            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)
            grad = torch.autograd.grad(loss, whirlpool_strength_vars, only_inputs=True)[
                0
            ]

            whirlpool_strength_vars = (
                whirlpool_strength_vars
                + self.step_size * self.normalise_tensor(grad.unsqueeze(1)).squeeze(1)
            )
            whirlpool_strength_vars = self.project_tensor(
                whirlpool_strength_vars.unsqueeze(1), self.epsilon
            ).squeeze(1)
            whirlpool_strength_vars = whirlpool_strength_vars.detach()
            whirlpool_strength_vars.requires_grad = True

        adv_inputs = swirl_image(
            inputs,
            whirlpool_centers,
            torch.abs(whirlpool_strength_vars),
            self.whirlpool_radius,
            self.whirlpool_min_strength,
        )

        return adv_inputs.detach()


class WhirlpoolAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = WhirlpoolAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            num_whirlpools=args.num_whirlpools,
            whirlpool_radius=args.whirlpool_radius,
            whirlpool_min_strength=args.whirlpool_min_strength,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return WhirlpoolAttack(model, args)

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import attacks
from attacks.attacks import AttackInstance
import config


def get_pixels(image, pixel_size):
    """
    Pixelates an image by selecting every pixel_size numebered pixel.
    """

    pixelated_image = image[..., ::pixel_size, ::pixel_size].clone().detach()
    return pixelated_image


def pixelate_image(image, pixelated_image, pixel_variables):
    """
    Pixelates the image, depending on the value of each of the entries in pixel_variables.
    """

    batch_size, num_channels, pixelated_height, pixelated_width = pixelated_image.size()
    _, _, height, width = image.size()
    pixel_size = height // pixelated_height

    # Take the image and put it into blocks, then average the blocks, and return to the original iamge (note the averaging is done
    # by interpolating to the pixel function, which is equivalent)
    chunked_image = (
        image.view(
            batch_size,
            num_channels,
            pixelated_height,
            pixel_size,
            pixelated_width,
            pixel_size,
        )
        .transpose(3, 4)
        .contiguous()
    )
    pixel_variables = pixel_variables.unsqueeze(-1).unsqueeze(-1)
    return_image = (
        chunked_image * (1 - pixel_variables)
        + (pixelated_image.unsqueeze(-1).unsqueeze(-1)) * pixel_variables
    )
    return_image = (
        return_image.transpose(3, 4)
        .contiguous()
        .view(batch_size, num_channels, height, width)
    )

    return return_image.clamp(0, 1)


class PixelAdversary(nn.Module):
    """
    This carries out the Pixel attack, which works by pixelating the image in a differentiable manner.

    Parameters
    ----

    epislon (float):
         epsilon used in attack

    num_steps (int):
         number of steps used in the opitimsation loop

    step_size (flaot):
         step size used in the optimisation loop

    pixel_size (int):
        size of the pixels used in the pixelation

    distance_metric(str):
        distance metric used in the attack, either 'l2' or 'linf'
    """

    def __init__(self, epsilon, num_steps, step_size, distance_metric, pixel_size):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.pixel_size = pixel_size
        self.distance_metric = distance_metric

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: attacks.attacks.normalize_l2(x)
            self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )
        elif distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.clamp(x, -epsilon, epsilon)
            self.normalise_tensor = lambda x: torch.sign(x)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        """
        Implements the pixel attack, which works by pixelating the image in a differentiable manner.
        """
        batch_size, num_channels, height, width = inputs.size()

        even_pixelation = not (
            height % self.pixel_size == 0 and width % self.pixel_size == 0
        )
        if even_pixelation:
            inputs = F.interpolate(
                inputs,
                size=(
                    math.ceil(height / self.pixel_size) * self.pixel_size,
                    math.ceil(width / self.pixel_size) * self.pixel_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

        pixelated_image = get_pixels(inputs, self.pixel_size)

        pixel_vars = (
            self.epsilon
            * random.random()
            * self.normalise_tensor(
                torch.rand(
                    batch_size,
                    num_channels,
                    math.ceil(height / self.pixel_size),
                    math.ceil(width / self.pixel_size),
                    device=config.device,
                )
            )
        )

        pixel_vars = torch.abs(pixel_vars).clamp(0, 1)
        pixel_vars.requires_grad = True

        for _ in range(0, self.num_steps):
            adv_inputs = pixelate_image(inputs, pixelated_image, pixel_vars)

            if even_pixelation:
                adv_inputs = adv_inputs[..., :height, :width]

            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)
            grad = torch.autograd.grad(loss, pixel_vars, only_inputs=True)[0]

            pixel_vars = pixel_vars + self.step_size * self.normalise_tensor(grad)
            pixel_vars = self.project_tensor(pixel_vars, self.epsilon)
            pixel_vars = pixel_vars.clamp(0, 1)
            pixel_vars = pixel_vars.detach()
            pixel_vars.requires_grad = True

        adv_inputs = pixelate_image(inputs, pixelated_image, pixel_vars)

        if even_pixelation:
            adv_inputs = adv_inputs[..., :height, :width]

        return adv_inputs.detach()


class PixelAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = PixelAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            pixel_size=args.pixel_size,
            distance_metric=args.distance_metric,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return PixelAttack(model, args)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import attacks.attacks
import config


def wood_noise(noise, image_size, num_rings, normalising_constant):
    """We make the wood texture. This is done by taking the basic function f(x,y) = sin(x^2 + y^2), which maps concentric circles,
    and then adding some (optimisable) turbulence to the inputs to optimse the shape of the wood.
    """

    batch_size, _, height, width = image_size
    sin_frequency = math.pi * num_rings

    # We generate the coordinates required of the noise
    xs, ys = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )
    xs, ys = xs.float().to(config.device), ys.float().to(config.device)
    xs, ys = xs.expand(batch_size, -1, -1), ys.unsqueeze(0).expand(batch_size, -1, -1)

    # Make the corrdinates between -1 and 1
    x_value = 2 * (xs - width / 2) / width
    y_value = 2 * (ys - height / 2) / height

    # We apply f(x,y) = sin(x^2 + y^2) to the coordinatesm and also apply an
    # interpolation of the noise to the coordinates. The interpolation is
    # done to make the distortion more smooth.
    dist = torch.sqrt(
        x_value * x_value + y_value * y_value
    ) + torch.nn.functional.interpolate(
        noise, (height, width), mode="bilinear"
    ).squeeze(
        1
    )
    sin_value = torch.abs(torch.sin(sin_frequency * dist)) ** (1 / normalising_constant)

    return sin_value.unsqueeze(1)


def apply_wood_noise(image, noise):
    return (image * noise).clamp(0, 1)


class WoodAdversary(nn.Module):
    """
    Implements the Wood attack, which works by overloaying ring-like patterns on the image which are then optimised.

    Parameters
    ----------

    epsilon : float
        The amount that the image can be perturbed from the original image.

    num_steps : int
        The number of steps to optimise the noise for.

    step_size : float
        The step size to use when optimising the noise.

    noise_resolution : int
        The resolution of the noise to use. This is the resolution of the noise that is optimised, and then is interpolated between.

    num_rings : int
        The number of rings to use in the wood texture.
    """

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        noise_resolution,
        num_rings,
        random_init,
        normalising_constant,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.noise_resolution = noise_resolution
        self.num_rings = num_rings
        self.random_init = random_init
        self.normalising_constant = normalising_constant

    def forward(self, model, inputs, targets):
        batch_size, _, height, width = inputs.shape

        if self.random_init:
            noise_variables = self.epsilon * torch.rand(
                (batch_size, 1, self.noise_resolution, self.noise_resolution)
            ).to(config.device)
        else:
            noise_variables = torch.zeros(
                (batch_size, 1, self.noise_resolution, self.noise_resolution)
            ).to(config.device)

        noise_variables.requires_grad_()

        # begin optimizing the inner loop
        for i in range(self.num_steps):
            noise = wood_noise(
                noise_variables,
                inputs.size(),
                self.num_rings,
                self.normalising_constant,
            )
            adv_inputs = apply_wood_noise(inputs, noise)
            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)

            grad = torch.autograd.grad(loss, noise_variables)[0]
            grad = torch.sign(grad)

            noise_variables = noise_variables + self.step_size * grad
            noise_variables = noise_variables.clamp(-self.epsilon, self.epsilon)

            noise_variables = noise_variables.detach()
            noise_variables.requires_grad_()

        noise = wood_noise(
            noise_variables, inputs.size(), self.num_rings, self.normalising_constant
        )
        adv_inputs = apply_wood_noise(inputs, noise)

        return adv_inputs


class WoodAttack(attacks.attacks.AttackInstance):
    def __init__(self, model, args):
        super(WoodAttack, self).__init__(model, args)
        self.attack = WoodAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            noise_resolution=args.wood_noise_resolution,
            num_rings=args.wood_num_rings,
            random_init=args.wood_random_init,
            normalising_constant=args.wood_normalising_constant,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return WoodAttack(model, args)

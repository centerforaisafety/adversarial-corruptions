import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import attacks
import config
from attacks.attacks import AttackInstance


def fog_creator(fog_vars, bsize, mapsize, wibbledecay):
    """This carries out the diamond square algorithm. Here:
    Parameters
    ---
    fog_vars: list of Tensor
       These are the perturbations which we are optimising. Every group of three entries correspond to one size of grid,
       and each entry within the group of three corresponds to one type of pertubation location (perturbation in the centre of squares,
       pertubations in the centre of diamonds on odd rows, and perturbations on the centre of dimaodns on even rows).
    b_size: int
       batch size of generated noise
    mapsize: int
       size of noise grid, has to be a power of two
    wibbledecay: float
       Controls how quickly the noise decays as the fog decreases (intuitively, this controls the "size" of the structure in the fog)
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = torch.from_numpy(
        np.zeros((bsize, mapsize, mapsize), dtype=np.float32)
    ).to(config.device)
    maparray[:, 0, 0] = 0
    stepsize = mapsize
    wibble = 1

    var_num = 0

    def wibbledmean(array, var_num):
        """given an array describing the sum of each corner, add the required pertubation to the array."""
        result = array / 4.0 + wibble * fog_vars[var_num]
        return result

    def fillsquares(var_num):
        """Go along each square, and calculate the value of each squares middle as the average of its corners, and then
        add some perturbation."""

        cornerref = maparray[:, 0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + torch.roll(cornerref, -1, 1)
        squareaccum = squareaccum + torch.roll(squareaccum, -1, 2)
        maparray[
            :, stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum, var_num)
        return var_num + 1

    def filldiamonds(var_num):
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""

        mapsize = maparray.size(1)
        drgrid = maparray[
            :, stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[:, 0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + torch.roll(drgrid, 1, 1)
        lulsum = ulgrid + torch.roll(ulgrid, -1, 2)
        ltsum = ldrsum + lulsum
        maparray[
            :, 0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(ltsum, var_num)
        var_num += 1
        tdrsum = drgrid + torch.roll(drgrid, 1, 2)
        tulsum = ulgrid + torch.roll(ulgrid, -1, 1)
        ttsum = tdrsum + tulsum
        maparray[
            :, stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize
        ] = wibbledmean(ttsum, var_num)
        return var_num + 1

    while stepsize >= 2:  # For every square and diamond size, fill the array
        var_num = fillsquares(var_num)
        var_num = filldiamonds(var_num)
        stepsize //= 2
        wibble /= wibbledecay

    return torch.abs((maparray)).reshape(bsize, 1, mapsize, mapsize).clamp(0, 1)


def apply_fog(
    inputs, fog, grey=torch.tensor([0.485, 0.45, 0.406], device=config.device)
):
    return inputs * (1 - fog) + fog * grey.reshape(1, 3, 1, 1)


class FogAdversary(nn.Module):
    """
    This class implements tbe fog attack. This attack creates fog using the diamond square algorithm, which generates
    self-similar noise by repeatedly tiling the grid by squares, followed by diamonds. The attack optimises the noise by replacing
    the usual random perturbations which are used to generate the noise in the classical algorithm, by optimisable variables.

    Parameters
    ---
    epsilon: float
        This controls the size of the added perturbations.
    wibbledecay: float
       As the agorithm recurses, and the granularity of the tiling grid gets smaller, the perturbations must also decrease (this is a
       one wishes to keep the density of the fog constant, so more grid squares means less "budget" per square). This argument controls
       the speed of this decay.
    num_steps: int
        How many optimisation steps are taken by the algorithm
    step_size: flaot
        size of the optimisation steps
    distance_metric: string
       describes which distance metric is used to constrain the perutrbations used within the attack.
    """

    def __init__(self, epsilon, num_steps, step_size, distance_metric, wibbledecay):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.wibbledecay = wibbledecay
        if distance_metric == "linf":
            self.clamp_tensor = lambda x, epsilon: torch.clamp(x, -epsilon, epsilon)

        elif distance_metric == "l2":
            self.clamp_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )

        else:
            raise f"The distance metric {distance_metric} is not supported"

    def forward(self, model, inputs, targets):
        """
        :param model: the classifier's forward method
        :param inputs: batch of images
        :param targets: true labels
        :return: perturbed batch of images
        """
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        bsize = inputs.size(0)
        height = inputs.size(-1)

        bits = math.ceil(math.log2(height))
        nearest_power_of_2 = 2**bits
        padding = (nearest_power_of_2 - height) // 2
        # create fog variables
        fog_vars = []
        for i in range(bits):
            for j in range(3):
                fog_vars.append(
                    self.epsilon
                    * torch.rand(
                        (bsize, 2**i, 2**i),
                        requires_grad=True,
                        device=config.device,
                    )
                )

        for i in range(self.num_steps):
            fog = fog_creator(fog_vars, bsize, nearest_power_of_2, self.wibbledecay)
            if padding != 0:
                fog = fog[:, :, padding:-padding, padding:-padding]

            adv_inputs = apply_fog(inputs, fog)
            logits = model(adv_inputs)

            loss = F.cross_entropy(logits, targets)

            grads = torch.autograd.grad(loss, fog_vars, only_inputs=True)

            for i in range(len(fog_vars)):
                grad = grads[i]
                grad = torch.sign(grad)

                fog_vars[i] = fog_vars[i] + self.step_size * grad

                fog_vars[i] = self.clamp_tensor(fog_vars[i], self.epsilon)
                fog_vars[i].detach()
                fog_vars[i].requires_grad_()

        fog = fog_creator(fog_vars, bsize, nearest_power_of_2, self.wibbledecay)
        if padding != 0:
            fog = fog[:, :, padding:-padding, padding:-padding]

        return apply_fog(inputs, fog).detach()


class FogAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = FogAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            wibbledecay=args.fog_wibbledecay,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return FogAttack(model, args)

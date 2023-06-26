import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import attacks
import config
from attacks.attacks import AttackInstance


def get_gabor(k_size, sigma, Lambda, theta):
    """Generates a single gabor kernel"""
    y, x = torch.meshgrid(
        [torch.linspace(-0.5, 0.5, k_size), torch.linspace(-0.5, 0.5, k_size)]
    )
    y = y.to(config.device)
    x = x.to(config.device)
    rotx = x * torch.cos(theta) + y * torch.sin(theta)
    roty = -x * torch.sin(theta) + y * torch.cos(theta)
    g = torch.zeros(y.shape)
    g = torch.exp(
        -0.5 * (rotx**2 / (sigma + 1e-3) ** 2 + roty**2 / (sigma + 1e-3) ** 2)
    )
    g = g * torch.cos(2 * np.pi * Lambda * rotx)
    return g


def get_gabor_with_sides(k_size, sigma, Lambda, theta, sides=3):
    """Creates "sides" for the gabor kernel"""
    g = get_gabor(k_size, sigma, Lambda, theta)
    for i in range(1, sides):
        g += get_gabor(k_size, sigma, Lambda, theta + np.pi * i / sides) / sides
    return g


def create_gabor_kernel(batch_size, k_size, sigma, sides):
    """Makes a gabor Kernel or in this case, a superpostion of gabor kernels, each angled in a different direction,"""
    kernels = []
    theta = np.pi * torch.rand(1).to(config.device)
    gamma = ((k_size / 4.0 - 3) * torch.rand(1) + 3).to(config.device)

    for b in range(batch_size):
        kernels.append(get_gabor_with_sides(k_size, sigma, gamma, theta, sides))

    gabor_kernel = torch.cat(kernels, 0).view(-1, 1, k_size, k_size)

    return gabor_kernel


def generate_random_distribution(
    g_vars, mask, image, batch_size, k_size, sigma, sides, size
):
    """Generates the required Gabor noise in a differentiable manner.

    Parameters
    ---
    g_vars: list of Tensors
       This is a list of tensors, each of size (batch_size, image_height,image_width), one for each image channel. They repesent the sparse matrices used
       in the gabor convolution

    image: Tensor
        This is the image which we are adding the noise to. Used for clipping reasons

    batch_size: int
        size of image batch

    k_size: int
       size of kernel

    sigma: float
       float parameter for gabor kernel

    sides: int
       How many gabor kernels are stacked on top of each other (varying the theta parameters)
    """

    # Although it seems weird to recreate the gabor kernel every iteration, we found that this lead to the attack being less correlated with our other attacks and did not lead to large accuracy problems.
    gabor_kernel = create_gabor_kernel(batch_size, k_size, sigma, sides)
    res = []

    g_vars = g_vars * mask

    grid_size = g_vars[0].shape[-1]
    convolution_size = grid_size - k_size + 1
    padding_size = (size - convolution_size) // 2

    for (
        g_conv
    ) in (
        g_vars
    ):  # Due to weirdness with grouped convolution, the for loop is more efficient than using a single grouped convolution
        batch_size = g_conv.size(0)
        g_conv = g_conv.view(1, batch_size, g_conv.size(-2), g_conv.size(-1))
        g_conv = F.conv2d(
            g_conv,
            weight=gabor_kernel,
            stride=1,
            groups=batch_size,
            padding=padding_size,
        )
        g_conv = g_conv.view(batch_size, 1, g_conv.shape[-1], g_conv.shape[-2])
        res.append(g_conv)

    # Concatenates all of the images in the correct [batch,channel,height,width] form.
    out = torch.concat(res, dim=1)
    # As the noise is added to the image, we need to clamp the noise to the range [0-image,1-image] so that the adversarial example is in the range [0,1]
    out = torch.clamp(out, 0 - image, 1 - image)
    return out


class GaborAdversary(nn.Module):
    """This class implements a gabor-noise based adversary. Gabor noise (explained in more detail here https://arxiv.org/pdf/1906.03455.pdf), is
    a noise which is generated by convolving a "Gabor Kernel", over a randomly generated sparse matrix - a matrix of mostly zeros, where the values of
    the non-zero elements are randomly generated (in our case, they are generated by a uniform distribution in the range -epsilon to +epsilon).

    Parameters
    ---

    num_steps: int
        Number of optimisaiton steps taken when generating the Gabor noise

    epsilon: float
        The range of the values in the sparse matrix

    weight_density: float
        The density of non-zero values in the randomly generated sparse matrix

    kernel_size: int
        The size of the Gabor kernel

    sigma: float
        A parameter of the gabor kernel, controlling how quickly the values in the kernel reduce as you
        move away from the center.

    """

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        distance_metric,
        kernel_size,
        sides,
        sigma,
        weight_density,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.sides = sides
        self.sigma = sigma
        self.step_size = step_size
        self.weight_density = weight_density

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: attacks.attacks.normalize_l2(x)

            self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )
        elif distance_metric == "linf":
            self.normalise_tensor = lambda x: torch.sign(x)
            self.project_tensor = lambda x, epsilon: torch.clamp(x, -epsilon, epsilon)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        """This computes the acttual attack. The attack optimises the sparse matrix used in noise generation with the aim of maximisng classifier loss,
        using a seperate instance of Gabor noise for each image channel."""
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        batch_size, num_image_channels, img_size, _ = inputs.size()
        # Initalise the base
        sp_conv_numpy = sparse.random(
            num_image_channels * img_size * batch_size,
            img_size,
            density=self.weight_density,
            format="csr",
        )
        gabor_vars = (
            torch.FloatTensor(sp_conv_numpy.todense())
            .view(num_image_channels, batch_size, img_size, img_size)
            .to(config.device)
        )

        gabor_vars = self.normalise_tensor(gabor_vars) * self.epsilon * random.random()

        mask = (gabor_vars != 0).float()

        gabor_vars.requires_grad = True

        for i in range(self.num_steps):
            gabor_noise = generate_random_distribution(
                gabor_vars,
                mask,
                inputs,
                batch_size,
                self.kernel_size,
                self.sigma,
                self.sides,
                img_size,
            )
            logits = model(inputs + gabor_noise)
            loss = F.cross_entropy(logits, targets)

            grad = torch.autograd.grad(loss, gabor_vars)[0]

            gabor_vars = gabor_vars + self.step_size * self.normalise_tensor(grad)

            gabor_vars = self.project_tensor(gabor_vars, self.epsilon)
            gabor_vars = gabor_vars.detach()
            gabor_vars.requires_grad = True

        gabor_noise = generate_random_distribution(
            gabor_vars,
            mask,
            inputs,
            batch_size,
            self.kernel_size,
            self.sigma,
            self.sides,
            img_size,
        )
        adv_inputs = inputs + gabor_noise

        return adv_inputs


class GaborAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = GaborAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            kernel_size=args.gabor_kernel_size,
            sides=args.gabor_sides,
            sigma=args.gabor_sigma,
            weight_density=args.gabor_weight_density,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return GaborAttack(model, args)
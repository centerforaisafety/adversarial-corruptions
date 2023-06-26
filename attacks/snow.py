import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import attacks
import config
from attacks.attacks import AttackInstance


def make_snow_kernels(
    num_images, num_layers, snow_flake_length, gaussian_kernel_size=7, gaussian_sigma=5
):
    """
    This function generates the kernels which are passed over sparse matrices, to generate the snowflakes.
    Kernels generated by this function consist of a single line at a random orientation, where the line has
    varying levels of thickness along its length.

    num_images: number of images to which the kernels are applied
    num_layers: number of layers of snow which make up the adversarial perturbation
    snow_flake_length: length of the snowflakes which are to be produced by the code.
    gaussian_kernel_size: the size of the kernel which is used to make the line thicked and "more snow-flake-like"
    gaussian_sigma: the size of the

    """
    kernels = []
    for i in range(num_images):
        # Each iteration of the outer loop creates all the kernels for a single image
        # Randomly flip the direction of the snow, kept constant across a single image.
        flip = np.random.uniform() < 0.5

        # Each iteration of the inner loop creates a single kernel, for a single layer of snow. (there are multiple layers per image)
        for j in range(num_layers):
            kernel_size = snow_flake_length
            mid = kernel_size // 2

            k_npy = np.zeros((kernel_size, kernel_size))  # The kernel tensor

            # Get a weigthed line. Returns rr and cc as the x and y coordinates of the line points, and val as the values in those points
            rr, cc, val = weighted_line(
                mid,
                mid,
                np.random.randint(mid + 1, kernel_size),
                np.random.randint(mid + 1, kernel_size),
                np.random.choice([1, 3, 5], p=[0.6, 0.3, 0.1]),
                mid,
                kernel_size,
            )

            k_npy[rr, cc] = val
            k_npy[: mid + 1, : mid + 1] = k_npy[::-1, ::-1][
                : (mid + 1), : (mid + 1)
            ]  # The line generated by the algorithm is only in one quadrant, this extends it symetrically across the whole kernel

            if flip:
                k_npy = k_npy[:, ::-1]

            kernel = torch.FloatTensor(k_npy.copy()).view(1, kernel_size, kernel_size)

            gaussian_blur = torchvision.transforms.GaussianBlur(
                gaussian_kernel_size, gaussian_sigma
            )  # Add some blurring to make the line a bit thicker
            kernel = gaussian_blur(kernel)

            kernels.append(kernel)

    kernels = torch.stack(kernels).view(
        [num_images, num_layers, kernel_size, kernel_size]
    )
    return kernels


# Pretty ad-hoc line creation algorithm, taken from https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays/47381058#47381058


def weighted_line(c0, r0, c1, r1, w, rmin=0, rmax=np.inf):
    """
    This function generates a 2D-line with weighted values along its direction. Lines point
    either to the top right or top left, and are limited in their slope.

    c0,r0 :start point of line (x,y) format.
    c1,r1 :nd point of line (x,y) format.
    w : controls the width of the line
    rmin: controls the distance along the line at which the line begins to take values
    rmax: controls the ending distance of the line.

    """

    # If the line's slope is too high (above one), the algorithm below doesn't work. So reflect the line in
    # y = x, draw it, then reflect back afterwards
    if abs(c1 - c0) < abs(r1 - r0):
        xx, yy, val = weighted_line(r0, c0, r1, c1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # If the line points backwards, make sure it points upward (again, as the algorithm wouldn't work. This keep the line equivalent)
    if c0 > c1:
        return weighted_line(c1, r1, c0, r0, w, rmin=rmin, rmax=rmax)

    # Take the slope of the line
    slope = (r1 - r0) / (c1 - c0)
    # Adjust the thickness of the line, depending on the slope, for anti-aliasing
    w *= np.sqrt(1 + np.abs(slope)) / 2

    # Get the x values along the the line
    x = np.arange(c0, c1 + 1, dtype=float)

    # generate the line using the y = mx + c form (the expression at the end simply calculates the y intercept.)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    # Add thickness to the line depending on it slope, steeper lines are thicker
    thickness = np.ceil(w / 2)
    yy = np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(
        1, -1
    )  # yy are the y coordinates of the "extra line-thickening pixels"
    xx = np.repeat(x, yy.shape[1])
    # Calculate the values taken by the "line pixels"
    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()
    mask = np.logical_and.reduce(
        (yy >= rmin, yy < rmax, vals > 0)
    )  # keep only values which are within the rmin and rmax rangle, and which are strictly positive.

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def trapez(y, y0, w):
    # Calculate the value assigned to each line pixel.
    # y: the line pixels, each row of this matrix corresponds to y values at some particular x value
    # y0: the centre of the line

    return np.clip(
        np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1
    )  # As you move away from the centre of the line, the weight assigned to the pixels decreases linearly


def apply_snow_kernels(inputs, kernels):
    """Apply a set of kernels to a set of images. The ith kernel is applied to the ith input image.

    inputs: input images
    kernels: inputs kernels
    """

    pad = kernels.size(-1) // 2
    batch_size, num_layers, height, width = inputs.size()
    inputs = inputs.view(1, batch_size * num_layers, height, width)
    snow_grid = F.conv2d(inputs, kernels, padding=pad, groups=batch_size)
    return snow_grid.squeeze(0)


def get_location_mask(intensities, grid_size):
    """
    Generate a binary mask indicating the locations of the snowflakes.

    intensities: the variables which control the values of non-zero grid entries
    grid_size: roughly how far away are snowfakes
    """

    mask = torch.zeros(intensities.size())
    index = torch.rand(mask.size()) < 1 / grid_size**2
    mask[index] = 1
    return mask


def make_snow(flake_intensities, mask, kernels, normalizing_constant):
    flake_intensities = (flake_intensities * mask) ** normalizing_constant
    return apply_snow_kernels(flake_intensities, kernels)


def apply_snow(img, snow, discolor):
    """
    Applying snowflakes to an image, and discoloring the image to give a wintery look.

    img: Image to which the snow is to be appled
    snow: Tensor holding the snow noise
    discolor: Controls level of interpolation between original image and discoloured version
    """
    snow = snow.to(config.device).unsqueeze(1)
    batch_size, channels, height, width = img.shape

    out = (1 - discolor) * img + discolor * torch.max(
        img,
        (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]) * 1.5
        + 0.5,
    )
    return torch.clamp(out + snow[..., 0:height, 0:width], 0, 1)


class SnowAdversary(nn.Module):
    """
    Implements the snow attack. This attack works by first generating several matrices who's only non-zero
    entries appear in a regular grid. A special form of convolution is then applied to the matrix, such that
    each of the non-zero entries turns into a single snowflake, who's size is controlled by the value of that
    matrix entry. These values are then optimised to cause maximal disruption to the input model's classification.

    Parameters
    ----------

    epsilon (float):
        Controls the strength of the attack, by changing the size of the vaalues in the kernels which are convolved.

    distance_metric (str):
        The distance metric used to limit the magnitude of perturbations to the snow variable. Must be either "l2" or "linf".

    num_steps (int):
        The number of steps the optimisation algorithm takes.

    step_size (float):
        The size of each optimisation algoirthm step.

    flake_size (int):
        Size of snowflakes (size of the kernel we convolve with). Must be odd.

    num_layers (int):
        Number of different layers of snow applied to each image.

    grid_size (int):
        Distance between adjactent snow flakes (non-zero matrix entries) in the snow flake grids.

    snow_init (float):
        Value used when initalising the values with of the spare matrices.

    image_discolour (float):
        Strength of filter applied to image to make surroundings look more "wintery".

    normalising_constant (float):
        Snow matrix entries are raised to this variable. Makes snow flakes more sparse.

    gaussian_kernel_size (int):
        Size of gaussian blur used to blur the snow kernels. Affects thickness of flakes. Should be odd.

    sigma_range (float):
        Range of the gaussian blur for the snow kernels. Affects range of flake thickness.

    """

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        distance_metric,
        flake_size,
        num_layers,
        grid_size,
        snow_init,
        image_discolour,
        normalising_constant,
        gaussian_kernel_size,
        sigma_range,
    ):
        super().__init__()
        # Hyperparameters
        # Controls the allowed size of the values in the spare matrices we convolve with
        self.epsilon = epsilon
        # The distance metric used to limit the magnitude of perturbations to the snow variable
        self.distance_metric = distance_metric
        self.num_steps = (
            num_steps  # The number of steps the optimisation algorithm takes
        )
        self.step_size = step_size  # The size of each optimisation algoirthm step

        # Size of snowflakes (size of the kernel we convolve with). Must be odd.
        self.flake_size = flake_size
        # Number of different layers of snow applied to each image
        self.num_layers = num_layers
        # Distance between adjactent snow flakes (non-zero matrix entries) in the snow flake grids.
        self.grid_size = grid_size

        # Value used when initalising the values with of the spare matrices.
        self.snow_init = snow_init
        # Strength of filter applied to image to make surroundings look more "wintery"
        self.image_discolour = image_discolour
        # Snow matrix entries are raised to this variable. Makes grids more sparse.
        self.normalizing_constant = normalising_constant

        # Size of gaussian blur used to blur the snow kernels. Affects thickness of flakes. Should be odd.
        self.kernel_size = gaussian_kernel_size
        # Range of the gaussian blur for the snow kernels. Affects range of flake thickness.
        self.sigma_range = sigma_range

        if distance_metric == "l2":
            self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )
        elif distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.clamp(x, 0, epsilon)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        batch_size, _, image_size, _ = inputs.size()  # Assumes image is square
        # intensity_grid_size = math.ceil(image_size / self.grid_size) * self.grid_size
        intensity_grid_size = image_size

        # Initalise the variables controlling the non-zero matrix entries.
        flake_intensities = torch.exp(
            -1.0
            / (self.snow_init)
            * (
                torch.rand(
                    batch_size,
                    self.num_layers,
                    intensity_grid_size,
                    intensity_grid_size,
                )
            )
        ).to(config.device)

        num_vars = flake_intensities.view(flake_intensities.size(0), -1).size(1)

        flake_intensities = self.project_tensor(flake_intensities, self.epsilon)
        mask = get_location_mask(flake_intensities, self.grid_size).to(config.device)

        flake_intensities = flake_intensities * mask

        # create initial variables
        flake_intensities.requires_grad_()

        kernels = make_snow_kernels(
            batch_size,
            self.num_layers,
            self.flake_size,
            self.kernel_size,
            self.sigma_range,
        ).to(config.device)

        # begin optimizing the inner loop
        for i in range(self.num_steps):
            snow_noise = make_snow(
                flake_intensities, mask, kernels, self.normalizing_constant
            )
            adv_inputs = apply_snow(inputs.detach(), snow_noise, self.image_discolour)

            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)

            grad = torch.autograd.grad(loss, flake_intensities, only_inputs=True)[0]
            grad = torch.sign(grad)

            flake_intensities = flake_intensities + self.step_size * grad
            flake_intensities = self.project_tensor(flake_intensities, self.epsilon)

            flake_intensities = flake_intensities.detach()
            flake_intensities.requires_grad_()

        snow_noise = make_snow(
            flake_intensities, mask, kernels, self.normalizing_constant
        )
        adv_inputs = apply_snow(
            inputs.detach(), snow_noise, self.image_discolour
        ).clamp(0, 1)

        return adv_inputs


class SnowAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = SnowAdversary(
            epsilon=args.epsilon,
            distance_metric=args.distance_metric,
            num_steps=args.num_steps,
            step_size=args.step_size,
            flake_size=args.snow_flake_size,
            num_layers=args.snow_num_layers,
            grid_size=args.snow_grid_size,
            snow_init=args.snow_init,
            image_discolour=args.snow_image_discolour,
            normalising_constant=args.snow_normalizing_constant,
            gaussian_kernel_size=args.snow_kernel_size,
            sigma_range=(args.snow_sigma_range_lower, args.snow_sigma_range_upper),
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return SnowAttack(model, args)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import config
from attacks.attacks import AttackInstance


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x**2 + y**2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-((distance - mu) ** 2) / (2 * sigma**2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma**2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = x**2 + y**2
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1 :] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        # apply rotation
        kernel_angle_increased = (
            torchvision.transforms.functional.rotate(
                torch.from_numpy(thin_kernel_0).unsqueeze(0),
                angle,
                torchvision.transforms.InterpolationMode.BILINEAR,
            )
            .squeeze(0)
            .numpy()
        )
        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = abs(kernel_angle) == 1  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


class CannyFilter(nn.Module):
    """
    Implements the "Canny edge detector" as a PyTorch module. See https://en.wikipedia.org/wiki/Canny_edge_detector for
    details on the algorithm. Implementatation is based on the one here https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed.
    """

    def __init__(self, k_gaussian=3, mu=0, sigma=1, k_sobel=3):
        super(CannyFilter, self).__init__()
        # device
        self.device = config.device

        with torch.no_grad():
            gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
            self.gaussian_filter = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=k_gaussian,
                padding=k_gaussian // 2,
                bias=False,
            )
            self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

            sobel_2D = get_sobel_kernel(k_sobel)
            self.sobel_filter_x = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=k_sobel,
                padding=k_sobel // 2,
                bias=False,
            )
            self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

            self.sobel_filter_y = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=k_sobel,
                padding=k_sobel // 2,
                bias=False,
            )
            self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

            # thin

            thin_kernels = get_thin_kernels()
            directional_kernels = np.stack(thin_kernels)

            self.directional_filter = nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=thin_kernels[0].shape,
                padding=thin_kernels[0].shape[-1] // 2,
                bias=False,
            )
            self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

            # hysteresis

            hysteresis = np.ones((3, 3)) + 0.25
            self.hysteresis = nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
            ).to(self.device)
            self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

            self.gaussian_filter = self.gaussian_filter.to(self.device)
            self.sobel_filter_x = self.sobel_filter_x.to(self.device)
            self.sobel_filter_y = self.sobel_filter_y.to(self.device)
            self.directional_filter = self.directional_filter.to(self.device)
            self.hysteresis = self.hysteresis.to(self.device)

    def forward(self, img, threshold):
        with torch.no_grad():
            # set the setps tensors
            B, C, H, W = img.shape
            blurred = torch.zeros((B, C, H, W)).to(self.device)
            grad_x = torch.zeros((B, 1, H, W)).to(self.device)
            grad_y = torch.zeros((B, 1, H, W)).to(self.device)
            grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
            grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

            # gaussian

            for c in range(C):
                blurred[:, c : c + 1] = self.gaussian_filter(img[:, c : c + 1])

                grad_x = grad_x + self.sobel_filter_x(blurred[:, c : c + 1])
                grad_y = grad_y + self.sobel_filter_y(blurred[:, c : c + 1])

            # thick edges

            grad_x, grad_y = grad_x / C, grad_y / C
            grad_magnitude = (grad_x**2 + grad_y**2) ** 0.5
            grad_orientation = torch.atan(grad_y / grad_x)
            grad_orientation = (
                grad_orientation * (360 / np.pi) + 180
            )  # convert to degree
            grad_orientation = (
                torch.round(grad_orientation / 45) * 45
            )  # keep a split by 45

            # thin edges

            directional = self.directional_filter(grad_magnitude)
            # get indices of positive and negative directions
            positive_idx = (grad_orientation / 45) % 8
            negative_idx = ((grad_orientation / 45) + 4) % 8
            thin_edges = grad_magnitude.clone()
            # non maximum suppression direction by direction
            for pos_i in range(4):
                neg_i = pos_i + 4
                # get the oriented grad for the angle
                is_oriented_i = (positive_idx == pos_i) * 1
                is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
                pos_directional = directional[:, pos_i]
                neg_directional = directional[:, neg_i]
                selected_direction = torch.stack([pos_directional, neg_directional])

                # get the local maximum pixels for the angle
                is_max = selected_direction.min(dim=0)[0] > 0.0
                is_max = torch.unsqueeze(is_max, dim=1)

                # apply non maximum suppression
                to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
                thin_edges[to_remove] = 0.0

            # thresholds

            thin_edges = thin_edges / (
                torch.max(thin_edges.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
            )

            thin_edges = thin_edges > threshold

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges


def apply_mask(inputs, mask, edge_vars):
    return inputs + mask * edge_vars


class EdgeAdversary(nn.Module):
    """
    Implements the edge attack, which is only allowed to pertubs the edges of the image.
    """

    def __init__(self, epsilon, num_steps, step_size, threshold):
        super().__init__()
        self.epsilon = epsilon

        self.num_steps = num_steps
        self.step_size = step_size

        self.threshold = threshold

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.clone().to(config.device), targets.to(config.device)

        # We initalise the interpolation matrix, on which the inner loop performs PGD.
        filter = CannyFilter()

        edge_mask = (filter(inputs, threshold=self.threshold)[-1]).repeat(1, 3, 1, 1)

        inputs[edge_mask] = 0.5
        edge_vars = torch.rand(size=inputs.shape).to(config.device)
        edge_vars[~edge_mask] = 0.0
        edge_vars.requires_grad = True

        # The inner loop
        for i in range(self.num_steps):
            adv_inputs = apply_mask(inputs, edge_mask, edge_vars)
            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)

            # Typical PGD implementation
            grad = torch.autograd.grad(loss, edge_vars)[0]
            grad = torch.sign(grad)

            edge_vars = edge_vars + self.step_size * grad
            edge_vars = torch.clamp(edge_vars, -self.epsilon, self.epsilon)

            edge_vars = edge_vars.detach()
            edge_vars.requires_grad = True

        adv_inputs = apply_mask(inputs, edge_mask, edge_vars)

        return adv_inputs


class EdgeAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = EdgeAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            threshold=args.edge_threshold,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return EdgeAttack(model, args)

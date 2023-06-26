import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from attacks.attacks import AttackInstance
from attacks.hsv import hsv2rgb


sin_60 = math.sin(math.pi / 3)


def get_random_circle(image_size, radius):
    """Returns a random circle"""
    batch_size, num_channels, height, width = image_size

    x = random.randrange(0, width)
    y = random.randrange(0, height)

    xx, yy = torch.meshgrid(
        torch.arange(width).to(config.device),
        torch.arange(height).to(config.device),
        indexing="xy",
    )

    circle = (xx - x) ** 2 + (yy - y) ** 2 < radius**2
    return circle


def get_random_equilateral_triangle(image_size, edge_size):
    """Returns a random triangle"""
    batch_size, num_channels, height, width = image_size
    triangle_mesh = torch.zeros((height, width), dtype=bool).to(config.device)

    x_1, y_1 = random.randrange(edge_size + 1, width - edge_size - 1), random.randrange(
        edge_size + 1, height - edge_size - 1
    )

    theta_1 = random.random() * 2 * math.pi
    x_2, y_2 = int(x_1 + math.cos(theta_1) * edge_size), int(
        y_1 + math.sin(theta_1) * edge_size
    )

    theta_2 = theta_1 + math.pi / 3
    x_3, y_3 = int(x_1 + math.cos(theta_2) * edge_size), int(
        y_1 + math.sin(theta_2) * edge_size
    )

    p1, p2, p3 = (x_1, y_1), (x_2, y_2), (x_3, y_3)

    fill_triangle(p1, p2, p3, triangle_mesh)

    return triangle_mesh


# Adapted from https://github.com/SpiderMaf/PiPicoDsply/blob/main/filled-triangles.py


def fill_triangle(p1, p2, p3, fill_tensor):
    triangle_coords = [p1, p2, p3]

    triangle_coords_sorted = sorted(triangle_coords, key=lambda p: p[1])

    p1, p2, p3 = triangle_coords_sorted

    # filled triangle routines ported From http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    if p2[1] == p3[1]:
        return fill_flat_triangle(p2, p1, p3, fill_tensor)
    else:
        if p1[1] == p2[1]:
            fill_flat_triangle(p1, p3, p2, fill_tensor)
        else:
            newx = int(
                p1[0] + (float(p2[1] - p1[1]) / float(p3[1] - p1[1])) * (p3[0] - p1[0])
            )
            newy = p2[1]
            p_new = (newx, newy)
            fill_flat_triangle(p2, p3, p_new, fill_tensor)
            fill_flat_triangle(p2, p1, p_new, fill_tensor)


def fill_flat_triangle(p1, p2, p3, tensor):
    # filled triangle routines ported From http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html

    input_height = tensor.shape[1]
    input_width = tensor.shape[0]

    if p1[0] > p3[0]:
        p1, p3 = p3, p1

    height = p2[1] - p1[1]

    if height == 0:
        xs = torch.arange(p1[0], p3[0])
        tensor[xs] = True
        return

    slope1 = (p2[0] - p1[0]) / height
    slope2 = (p2[0] - p3[0]) / height

    x_1 = p1[0]
    x_2 = p3[0]

    if height > 0:
        height_range = range(p1[1], p1[1] + int(height))
    else:
        height_range = range(p1[1], p1[1] + int(height), -1)
        slope1, slope2 = -slope1, -slope2

    for curr_h in height_range:
        if not 0 < curr_h < tensor.shape[1]:
            continue

        xs = torch.arange(max(0, int(x_1)), max(min(int(x_2), input_height), 0))
        tensor[xs, curr_h] = True

        x_1 += slope1
        x_2 += slope2


def get_random_square(image_size, edge_size):
    batch_size, num_channels, height, width = image_size
    square_mask = torch.zeros((height, width), dtype=bool).to(config.device)

    x_1, y_1 = random.randrange(
        edge_size + 1, int(width - edge_size - 1)
    ), random.randrange(edge_size + 1, int(height - edge_size - 1))
    theta_1 = random.random() * math.pi * 2
    theta_2 = theta_1 + math.pi / 2

    dx_1, dy_1 = edge_size * math.cos(theta_1), edge_size * math.sin(theta_1)
    dx_2, dy_2 = edge_size * math.cos(theta_2), edge_size * math.sin(theta_2)

    x_2, y_2 = int(x_1 + dx_1), int(y_1 + dy_1)
    x_3, y_3 = int(x_1 + dx_2), int(y_1 + dy_2)
    x_4, y_4 = int(x_1 + dx_1 + dx_2), int(y_1 + dy_1 + dy_2)

    p1, p2, p3, p4 = (x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)

    fill_square(p1, p2, p3, p4, square_mask)

    return square_mask


def fill_square(p1, p2, p3, p4, tensor):
    fill_triangle(p1, p2, p4, tensor)
    fill_triangle(p1, p3, p4, tensor)


def get_random_shape_mask(image_size, shape_size):
    shape = random.randrange(0, 3)

    if shape == 0:
        return get_random_circle(image_size, shape_size / 2)
    elif shape == 1:
        return get_random_equilateral_triangle(
            image_size, int(shape_size * (1.5 / sin_60))
        )
    elif shape == 2:
        return get_random_square(image_size, shape_size)


def get_shape_masks(image_size, num_shapes, shape_size):
    shape_mask_batch = []
    for _ in range(image_size[0]):
        shape_masks = []
        for _ in range(0, num_shapes):
            shape_mask = get_random_shape_mask(image_size, shape_size)
            shape_masks.append(shape_mask.to(config.device))

        shape_masks = torch.stack(shape_masks, dim=0)
        shape_mask_batch.append(shape_masks)
    shape_mask_batch = torch.stack(shape_mask_batch, dim=0)

    return shape_mask_batch


def init_colours(batch_size, num_shapes, min_value_valence, min_value_saturation):
    colours = torch.zeros((batch_size, 3, num_shapes, 1)).to(config.device)

    colours[:, 0:1, :, 0:1] = torch.rand((batch_size, 1, num_shapes, 1)).to(
        config.device
    )
    colours[:, 1:2, :, 0:1] = (
        torch.rand((batch_size, 1, num_shapes, 1)).to(config.device)
        * (1 - min_value_saturation)
        + min_value_saturation
    )
    colours[:, 2:3, :, 0:1] = (
        torch.rand((batch_size, 1, num_shapes, 1)).to(config.device)
        * (1 - min_value_valence)
        + min_value_valence
    )

    colours = hsv2rgb(colours)

    return colours.squeeze(-1).transpose(1, 2)


def colour_shapes(shape_masks, colour):
    shape_masks_extended = shape_masks.unsqueeze(2).repeat(1, 1, 3, 1, 1)

    # broadcasting to give each shape a colour
    shapes = shape_masks_extended.float() * colour.unsqueeze(-1).unsqueeze(-1)
    return shapes


def add_shapes(image_batch, shape_masks, edge_mask, colour, transparency):
    image_batch = image_batch.clone()

    shape_masks_extended = shape_masks.unsqueeze(2)
    # broadcasting to give each shape a colou
    shapes_colour = shape_masks_extended.float() * colour.unsqueeze(-1).unsqueeze(-1)
    shapes_colour = shapes_colour.sum(dim=1)
    shape_masks_combined = shape_masks_extended.sum(dim=1) > 0
    image_uncoloured = (
        1 - shape_masks_combined.float() * (1 - transparency)
    ) * image_batch

    image_batch = image_uncoloured + shapes_colour

    for i in range(image_batch.size(0)):
        image_batch[i][..., edge_mask[i]] = 0
    return image_batch


def get_edge_masks(shape_mask, edge_width):
    num_shapes, height, width = shape_mask.shape
    kernels = torch.ones(num_shapes, 1, edge_width, edge_width).to(config.device)

    # Using convolutions to find a larger outline to the shape
    bigger_shape = F.conv2d(
        shape_mask.unsqueeze(0).float(),
        kernels,
        padding=edge_width // 2,
        groups=num_shapes,
    ).squeeze(0)

    bigger_shape_mask = bigger_shape > 0
    edge_mask_seperate = torch.logical_and(
        bigger_shape_mask, torch.logical_not(shape_mask.squeeze(0).squeeze(0) > 0)
    )

    edge_mask = edge_mask_seperate > 0
    return edge_mask


def add_variables(
    image_batch,
    shape_masks,
    colour_variables,
    edge_mask,
    edge_variables,
    transparency,
    colour_init,
    same_color,
):
    batch_size, num_shapes, height, width = shape_masks.size()

    shape_masks_extended = shape_masks.unsqueeze(2)
    # broadcasting to give each shape a colou
    shape_masks_combined = shape_masks_extended.sum(dim=1) > 0
    if same_color:
        shapes_colour = shape_masks_extended.float() * (
            colour_init + colour_variables
        ).unsqueeze(-1).unsqueeze(-1)
        shapes_colour = shapes_colour.sum(dim=1)
    else:
        shapes_colour = shape_masks_extended.float() * colour_init.unsqueeze(
            -1
        ).unsqueeze(-1)
        shapes_colour = (
            shapes_colour.sum(dim=1) + shape_masks_combined.float() * colour_variables
        )
    image_uncoloured = (
        1 - shape_masks_combined.float() * (1 - transparency)
    ) * image_batch

    image_batch = image_uncoloured + shapes_colour

    edge_mask = edge_mask.sum(1) > 0
    for i in range(image_batch.size(0)):
        image_batch[i][..., edge_mask[i]] = 0

    edge_colour = (
        torch.abs(edge_variables).repeat(1, 3, 1, 1) * edge_mask.unsqueeze(1).float()
    )
    image_batch = image_batch + edge_colour

    return image_batch.clamp(0, 1)


class KaleidoscopeAdversary(nn.Module):
    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        num_shapes,
        shape_size,
        min_value_valence,
        min_value_saturation,
        transparency,
        edge_width,
    ):
        """
        Implements the Kaleidoscope Adversary, whcih involves adding random shapes to an image and then optimising their colour.

        Parameters
        ---
        epsilon: float
            The maximum amount of distortion to add to the image. This is the maximum L2 distance between the original image and the adversarial image.

        num_steps: int
            The number of steps to take when optimising the colour of the shapes.

        step_size: float
            The step size to take when optimising the colour of the shapes.

        num_shapes: int
            The number of shapes to add to the image.

        shape_size: int
            The size of the shapes to add to the image, in pixels.

        min_value_valence: float
            The minimum value of the valence channel of the colour of the shapes. This is to ensure that the shapes are colourful.

        min_value_saturation: float
            The minimum value of the saturation channel of the colour of the shapes. This is to ensure that the shapes are not too dark.

        transparency: float
            The transparency of the shapes. This is the amount of the original image that is visible through the shapes.


        """

        super().__init__()
        self.epsilon = epsilon

        self.num_steps = num_steps
        self.step_size = step_size
        self.num_shapes = num_shapes
        self.shape_size = shape_size
        self.min_value_valence = min_value_valence
        self.min_value_saturation = min_value_saturation
        self.transparency = transparency
        self.edge_width = edge_width
        self.same_color = True

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        batch_size, _, height, width = inputs.size()

        # We initalise the interpolation matrix, on which the inner loop performs PGD.

        colour_init = init_colours(
            batch_size,
            self.num_shapes,
            self.min_value_valence,
            self.min_value_saturation,
        )
        if self.same_color:
            colour_vars = (
                (torch.rand(size=colour_init.size()).to(config.device) - 0.5)
                * self.epsilon
                * 2
            )
        else:
            colour_vars = (
                (torch.rand(size=inputs.shape).to(config.device) - 0.5)
                * self.epsilon
                * 2
            )
        colour_vars.requires_grad = True
        edge_vars = (
            torch.rand(size=(batch_size, 1, height, width)).to(config.device)
            * self.epsilon
        )
        edge_vars.requires_grad = True

        shape_mask = get_shape_masks(inputs.size(), self.num_shapes, self.shape_size)
        edge_mask = torch.stack(
            [
                get_edge_masks(shape_mask[i], self.edge_width)
                for i in range(shape_mask.size(0))
            ],
            dim=0,
        )

        # Optimize colors
        for k in range(self.num_steps):
            adv_inputs = add_variables(
                inputs,
                shape_mask,
                colour_vars,
                edge_mask,
                edge_vars,
                self.transparency,
                colour_init,
                self.same_color,
            )

            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)

            colour_grads, edge_grads = torch.autograd.grad(
                loss, [colour_vars, edge_vars]
            )

            colour_vars = colour_vars + self.step_size * torch.sign(colour_grads)
            colour_vars = torch.clamp(colour_vars, -self.epsilon, self.epsilon)

            colour_vars = colour_vars.detach()
            colour_vars.requires_grad = True

            edge_vars = edge_vars + self.step_size * torch.sign(edge_grads)
            edge_vars = torch.clamp(edge_vars, -self.epsilon, self.epsilon)

            edge_vars = edge_vars.detach()
            edge_vars.requires_grad = True

        adv_inputs = add_variables(
            inputs,
            shape_mask,
            colour_vars,
            edge_mask,
            edge_vars,
            self.transparency,
            colour_init,
            self.same_color,
        )
        return adv_inputs


class KaleidoscopeAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = KaleidoscopeAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            num_shapes=args.kaleidoscope_num_shapes,
            shape_size=args.kaleidoscope_shape_size,
            min_value_valence=args.kaleidoscope_min_value_valence,
            min_value_saturation=args.kaleidoscope_min_value_saturation,
            transparency=args.kaleidoscope_transparency,
            edge_width=args.kaleidoscope_edge_width,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return KaleidoscopeAttack(model, args)

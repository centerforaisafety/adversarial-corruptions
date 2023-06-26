import torch

from attacks.attacks import AttackInstance
import attacks
import random
import config
import torch.nn.functional as F
import torch.nn as nn


def shift_image(image, shift_vars):
    _, shift_height, shift_width, _ = shift_vars.shape
    batch_size, channels, height, width = image.shape

    num_repeats_height, num_repeats_width = height // shift_height, width // shift_width

    base_coords = torch.stack(
        torch.meshgrid(torch.arange(height), torch.arange(width), indexing="xy"),
        axis=-1,
    )
    base_coords = (
        (base_coords / torch.tensor([height, width], dtype=torch.float32)) - 0.5
    ) * 2
    base_coords = base_coords.to(config.device)

    shifted_coords_repeat = torch.repeat_interleave(
        shift_vars, num_repeats_height, axis=1
    )
    shifted_coords_repeat = torch.repeat_interleave(
        shifted_coords_repeat, num_repeats_width, axis=2
    )

    shifted_coords = base_coords + shifted_coords_repeat

    return torch.nn.functional.grid_sample(image, shifted_coords, padding_mode="zeros")


class KlotskiAdversary(nn.Module):

    """
    Implements the Klotski attack, which works by cutting the image into blocks and shifting them around.

    Arguments
    ---

    epsilon: float
        The maximum distortion on each block

    num_steps: int
        The number of steps to take in the attack

    step_size: float
        The step size to take in the attack

    distance_metric: str
        The distance metric to use, either 'l2' or 'linf'
    """

    def __init__(self, epsilon, num_steps, step_size, distance_metric, num_blocks):
        super(KlotskiAdversary, self).__init__()

        self.step_size = step_size
        self.distance_metric = distance_metric
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.num_blocks = num_blocks

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
        batch_size, num_channels, height, width = inputs.shape

        if self.distance_metric == "l2":
            shift_vars = (
                self.epsilon
                * self.normalise_tensor(
                    torch.rand(
                        batch_size,
                        self.num_blocks,
                        self.num_blocks,
                        2,
                        device=config.device,
                    )
                )
                * random.random()
            )
        if self.distance_metric == "linf":
            shift_vars = self.epsilon * torch.rand(
                batch_size, self.num_blocks, self.num_blocks, 2, device=config.device
            )

        shift_vars.requires_grad = True

        for _ in range(0, self.num_steps):
            adv_inputs = shift_image(inputs, shift_vars)

            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)
            grad = torch.autograd.grad(loss, shift_vars, only_inputs=True)[0]

            shift_vars = shift_vars + self.step_size * self.normalise_tensor(
                grad.unsqueeze(1)
            ).squeeze(1)
            shift_vars = self.project_tensor(
                shift_vars.unsqueeze(1), self.epsilon
            ).squeeze(1)
            shift_vars = shift_vars.detach()
            shift_vars.requires_grad = True

        adv_inputs = shift_image(inputs, shift_vars)

        return adv_inputs.detach()


class KlotskiAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = KlotskiAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            num_blocks=args.klotski_num_blocks,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return KlotskiAttack(model, args)

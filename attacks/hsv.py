import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import attacks
import config
from attacks.attacks import AttackInstance


def rgb2hsv(rgb):
    """
    rgb: an input image tensor in the RGB colour format.
    """
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb(hsv):
    """
    hsv: an input tensor in the HSV colour format.
    """
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def add_hsv_perturbation(vars, images, epsilon, kernel_size, sigma):
    """
    Adds the relevant noise in HSV colour space (plus some extra blurring to make it more visually interesting).

    vars: the perturbation which is aded to the images.
    images: the input images which are to be perturbed.
    """

    vars = torchvision.transforms.functional.gaussian_blur(
        vars, kernel_size=kernel_size, sigma=sigma
    )

    images = rgb2hsv(images)
    images = torch.clamp(images + vars, 0, 1)
    images = torch.clamp(hsv2rgb(images), 0, 1)

    return images


class HSVAdversary(nn.Module):
    """
    Implements the HSV attack, which works by translating an image to the HSV colour space, adding a perturbation in the HSV colour space, and then translating back
    to the RGB colour space.

    Parameters
    ---

    epsilon: float
        The maximum perturbation allowed in the HSV colour space.

    num_steps: int
        The number of steps to take in the attack.

    step_size: float
        The step size to take in the attack.

    distance_metric: str
        The distance metric to use in the attack. Can be either "linf" or "l_2".

    kernel_size: int
        The kernel size to use in the blurring step.

    sigma: float
        The sigma to use in the blurring step.
    """

    def __init__(
        self, epsilon, num_steps, step_size, distance_metric, kernel_size, sigma
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.sigma = sigma

        if distance_metric == "linf":
            self.clamp_tensor = lambda x, epsilon: torch.clamp(x, -epsilon, epsilon)

        elif distance_metric == "l2":
            self.clamp_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )

        else:
            raise ValueError(f"The distance metric {distance_metric} is not supported")

    def forward(self, model, inputs, targets):
        batch_size, channels, height, width = inputs.shape
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        # Initalise the perturbation in HSV space, which is then optimised by the inner loop.
        hsv_perturb = self.epsilon * torch.rand(inputs.shape, device=config.device).to(
            config.device
        )
        hsv_perturb.requires_grad = True

        for i in range(self.num_steps):
            adv_inputs = add_hsv_perturbation(
                hsv_perturb, inputs.detach(), self.epsilon, self.kernel_size, self.sigma
            )
            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)

            # PGD
            grad = torch.autograd.grad(loss, hsv_perturb, only_inputs=True)[0]
            hsv_perturb = hsv_perturb + self.step_size * torch.sign(grad)
            hsv_perturb = hsv_perturb.clamp(-self.epsilon, self.epsilon)

            hsv_perturb = hsv_perturb.detach()
            hsv_perturb.requires_grad_()

        adv_inputs = add_hsv_perturbation(
            hsv_perturb, inputs.detach(), self.epsilon, self.kernel_size, self.sigma
        )
        adv_inputs = (adv_inputs).clamp(0, 1)

        return adv_inputs


class HsvAttack(AttackInstance):
    def __init__(self, model, args):
        super(HsvAttack, self).__init__(model, args)
        self.attack = HSVAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            kernel_size=args.hsv_kernel_size,
            sigma=args.hsv_sigma,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return HsvAttack(model, args)

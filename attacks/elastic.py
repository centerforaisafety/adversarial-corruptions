import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import attacks
import config
from attacks.attacks import AttackInstance

base_flow = None


def get_base_flow(shape):
    _, _, height, width = shape
    xflow, yflow = torch.meshgrid(
        torch.linspace(-1, 1, height, device=config.device),
        torch.linspace(-1, 1, width, device=config.device),
        indexing="xy",
    )
    base_flow = torch.stack((xflow, yflow), dim=-1)
    base_flow = torch.unsqueeze(base_flow, dim=0)

    return base_flow


def flow(image, flow_variables, scale_factor):
    flow_variables = F.interpolate(
        flow_variables, scale_factor=scale_factor, mode="bicubic"
    )

    if not hasattr(flow, "base_flow"):
        flow.base_flow = get_base_flow(image.shape)

    flow_image = F.grid_sample(
        image, flow.base_flow + flow_variables.permute([0, 2, 3, 1]), mode="bilinear"
    ).requires_grad_()

    return flow_image


class ElasticAdversary(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, scale_factor):
        super().__init__()
        """
        Implementaiton of the Elastic attack, functoning by perturbing the image using an optimisable flow field.

        Parameters
        ----
        
            epsilon (float): maximum perturbation
            num_steps (int): number of steps
            step_size (float): step size
        """
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.step_size = step_size
        self.scale_factor = scale_factor
        self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
            x, 0, epsilon
        )

    def forward(self, model, inputs, targets):
        batch_size, _, height, width = inputs.size()
        inputs.requires_grad = True

        epsilon = self.epsilon
        step_size = self.step_size

        flow_vars = 1.0 * (
            torch.rand(
                (
                    batch_size,
                    2,
                    height // self.scale_factor,
                    width // self.scale_factor,
                ),
                device=config.device,
            )
            * 2
            - 1
        )
        flow_vars = self.project_tensor(flow_vars, epsilon)
        flow_vars.requires_grad = True

        for _ in range(self.num_steps):
            adv_inputs = flow(inputs, flow_vars, self.scale_factor)
            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)

            grad = torch.autograd.grad(loss, flow_vars)[0]

            flow_vars = flow_vars + step_size * torch.sign(grad)
            flow_vars = self.project_tensor(flow_vars, epsilon)
            flow_vars = flow_vars.detach()
            flow_vars.requires_grad = True

        adv_inputs = flow(inputs, flow_vars, self.scale_factor)

        return adv_inputs


class ElasticAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = ElasticAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            scale_factor=args.elastic_upsample_factor,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return ElasticAttack(model, args)

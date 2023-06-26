import torch
import torch.nn.functional as F

import attacks
import config
from attacks.attacks import AttackInstance


class FGSM(AttackInstance):
    def __init__(self, model, args):
        super(FGSM, self).__init__(model, args)
        self.epsilon = args.epsilon
        self.step_size = args.step_size

        if args.distance_metric == "l2":
            self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )
        elif args.distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.clamp(x, -epsilon, epsilon)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {args.distance_metric}"
            )

    def generate_attack(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        delta = torch.zeros_like(inputs)

        delta = delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        adv_inputs = inputs + delta
        logits = self.model(adv_inputs)
        loss = F.cross_entropy(logits, targets)
        grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]

        adv_delta = self.step_size * torch.sign(grad)
        adv_delta = self.project_tensor(adv_delta, self.epsilon)
        adv_inputs = torch.clamp(inputs + adv_delta, 0, 1)

        return adv_inputs.detach()


def get_attack(model, args):
    return FGSM(model, args)

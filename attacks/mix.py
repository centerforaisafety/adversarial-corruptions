import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import attacks
import config
from attacks.attacks import AttackInstance


def dynamic_interpolation(
    image, perturbed_image, interpolation_matrix, kernel_size, sigma
):
    """Performs dynamic interpolation, as described above.

    image: the original input image.
    perturbed_image: the perturbed image which is interpolated with to create the example.
    interpolation_matrix: matrix used to do element-wise interpolation bertween two matrices.

    returns the adversarial example formed by interpolation between the original and perturbed image.
    """
    image = image.detach()
    interpolation_matrix = torch.abs(interpolation_matrix)
    interpolation_matrix = torchvision.transforms.functional.gaussian_blur(
        interpolation_matrix, kernel_size=kernel_size, sigma=sigma
    )

    adv_img = (
        interpolation_matrix * perturbed_image + (1 - interpolation_matrix) * image
    )
    # Need to make sure our adversarial image does not violate the pixel range constraints.
    adv_img = torch.clamp(adv_img, 0, 1)

    return adv_img


def get_mix_images(images, targets):
    """Given a list of images 'i' and their classes, returns another list of images sampled from that list 'i', such that
    the class of the nth images in the new list are different from the classes of the nth image in the original list.
    """

    targets = targets.detach().cpu()
    batch_size = len(targets)
    first_target = targets[0]

    all_same = True
    for i in targets:
        if i != first_target:
            all_same = False
            break

    if all_same:
        print(
            """Your input batch contains images which are all from the same class. Due to the the mix attack
        functions, the attack will not work on this batch. This should happen at a rate of (1/num_classes)^batch_size
        . If you are seeing this message frequently, please increase the randomness/size
        of input batches."""
        )
        return images

    returned_target_indices = torch.arange(batch_size).to(config.device)

    for i in range(batch_size):
        curr_target = targets[i]
        for j in torch.randperm(batch_size):
            if curr_target != targets[j]:
                returned_target_indices[i] = j
                break

    return images[returned_target_indices].detach()


class MixAdversary(nn.Module):
    """Implementation of the blur attack, which involves interpolating between an image and its blurred version,
    optimising for the correct level of interpolation.

    Parameters
    ---
    Most parameters are the same when compared to other attacks apart from.

    blur_kernel_size: int
     Size of the gaussian kernel used to add the blur

    sigma: float
      Standard deviation of the gaussian kernel within the blurring

    blur_init: float
       The initialisation value of the blur tensor"""

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        distance_metric,
        interp_kernel_size,
        interp_kernel_sigma,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.interp_kernel_size = interp_kernel_size
        self.interp_kernel_sigma = interp_kernel_sigma

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: attacks.attacks.normalize_l2(x)
            self.project_tensor = lambda x, epsilon: attacks.attacks.tensor_clamp_l2(
                x, 0, epsilon
            )
        elif distance_metric == "linf":
            self.normalise_tensor = lambda x: torch.sign(x)
            self.project_tensor = lambda x, epsilon: torch.clamp(x, 0, epsilon)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        a_start = time.time()
        inputs, targets = (
            inputs.to(config.device).detach(),
            targets.to(config.device).detach(),
        )
        """

        Implementation of the mix attack, which involves interpolating between an image and a random other image in the dataset (with a different class),
        optmising the element-wise interpolation matrix.

        model: the model to be attacked.
        inputs: batch of unmodified images.
        targets: true labels.

        returns: adversarially perturbed images.
        """

        batch_size, _, height, width = inputs.size()

        # We initalise the interpolation matrix, on which the inner loop performs PGD.

        interp_matrix = self.epsilon * self.normalise_tensor(
            torch.rand(
                (batch_size, 1, height, width), device=config.device, requires_grad=True
            )
        )
        mix_images = get_mix_images(inputs, targets).detach()

        # The inner loop
        for i in range(self.num_steps):
            t_loop = time.time()

            adv_inputs = dynamic_interpolation(
                inputs,
                mix_images,
                interp_matrix,
                self.interp_kernel_size,
                self.interp_kernel_sigma,
            )
            logits = model(adv_inputs)
            loss = F.cross_entropy(logits, targets)

            grad = torch.autograd.grad(loss, interp_matrix, only_inputs=True)[0]
            grad = self.normalise_tensor(grad)

            interp_matrix = interp_matrix + self.step_size * grad
            interp_matrix = self.project_tensor(interp_matrix, self.epsilon)

            interp_matrix = interp_matrix.detach()
            interp_matrix.requires_grad = True
            t_end_loop = time.time()

        adv_inputs = dynamic_interpolation(
            inputs,
            mix_images,
            interp_matrix,
            self.interp_kernel_size,
            self.interp_kernel_sigma,
        )
        a_end = time.time()

        return adv_inputs.detach()


class MixAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = MixAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            interp_kernel_size=args.mix_interp_kernel_size,
            interp_kernel_sigma=args.mix_interp_kernel_sigma,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return MixAttack(model, args)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from attacks.attacks import AttackInstance


def dct(x, norm="ortho"):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.reshape(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm="ortho"):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.reshape(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm="ortho"):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x - 128, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm="ortho"):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2) + 128


# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def image_to_patches(image):
    """Splits an image into 8 by 8 patches, each of which has the jpeg compression applied seperately"""
    k = 8
    batch_size, num_channels, height, width = image.shape

    image_reshaped = image.view(batch_size, num_channels, height // k, k, width // k, k)
    image_transposed = torch.transpose(image_reshaped, 3, 4)
    return image_transposed.reshape(batch_size, num_channels, -1, k, k)


# -3. Block joining
def patches_to_image(patches, image_shape):
    """Reverses the process of splitting an image into patches"""

    k = 8
    batch_size, num_channels, height, width = image_shape
    image_reshaped = patches.view(
        batch_size, num_channels, height // k, width // k, k, k
    )
    image_transposed = torch.transpose(image_reshaped, 3, 4)

    return image_transposed.reshape(batch_size, num_channels, height, width)


def ycbcr_to_rgb_jpeg(image):
    """Converts a batch of YCbCr image tensors to RGB image tensor, using the jpeg conversion matrix"""

    matrix_yuv = torch.tensor(
        np.array(
            [[1.0, 0.0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32,
        ).T,
        device=config.device,
    )

    shift_yuv = torch.FloatTensor([0, -128, -128]).to(config.device)

    image = image.permute(0, 2, 3, 1)
    image = torch.tensordot(image + shift_yuv, matrix_yuv, dims=1)
    image = image.permute(0, 3, 1, 2)

    return image


def rgb_to_ycbcr_jpeg(image):
    """Converts a batch of RGB image tensors to YCbCr image tensors"""

    matrix_rgb = torch.as_tensor(
        np.array(
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
            dtype=np.float32,
        ).T,
        device=config.device,
    )

    shift_rgb = torch.as_tensor([0.0, 128.0, 128.0], device=config.device)

    image = image.permute(0, 2, 3, 1)
    image = torch.tensordot(image, matrix_rgb, dims=1) + shift_rgb
    image = image.permute(0, 3, 1, 2)

    return image


def jpeg_encode(image, noise_vars):
    """Applies JPEG compression to an image, and adds the noise variables to the quantized coefficients"""

    # Whole algorithm functions with the assumption images are in the range [0, 255], so transform to this range
    noise_vars = noise_vars * 255
    image = image * 255

    # "Compression"
    # Change to the ycbr colour space, which is more suited for compression due to the human eye's sensitivity to luminance (which is concentrated in  the Y channel)
    image = rgb_to_ycbcr_jpeg(image)
    comp = image_to_patches(image)  # Split the image into 8x8 blocks
    # Apply the dicrete cosine transform to each block, changing it from the spatial domain to the frequency domain
    comp = dct_2d(comp)

    comp = (
        comp + noise_vars
    )  # Add the noise variables to the frequency domain coefficients

    return comp


def jpeg_decode(components, image_shape):
    """
    Transforms from a jpeg compressed image to a normal image.
    """
    comp = idct_2d(components)
    image = patches_to_image(comp, image_shape)
    image = ycbcr_to_rgb_jpeg(image)

    image = torch.clamp(image / 255.0, 0, 1)

    return image


def apply_jpeg(image, quantiz_err_vars):
    """
    Applies JPEG compression to an image, and adds the noise variables to the quantized coefficients.
    """
    jpeg_image = jpeg_encode(image, quantiz_err_vars)
    rgb_image = jpeg_decode(jpeg_image, image.shape)
    return rgb_image


class JPEGAdversary(nn.Module):
    """
    Implements the JPEG attack, which works by transforming an image into the JPEG domain,
    adding noise to the quantization error, and then transforming it back into the RGB domain. Has been
    edited from the original implementation in https://arxiv.org/pdf/1908.08016.pdf.

    Parameters
    ---

    num_steps : int
        The number of steps to take in the optimization process.

    step_size : float
        The step size to take in the optimization process.

    epislon: float
        The maximum amount of noise to add to the quantization error.


    """

    def __init__(self, epsilon, num_steps, step_size):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def forward(self, model, inputs, targets):
        batch_size, num_channels, height, width = inputs.size()

        if height % 8 != 0 or width % 8 != 0:
            raise Exception

        noise_vars = self.epsilon * torch.rand(
            (batch_size, num_channels, height // 8 * width // 8, 8, 8)
        ).to(config.device)
        noise_vars.requires_grad = True

        for _ in range(self.num_steps):
            adv_inputs = apply_jpeg(inputs.detach(), noise_vars)

            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)

            grads = torch.autograd.grad(loss, noise_vars)[0]

            noise_vars = noise_vars + self.step_size * torch.sign(grads)
            noise_vars = torch.clamp(noise_vars, -self.epsilon, self.epsilon)

        return apply_jpeg(inputs.detach(), noise_vars)


class JPEGAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = JPEGAdversary(
            epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return JPEGAttack(model, args)

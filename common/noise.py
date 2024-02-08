import colorednoise as cn
import torch
from torch import Tensor


def sample_gaussian_noise(shape: tuple) -> Tensor:
    return torch.normal(torch.zeros(shape), 1.0)


def sample_pink_noise(shape: tuple) -> Tensor:
    return sample_colored_noise(shape, beta=1.0)


def sample_colored_noise(shape: tuple, beta: float) -> Tensor:
    noise = cn.powerlaw_psd_gaussian(beta, shape)
    return torch.from_numpy(noise)


def sample_ou_noise(shape: tuple) -> Tensor:
    raise NotImplementedError()

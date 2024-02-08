import os
import random
from typing import Callable, Iterable

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import Module


def get_linear_schedule(start: float, end: float, end_fraction: float) -> Callable:
    def linear_schedule(progress: float) -> float:
        if progress > end_fraction:
            return end
        else:
            return start + progress * (end - start) / end_fraction

    return linear_schedule


def polyak_update(
    parameters: Iterable[Tensor], target_parameters: Iterable[Tensor], tau: float
) -> None:
    with torch.no_grad():
        for parameter, target_parameter in zip(parameters, target_parameters):
            target_parameter.data.mul_(1 - tau)
            torch.add(
                target_parameter.data,
                parameter.data,
                alpha=tau,
                out=target_parameter.data,
            )


def hard_update(target: Module, source: Module) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def euclidean_distance(x: ndarray, y: ndarray) -> float:
    return np.sqrt(((x - y) ** 2).sum())


def join_parameters(
    algorithm_config: dict,
    agent_config: dict,
    buffer_config: dict,
    optimizer_config: dict | list[dict],
) -> dict:
    parameters = {**algorithm_config, **agent_config, **buffer_config}

    if isinstance(optimizer_config, dict):
        parameters = parameters | optimizer_config
    elif isinstance(optimizer_config, list):
        optimizer_actor_config = optimizer_config[0]
        optimizer_critics_config = optimizer_config[1]

        for key, value in optimizer_actor_config.items():
            parameters["act_" + key] = value

        for key, value in optimizer_critics_config.items():
            parameters["crit_" + key] = value
    else:
        raise ValueError("Dictionary or list expected")

    return parameters


def apply_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

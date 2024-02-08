from abc import ABC, abstractmethod
from typing import Any

import torch


class Agent(ABC):
    def __init__(
        self,
        agent_config: dict[str, Any],
    ) -> None:
        self.agent_config = agent_config

    @abstractmethod
    def act(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: torch.device) -> None:
        raise NotImplementedError

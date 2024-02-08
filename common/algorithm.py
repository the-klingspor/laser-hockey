from abc import ABC, abstractmethod
from typing import Any

import torch

from common.agent import Agent
from common.replay_buffer import ReplayBuffer
from environment.hockey_env import HockeyEnv


class Algorithm(ABC):
    def __init__(
        self,
        env: HockeyEnv,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        algorithm_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        gamma: float,
    ) -> None:
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.algorithm_config = algorithm_config
        self.optimizer_config = optimizer_config
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.set_device(self.device)

    @abstractmethod
    def train_agent(self, *args, **kwargs) -> None:
        raise NotImplementedError

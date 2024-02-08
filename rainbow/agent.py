import json
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

sys.path.append("..")

from numpy import ndarray

from common.agent import Agent
from common.mlp import MLP
from common.noisy_linear import NoisyLinear
from common.observation import DistHockeyObservation, HockeyObservation
from common.reward import (
    BaselineReward,
    DistributionalReward,
    SimpleReward,
    SparseReward,
)

QNetwork = MLP


class ActionConverter:
    def __init__(self, mode: str) -> None:
        with open(f"actions/conversion_{mode}.json") as f:
            action_list = json.load(f)
        self.index2action = dict(enumerate(action_list))
        self.action2index = {
            tuple(action): index for index, action in self.index2action.items()
        }

    def __call__(self, index: int) -> list:
        return self.index2action[index]


class DuelingQNetwork(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation_function: str,
        use_noisy_linear: bool,
    ) -> None:
        super().__init__()
        self.activation = self.get_activation_function(activation_function)

        # Initialize network components
        self.hidden_layers = self.create_hidden_layers(input_dim, hidden_layers)
        self.value_stream, self.advantage_stream = self.create_streams(
            self.hidden_layers[-2].out_features, output_dim, use_noisy_linear
        )

    @staticmethod
    def get_activation_function(name: str) -> Module:
        if name == "relu":
            return nn.ReLU
        raise NotImplementedError(f"Activation function '{name}' not implemented")

    def create_hidden_layers(
        self, input_dim: int, hidden_dims: list[int]
    ) -> ModuleList:
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, dim), self.activation()])
            last_dim = dim
        return ModuleList(layers)

    def create_streams(
        self, input_dim: int, output_dim: int, use_noisy_linear: bool
    ) -> tuple[nn.Sequential, nn.Sequential]:
        linear = NoisyLinear if use_noisy_linear else nn.Linear

        value_stream = nn.Sequential(
            linear(input_dim, input_dim), self.activation(), linear(input_dim, 1)
        )

        advantage_stream = nn.Sequential(
            linear(input_dim, input_dim),
            self.activation(),
            linear(input_dim, output_dim),
        )
        return value_stream, advantage_stream

    def reset_noise(self):
        self.reset_noise_in_stream(self.value_stream)
        self.reset_noise_in_stream(self.advantage_stream)

    def reset_noise_in_stream(self, stream: nn.Sequential):
        for layer in stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            x = layer(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantages into Q-values
        # using Dueling Network architecture formula
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DistributionalDuelingQNetwork(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation_function: str,
        use_noisy_linear: bool,
        v_min: float,
        v_max: float,
        atom_size: int,
    ) -> None:
        super().__init__()
        self.activation = self.get_activation_function(activation_function)
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.output_dim = output_dim

        # Initialize network components
        self.hidden_layers = self.create_hidden_layers(input_dim, hidden_layers)
        self.value_stream, self.advantage_stream = self.create_streams(
            self.hidden_layers[-2].out_features, output_dim, use_noisy_linear, atom_size
        )
        support = torch.linspace(v_min, v_max, atom_size)
        self.register_buffer("support", support)

    def get_activation_function(self, name: str) -> Module:
        if name == "relu":
            return nn.ReLU
        raise NotImplementedError(f"Activation function '{name}' not implemented")

    def create_hidden_layers(
        self, input_dim: int, hidden_dims: list[int]
    ) -> ModuleList:
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, dim), self.activation()])
            last_dim = dim
        return nn.ModuleList(layers)

    def create_streams(
        self, input_dim: int, output_dim: int, use_noisy_linear: bool, atom_size: int
    ) -> tuple[Sequential, Sequential]:
        linear = NoisyLinear if use_noisy_linear else nn.Linear

        value_stream = nn.Sequential(
            linear(input_dim, input_dim),
            self.activation(),
            linear(input_dim, atom_size),
        )

        advantage_stream = nn.Sequential(
            linear(input_dim, input_dim),
            self.activation(),
            linear(input_dim, output_dim * atom_size),
        )

        return value_stream, advantage_stream

    def reset_noise(self):
        self.reset_noise_in_stream(self.value_stream)
        self.reset_noise_in_stream(self.advantage_stream)

    def reset_noise_in_stream(self, stream: Sequential) -> None:
        for layer in stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        dist = self.distribution(x)
        return (dist * self.support).sum(dim=2)

    def distribution(self, x: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            x = layer(x)

        # Compute value and advantage distributions
        value = self.value_stream(x).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(x).view(-1, self.output_dim, self.atom_size)

        Q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(Q_atoms, dim=-1).clamp(min=1e-3)

        return dist


class RainbowAgent(Agent):
    def __init__(self, agent_config: dict[str, Any]):
        super().__init__(agent_config)

        self.initialize_defaults(agent_config)
        self.initialize_networks(agent_config)
        self.initialize_rewards_observations(agent_config)

        self.action_converter = ActionConverter(mode=agent_config["action_mode"])

        self.last_observations = [
            torch.zeros((agent_config["dim_states"],))
            for _ in range(agent_config["num_stacked_observations"])
        ]
        self.set_device("cpu")

    @staticmethod
    def initialize_defaults(agent_config):
        if "num_stacked_observations" not in agent_config:
            agent_config["num_stacked_observations"] = 1

    def initialize_networks(self, agent_config):
        network, kwargs = self.determine_network_type(agent_config)
        self.Q = network(**kwargs)
        self.Q_target = network(**kwargs)
        self.hard_update_target()

    def determine_network_type(self, agent_config):
        kwargs = {
            "input_dim": agent_config["num_stacked_observations"]
            * agent_config["dim_states"],
            "hidden_layers": agent_config["hidden_layers"],
            "output_dim": agent_config["num_actions"],
            "activation_function": agent_config["activation_function"],
            "use_noisy_linear": agent_config["use_noisy_linear"],
        }

        if agent_config["is_dueling"] and not agent_config["is_distributional"]:
            return DuelingQNetwork, kwargs
        elif agent_config["is_dueling"]:
            kwargs.update(
                {
                    "v_min": agent_config["v_min"],
                    "v_max": agent_config["v_max"],
                    "atom_size": agent_config["atom_size"],
                }
            )
            return DistributionalDuelingQNetwork, kwargs
        else:
            return QNetwork, kwargs

    def initialize_rewards_observations(self, agent_config):
        reward_strategies = {
            "sparse": SparseReward,
            "simple": SimpleReward,
            "baseline": BaselineReward,
            "distributional": DistributionalReward,
        }
        observation_strategies = {
            "hockey": HockeyObservation,
            "dist": DistHockeyObservation,
        }

        self.reward = reward_strategies.get(
            agent_config["reward"], self.not_implemented
        )
        self.observation = observation_strategies.get(
            agent_config["observation"], self.not_implemented
        )

    def not_implemented(self):
        raise NotImplementedError("Type not supported")

    def act(
        self, observation: ndarray, epsilon: float = 0, return_int: bool = False
    ) -> tuple[list, int] | list:
        observation = self.observation.augment(observation)
        observation_t = torch.from_numpy(observation.astype(np.float32))

        self.last_observations.pop(0)
        self.last_observations.append(observation_t)
        observation_t = torch.cat(self.last_observations, dim=0).to(self.device)

        if torch.rand(size=()) > epsilon:  # Greedy action
            with torch.no_grad():
                prediction = self.Q(observation_t)
            action = prediction.argmax(dim=-1).detach().cpu().item()
        else:  # Uniform random action
            action = torch.randint(
                high=self.agent_config["num_actions"], size=()
            ).item()
        vec_action = self.action_converter(action)
        return (vec_action, action) if return_int else vec_action

    def train(self) -> None:
        self.Q.train()
        self.Q_target.train()

    def eval(self) -> None:
        self.Q.eval()
        self.Q_target.eval()

    def save(self, path: str) -> None:
        torch.save(self.Q.state_dict(), path)

    def load(self, path: str) -> None:
        self.Q.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        self.hard_update_target()

    def set_device(self, device: torch.device | str) -> None:
        self.device = device
        self.Q.to(device)
        self.Q_target.to(device)

    def hard_update_target(self) -> None:
        self.Q_target.load_state_dict(self.Q.state_dict())

    def reset_observations(self) -> None:
        self.last_observations = [
            torch.zeros((self.agent_config["dim_states"],))
            for _ in range(self.agent_config["num_stacked_observations"])
        ]

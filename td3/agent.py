import sys
from typing import Any

import numpy as np
import torch

sys.path.append("..")

from common.agent import Agent
from common.mlp import MLP
from common.observation import DistHockeyObservation, HockeyObservation
from common.reward import SimpleReward, SparseReward

QNetwork = MLP
Actor = MLP


class TD3Agent(Agent):
    def __init__(self, agent_config: dict[str, Any]) -> None:
        super().__init__(agent_config)

        if "num_stacked_observations" not in agent_config:
            agent_config["num_stacked_observations"] = 1

        # Initialize training networks
        self.actor = Actor(
            input_dim=agent_config["num_stacked_observations"]
            * agent_config["dim_states"],
            hidden_layers=agent_config["actor_hidden_layers"],
            output_dim=agent_config["num_actions"],
            activation_function=agent_config["activation_function"],
            squash=True,
            use_noisy_linear=False,
        )

        self.critic_1 = QNetwork(
            input_dim=agent_config["num_stacked_observations"]
            * agent_config["dim_states"]
            + agent_config["num_actions"],
            hidden_layers=agent_config["critic_hidden_layers"],
            output_dim=1,
            activation_function=agent_config["activation_function"],
            squash=False,
            use_noisy_linear=False,
        )

        self.critic_2 = QNetwork(
            input_dim=agent_config["num_stacked_observations"]
            * agent_config["dim_states"]
            + agent_config["num_actions"],
            hidden_layers=agent_config["critic_hidden_layers"],
            output_dim=1,
            activation_function=agent_config["activation_function"],
            squash=False,
            use_noisy_linear=False,
        )

        # Initialize target networks
        self.actor_target = Actor(
            input_dim=agent_config["num_stacked_observations"]
            * agent_config["dim_states"],
            hidden_layers=agent_config["actor_hidden_layers"],
            output_dim=agent_config["num_actions"],
            activation_function=agent_config["activation_function"],
            squash=True,
            use_noisy_linear=False,
        )

        self.critic_target_1 = QNetwork(
            input_dim=agent_config["num_stacked_observations"]
            * agent_config["dim_states"]
            + agent_config["num_actions"],
            hidden_layers=agent_config["critic_hidden_layers"],
            output_dim=1,
            activation_function=agent_config["activation_function"],
            squash=False,
            use_noisy_linear=False,
        )

        self.critic_target_2 = QNetwork(
            input_dim=agent_config["num_stacked_observations"]
            * agent_config["dim_states"]
            + agent_config["num_actions"],
            hidden_layers=agent_config["critic_hidden_layers"],
            output_dim=1,
            activation_function=agent_config["activation_function"],
            squash=False,
            use_noisy_linear=False,
        )
        self.num_actions = agent_config["num_actions"]

        # Copy parameters to the target networks
        self.hard_target_update()

        device = agent_config["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise Exception("CUDA is not available, but was required")

        self.set_device(device)

        reward_str = agent_config["reward"]

        if reward_str == "sparse":
            self.reward = SparseReward
        elif reward_str == "simple":
            self.reward = SimpleReward
        else:
            raise NotImplementedError

        observation_str = agent_config["observation"]

        if observation_str == "hockey":
            self.observation = HockeyObservation
        elif observation_str == "dist":
            self.observation = DistHockeyObservation
        else:
            raise NotImplementedError

        self.last_observations = [
            torch.zeros((agent_config["dim_states"],))
            for _ in range(agent_config["num_stacked_observations"])
        ]

    def act(self, observation: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
        # Augment input data
        observation_augmented = self.observation.augment(observation)
        observation_t = torch.from_numpy(observation_augmented.astype(np.float32))

        # Handle stacked observations
        self.last_observations.pop(0)
        self.last_observations.append(observation_t)
        observation_t = torch.cat(self.last_observations, dim=0).to(self.device)

        # Perform action
        with torch.no_grad():
            action = self.actor(observation_t)

        action = action.detach().cpu()

        if noise is None:
            noise = np.zeros_like(action)

        # Prepare final output
        action += noise
        action = action.clamp(-1, 1)
        action = action.squeeze().numpy()

        return action

    def train(self) -> None:
        self.actor.train()
        self.actor_target.train()
        self.critic_1.train()
        self.critic_target_1.train()
        self.critic_2.train()
        self.critic_target_2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.actor_target.eval()
        self.critic_1.eval()
        self.critic_target_1.eval()
        self.critic_2.eval()
        self.critic_target_2.eval()

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_1_state_dict": self.critic_1.state_dict(),
                "critic_2_state_dict": self.critic_2.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(
            path, map_location=torch.device(self.agent_config["device"])
        )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])

        self.hard_target_update()

    def hard_target_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

    def set_device(self, device: torch.device | str) -> None:
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic_1.to(device)
        self.critic_target_1.to(device)
        self.critic_2.to(device)
        self.critic_target_2.to(device)

    def reset_observations(self):
        self.last_observations = [
            torch.zeros((self.agent_config["dim_states"],))
            for _ in range(self.agent_config["num_stacked_observations"])
        ]

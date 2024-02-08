import os
import pickle
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from common.observation import Observation
from common.reward import Reward


@dataclass
class ReplayData:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: int,
        action_shape: int,
        action_dtype_str: str,
        buffer_size: int,
    ) -> None:
        if action_dtype_str == "float32":
            action_dtype = torch.float32
        elif action_dtype_str == "uint8":
            action_dtype = torch.uint8
        else:
            raise NotImplementedError

        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.pos = 0
        self.full = False

        self.observations = torch.empty(
            (self.buffer_size, self.observation_shape), dtype=torch.float32
        )

        self.next_observations = torch.empty(
            (self.buffer_size, self.observation_shape), dtype=torch.float32
        )

        if action_shape == 1:  # DQN case
            self.actions = torch.empty((self.buffer_size,), dtype=action_dtype)
        else:
            self.actions = torch.empty(
                (self.buffer_size, self.action_shape), dtype=action_dtype
            )
        self.rewards = torch.empty((self.buffer_size,), dtype=torch.float32)
        self.dones = torch.empty((self.buffer_size), dtype=bool)

    def add(
        self,
        observation: Tensor,
        next_observation: Tensor,
        action: Tensor,
        reward: Tensor,
        done: bool,
    ) -> None:
        self.observations[self.pos] = observation.clone()
        self.next_observations[self.pos] = next_observation.clone()
        self.actions[self.pos] = action.clone()
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get(self, batch_indices: Tensor | list) -> ReplayData:
        # Extract the samples
        observations = self.observations[batch_indices]
        next_observations = self.next_observations[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        dones = self.dones[batch_indices]

        return ReplayData(observations, next_observations, actions, rewards, dones)

    def sample(self, batch_size: int) -> ReplayData:
        max_idx = self.buffer_size if self.full else self.pos
        batch_indices = torch.randint(0, max_idx, size=(batch_size,))

        return self.get(batch_indices)

    def save(self, path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(self.observations, os.path.join(path, "observations.pt"))
        torch.save(self.next_observations, os.path.join(path, "next_observations.pt"))
        torch.save(self.actions, os.path.join(path, "actions.pt"))
        torch.save(self.rewards, os.path.join(path, "rewards.pt"))
        torch.save(self.dones, os.path.join(path, "dones.pt"))

        parameters = {
            "buffer_size": self.buffer_size,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "pos": self.pos,
            "full": self.full,
        }

        with open(os.path.join(path, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)

    def load(self, path: str) -> None:
        self.observations = torch.load(os.path.join(path, "observations.pt"))
        self.next_observations = torch.load(os.path.join(path, "next_observations.pt"))
        self.actions = torch.load(os.path.join(path, "actions.pt"))
        self.rewards = torch.load(os.path.join(path, "rewards.pt"))
        self.dones = torch.load(os.path.join(path, "dones.pt"))

        with open(os.path.join(path, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)

        self.buffer_size = parameters["buffer_size"]
        self.observation_shape = parameters["observation_shape"]
        self.action_shape = parameters["action_shape"]
        self.pos = parameters["pos"]
        self.full = parameters["full"]

    def poll_folders(
        self,
        folders: list[str],
        augmentation: Observation,
        reward_function: Reward,
        action_to_index: Optional[dict] = None,
        num_stacked_observations: int = 1,
    ) -> None:
        last_observations = [
            torch.zeros((self.observation_shape // num_stacked_observations,))
            for _ in range(num_stacked_observations)
        ]

        for folder in folders:
            # Get list of all .npz files in the folder
            npz_files = [
                file
                for file in os.listdir(folder)
                if file.endswith(".npz") and valid(folder, file)
            ]

            random.shuffle(npz_files)

            npz_files = npz_files[:416]

            print(f"Loading from {len(npz_files)} files...")

            # Loop over the .npz files
            for file in npz_files:
                file_path = os.path.join(folder, file)
                with np.load(file_path, allow_pickle=True) as data:
                    data_dict = data["arr_0"].item()
                for (
                    observation,
                    action,
                    next_observation,
                    reward,
                    done,
                    _,
                    info,
                ) in data_dict["transitions"]:
                    observation = torch.from_numpy(
                        augmentation.augment(np.asarray(observation).astype(np.float32))
                    )
                    last_observations.append(observation)
                    last_observations.pop(0)

                    observation = torch.cat(last_observations, dim=0)

                    if action_to_index is not None:
                        action = torch.tensor(
                            action_to_index[tuple(action)], dtype=torch.uint8
                        )
                    else:
                        action = torch.tensor(action, dtype=torch.float32)

                    next_observation = torch.from_numpy(
                        augmentation.augment(
                            np.asarray(next_observation).astype(np.float32)
                        )
                    )

                    next_observation = torch.cat(
                        last_observations[1:] + [next_observation], dim=0
                    )

                    reward = reward_function.get(reward, info)
                    reward = torch.tensor(reward, dtype=torch.float32)

                    self.add(observation, next_observation, action, reward, done)

                    if done:
                        last_observations = [
                            torch.zeros(
                                (self.observation_shape // num_stacked_observations,)
                            )
                            for _ in range(num_stacked_observations)
                        ]


def valid(folder, file):
    file_path = os.path.join(folder, file)
    with np.load(file_path, allow_pickle=True) as data:
        data_dict = data["arr_0"].item()

    allowed_users = [
        "AlphaPuck:MuZero",
        "Arizona Codeyotes:TD3",
        "The Psychedelic Policy Pioneers:ppo",
        "The Psychedelic Policy Pioneers:td3",
        "AlphaPuck:DecQN",
        "Eigentor:SAC",
        "Eigentor:TD3",
        "HC_Slavoj:TD3",
        "RLcochet:DDPG",
        "HC_Slavoj:SAC",
    ]

    is_allowed_user = False
    for user in allowed_users:
        if user in (data_dict["player_one"], data_dict["player_two"]):
            is_allowed_user = True
            break

    return is_allowed_user

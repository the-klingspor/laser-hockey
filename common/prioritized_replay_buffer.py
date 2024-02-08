import os
import pickle
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(1, "..")

import random

from torch import Tensor

from common.replay_buffer import ReplayBuffer, ReplayData
from common.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_shape: int,
        action_shape: int,
        action_dtype_str: str,
        buffer_size: int,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__(observation_shape, action_shape, action_dtype_str, buffer_size)

        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(
        self,
        observation: Tensor,
        next_observation: Tensor,
        action: Tensor,
        reward: float,
        done: bool,
    ) -> None:
        idx = self.pos
        super().add(observation, next_observation, action, reward, done)
        self.sum_tree[idx] = self.max_priority**self.alpha
        self.min_tree[idx] = self.max_priority**self.alpha

    def sample(self, batch_size: int) -> tuple[ReplayData, Tensor, list]:
        end = self.buffer_size - 1 if self.full else self.pos - 1
        segment_length = self.sum_tree.sum(start=0, end=end) / batch_size

        batch_indices = []
        priorities = []
        for i in range(batch_size):
            prefixsum = (i + random.random()) * segment_length
            idx = self.sum_tree.find_prefixsum_idx(prefixsum=prefixsum)
            batch_indices.append(idx)
            priorities.append(self.sum_tree[idx])

        sum_priorities = self.sum_tree.sum()
        min_priority = self.min_tree.min() / sum_priorities
        max_weight = (min_priority * (end + 1)) ** (-self.beta)

        # Compute importance sampling weights
        weights = [
            (priority / sum_priorities * (end + 1)) ** (-self.beta) / max_weight
            for priority in priorities
        ]
        weights = torch.tensor(weights)

        return (self.get(batch_indices), weights, batch_indices)

    def update_priorities(
        self, batch_indices: Tensor | list, priorities: Tensor | list
    ) -> None:
        for idx, priority in zip(batch_indices, priorities):
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def save(self, path: str) -> None:
        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(self.observations, os.path.join(path, "observations.pt"))
        torch.save(self.next_observations, os.path.join(path, "next_observations.pt"))
        torch.save(self.actions, os.path.join(path, "actions.pt"))
        torch.save(self.rewards, os.path.join(path, "rewards.pt"))
        torch.save(self.dones, os.path.join(path, "dones.pt"))

        with open(os.path.join(path, "sum_tree.pkl"), "wb") as f:
            pickle.dump(self.sum_tree, f)

        with open(os.path.join(path, "min_tree.pkl"), "wb") as f:
            pickle.dump(self.min_tree, f)

        parameters = {
            "buffer_size": self.buffer_size,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "pos": self.pos,
            "full": self.full,
            "alpha": self.alpha,
            "beta": self.beta,
        }

        with open(os.path.join(path, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)

    def load(self, path: str) -> None:
        self.observations = torch.load(os.path.join(path, "observations.pt"))
        self.next_observations = torch.load(os.path.join(path, "next_observations.pt"))
        self.actions = torch.load(os.path.join(path, "actions.pt"))
        self.rewards = torch.load(os.path.join(path, "rewards.pt"))
        self.dones = torch.load(os.path.join(path, "dones.pt"))

        with open(os.path.join(path, "sum_tree.pkl"), "rb") as f:
            self.sum_tree = pickle.load(f)

        with open(os.path.join(path, "min_tree.pkl"), "rb") as f:
            self.min_tree = pickle.load(f)

        with open(os.path.join(path, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)

        self.buffer_size = parameters["buffer_size"]
        self.observation_shape = parameters["observation_shape"]
        self.action_shape = parameters["action_shape"]
        self.pos = parameters["pos"]
        self.full = parameters["full"]
        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]

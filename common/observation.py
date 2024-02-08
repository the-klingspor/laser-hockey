from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from common.utils import euclidean_distance
from environment.hockey_env import SCALE, H, W


class Observation(ABC):
    @staticmethod
    @abstractmethod
    def augment(observation: ndarray) -> ndarray:
        raise NotImplementedError


class HockeyObservation(Observation):
    @staticmethod
    def augment(observation: ndarray) -> ndarray:
        return observation


class DistHockeyObservation(Observation):
    @staticmethod
    def augment(observation: ndarray) -> ndarray:
        dim_states = 27

        observation_augmented = np.empty(dim_states)
        observation_augmented[: observation.shape[0]] = observation

        player_1 = observation[0:2]
        player_2 = observation[6:8]
        puck = observation[12:14]
        goal_1 = np.array([W / 2 - 250 / SCALE, H / 2])
        goal_2 = np.array([W / 2 + 250 / SCALE, H / 2])

        # Augment by adding distances
        observation_augmented[18] = euclidean_distance(player_1, player_2)
        observation_augmented[19] = euclidean_distance(player_1, puck)
        observation_augmented[20] = euclidean_distance(player_2, puck)
        observation_augmented[21] = euclidean_distance(player_1, goal_1)
        observation_augmented[22] = euclidean_distance(player_1, goal_2)
        observation_augmented[23] = euclidean_distance(player_2, goal_1)
        observation_augmented[24] = euclidean_distance(player_2, goal_2)
        observation_augmented[25] = euclidean_distance(puck, goal_1)
        observation_augmented[26] = euclidean_distance(puck, goal_2)

        return observation_augmented

from abc import ABC, abstractmethod


class Reward(ABC):
    @staticmethod
    @abstractmethod
    def get(sparse_reward: float, info: dict) -> float:
        raise NotImplementedError


class SparseReward(Reward):
    @staticmethod
    def get(sparse_reward: float, info: dict) -> float:
        return 10 * info["winner"]


class SimpleReward(Reward):
    @staticmethod
    def get(sparse_reward: float, info: dict) -> float:
        proxy_reward = (
            10 * info["winner"]
            + 0.05 * info["reward_closeness_to_puck"]
            + 1.0 * info["reward_touch_puck"]
            + 3.0 * info["reward_puck_direction"]
        )

        return proxy_reward


class BaselineReward(Reward):
    @staticmethod
    def get(sparse_reward: float, info: dict) -> float:
        proxy_reward = 10 * info["winner"] + info["reward_closeness_to_puck"]

        return proxy_reward


class DistributionalReward(Reward):
    @staticmethod
    def get(sparse_reward: float, info: dict) -> float:
        proxy_reward = (
            10 * info["winner"]
            + 0.0167 * info["reward_closeness_to_puck"]
            + 0.3333 * info["reward_touch_puck"]
            + 1.0 * info["reward_puck_direction"]
        )

        return proxy_reward

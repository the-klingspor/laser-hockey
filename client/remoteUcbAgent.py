import sys
from typing import Any

import numpy as np

sys.path.append("")

from client.backend.client import Client
from client.remoteControllerInterface import RemoteControllerInterface
from common.loading import load_agents


class RemoteUcbAgent(RemoteControllerInterface):
    def __init__(
        self,
        identifier: str,
        agents_config: list[dict[str, Any]],
        exploration_factor: float = 1.0,
        performances: np.array = None,
    ) -> None:
        self.agents = load_agents(agents_config)
        RemoteControllerInterface.__init__(self, identifier=identifier)

        self.current_agent_index = 0
        self.current_agent = self.agents[self.current_agent_index]
        if performances:
            self.avg_performances = performances
        else:
            self.avg_performances = np.zeros(len(self.agents))

        self.play_counts = np.ones(len(self.agents))
        self.num_games = 1
        self.exploration_hyperparameter = exploration_factor

    def remote_act(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        return self.current_agent.act(obs)

    def select_next_agent(self, win: int) -> None:
        # Update play counts and time step
        self.play_counts[self.current_agent_index] += 1
        self.num_games += 1

        # Update UCB exploration term for all agents
        exploration_terms = np.sqrt(
            6
            * self.exploration_hyperparameter
            * np.log(self.num_games)
            / self.play_counts
        )

        # Update average performance and UCB for the last played agent based on the win
        current_agent_avg_performance = (
            (self.play_counts[self.current_agent_index] - 1)
            * self.avg_performances[self.current_agent_index]
            + win
        ) / self.play_counts[self.current_agent_index]

        self.avg_performances[self.current_agent_index] = current_agent_avg_performance

        ucbs = self.avg_performances + exploration_terms

        # Choose the next agent with the highest UCB value
        self.current_agent_index = np.argmax(ucbs)
        self.current_agent = self.agents[self.current_agent_index]


if __name__ == "__main__":
    agents_info = [
        {
            "config_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\agent_config.json",
            "agent_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\best_agent_1",
        },
        {
            "config_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\agent_config.json",
            "agent_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\best_agent_2",
        },
        {
            "config_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\agent_config.json",
            "agent_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\best_agent_3",
        },
        {
            "config_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\agent_config.json",
            "agent_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\best_agent_4",
        },
        {
            "config_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\agent_config.json",
            "agent_path": r"C:\Users\Joschi\Documents\Studium\Vorlesungen SS 23\laser-hockey\td3\checkpoints\td3_simple_reward_256x256_dist_pretrained_long\best_agent_5",
        },
    ]

    exploration_factor = 1.0

    controller = RemoteUcbAgent("some_weak_agent", agents_info, exploration_factor)

    username = "user0"
    password = "1234"

    # Play n (None for an infinite amount) games and quit
    client = Client(
        username=username,  # Testuser
        password=password,
        controller=controller,
        output_path=f"data/{username}",  # rollout buffer with finished games will be saved in here
        interactive=False,
        op="start_queuing",
        num_games=25,
    )

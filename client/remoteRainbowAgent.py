import json
import sys
from typing import Any

import numpy as np

sys.path.append("..")

from client.backend.client import Client
from client.remoteControllerInterface import RemoteControllerInterface
from rainbow.agent import RainbowAgent


class RemoteRainbowAgent(RainbowAgent, RemoteControllerInterface):
    def __init__(self, agent_config: dict[str, Any]) -> None:
        RainbowAgent.__init__(self, agent_config)
        RemoteControllerInterface.__init__(self, identifier="Rainbow")

    def remote_act(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(self.act(obs))


if __name__ == "__main__":
    # Fallback
    # path_agent = "../rainbow/checkpoints/rainbow_tournament_second_09.08.2023 15_26_15/agent_1200000"
    # path_config = "../rainbow/checkpoints/rainbow_tournament_second_09.08.2023 15_26_15/agent_config.json"

    path_agent = "../rainbow/checkpoints/rainbow_tournament_second_09.08.2023 15_26_15/agent_2800000"
    path_config = "../rainbow/checkpoints/rainbow_tournament_second_09.08.2023 15_26_15/agent_config.json"

    with open(path_config) as f:
        agent_config = json.load(f)
    agent_config["device"] = "cpu"

    controller = RemoteRainbowAgent(agent_config=agent_config)
    controller.load(path_agent)

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
        num_games=None,
    )

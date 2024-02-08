import json
import sys
from typing import Any

import numpy as np

sys.path.append("")

from client.backend.client import Client
from client.remoteControllerInterface import RemoteControllerInterface
from td3.agent import TD3Agent


class RemoteTD3Agent(TD3Agent, RemoteControllerInterface):
    def __init__(self, agent_config: dict[str, Any]) -> None:
        TD3Agent.__init__(self, agent_config)
        RemoteControllerInterface.__init__(self, identifier="some_weak_agent")

    def remote_act(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        return self.act(obs)


if __name__ == "__main__":
    """
    path_agent = (
        r"..\td3\checkpoints\test_agent_server\best_agent"
    )
    path_config = (
        r"..\td3\checkpoints\test_agent_server\agent_config.json"
    )
    """
    path_agent = "..\\td3\\checkpoints\\td3_simple_reward_256x256_dist_strong-self_b1mio_s20mio_iter2\\best_agent_5"
    path_config = "..\\td3\\checkpoints\\td3_simple_reward_256x256_dist_strong-self_b1mio_s20mio_iter2\\agent_config.json"

    with open(path_config) as f:
        agent_config = json.load(f)
    agent_config["device"] = "cpu"

    controller = RemoteTD3Agent(agent_config=agent_config)
    controller.load(path_agent)

    # username = "user0"
    # password = "1234"

    username = "Arizona Codeyotes"
    password = "gaang5Co2l"

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

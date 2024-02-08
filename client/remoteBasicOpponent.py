import sys

import numpy as np

sys.path.append("..")

from client.backend.client import Client
from client.remoteControllerInterface import RemoteControllerInterface
from environment.hockey_env import BasicOpponent


class RemoteBasicOpponent(BasicOpponent, RemoteControllerInterface):
    def __init__(self, weak, keep_mode=True):
        BasicOpponent.__init__(self, weak=weak, keep_mode=keep_mode)
        RemoteControllerInterface.__init__(self, identifier="StrongBasicOpponent")

    def remote_act(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        return self.act(obs)


if __name__ == "__main__":
    controller = RemoteBasicOpponent(weak=False)

    username = "user0"

    # Play n (None for an infinite amount) games and quit
    client = Client(
        username=username,  # Testuser
        password="1234",
        controller=controller,
        output_path=f"data/{username}",  # rollout buffer with finished games will be saved in here
        interactive=False,
        op="start_queuing",
        num_games=2,
    )

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(
    #     username="user0",
    #     password="1234",
    #     controller=controller,
    #     output_path="/tmp/ALRL2020/client/user0",
    # )

import json
from typing import Any

from common.agent import Agent
from rainbow.agent import RainbowAgent
from sac.agent import SACAgent
from td3.agent import TD3Agent


def load_agents(agent_infos: list[dict[str, Any]]) -> list[Agent]:
    if agent_infos:
        print("Loading:")
    agents = []

    for i, agent_info in enumerate(agent_infos):
        with open(agent_info["config_path"]) as f:
            print(f"[{i}]: {agent_info['config_path']}")
            agent_config = json.load(f)

        # ToDo: so far only used for evaluation/acting
        agent_config["device"] = "cpu"

        if agent_config["type"] == "td3":
            agent = TD3Agent(agent_config)
        elif agent_config["type"] == "sac":
            agent = SACAgent(agent_config)
        elif agent_config["type"] == "rainbow":
            agent = RainbowAgent(agent_config)
        else:
            raise Exception("Not a type of continuous agent that is supported")

        agent.load(agent_info["agent_path"])
        agents.append(agent)

    return agents

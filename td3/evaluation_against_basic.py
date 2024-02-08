import json
import sys

import numpy as np

sys.path.insert(0, "..")

import environment.hockey_env as h_env
from td3.agent import TD3Agent

if __name__ == "__main__":
    # Settings
    mode = "weak"
    evaluate_every = 100
    num_games = 1000
    render = False

    # Prepare the environment and opponents
    env = h_env.HockeyEnv()

    strong_agent = h_env.BasicOpponent(weak=False)
    weak_agent = h_env.BasicOpponent(weak=False)
    strong_agent.agent_config = {"name": "strong"}
    weak_agent.agent_config = {"name": "weak"}

    # Load our trained agent
    path_agent = ("agent_weights")

    path_config = ("td3_agent_config.json")

    if mode == "weak":
        opponent = weak_agent
    elif mode == "strong":
        opponent = strong_agent
    else:
        raise NotImplementedError

    with open(path_config) as f:
        agent_config = json.load(f)
    agent_config["device"] = "cpu"

    agent = TD3Agent(agent_config)
    agent.load(path_agent)

    results = []
    for i in range(num_games):
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        # Play a game
        for _ in range(env.max_timesteps):
            if render:
                env.render()
            a1 = agent.act(obs)
            a2 = opponent.act(obs_agent2)
            obs, r, d, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            if d:
                break

        # Evaluate outcome
        if info["winner"] == 1:
            results.append(np.array([1, 0, 0]))
        elif info["winner"] == -1:
            results.append(np.array([0, 0, 1]))
        else:
            results.append(np.array([0, 1, 0]))

        # Report current standings
        if (i + 1) % evaluate_every == 0:
            avg_results = np.mean(results, axis=0)
            print(f"After {i+1} games: {avg_results[0]:4.3f}% wins, {avg_results[1]:4.3f}% losses and {avg_results[2]: 4.3f}% draws.")

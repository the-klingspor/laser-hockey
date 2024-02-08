import json
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "..")

import environment.hockey_env as h_env
from common.utils import apply_random_seed
from rainbow.agent import RainbowAgent

if __name__ == "__main__":
    seed = 1
    apply_random_seed(seed)

    # Settings
    mode = "weak"
    evaluate_every = 100
    num_games = 1000
    render = False

    # Prepare the environment and opponents
    env = h_env.HockeyEnv(seed=seed)

    strong_agent = h_env.BasicOpponent(weak=False)
    weak_agent = h_env.BasicOpponent(weak=False)
    strong_agent.agent_config = {"name": "strong"}
    weak_agent.agent_config = {"name": "weak"}

    # Load a trained agent that was only trained against weak and strong opponents
    path_agent = "checkpoints/rainbow_experiment_6_new_08.08.2023 16_56_52/best_agent"

    path_config = (
        "checkpoints/rainbow_experiment_6_new_08.08.2023 16_56_52/agent_config.json"
    )

    if mode == "weak":
        opponent = weak_agent
    elif mode == "strong":
        opponent = strong_agent
    else:
        raise NotImplementedError

    with open(path_config) as f:
        agent_config = json.load(f)
    agent_config["device"] = "cpu"

    agent = RainbowAgent(agent_config)
    agent.load(path_agent)

    results = []
    for i in tqdm(range(num_games)):
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

    avg_results = np.mean(results, axis=0)
    print(
        f"After {num_games} games: {avg_results[0]:4.3f}% wins, {avg_results[1]:4.3f}% "
        f"losses and {avg_results[2]: 4.3f}% draws."
    )

import concurrent.futures
from typing import Dict

import numpy as np

from common.agent import Agent
from environment.hockey_env import HockeyEnv


def evaluate(
    env: HockeyEnv,
    agent,
    opponent,
    n_eval_episodes: int,
    appendix: str,
) -> Dict[str, float]:
    rewards = []
    proxy_rewards = []
    wins = []

    for _ in range(n_eval_episodes):
        total_reward = 0
        total_proxy_reward = 0
        observation, _ = env.reset()

        if hasattr(agent, "reset_observations"):
            agent.reset_observations()

        observation_opponent = env.obs_agent_two()

        for _ in range(env.max_timesteps):
            action_one = agent.act(observation)
            action_two = opponent.act(observation_opponent)

            actions = np.hstack([action_one, action_two])
            observation, reward, done, _, info = env.step(actions)
            info["done"] = done

            observation_opponent = env.obs_agent_two()

            if isinstance(agent, Agent):
                total_reward += reward
                total_proxy_reward += agent.reward.get(reward, info)

            if done:
                break

        win = 1 if env.winner == 1 else 0
        wins.append(win)
        sparse_reward = 10 * env.winner

        rewards.append(sparse_reward)
        proxy_rewards.append(total_proxy_reward)

    # Compute mean reward and win percentage
    mean_total_reward = np.mean(rewards).item()
    mean_total_proxy_reward = np.mean(proxy_rewards).item()
    win_percentage = np.mean(wins).item()

    key_reward = "mean_total_reward"
    key_proxy_reward = "mean_total_proxy_reward"
    key_wins = "win_percentage"

    if appendix:
        key_reward += "_" + appendix
        key_proxy_reward += "_" + appendix
        key_wins += "_" + appendix

    dict = {
        key_reward: mean_total_reward,
        key_proxy_reward: mean_total_proxy_reward,
        key_wins: win_percentage,
    }

    return dict


def evaluate_against_many(
    env: HockeyEnv,
    agent: Agent,
    opponents: list,
    n_eval_episodes: int,
    appendix: str,
) -> Dict[str, float]:
    results = {}
    for opponent in opponents:
        full_appendix = opponent.agent_config["name"]
        if appendix != "":
            full_appendix += "_" + appendix
        results |= evaluate(env, agent, opponent, n_eval_episodes, full_appendix)

    win_percentages = []
    for key, value in results.items():
        if "win_percentage" in key:
            win_percentages.append(value)

    avg_win_percentage = np.mean(win_percentages)
    if appendix:
        results["avg_win_percentage_" + appendix] = avg_win_percentage
    else:
        results["avg_win_percentage"] = avg_win_percentage

    return results


# Helper function for evaluating an opponent
def evaluate_opponent_chunk(args):
    env, agent, opponent, n_eval_episodes, appendix = args
    full_appendix = opponent.agent_config["name"]
    if appendix:
        full_appendix += "_" + appendix
    return evaluate(env, agent, opponent, n_eval_episodes, full_appendix)


def evaluate_against_many_new(
    env: HockeyEnv, agent: Agent, opponents: list, n_eval_episodes: int, appendix: str
) -> Dict[str, float]:
    results = {}

    # Prepare the chunks for evaluation
    chunks = [
        (env, agent, opponent, n_eval_episodes, appendix) for opponent in opponents
    ]

    # Use ProcessPoolExecutor for multiprocessing with chunking
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Evaluate all opponents in parallel
        future_results = executor.map(evaluate_opponent_chunk, chunks)

        # Gather the results
        for _, opponent_results in zip(opponents, future_results):
            results |= opponent_results

    win_percentages = [
        value for key, value in results.items() if "win_percentage" in key
    ]
    avg_win_percentage = np.mean(win_percentages)

    if appendix:
        results["avg_win_percentage_" + appendix] = avg_win_percentage
    else:
        results["avg_win_percentage"] = avg_win_percentage

    return results

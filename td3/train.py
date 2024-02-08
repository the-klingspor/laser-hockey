import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append("..")

from agent import TD3Agent
from algorithm import TD3Algorithm

import common.logger as logger
from common.evaluation import evaluate_against_many
from common.flags import parser
from common.loading import load_agents
from common.prioritized_replay_buffer import PrioritizedReplayBuffer
from common.replay_buffer import ReplayBuffer
from common.utils import apply_random_seed, join_parameters
from environment.hockey_env import HockeyEnv

if __name__ == "__main__":
    # Load all config dicts
    with open(f"td3_agent_config.json") as f:
        agent_config = json.load(f)

    with open(f"td3_algorithm_config.json") as f:
        algorithm_config = json.load(f)

    with open(f"td3_buffer_config.json") as f:
        buffer_config = json.load(f)

    with open(f"td3_optimizer_config.json") as f:
        optimizer_config = json.load(f)

    # Join all parameters for wandb logging
    joint_config = join_parameters(
        algorithm_config, agent_config, buffer_config, optimizer_config
    )
    args = parser.parse_args()
    algorithm_config["log"] = args.wandb_key != "<your_api_key_here>"
    logger.setup_logger(args, joint_config)

    apply_random_seed(args.seed)

    # Create environment and replay buffer
    env = HockeyEnv(seed=args.seed)
    if algorithm_config["is_per"]:
        replay_buffer = PrioritizedReplayBuffer(
            agent_config["num_stacked_observations"]
            * buffer_config["observation_shape"],
            buffer_config["action_shape"],
            buffer_config["action_dtype_str"],
            buffer_config["buffer_size"],
            buffer_config["alpha"],
            buffer_config["beta"],
        )
    else:
        replay_buffer = ReplayBuffer(
            agent_config["num_stacked_observations"]
            * buffer_config["observation_shape"],
            buffer_config["action_shape"],
            buffer_config["action_dtype_str"],
            buffer_config["buffer_size"],
        )

    # Load filled buffer
    if "buffer_path" in algorithm_config:
        replay_buffer.load(algorithm_config["buffer_path"])
        print(f"Buffer full: {replay_buffer.full}")

    # Create agent
    agent = TD3Agent(agent_config)
    if "agent_weights" in algorithm_config:
        print(f"Load agent weights: {algorithm_config['agent_weights']}")
        agent.load(algorithm_config["agent_weights"])

    # Create algorithm
    algorithm = TD3Algorithm(
        env,
        agent,
        replay_buffer,
        algorithm_config,
        optimizer_config,
        algorithm_config["gamma"],
    )

    # Run algorithm
    start_time = datetime.now().strftime("%d.%m.%Y %H_%M_%S")
    path = os.path.join(os.getcwd(), "checkpoints", args.group_name + "_" + start_time)
    os.mkdir(path)

    algorithm.train_agent(algorithm_config["total_timesteps"])

    # Save algorithm setup
    agent_config["name"] = args.group_name + "_" + str(args.seed)
    agent_config["type"] = "td3"
    with open(os.path.join(path, "agent_config.json"), "w") as f:
        json.dump(agent_config, f)

    algorithm_config["seed"] = args.seed
    with open(os.path.join(path, "algorithm_config.json"), "w") as f:
        json.dump(algorithm_config, f)

    with open(os.path.join(path, "buffer_config.json"), "w") as f:
        json.dump(buffer_config, f)

    with open(os.path.join(path, "optimizer_config.json"), "w") as f:
        json.dump(optimizer_config, f)

    algorithm.best_agent.save(os.path.join(path, "best_agent"))
    algorithm.agent.save(os.path.join(path, "final_agent"))
    algorithm.replay_buffer.save(os.path.join(path, "buffer"))
    np.save(os.path.join(path, "wins"), algorithm.wins)
    np.save(os.path.join(path, "rewards"), algorithm.rewards)

    logger.save_directory(path, algorithm_config["log"])
    print(f"Run path: {path}")

    # Final evaluation
    print("Final evaluation:")
    if algorithm_config["evaluate"] == "best":
        eval_agent = algorithm.best_agent
    elif algorithm_config["evaluate"] == "final":
        eval_agent = algorithm.agent

    algorithm.best_agent.eval()

    results = evaluate_against_many(
        algorithm.env,
        eval_agent,
        algorithm.eval_opponents,
        algorithm_config["n_eval_episodes_final"],
        appendix="final",
    )
    logger.log(results, algorithm_config["log"])

    opponents_eval = load_agents(algorithm_config["opponents_eval_final"])
    if len(opponents_eval) > 0:
        print("Evaluate against unseen opponents")
        results = evaluate_against_many(
            algorithm.env,
            eval_agent,
            opponents_eval,
            algorithm_config["n_eval_episodes_final"],
            appendix="final_unseen",
        )
        logger.log(results, algorithm_config["log"])

import json
import os
import sys
from datetime import datetime

import torch

sys.path.append("..")

from agent import RainbowAgent
from algorithm import RainbowAlgorithm

import common.logger as logger
from common.evaluation import evaluate_against_many
from common.flags import parser
from common.loading import load_agents
from common.prioritized_replay_buffer import PrioritizedReplayBuffer
from common.replay_buffer import ReplayBuffer
from common.utils import apply_random_seed, join_parameters
from environment.hockey_env import HockeyEnv

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all config dicts
    with open(f"configs/agent_config_tournament.json") as f:
        agent_config = json.load(f)

    with open(f"configs/algorithm_config_tournament.json") as f:
        algorithm_config = json.load(f)

    with open(f"configs/buffer_config_tournament.json") as f:
        buffer_config = json.load(f)

    with open(f"configs/optimizer_config_tournament.json") as f:
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

    # Load buffer
    if path := algorithm_config["buffer_load_path"]:
        replay_buffer.load(path)
    else:
        raise ValueError("Buffer path must be given for tournament training")

    # Create agent
    agent = RainbowAgent(agent_config)

    if agent_path := algorithm_config["agent_weights"]:
        print(f"Load agent weights: {agent_path}")
        agent.load(agent_path)
    else:
        raise ValueError("Agent weights must be given for tournament training")

    agent.set_device(device)

    # Create algorithm
    algorithm = RainbowAlgorithm(
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

    algorithm.train_agent_tournament(algorithm_config["total_timesteps"], path)

    # Save algorithm setup
    agent_config["name"] = args.group_name + "_" + str(args.seed)
    agent_config["type"] = "rainbow"
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
    logger.save_directory(path, algorithm_config["log"])
    print(f"Run path: {path}")

    # Final evaluation
    algorithm.best_agent.eval()
    print("Final evaluation")
    results = evaluate_against_many(
        algorithm.env,
        algorithm.best_agent,
        algorithm.eval_opponents,
        algorithm.num_eval_episodes_final,
        appendix="final",
    )

    opponents_eval = load_agents(algorithm_config["opponents_eval_final"])
    if opponents_eval:
        print("Evaluate against unseen opponents")
        results |= evaluate_against_many(
            algorithm.env,
            algorithm.best_agent,
            opponents_eval,
            algorithm.num_eval_episodes_final,
            appendix="final",
        )

    logger.log(results, algorithm_config["log"])

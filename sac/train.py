import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from common.utils import *
import common.logger as logger
from sac.agent import SACAgent
from sac.algorithm import SACTrainer
from environment.hockey_env import HockeyEnv
from common.replay_buffer import ReplayBuffer

if __name__ == '__main__':

    dirname = time.strftime(f'%y%m%d_%H%M%S_{random.randint(0, 1e6):06}', time.gmtime(time.time()))
    abs_path = os.path.dirname(os.path.realpath(__file__))

    # Load all config dicts
    with open(f"./sac/sac_agent_config.json") as f:
        config_all = json.load(f)

    args = ArgumentParser().parse_args()
    args.algo = "SAC"
    args.project = "laser-hockey"
    args.save_name = "SAC_init"
    args.name = "SAC_init"
    args.wandb_key = "36904af281741d9d46cf09fb7f2ec083bd844ad8"
    args.group_name = "SAC_init"
    args.run_name = "reward_augment_state_augment_sr"
    args.selfplay = False

    agent_config = config_all["agent_config"]
    buffer_config = config_all["buffer_config"]
    algorithm_config = config_all["algorithm_config"]
    env_config = config_all["env_config"]

    start_time = datetime.now().strftime("%d.%m.%Y %H_%M_%S")
    path = os.path.join(os.getcwd(), "checkpoints", args.group_name + "_" + start_time)
    os.makedirs(path, exist_ok=True)

    args.lr = algorithm_config["lr"]
    args.batch_size = agent_config["batch_size"]

    args.path_aux = os.path.dirname(os.path.realpath(__file__))
    logger.setup_logger(args)

    env = HockeyEnv(verbose=(not env_config["quiet"]))
    
    # merge them together 
    agent_config.update(algorithm_config)
    agent_config.update(buffer_config)
    agent_config["best_agent_save_path"] = path
    
    buffer_observation_shape = 18 if not agent_config["augment_state"] else buffer_config["observation_shape"]

    replay_buffer = ReplayBuffer(
        buffer_observation_shape,
        buffer_config["action_shape"],
        buffer_config["action_dtype_str"],
        buffer_config["buffer_size"])

    agent = SACAgent(agent_config=agent_config)
    
    if agent_config["pretrained_agent_path"]:
        agent.load(
            fpath=os.path.join(agent_config["pretrained_agent_path"])    # "best_agent"
        )
        # agent.buffer.load(os.path.join(agent_config["pretrained_agent_path"], "buffer"))

    algorithm = SACTrainer(logger=logger, env=env, agent=agent, config=agent_config)

    if agent_config["eval_only"]:
        algorithm.evaluate()
    else:
        algorithm.train_agent()

    
    algorithm.agent.save(os.path.join(path, "final_agent"))
    algorithm.agent.buffer.save(os.path.join(path, "buffer"))
    np.save(os.path.join(path, "wins"), algorithm.wins)
    np.save(os.path.join(path, "rewards"), algorithm.rewards)

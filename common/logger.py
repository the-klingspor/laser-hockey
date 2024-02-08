import os
from typing import Any, Dict, Optional

import wandb


def setup_logger(args, parameters: Dict[str, Any] = {}) -> None:
    """
    Initialize the logging with weights and biases.
    """
    if args.wandb_key == "<your_api_key_here>":
        return

    _ = os.system("wandb login {}".format(args.wandb_key))
    os.environ["WANDB_API_KEY"] = args.wandb_key

    save_path = os.path.join(args.path_aux, "checkpoints")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(
        project=args.project,
        group=args.group_name,
        name=args.run_name,
        dir=save_path,
        entity="arizona_codeyotes",
        settings=wandb.Settings(start_method="thread"),
    )

    wandb.config.update(vars(args) | parameters)


def log(dict_to_log, log: bool, step: Optional[int] = None) -> None:
    if log:
        wandb.log(dict_to_log, step=step)


def save_directory(path, log: bool) -> None:
    if log:
        wandb.save(os.path.join(path, "*"))


def save_file(path, log: bool) -> None:
    if log:
        wandb.save(path)


def print_episode_info(
    game_outcome,
    episode_counter,
    step,
    total_reward,
    epsilon=None,
    touched=None,
    opponent=None,
):
    padding = 8 if game_outcome == 0 else 0
    msg_string = "{} {:>4}: Done after {:>3} steps. \tReward: {:<15}".format(
        " " * padding, episode_counter, step + 1, round(total_reward, 4)
    )

    if touched is not None:
        msg_string = "{}Touched: {:<15}".format(msg_string, int(touched))

    if epsilon is not None:
        msg_string = "{}Eps: {:<5}".format(msg_string, round(epsilon, 2))

    if opponent is not None:
        msg_string = "{}\tOpp: {}".format(msg_string, opponent)

    print(msg_string)

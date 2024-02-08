import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Setting and logging
parser.add_argument(
    "--wandb_key", default="<your_api_key_here>", type=str, help="API key for W&B."
)

parser.add_argument(
    "--project",
    default="laser-hockey",
    type=str,
    help="Name of the project - relates to W&B project "
    "names. In --savename default setting part of "
    "the savename.",
)

parser.add_argument(
    "--group_name",
    default="",
    type=str,
    help="Name of the group - relates to W&B group names - all runs "
    "with same setup but different seeds are logged into one "
    "group.",
)

parser.add_argument(
    "--run_name", default="", type=str, help="Name of an individual run"
)  # ToDo

parser.add_argument(
    "--path_aux",
    type=str,
    default=os.getcwd(),
    help="Base path for logging and saving checkpoints",
)

parser.add_argument(
    "--path_pretrained",
    type=str,
    help="Path to a pretrained model to load for continued training",
)

# General training
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for numpy and PyTorch"
)

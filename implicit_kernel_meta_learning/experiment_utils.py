import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .parameters import GUILD_RUNS_DIR


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def retrieve_paths_from_identifier(identifier):
    """Get the expanded paths from an identifier"""
    path = filter(
        lambda x: x.is_dir() and x.name.startswith(identifier), GUILD_RUNS_DIR.iterdir()
    )
    path = list(path)
    return path


class ExperimentLogger:
    def __init__(self, config, save_path):
        """State keeps metrics, config all configuration parameters"""
        self.state = {
            "train": pd.DataFrame(columns=["loss", "accuracy", "t"]),
            "val": pd.DataFrame(columns=["loss", "accuracy", "t"]),
            "best_model_state_dict": None,
            "current_model_state_dict": None,
            "current_opt_state_dict": None,
        }
        self.config = config
        self.save_path = save_path
        # Make save directory if it doesn't exist
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def log_train(self, loss, accuracy, t):
        self.state["train"] = self.state["train"].append(
            pd.DataFrame(data=[[loss, accuracy, t]], columns=["loss", "accuracy", "t"]),
            ignore_index=True,
        )

    def log_val(self, loss, accuracy, t):
        self.state["val"] = self.state["val"].append(
            pd.DataFrame(data=[[loss, accuracy, t]], columns=["loss", "accuracy", "t"]),
            ignore_index=True,
        )

    def log_best_model(self, state_dict):
        self.state["best_model_state_dict"] = state_dict

    def log_current_model(self, state_dict):
        self.state["current_model_state_dict"] = state_dict

    def log_opt(self, state_dict):
        self.state["current_opt_state_dict"] = state_dict

    def dump(self):
        torch.save(self.state, self.save_path / "checkpoint.tar")
        with open(self.save_path / "config.pkl", "wb") as f:
            pickle.dump(self.config, f)

    def load(self):
        self.state = torch.load(self.save_path / "checkpoint.tar")
        with open(self.save_path / "config.pkl", "rb") as f:
            self.config = pickle.load(f)

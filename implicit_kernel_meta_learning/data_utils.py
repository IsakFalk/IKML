import itertools
import logging
import os
import pickle
import random

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from implicit_kernel_meta_learning.parameters import PROCESSED_DATA_DIR


def save_experiment(model_dict, file_name):
    """Save an experiment in the form of the model dictionary"""
    # Create save directory
    experiment_dir = PROCESSED_DATA_DIR / file_name
    experiment_dir.mkdir(parents=False, exist_ok=True)
    # Process data
    save_dict = {}
    for model, data in model_dict.items():
        temp_dict = {}
        # Save the algorithm
        torch.save(data["alg"], experiment_dir / f"{model}.pt")
        temp_dict["alg_path"] = experiment_dir / f"{model}.pt"
        # Dump the data
        # Note that data need to be in the right format first
        # So tensors to numpy arrays and so on
        temp_dict["meta_test"] = data["meta_test"]
        temp_dict["meta_train"] = data["meta_train"]
        save_dict[model] = temp_dict
    with open(experiment_dir / "experiment.dill", "wb") as f:
        dill.dump(save_dict, f)
        logging.info("Saved experiment {} to disk".format(file_name))


def load_experiment(file_name):
    """Load an experiment in the form of the model dictionary"""
    # Experiment directory
    experiment_dir = PROCESSED_DATA_DIR / file_name
    with open(experiment_dir / "experiment.dill", "rb") as f:
        experiment_dict = dill.load(f)

    # Load torch algorithms
    for model, d in experiment_dict.items():
        experiment_dict[model]["alg"] = torch.load(d["alg_path"])
    return experiment_dict


class AirQualityDataLoader:
    def __init__(self, k_support, k_query, split="train", forecast=False):
        self.data_dir = PROCESSED_DATA_DIR / "beijing_air_quality"
        self.k_support = k_support
        self.k_query = k_query
        self.k = k_query + k_support
        self.split = split
        self.forecast = forecast
        self.stations = [
            "aotizhongxin",
            "dingling",
            "guanyuan",
            "gucheng",
            "tiantan",
            "shunyi",
            "changping",
            "nongzhanguan",
            "wanliu",
            "huairou",
            "wanshouxigong",
            "dongsi",
        ]
        self._load_split()
        self.output_col = "PM2.5"
        self.feature_cols = [
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "t",
        ]

    def _load_split(self):
        with open(self.data_dir / f"{self.split}.pkl", "rb") as f:
            self.data_dict = pickle.load(f)

    def sample(self, station=None):
        # Sample station uniformly if not given
        if station is None:
            station = random.choice(self.stations)

        df = self.data_dict[station]
        end = len(df) - self.k
        start = np.random.randint(0, end)
        task_df = df.iloc[start : start + self.k, :].copy()
        # Instead of using time, we just append t from 1 to self.k
        # this means that we still have a sense of temporal direction
        task_df["t"] = np.arange(self.k)
        task_data = task_df.loc[:, self.feature_cols].to_numpy()
        task_output = task_df.loc[:, self.output_col].to_numpy()

        bool_train_mask = np.ones(self.k, dtype=bool)
        bool_train_mask[self.k_support :] = False
        if not self.forecast:
            bool_train_mask = bool_train_mask[np.random.permutation(self.k)]
        bool_val_mask = ~bool_train_mask

        X_tr = task_data[bool_train_mask]
        y_tr = task_output[bool_train_mask]

        X_val = task_data[bool_val_mask]
        y_val = task_output[bool_val_mask]

        task = {
            "train": (torch.tensor(X_tr), torch.tensor(y_tr).reshape(-1, 1)),
            "valid": (torch.tensor(X_val), torch.tensor(y_val).reshape(-1, 1)),
            "full": task_df,
        }
        return task


class GasSensorDataLoader:
    def __init__(
        self, k_support, k_query, split="train", lag=3, forecast=False, t=True
    ):
        self.data_dir = PROCESSED_DATA_DIR / "gas_sensor"
        self.k_support = k_support
        self.k_query = k_query
        self.k = k_query + k_support
        self.split = split
        self.experiments = ["{}".format(i) for i in range(13)]
        self._load_data()
        self._load_split()
        self.output_col = "r2"
        self.lag = lag
        self.forecast = forecast
        self.t = t
        self.feature_cols = ["r1"] + ["r{}".format(i) for i in range(3, 15)]
        if self.t:
            self.feature_cols += ["t"]

    def _load_data(self):
        with open(self.data_dir / f"df_dict.pkl", "rb") as f:
            self.df_dict = pickle.load(f)

    def _load_split(self):
        with open(self.data_dir / f"{self.split}_dict.pkl", "rb") as f:
            self.id_dict = pickle.load(f)

    def sample(self, exp=None):
        # Sample experiment uniformly if not given
        if exp is None:
            exp = random.choice(self.experiments)

        # Sample start and end point
        num_intervals = len(self.id_dict[exp])
        task_size = 0
        while task_size <= self.k:
            id_sample = np.random.randint(0, num_intervals)
            start, end = self.id_dict[exp].iloc[id_sample, :].values.squeeze().tolist()
            task_df = self.df_dict[exp].loc[start:end, :].copy()

            task_df[self.output_col] = task_df[self.output_col].shift(-self.lag)
            if self.t:
                task_df["t"] = np.arange(len(task_df))
            task_df = task_df.dropna(how="any")
            task_df = task_df[self.feature_cols + [self.output_col]]
            task_size = len(task_df)

        task_data = task_df.loc[:, self.feature_cols].to_numpy()
        task_output = task_df.loc[:, self.output_col].to_numpy()
        if not self.forecast:
            perm_idx = np.random.permutation(len(task_data))[: self.k]

            X_tr = task_data[perm_idx[: self.k_support]]
            y_tr = task_output[perm_idx[: self.k_support]]
            assert X_tr.shape[0] == self.k_support, "X_tr.shape: {}, k_tr: {}".format(
                X_tr.shape, self.k_support
            )
            assert y_tr.shape[0] == self.k_support, "y_tr.shape: {}, k_tr: {}".format(
                y_tr.shape, self.k_support
            )

            X_val = task_data[perm_idx[self.k_support : self.k]]
            y_val = task_output[perm_idx[self.k_support : self.k]]
            assert X_val.shape[0] == self.k_query, "X_val.shape: {}, k_val: {}".format(
                X_val.shape, self.k_query
            )
            assert y_val.shape[0] == self.k_query, "y_val.shape: {}, k_val: {}".format(
                y_val.shape, self.k_query
            )
        else:
            X_tr = task_data[: self.k_support]
            y_tr = task_output[: self.k_support]
            assert X_tr.shape[0] == self.k_support, "X_tr.shape: {}, k_tr: {}".format(
                X_tr.shape, self.k_support
            )
            assert y_tr.shape[0] == self.k_support, "y_tr.shape: {}, k_tr: {}".format(
                y_tr.shape, self.k_support
            )

            X_val = task_data[self.k_support : self.k]
            y_val = task_output[self.k_support : self.k]
            assert X_val.shape[0] == self.k_query, "X_val.shape: {}, k_val: {}".format(
                X_val.shape, self.k_query
            )
            assert y_val.shape[0] == self.k_query, "y_val.shape: {}, k_val: {}".format(
                y_val.shape, self.k_query
            )

        task = {
            "train": (torch.tensor(X_tr), torch.tensor(y_tr).reshape(-1, 1)),
            "valid": (torch.tensor(X_val), torch.tensor(y_val).reshape(-1, 1)),
            "full": task_df,
        }
        return task

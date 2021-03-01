import datetime
import pickle as pkl
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from implicit_kernel_meta_learning.data_utils import SpatialDataLoader
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.parameters import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

data_dir = RAW_DATA_DIR / "beijing_air_quality" / "PRSA_Data_20130301-20170228"
processed_dir = PROCESSED_DATA_DIR / "beijing_air_quality"


def load_and_clean_data():
    # Read in the csvs for each station
    # We remove any categorical
    def dt_parse(date_string):
        dt = datetime.datetime.strptime(date_string, "%Y %m %d %H")
        return dt

    # Read and massage the data.
    # We first format the index to be based on the datetime.
    # Then we drop columns which are not continuous or
    # are not part of the problem.
    # Finally we remove any rows with missing values.
    df_dict = {}
    for f in data_dir.glob("*.csv"):
        df = (
            pd.read_csv(f, parse_dates={"datetime": [1, 2, 3, 4]}, date_parser=dt_parse)
            .set_index("datetime")
            .drop(["PM10", "No", "wd", "WSPM", "station"], axis=1)
            .dropna(how="any")
        )
        df_dict[f.name.split("_")[2].lower()] = df

    return df_dict


df_dict = load_and_clean_data()

### Train / val / test splits
# Split on the temporal index
# We split into 0.64, 0.16, 0.2
# according to usual few-shot learning splits
def split_data():
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for name, df in df_dict.items():
        index_length = len(df.index)
        split_points = [int(index_length * 0.64), int(index_length * (0.64 + 0.16))]
        train_df = df.iloc[: split_points[0]]
        val_df = df.iloc[split_points[0] + 1 : split_points[1]]
        test_df = df.iloc[split_points[1] + 1 :]
        train_dict[name] = train_df
        val_dict[name] = val_df
        test_dict[name] = test_df

    return train_dict, val_dict, test_dict


train_dict, val_dict, test_dict = split_data()

# Sum up number of experiments
train_n = 0
for val in train_dict.values():
    train_n += len(val)
print("train size: {}".format(train_n))

val_n = 0
for val in val_dict.values():
    val_n += len(val)
print("val size: {}".format(val_n))

test_n = 0
for val in test_dict.values():
    test_n += len(val)
print("test size: {}".format(test_n))


def dump_data_dicts(data_dir, train_dict, val_dict, test_dict):
    # Dump data
    with open(data_dir / "train.pkl", "wb") as f:
        pkl.dump(train_dict, f)
    with open(data_dir / "valid.pkl", "wb") as f:
        pkl.dump(val_dict, f)
    with open(data_dir / "test.pkl", "wb") as f:
        pkl.dump(test_dict, f)


dump_data_dicts(processed_dir, train_dict, val_dict, test_dict)

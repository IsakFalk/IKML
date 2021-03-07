import pickle as pkl

import pandas as pd
from implicit_kernel_meta_learning.parameters import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

# Create data directories
data_dir = RAW_DATA_DIR / "gas_sensor"
data_dir.mkdir(exist_ok=True)

processed_dir = PROCESSED_DATA_DIR / "gas_sensor"
processed_dir.mkdir(exist_ok=True)

fig_dir = FIGURES_DIR / "gas_sensor"
fig_dir.mkdir(exist_ok=True)


# Helpers to load and process data
def clean_df(df):
    cols = df.columns
    cols = [col.split("(")[0].replace(" ", "_")[:-1].lower() for col in cols]
    df.columns = cols
    df = df.set_index(df["time"])
    return df


def extract_start_and_end(df):
    df["heater_active"] = (df["heater_voltage"] > 0.3).astype(int)
    df["diff_heater_active"] = df["heater_active"].diff(1)
    df["ts_start"] = (df["diff_heater_active"] == -1).astype(int)
    df["ts_end"] = (df["diff_heater_active"] == 1).astype(int)
    return df


def read_in_all_csvs():
    df_dict = {}
    file_list = list(data_dir.glob("*.csv"))
    file_list = sorted(file_list, key=lambda x: x.name)
    file_dict = {"{}".format(i): fp for (i, fp) in enumerate(file_list)}
    for i, path in file_dict.items():
        df = pd.read_csv(path)
        df = clean_df(df)
        df_dict["{}".format(i)] = df

    return df_dict


def extract_id_dfs(df_dict):
    id_dict = {}
    for exp_no, df in df_dict.items():
        event_df = extract_start_and_end(df).loc[:, ["ts_start", "ts_end"]]
        event_df["event"] = event_df["ts_start"] + event_df["ts_end"]
        event_df = event_df.query("event == 1.0")
        event_df = event_df.drop("event", axis=1)
        event_df = event_df.reset_index()
        # Need to start at ts_start, not ts_end
        for index, row in event_df.iterrows():
            if row["ts_start"] == 1.0:
                start_idx = index
                break
        # Also stop at the correct point
        for index, row in event_df.sort_index(axis=0, ascending=False).iterrows():
            if row["ts_end"] == 1.0:
                end_idx = index
                break
        # Fix event df so that we start on a start event
        # and end on an end event
        event_df = event_df.loc[start_idx:end_idx, :]
        start_df = event_df.query("ts_start == 1").filter(items=["time"])
        start_df.columns = ["start_time"]
        start_df = start_df.reset_index().drop("index", axis=1)
        end_df = event_df.query("ts_end == 1").filter(items=["time"])
        end_df.columns = ["end_time"]
        end_df = end_df.reset_index().drop("index", axis=1)
        task_df = start_df.merge(end_df, left_index=True, right_index=True)
        id_dict[exp_no] = task_df
    return id_dict


split_pct = [0.64, (0.64 + 0.12)]


def create_split_dict(id_dict):
    """Split each ts according to split_pct"""
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for exp_no, df in id_dict.items():
        n = len(df)
        split_tr_idx = int(n * split_pct[0])
        split_valid_idx = int(n * split_pct[1])
        train_dict[exp_no] = df[:split_tr_idx]
        valid_dict[exp_no] = df[split_tr_idx + 1 : split_valid_idx]
        test_dict[exp_no] = df[split_valid_idx + 1 :]
    return train_dict, valid_dict, test_dict


def main():
    # 0. Read all csvs into dfs
    df_dict = read_in_all_csvs()

    # 1. For each csv file, make a df with columns "start", "end"
    # having as entries the timestamps
    id_dict = extract_id_dfs(df_dict)
    for df in id_dict.values():
        assert all(
            df["start_time"] < df["end_time"]
        ), "start_time must be before end_time for each event"

    # 2. For each of the extracted id dataframes
    # Split this up into train, val and test
    train_dict, valid_dict, test_dict = create_split_dict(id_dict)

    print("Number of base tasks")
    # Sum up number of experiments
    train_n = 0
    for val in train_dict.values():
        train_n += len(val)
    print("train: {}".format(train_n))

    val_n = 0
    for val in valid_dict.values():
        val_n += len(val)
    print("val: {}".format(val_n))

    test_n = 0
    for val in test_dict.values():
        test_n += len(val)
    print("test: {}".format(test_n))

    with open(processed_dir / "train_dict.pkl", "wb") as f:
        pkl.dump(train_dict, f)
    with open(processed_dir / "valid_dict.pkl", "wb") as f:
        pkl.dump(valid_dict, f)
    with open(processed_dir / "test_dict.pkl", "wb") as f:
        pkl.dump(test_dict, f)

    with open(processed_dir / "train_dict.pkl", "rb") as f:
        train_ = pkl.load(f)
    with open(processed_dir / "valid_dict.pkl", "rb") as f:
        valid_ = pkl.load(f)
    with open(processed_dir / "test_dict.pkl", "rb") as f:
        test_ = pkl.load(f)

    # Make sure everything is fine
    for i in train_dict.keys():
        pd.testing.assert_frame_equal(train_[i], train_dict[i])
        pd.testing.assert_frame_equal(valid_[i], valid_dict[i])
        pd.testing.assert_frame_equal(test_[i], test_dict[i])

    # Save data dict
    with open(processed_dir / "df_dict.pkl", "wb") as f:
        pkl.dump(df_dict, f)


main()

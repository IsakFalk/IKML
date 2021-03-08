import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from implicit_kernel_meta_learning.parameters import GUILD_RUNS_DIR


def load_ids_from_batch(gid):
    """Load all results from a batch run"""
    complete_dir = glob(str(GUILD_RUNS_DIR / gid) + "*")[0]
    all_dirs = glob(os.path.join(complete_dir, "*"))
    all_dirs = [Path(p).name for p in all_dirs]
    return all_dirs


def load_result(gid):
    complete_dir = glob(str(GUILD_RUNS_DIR / gid) + "*")[0]
    result_path = os.path.join(complete_dir, "result.pkl")
    result = pd.read_pickle(result_path)
    return result


def get_holdout_perf(results, col):
    new_list = []
    for result in results:
        new_list.append(result[col])
    arr = np.sqrt(np.array(new_list))
    return arr.mean(), arr.std()


def main(guild_id):
    ids = load_ids_from_batch(guild_id)

    results = []
    for gid in ids:
        results.append(load_result(gid))

    print("{}: Average RMSE +/- 1 std\n".format(results[0]["name"]))

    if results[0]["name"] == "Gaussian Oracle":
        print(
            "Holdout meta-test error: {:.4f} +/- {:.4f} std".format(
                *get_holdout_perf(results, "holdout_meta_test_error")
            )
        )
    else:
        print(
            "Holdout meta-valid error: {:.4f} +/- {:.4f} std".format(
                *get_holdout_perf(results, "holdout_meta_valid_error")
            )
        )
        print(
            "Holdout meta-test error: {:.4f} +/- {:.4f} std".format(
                *get_holdout_perf(results, "holdout_meta_test_error")
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--guild_id", type=str)
    args = parser.parse_args()
    main(**vars(args))

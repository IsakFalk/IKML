import argparse
import logging
import pickle as pkl
import warnings
from collections import OrderedDict

import learn2learn as l2l
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from implicit_kernel_meta_learning.algorithms import LearnedBiasRidgeRegression
from implicit_kernel_meta_learning.data_utils import GasSensorDataLoader
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.kernels import (
    BochnerKernel,
    GaussianKernel,
    LinearKernel,
)
from implicit_kernel_meta_learning.parameters import FIGURES_DIR, PROCESSED_DATA_DIR
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

warnings.filterwarnings("ignore")


def visualise_run(result):
    t_val = result["meta_val_every"] * np.arange(len(result["meta_valid_error"]))
    t = np.arange(len(result["meta_train_error"]))
    fig, ax = plt.subplots()
    ax.plot(t, result["meta_train_error"], label="Meta train MSE")
    ax.plot(t_val, result["meta_valid_error"], label="Meta val MSE")
    ax.legend()
    ax.set_title(
        "meta-(val, test) holdout MSE: ({:.4f}, {:.4f})".format(
            result["holdout_meta_valid_error"][0], result["holdout_meta_test_error"][0]
        )
    )
    return fig, ax

def fast_adapt_ker(batch, model, loss, device):
    # Unpack data
    X_tr, y_tr = batch["train"]
    X_tr = X_tr.to(device).float()
    y_tr = y_tr.to(device).float()
    X_val, y_val = batch["valid"]
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    # adapt algorithm
    model.fit(X_tr, y_tr)

    # Predict
    y_hat = model.predict(X_val)
    return loss(y_val, y_hat)

def main(
    seed,
    k_support,
    k_query,
    num_iterations,
    meta_batch_size,
    meta_val_batch_size,
    meta_val_every,
    holdout_size,
    lam,
    meta_lr,
):
    result = OrderedDict(
        meta_train_error=[],
        meta_valid_error=[],
        holdout_meta_test_error=[],
        holdout_meta_valid_error=[],
        meta_val_every=meta_val_every,
        num_iterations=num_iterations,
        name="linear",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed, False)

    # Load train/validation/test data
    traindata = GasSensorDataLoader(k_support, k_query, split="train", t=True)
    valdata = GasSensorDataLoader(k_support, k_query, split="valid", t=True)
    testdata = GasSensorDataLoader(k_support, k_query, split="test", t=True)

    # Holdout errors
    valid_batches = [valdata.sample() for _ in range(holdout_size)]
    test_batches = [testdata.sample() for _ in range(holdout_size)]

    # Define model
    in_dim = 14
    model = LearnedBiasRidgeRegression(in_dim, torch.log(torch.tensor(lam))).to(device)
    opt = optim.Adam(model.parameters(), meta_lr)
    loss = nn.MSELoss("mean")

    # Keep best model around
    best_val_iteration = 0
    best_val_mse = np.inf
    current_val_mse = 0.0

    for iteration in range(num_iterations):
        validate = True if iteration % meta_val_every == 0 else False

        train_batches = [traindata.sample() for _ in range(meta_batch_size)]
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        for train_batch in train_batches:
            evaluation_error = fast_adapt_ker(
                batch=train_batch, model=model, loss=loss, device=device,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
        if validate:
            val_batches = [valdata.sample() for _ in range(meta_val_batch_size)]
            for val_batch in val_batches:
                evaluation_error = fast_adapt_ker(
                    batch=val_batch, model=model, loss=loss, device=device,
                )
                meta_valid_error += evaluation_error.item()
            meta_valid_error /= meta_val_batch_size
            result["meta_valid_error"].append(meta_valid_error)
            print("Iteration {}".format(iteration))
            print("meta_valid_error: {}".format(meta_valid_error))
            if meta_valid_error < best_val_mse:
                best_val_iteration = iteration
                best_val_mse = meta_valid_error
                best_state_dict = model.state_dict()

            print("lam {}\n, bias {}".format(torch.exp(model.log_lam).item(), model.bias))

        meta_train_error /= meta_batch_size
        result["meta_train_error"].append(meta_train_error)
        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    # Load best model
    print("best_valid_iteration: {}".format(best_val_iteration))
    print("best_valid_mse: {}".format(best_val_mse))
    model.load_state_dict(best_state_dict)

    meta_valid_error = 0.0
    meta_test_error = 0.0
    for (valid_batch, test_batch) in zip(valid_batches, test_batches):
        evaluation_error = fast_adapt_ker(
            batch=valid_batch, model=model, loss=loss, device=device,
        )
        meta_valid_error += evaluation_error.item()
        evaluation_error = fast_adapt_ker(
            batch=test_batch, model=model, loss=loss, device=device,
        )
        meta_test_error += evaluation_error.item()

    meta_valid_error /= holdout_size
    meta_test_error /= holdout_size
    print("holdout_meta_valid_error: {}".format(meta_valid_error))
    print("holdout_meta_test_error: {}".format(meta_test_error))
    result["holdout_meta_valid_error"].append(meta_valid_error)
    result["holdout_meta_test_error"].append(meta_test_error)

    with open("result.pkl", "wb") as f:
        pkl.dump(result, f)

    # Visualise
    fig, ax = visualise_run(result)
    plt.tight_layout()
    fig.savefig("learning_curves.pdf", bbox_inches="tight")
    fig.savefig("learning_curves.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_support", type=int, default=10)
    parser.add_argument("--k_query", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument("--meta_val_batch_size", type=int, default=100)
    parser.add_argument("--meta_val_every", type=int, default=100)
    parser.add_argument("--holdout_size", type=int, default=3000)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    args = parser.parse_args()
    main(**vars(args))

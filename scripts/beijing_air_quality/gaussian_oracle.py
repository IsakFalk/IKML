import argparse
import itertools
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
from implicit_kernel_meta_learning.algorithms import RidgeRegression
from implicit_kernel_meta_learning.data_utils import AirQualityDataLoader
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


class RidgeRegression(nn.Module):
    def __init__(self, log_lam, kernel, device=None):
        super(RidgeRegression, self).__init__()
        self.log_lam = nn.Parameter(torch.tensor(log_lam))
        self.kernel = kernel
        self.alphas = None
        self.X_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)

        self.K = self.kernel(X, X)
        K_nl = self.K + torch.exp(self.log_lam) * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.X_tr = X

    def predict(self, X):
        return torch.matmul(self.kernel(X, self.X_tr), self.alphas)


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
    seed, k_support, k_query, holdout_size,
):
    result = OrderedDict(
        meta_train_error=[],
        meta_valid_error=[],
        holdout_meta_test_error=[],
        holdout_meta_valid_error=[],
        name="gaussian_oracle",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed, False)

    # s2s, lambdas
    s2s = np.geomspace(1e1, 1e12, 5)
    lambdas = np.geomspace(1e-6, 1e0, 5)

    # Gaussian Kernel
    loss = nn.MSELoss("mean")

    # Load test data
    testdata = AirQualityDataLoader(k_support, k_query, split="test")
    test_batches = [testdata.sample() for _ in range(holdout_size)]

    optimal_mse = np.inf
    optimal_s2 = None
    optimal_lam = None
    for (s2, lam) in itertools.product(s2s, lambdas):
        kernel = GaussianKernel(torch.log(torch.tensor(s2)))
        model = RidgeRegression(np.log(lam), kernel).to(device)
        meta_test_error = 0.0
        for test_batch in test_batches:
            evaluation_error = fast_adapt_ker(
                batch=test_batch, model=model, loss=loss, device=device,
            )
            meta_test_error += evaluation_error.item()
        meta_test_error /= holdout_size
        if meta_test_error < optimal_mse:
            optimal_mse = meta_test_error
            optimal_s2 = s2
            optimal_lam = lam

    print("oracle_holdout_meta_test_error: {}".format(optimal_mse))
    print("optimal_s2: {}".format(optimal_s2))
    print("optimal_lambda: {}".format(optimal_lam))
    result["holdout_meta_test_error"].append(optimal_mse)

    with open("result.pkl", "wb") as f:
        pkl.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_support", type=int, default=10)
    parser.add_argument("--k_query", type=int, default=10)
    parser.add_argument("--holdout_size", type=int, default=3000)
    args = parser.parse_args()
    main(**vars(args))

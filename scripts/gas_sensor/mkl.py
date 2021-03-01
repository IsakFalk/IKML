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
from implicit_kernel_meta_learning.algorithms import RidgeRegression
from implicit_kernel_meta_learning.data_utils import GasSensorDataLoader
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.kernels import (
    BochnerKernel,
    GaussianKernel,
    LinearKernel,
)
from implicit_kernel_meta_learning.parameters import FIGURES_DIR, PROCESSED_DATA_DIR

warnings.filterwarnings("ignore")
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


class MKLGaussianKernel(nn.Module):
    def __init__(self, s2_list, log_coeff):
        assert len(s2_list) == len(log_coeff)
        super(MKLGaussianKernel, self).__init__()
        self.s2_list = s2_list
        self.log_coeff = nn.ParameterList(
            [nn.Parameter(torch.tensor(c)) for c in log_coeff]
        )

    def apply_gaussian(self, x, y, gauss_idx):
        return torch.exp(-(0.5 / self.s2_list[gauss_idx] * torch.cdist(x, y) ** 2))

    def forward(self, x, y):
        K = 0.0
        s = 0.0
        for i, c in enumerate(self.log_coeff):
            s += torch.exp(c)
            K += torch.exp(c) * self.apply_gaussian(x, y, i)
        K /= s
        return K


def extract_histogram_from_mkl(kernel):
    # We transform s2s to be in the sigma format
    s2s = kernel.s2_list
    sigmas = 0.5 * (1.0 / np.array(s2s))
    log_sigmas = np.log10(sigmas)
    coeffs = [torch.exp(c).cpu().detach().numpy() for c in kernel.log_coeff]
    coeffs = np.array(coeffs)
    coeffs = coeffs / coeffs.sum()
    return pd.DataFrame({"sigmas": sigmas, "log_sigmas": log_sigmas, "coeff": coeffs})


def plot_bar_chart(df, width=0.2):
    label_pos = [x for x in df["log_sigmas"]]
    fig, ax = plt.subplots()
    ax.bar(label_pos, df["coeff"], label="Weight of kernel", width=width)
    ax.set_xlabel("$\log(\sigma^2)$")
    ax.legend()
    return fig, ax


def visualise_kernel(kernel, lim=(-0.5, 0.5), steps=300, device="cuda"):
    zero = torch.zeros(1, 2).to(device)  # Let 0 be the origin
    x = torch.linspace(lim[0], lim[1], steps).to(device)
    y = torch.linspace(lim[0], lim[1], steps).to(device)
    xgrid, ygrid = torch.meshgrid(x, y)

    scatter_grid = torch.stack((xgrid, ygrid), -1)
    scatter_grid = scatter_grid.reshape(-1, 2)
    xflat = scatter_grid[:, 0]
    yflat = scatter_grid[:, 1]
    z = kernel(scatter_grid, zero)
    z = z.reshape(steps, steps)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xgrid.cpu().detach().numpy(),
        ygrid.cpu().detach().numpy(),
        z.cpu().detach().numpy(),
        cmap=cm.inferno,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("$K((x, y), (0, 0))$")
    return fig, ax


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
    seed,
    k_support,
    k_query,
    num_iterations,
    meta_batch_size,
    meta_val_batch_size,
    meta_val_every,
    holdout_size,
    geom_start,
    geom_end,
    geom_steps,
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
        name="mkl",
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

    # Gaussian MKL kernel
    log_coeff = np.log(np.ones(geom_steps) / float(geom_steps))
    s2s = np.geomspace(geom_start, geom_end, num=geom_steps)
    kernel = MKLGaussianKernel(s2s, log_coeff)
    df = extract_histogram_from_mkl(kernel)
    plot_bar_chart(df)
    model = RidgeRegression(np.log(lam), kernel).to(device)
    opt = optim.Adam(model.parameters(), meta_lr)

    df = extract_histogram_from_mkl(model.kernel)
    fig, ax = plot_bar_chart(df)
    plt.tight_layout()
    fig.savefig(
        "mkl_distribution-init.pdf", bbox_inches="tight",
    )
    fig.savefig(
        "mkl_distribution-init.png", bbox_inches="tight",
    )
    df.to_csv("mkl_distribution-init.csv")

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

            df = extract_histogram_from_mkl(model.kernel)
            fig, ax = plot_bar_chart(df)
            plt.tight_layout()
            fig.savefig(
                "mkl_distribution-step{}.pdf".format(iteration), bbox_inches="tight",
            )
            fig.savefig(
                "mkl_distribution-step{}.png".format(iteration), bbox_inches="tight",
            )
            df.to_csv("mkl_distribution-step{}.csv".format(iteration))

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

    print("Final (s2, coeff) for mkl\n")
    for (s2, log_c) in zip(model.kernel.s2_list, model.kernel.log_coeff):
        print(f"s2: {s2:e}, coeff: {torch.exp(log_c).item()}")

    # Visualise
    fig, ax = visualise_run(result)
    plt.tight_layout()
    fig.savefig("learning_curves.pdf", bbox_inches="tight")
    fig.savefig("learning_curves.png", bbox_inches="tight")

    df = extract_histogram_from_mkl(model.kernel)
    fig, ax = plot_bar_chart(df)
    plt.tight_layout()
    fig.savefig(
        "mkl_distribution-final.pdf", bbox_inches="tight",
    )
    fig.savefig(
        "mkl_distribution-final.png", bbox_inches="tight",
    )
    df.to_csv("mkl_distribution-final.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_support", type=int, default=20)
    parser.add_argument("--k_query", type=int, default=20)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument("--meta_val_batch_size", type=int, default=100)
    parser.add_argument("--meta_val_every", type=int, default=100)
    parser.add_argument("--holdout_size", type=int, default=3000)
    parser.add_argument("--geom_start", type=float, default=1e0)
    parser.add_argument("--geom_end", type=float, default=1e8)
    parser.add_argument("--geom_steps", type=int, default=20)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    args = parser.parse_args()
    main(**vars(args))

import argparse
import pickle as pkl
import sys
from copy import deepcopy
from enum import Enum
from pathlib import Path

import learn2learn as l2l
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.decomposition as decomposition
import tikzplotlib as tpl
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.kernels import (
    BochnerKernel,
    CosKernel,
    GaussianKernel,
    LinearKernel,
)
from implicit_kernel_meta_learning.parameters import FIGURES_DIR
from implicit_kernel_meta_learning.utils import infer_dimensions, moving_average
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor
from tqdm import tqdm

sns.set_style()

# Save directory
save_dir = Path(".")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set up color palette
palette = sns.color_palette()


class FeatureMapRidgeRegression(nn.Module):
    """Like RidgeRegression but with an additional feature map phi: X \to Phi

    feature_map is a torch module which is learned together with the rest of the parameters"""

    def __init__(self, log_lam, kernel, feature_map, device=None):
        super(FeatureMapRidgeRegression, self).__init__()
        self.log_lam = nn.Parameter(torch.tensor(log_lam))
        self.kernel = kernel
        self.feature_map = feature_map
        self.alphas = None
        self.Phi_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        n = X.size()[0]

        Phi = self.feature_map(X)

        self.K = self.kernel(Phi, Phi)
        K_nl = self.K + torch.exp(self.log_lam) * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.Phi_tr = Phi

    def predict(self, X):
        return torch.matmul(self.kernel(self.feature_map(X), self.Phi_tr), self.alphas)

    def _kernel(self, X, Y):
        Phi_X = self.feature_map(X)
        Phi_Y = self.feature_map(Y)
        return self.kernel(Phi_X, Phi_Y)


class ExperimentColormap(Enum):
    GAUSS = palette[2]
    BOCHNER = palette[3]
    MKL_GOOD = palette[4]
    LSQ_BIAS = palette[5]
    MAML = palette[7]
    R2D2 = palette[1]
    GAUSS_ORACLE = "black"


def get_test_performance(alg, test_tasks, criterion):
    losses = []
    with torch.no_grad():
        for task in test_tasks:
            X_tr, Y_tr = task["train"]
            X_val, Y_val = task["val"]
            X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            Y_tr = Y_tr.squeeze(0).unsqueeze(-1)
            Y_val = Y_val.squeeze(0).unsqueeze(-1)
            alg.fit(X_tr, Y_tr)
            losses.append(criterion(alg.predict(X_val), Y_val).item())
    return np.array(losses).mean()


def get_test_performance_maml(alg, test_tasks, inner_steps, criterion):
    losses = []
    for task in test_tasks:
        X_tr, Y_tr = task["train"]
        X_val, Y_val = task["val"]
        X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
        X_val, Y_val = X_val.to(device), Y_val.to(device)
        Y_tr = Y_tr.squeeze(0).unsqueeze(-1)
        Y_val = Y_val.squeeze(0).unsqueeze(-1)

        # Make model and predict
        clone = alg.clone()
        for _ in range(inner_steps):
            error = criterion(clone(X_tr), Y_tr)
            clone.adapt(error)
        Y_pred = clone(X_val)
        losses.append(criterion(Y_pred, Y_val).item())
    return np.array(losses).mean()


def train_step(alg, opt, task, criterion):
    X_tr, Y_tr = task["train"]
    X_val, Y_val = task["val"]
    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    Y_tr = Y_tr.squeeze(0).unsqueeze(-1)
    Y_val = Y_val.squeeze(0).unsqueeze(-1)
    opt.zero_grad()
    alg.fit(X_tr, Y_tr)
    Y_pred = alg.predict(X_val)
    loss = criterion(Y_pred, Y_val)
    loss.backward()
    opt.step()


def train_step_maml(alg, opt, task, inner_steps, criterion):
    X_tr, Y_tr = task["train"]
    X_val, Y_val = task["val"]
    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    Y_tr = Y_tr.squeeze(0).unsqueeze(-1)
    Y_val = Y_val.squeeze(0).unsqueeze(-1)
    opt.zero_grad()

    # Make model and predict
    clone = alg.clone()
    for _ in range(inner_steps):
        error = criterion(clone(X_tr), Y_tr)
        clone.adapt(error)
    Y_pred = clone(X_val)
    loss = criterion(Y_val, Y_pred)
    loss.backward()
    opt.step()


def plot_ci(timeseries, t, fig, ax, color, alpha, **kwargs):
    mean = timeseries.mean(0)
    std = timeseries.std(0)
    ax.plot(t, mean, color=color, **kwargs)
    ax.fill_between(t, (mean - std), (mean + std), color=color, alpha=alpha)
    return fig, ax


def plot_kernel(kernel, fig, ax, xmin, xmax, gridsize, device, **kwargs):
    grid = torch.linspace(xmin, xmax, gridsize).reshape(1, -1, 1).to(device)
    zero_point = torch.tensor([[[0.0]]]).to(device)
    grid_kernel = kernel(zero_point, grid)
    ax.plot(
        grid.cpu().detach().numpy().squeeze(),
        grid_kernel.cpu().detach().numpy().squeeze(),
        **kwargs,
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


class KernelRegressionEnvironment:
    def __init__(
        self,
        num_basis_elements,
        d,
        kernel,
        X_bases_sigma=10.0,
        alpha_sigma=0.01,
        X_marginal_sigma=10.0,
        noise=0.1,
        device="cpu",
    ):
        self.num_basis_elements = num_basis_elements
        self.d = d
        # We don't want to training the environment
        for param in kernel.parameters():
            param.requires_grad = False
        self.kernel = kernel
        self.X_bases_sigma = X_bases_sigma
        self.alpha_sigma = alpha_sigma
        self.X_marginal_sigma = X_marginal_sigma
        self.noise = noise
        self.device = device

    def generate_task(self, n_tr, n_val):
        with torch.no_grad():
            X_bases = self.X_bases_sigma * torch.rand(
                self.num_basis_elements, self.d
            ).to(self.device)
            alphas = self.alpha_sigma * torch.randn(self.num_basis_elements).to(
                self.device
            )

            def f_(X):
                return self.kernel(X, X_bases) @ alphas

            X_tr = self.X_marginal_sigma * torch.rand(n_tr, self.d).to(self.device)
            X_val = self.X_marginal_sigma * torch.rand(n_val, self.d).to(self.device)

            Y_tr = f_(X_tr) + self.noise * torch.randn(n_tr).to(self.device)
            Y_val = f_(X_val) + self.noise * torch.randn(n_val).to(self.device)

        return X_tr, X_val, Y_tr, Y_val, f_


def t2np(x):
    return x.detach().cpu().numpy()


def grid_kernel(kernel, v, d):
    # Visualise kernel
    zeros = torch.zeros(1, d).to(device)
    endpoint = 0.4
    num_grid_points = 1000
    c = torch.linspace(-endpoint, endpoint, num_grid_points).unsqueeze(-1).to(device)
    cv = c * v
    ck = kernel(zeros, cv).squeeze()
    return t2np(c), t2np(ck)


def main(
    d,
    runs,
    num_iterations,
    meta_lr,
    num_basis_elements,
    X_bases_sigma,
    alpha_sigma,
    X_marginal_sigma,
    noise,
    k_support,
    k_query,
    save_every,
    num_test,
    geom_steps,
    geom_start,
    geom_end,
    env_latent_d,
    env_hidden_dim,
    boch_latent_d,
    boch_hidden_dim,
    maml_hidden_dim,
    maml_inner_lr,
    maml_num_steps,
    linear_hidden_dim,
    seed,
):
    # Generate meta-learning problem
    D = 10000

    # Run everything
    criterion = nn.MSELoss(reduction="mean")
    # Losses
    mkl_losses = np.zeros((runs, num_iterations // save_every))
    boch_losses = np.zeros((runs, num_iterations // save_every))
    maml_losses = np.zeros((runs, num_iterations // save_every))
    linear_losses = np.zeros((runs, num_iterations // save_every))
    oracle_losses = np.zeros(runs)
    ts = np.arange(0, num_iterations, save_every)

    # Kernel plotting
    boch_env_cks = []
    mkl_cks = []
    boch_cks = []
    linear_cks = []

    # Bochner variables
    kernel_latent_d = boch_latent_d
    kernel_hidden_dim = boch_hidden_dim

    # Other variables
    lam = 0.01

    env_latent_dist = torch.distributions.normal.Normal(0.0, 1.0)
    env_pf_map = nn.Sequential(
        nn.Linear(env_latent_d, env_hidden_dim),
        nn.ReLU(),
        nn.Linear(env_hidden_dim, env_hidden_dim),
        nn.ReLU(),
        nn.Linear(env_hidden_dim, d),
        nn.ReLU(),
    ).to(device)
    boch_env_kernel = BochnerKernel(
        env_latent_d, env_latent_dist, env_pf_map, device=device
    )

    boch_env_kernel.sample_features(D)
    boch_env_kernel.omegas *= 100.0

    # Collect random directions
    v_list = []
    for i in range(0, 5):
        v = torch.randn(1, d).to(device)
        v /= torch.norm(v)
        v_list.append(v)

    ck_list = []
    for i, v in enumerate(v_list):
        fig, ax = plt.subplots()
        c, ck = grid_kernel(boch_env_kernel, v, d)
        ck_list.append(ck)
        ax.plot(c, ck)
        ax.set_xlabel("$c$")
        ax.set_ylabel("$K(c v, 0)$")
        fig.savefig(
            save_dir / "boch_env_kernel-{}.png".format(i), format="png",
        )
        fig.savefig(
            save_dir / "boch_env_kernel-{}.pdf".format(i), format="pdf",
        )
    boch_env_cks.append(np.stack(ck_list))

    # Environment
    # We want the size to stay constant, so we make
    # X_bases_sigma *= 3/d
    env = KernelRegressionEnvironment(
        num_basis_elements=num_basis_elements,
        d=d,
        kernel=boch_env_kernel,
        X_bases_sigma=X_bases_sigma,
        alpha_sigma=alpha_sigma,
        X_marginal_sigma=X_marginal_sigma,
        noise=noise,
        device=device,
    )

    # We run it for `runs` times to get confidence intervals
    for run in range(runs):
        mkl_ck_run = {"init": [], "final": []}
        boch_ck_run = {"init": [], "final": []}
        linear_ck_run = {"init": [], "final": []}

        # Gaussian MKL kernel
        log_coeff = np.log(np.ones(geom_steps) / float(geom_steps))
        s2s = np.geomspace(geom_start, geom_end, num=geom_steps)
        mkl_kernel = MKLGaussianKernel(s2s, log_coeff)
        mkl_model = RidgeRegression(np.log(lam), mkl_kernel).to(device)
        mkl_opt = optim.Adam(mkl_model.parameters(), meta_lr)

        for i, v in enumerate(v_list):
            fig, ax = plt.subplots()
            c, ck = grid_kernel(mkl_kernel, v, d)
            mkl_ck_run["init"].append(ck)
            ax.plot(c, ck)
            ax.set_xlabel("$c$")
            ax.set_ylabel("$K(c v, 0)$")
            fig.savefig(
                save_dir / "mkl_kernel-init-{}-run{}.png".format(i, run), format="png"
            )
            fig.savefig(
                save_dir / "mkl_kernel-init-{}-run{}.pdf".format(i, run), format="pdf"
            )
        mkl_ck_run["init"] = np.stack(mkl_ck_run["init"])

        # Bochner Kernel
        latent_dist = torch.distributions.normal.Normal(0.0, 1.0)
        pf_map = nn.Sequential(
            nn.Linear(kernel_latent_d, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, d),
        ).to(device)
        boch_kernel = BochnerKernel(kernel_latent_d, latent_dist, pf_map).to(device)
        boch_model = RidgeRegression(np.log(lam), boch_kernel).to(device)
        boch_opt = optim.Adam(boch_model.parameters(), meta_lr)

        boch_kernel.sample_features(10000)
        for i, v in enumerate(v_list):
            fig, ax = plt.subplots()
            c, ck = grid_kernel(boch_kernel, v, d)
            boch_ck_run["init"].append(ck)
            ax.plot(c, ck)
            ax.set_xlabel("$c$")
            ax.set_ylabel("$K(c v, 0)$")
            fig.savefig(
                save_dir / "boch_kernel-init-{}-run{}.png".format(i, run), format="png"
            )
            fig.savefig(
                save_dir / "boch_kernel-init-{}-run{}.pdf".format(i, run), format="pdf"
            )
        boch_ck_run["init"] = np.stack(boch_ck_run["init"])

        # maml
        maml_net = nn.Sequential(
            nn.Linear(d, maml_hidden_dim),
            nn.ReLU(),
            nn.Linear(maml_hidden_dim, maml_hidden_dim),
            nn.ReLU(),
            nn.Linear(maml_hidden_dim, 1),
        ).to(device)
        maml_model = l2l.algorithms.MAML(maml_net, lr=maml_inner_lr).to(device)
        maml_opt = optim.Adam(maml_model.parameters(), meta_lr)

        # Linear (Bertinetto)
        linear_fm = nn.Sequential(
            nn.Linear(d, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, 1),
        ).to(device)
        linear_kernel = LinearKernel()
        linear_model = FeatureMapRidgeRegression(
            np.log(lam), linear_kernel, linear_fm
        ).to(device)
        linear_opt = optim.Adam(linear_model.parameters(), meta_lr)

        for i, v in enumerate(v_list):
            fig, ax = plt.subplots()
            c, ck = grid_kernel(linear_model._kernel, v, d)
            linear_ck_run["init"].append(ck)
            ax.plot(c, ck)
            ax.set_xlabel("$c$")
            ax.set_ylabel("$K(c v, 0)$")
            fig.savefig(
                save_dir / "linear_kernel-init-{}-run{}.png".format(i, run),
                format="png",
            )
            fig.savefig(
                save_dir / "linear_kernel-init-{}-run{}.pdf".format(i, run),
                format="pdf",
            )
        linear_ck_run["init"] = np.stack(linear_ck_run["init"])

        # Sample test tasks
        test_tasks = []
        for _ in range(num_test):
            X_tr, X_val, Y_tr, Y_val, _ = env.generate_task(k_support, k_query)
            test_tasks.append({"train": (X_tr, Y_tr), "val": (X_val, Y_val)})

        # Oracle
        # cross validate on test
        oracle_lams = [10 ** i for i in range(-8, 8)]
        best_oracle_loss = np.inf
        for oracle_lam in oracle_lams:
            oracle_model = RidgeRegression(
                torch.tensor(np.log(oracle_lam)), boch_env_kernel
            )
            cur_loss = get_test_performance(oracle_model, test_tasks, criterion)
            if cur_loss < best_oracle_loss:
                best_oracle_lam = oracle_lam
                best_oracle_loss = cur_loss
        oracle_losses[run] = best_oracle_loss

        for iteration in tqdm(range(num_iterations)):
            X_tr, X_val, Y_tr, Y_val, _ = env.generate_task(k_support, k_query)
            task = {"train": (X_tr, Y_tr), "val": (X_val, Y_val)}
            # Train
            train_step(mkl_model, mkl_opt, task, criterion)
            boch_kernel.sample_features(10000)
            train_step(boch_model, boch_opt, task, criterion)
            train_step_maml(maml_model, maml_opt, task, maml_num_steps, criterion)
            train_step(linear_model, linear_opt, task, criterion)

            # Get test performance
            if iteration % save_every == 0:
                print(f"\nIteration: {iteration}, losses")

                mkl_loss = get_test_performance(mkl_model, test_tasks, criterion)
                print(f"MKL: {mkl_loss}")

                boch_loss = get_test_performance(boch_model, test_tasks, criterion)
                print(f"boch: {boch_loss}")

                maml_loss = get_test_performance_maml(
                    maml_model, test_tasks, maml_num_steps, criterion
                )
                print(f"maml: {maml_loss}")

                linear_loss = get_test_performance(linear_model, test_tasks, criterion)
                print(f"linear: {linear_loss}")

                mkl_losses[run, iteration // save_every] = mkl_loss
                boch_losses[run, iteration // save_every] = boch_loss
                maml_losses[run, iteration // save_every] = maml_loss
                linear_losses[run, iteration // save_every] = linear_loss

        for i, v in enumerate(v_list):
            fig, ax = plt.subplots()
            c, ck = grid_kernel(mkl_kernel, v, d)
            mkl_ck_run["final"].append(ck)
            ax.plot(c, ck)
            ax.set_xlabel("$c$")
            ax.set_ylabel("$K(c v, 0)$")
            fig.savefig(
                save_dir / "mkl_kernel-final-{}-run{}.png".format(i, run), format="png"
            )
            fig.savefig(
                save_dir / "mkl_kernel-final-{}-run{}.pdf".format(i, run), format="pdf"
            )
        mkl_ck_run["final"] = np.stack(mkl_ck_run["final"])

        boch_kernel.sample_features(10000)
        for i, v in enumerate(v_list):
            fig, ax = plt.subplots()
            c, ck = grid_kernel(boch_kernel, v, d)
            boch_ck_run["final"].append(ck)
            ax.plot(c, ck)
            ax.set_xlabel("$c$")
            ax.set_ylabel("$K(c v, 0)$")
            fig.savefig(
                save_dir / "boch_kernel-final-{}-run{}.png".format(i, run), format="png"
            )
            fig.savefig(
                save_dir / "boch_kernel-final-{}-run{}.pdf".format(i, run), format="pdf"
            )
        boch_ck_run["final"] = np.stack(boch_ck_run["final"])

        for i, v in enumerate(v_list):
            fig, ax = plt.subplots()
            c, ck = grid_kernel(linear_model._kernel, v, d)
            linear_ck_run["final"].append(ck)
            ax.plot(c, ck)
            ax.set_xlabel("$c$")
            ax.set_ylabel("$K(c v, 0)$")
            fig.savefig(
                save_dir / "linear_kernel-final-{}-run{}.png".format(i, run),
                format="png",
            )
            fig.savefig(
                save_dir / "linear_kernel-final-{}-run{}.pdf".format(i, run),
                format="pdf",
            )
        linear_ck_run["final"] = np.stack(linear_ck_run["final"])

        mkl_cks.append(mkl_ck_run)
        boch_cks.append(boch_ck_run)
        linear_cks.append(linear_ck_run)

    # Save all data
    result = {
        "mkl_losses": mkl_losses,
        "boch_losses": boch_losses,
        "oracle_losses": oracle_losses,
        "linear_losses": linear_losses,
        "maml_losses": maml_losses,
        "boch_env_cks": boch_env_cks,
        "mkl_cks": mkl_cks,
        "boch_cks": boch_cks,
        "linear_cks": linear_cks,
        "c": c,
        "save_every": save_every,
        "dim": d,
    }
    with open("result.pkl", "wb") as f:
        pkl.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--meta_lr", type=float, default=0.01)
    parser.add_argument("--num_basis_elements", type=int, default=1)
    parser.add_argument("--X_bases_sigma", type=float, default=0.2)
    parser.add_argument("--alpha_sigma", type=float, default=1.0)
    parser.add_argument("--X_marginal_sigma", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--k_support", type=int, default=50)
    parser.add_argument("--k_query", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--geom_steps", type=int, default=20)
    parser.add_argument("--geom_start", type=float, default=1e-3)
    parser.add_argument("--geom_end", type=float, default=1e3)
    parser.add_argument("--env_latent_d", type=int, default=16)
    parser.add_argument("--env_hidden_dim", type=int, default=32)
    parser.add_argument("--boch_latent_d", type=int, default=16)
    parser.add_argument("--boch_hidden_dim", type=int, default=64)
    parser.add_argument("--maml_hidden_dim", type=int, default=64)
    parser.add_argument("--maml_inner_lr", type=float, default=0.001)
    parser.add_argument("--maml_num_steps", type=int, default=3)
    parser.add_argument("--linear_hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))

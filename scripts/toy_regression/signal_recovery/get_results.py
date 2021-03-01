import os
import pickle as pkl
from enum import Enum
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from implicit_kernel_meta_learning.parameters import FIGURES_DIR, GUILD_RUNS_DIR

sns.set_theme()

plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 16

# Set up color palette
palette = sns.color_palette()


class ExperimentColormap(Enum):
    GAUSS = palette[2]
    BOCHNER = palette[3]
    MKL_GOOD = palette[4]
    LSQ_BIAS = palette[5]
    MAML = palette[7]
    R2D2 = palette[1]
    GAUSS_ORACLE = "black"


def plot_ci(timeseries, t, fig, ax, color, alpha, **kwargs):
    mean = np.sqrt(timeseries).mean(0)
    std = np.sqrt(timeseries).std(0)
    ax.plot(t, mean, color=color, **kwargs)
    ax.fill_between(t, (mean - std), (mean + std), color=color, alpha=alpha)
    return fig, ax


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


# submission experiments
results_ids = load_ids_from_batch("1687f574")

results = []
for gid in results_ids:
    results.append(load_result(gid))

ts = results[0]["save_every"] * np.arange(results[0]["mkl_losses"].shape[1])

# Gaussian oracle
def prepare_oracle(timeseries):
    errors = []
    for ts in timeseries:
        errors.append(ts)
    errors = np.sqrt(np.array(errors)).squeeze()
    return errors


def plot_oracle(errors, fig, ax, alpha, alpha_fill, color, label):
    mean = errors.mean()
    std = errors.std()
    ax.axhline(mean, alpha=alpha, linestyle="--", color=color, label=label)
    ax.axhspan(mean - std, mean + std, alpha=alpha_fill, color=color)
    return fig, ax


# Plot learning curves
# upper_ylims = [1] * 6
upper_ylims = [0.04, 0.04, 0.5, 0.5, 0.6, 0.4]
alpha = 0.2
for i, result in enumerate(results):
    fig, ax = plt.subplots(figsize=(5, 4))
    # mkl
    plot_ci(
        result["mkl_losses"],
        ts,
        fig,
        ax,
        ExperimentColormap.MKL_GOOD.value,
        alpha,
        label="Gaussian MKL meta-KRR",
    )
    # maml
    plot_ci(
        result["maml_losses"],
        ts,
        fig,
        ax,
        ExperimentColormap.MAML.value,
        alpha,
        label="MAML",
    )
    # linear
    plot_ci(
        result["linear_losses"],
        ts,
        fig,
        ax,
        ExperimentColormap.R2D2.value,
        alpha,
        label="R2D2",
    )
    # bochner
    plot_ci(
        result["boch_losses"],
        ts,
        fig,
        ax,
        ExperimentColormap.BOCHNER.value,
        alpha,
        label="IKML",
    )
    # oracle
    oracle_errors = prepare_oracle(result["oracle_losses"])
    plot_oracle(
        oracle_errors,
        fig,
        ax,
        alpha=1.0,
        alpha_fill=alpha,
        color=ExperimentColormap.GAUSS_ORACLE.value,
        label="Oracle",
    )

    d = result["dim"]
    ax.set_xlabel("Iteration")
    if d == 1:
        ax.set_ylabel("RMSE")
    ax.set_title("$d = {}$".format(d))
    if d == 1:
        upper_ylim = upper_ylims[0]
    elif d == 2:
        upper_ylim = upper_ylims[1]
    elif d == 5:
        upper_ylim = upper_ylims[2]
    elif d == 10:
        upper_ylim = upper_ylims[3]
    elif d == 20:
        upper_ylim = upper_ylims[4]
    elif d == 30:
        upper_ylim = upper_ylims[5]
    ax.set_ylim([-0.05 * upper_ylim, upper_ylim])
    if d == 1:
        ax.legend()
    # if result["dim"] == 1:
    #     handles, labels = ax.get_legend_handles_labels()
    #     fig.legend(
    #         handles, labels, loc="upper right", ncol=1, facecolor="white", framealpha=1,
    #     )
    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "toy_regression"
        / "signal_recovery"
        / "bochner"
        / "learning_curves-d={}.pdf".format(result["dim"]),
        format="pdf",
    )
    fig.savefig(
        FIGURES_DIR
        / "toy_regression"
        / "signal_recovery"
        / "bochner"
        / "learning_curves-d={}.png".format(result["dim"]),
        format="png",
    )

# submission experiments
# One run
results_ids = load_ids_from_batch("48aaacb0")

results = []
for gid in results_ids:
    results.append(load_result(gid))

for result in results:
    c = result["c"].squeeze()

    # Each result is all runs
    boch_env_cks = result["boch_env_cks"][0]  # The bochner kernel stays fixed over runs
    mkl_cks = result["mkl_cks"]
    boch_cks = result["boch_cks"]
    linear_cks = result["linear_cks"]

    # Number of random vectors
    num_rand_vec = 5
    fig, ax = plt.subplots(2, num_rand_vec, figsize=(12, 4), sharex=True, sharey=True)

    # Note: first index is run, second is for random vector (after "init" / "final")
    # fix run to first run
    run = 0
    for i in range(num_rand_vec):
        ## Init
        # Boch env
        ax[0, i].plot(
            c, boch_env_cks[i], color="black", label="True Kernel",
        )

        # mkl
        ax[0, i].plot(
            c,
            mkl_cks[run]["init"][i],
            color=ExperimentColormap.MKL_GOOD.value,
            label="Gaussian MKL meta-KRR",
        )

        # boch
        ax[0, i].plot(
            c,
            boch_cks[run]["init"][i],
            color=ExperimentColormap.BOCHNER.value,
            label="IKML",
        )

        # linear
        # ax[0, i].plot(
        #     c,
        #     linear_cks[run]["init"][i],
        #     color=ExperimentColormap.R2D2.value,
        #     label="R2D2",
        # )

        ## Final
        # Boch env
        ax[1, i].plot(c, boch_env_cks[i], color="black", label="True Kernel")

        # mkl
        ax[1, i].plot(
            c,
            mkl_cks[run]["final"][i],
            color=ExperimentColormap.MKL_GOOD.value,
            label="Gaussian MKL meta-KRR",
        )

        # boch
        ax[1, i].plot(
            c,
            boch_cks[run]["final"][i],
            color=ExperimentColormap.BOCHNER.value,
            label="IKML",
        )

        # linear
        # ax[1, i].plot(
        #     c,
        #     linear_cks[run]["final"][i],
        #     color=ExperimentColormap.R2D2.value,
        #     label="R2D2",
        # )
        #
    ax[1, 2].set_xlabel("$t$")
    ax[0, 0].set_ylabel("$K(0, t \cdot v)$")
    ax[1, 0].set_ylabel("$K(0, t \cdot v)$")

    # Fix legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.1),
        facecolor="white",
        framealpha=1,
    )
    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR
        / "toy_regression"
        / "signal_recovery"
        / "bochner"
        / "kernels-d={}.png".format(result["dim"]),
        format="png",
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR
        / "toy_regression"
        / "signal_recovery"
        / "bochner"
        / "kernels-d={}.pdf".format(result["dim"]),
        format="pdf",
        bbox_inches="tight",
    )

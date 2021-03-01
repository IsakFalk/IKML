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
from implicit_kernel_meta_learning.parameters import FIGURES_DIR, GUILD_RUNS_DIR

sns.set_theme()


plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 16


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


# Set all ids
mkl_good_ids = load_ids_from_batch("76b9d54e")
lsq_bias_ids = load_ids_from_batch("991094a9")
maml_ids = load_ids_from_batch("1fe0bc64")
linear_ids = load_ids_from_batch("b7d06062")
gaussian_ids = load_ids_from_batch("b379fe63")
bochner_ids = load_ids_from_batch("8e98011c")
gaussian_oracle_ids = load_ids_from_batch("822c9430")

# Load all results
mkl_good_results = []
for gid in mkl_good_ids:
    mkl_good_results.append(load_result(gid))

lsq_bias_results = []
for gid in lsq_bias_ids:
    lsq_bias_results.append(load_result(gid))

maml_results = []
for gid in maml_ids:
    maml_results.append(load_result(gid))

linear_results = []
for gid in linear_ids:
    linear_results.append(load_result(gid))

bochner_results = []
for gid in bochner_ids:
    bochner_results.append(load_result(gid))

gaussian_results = []
for gid in gaussian_ids:
    gaussian_results.append(load_result(gid))

gaussian_oracle_results = []
for gid in gaussian_oracle_ids:
    gaussian_oracle_results.append(load_result(gid))

run_cols = ["meta_train_error", "meta_valid_error"]
num_cols = ["holdout_meta_valid_error", "holdout_meta_test_error"]


def npfy_results(result):
    run_cols = ["meta_train_error", "meta_valid_error"]
    num_cols = ["holdout_meta_valid_error", "holdout_meta_test_error"]
    for run_col in run_cols:
        result[run_col] = np.array(result[run_col])
    return result


def make_ts(results, col):
    new_list = []
    for result in results:
        new_list.append(result[col])
    return pd.DataFrame(data=np.stack(new_list))


# Get learning curves
palette = sns.color_palette()
figsize = (8, 6)


class ExperimentColormap(Enum):
    GAUSS = palette[2]
    BOCHNER = palette[3]
    MKL_GOOD = palette[4]
    LSQ_BIAS = palette[5]
    MAML = palette[7]
    R2D2 = palette[1]
    GAUSS_ORACLE = "black"


def plot_ci(timeseries, t, fig, ax, alpha, alpha_fill, **kwargs):
    # RMSE and not MSE
    timeseries = np.sqrt(timeseries)
    mean = timeseries.mean()
    std = timeseries.std()
    ax.plot(t, mean, alpha=alpha, **kwargs)
    ax.fill_between(
        t, (mean - std), (mean + std), alpha=alpha_fill, color=kwargs["color"]
    )
    return fig, ax


fig, ax = plt.subplots(figsize=figsize)
lw = 1.0
alpha = 0.3
alpha_fill = 0.1

# MKL GOOD
meta_train = make_ts(mkl_good_results, "meta_train_error")
meta_val_every = mkl_good_results[0]["meta_val_every"]
t = meta_val_every * np.arange(len(meta_train.T))
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.MKL_GOOD.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="MKL meta-KRR",
)

# LSQ bias
meta_train = make_ts(lsq_bias_results, "meta_train_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.LSQ_BIAS.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="LS Biased Regularization",
)

# MAML
meta_train = make_ts(maml_results, "meta_train_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.MAML.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="MAML",
)

# Gaussian
meta_train = make_ts(gaussian_results, "meta_train_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.GAUSS.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="Gaussian meta-KRR",
)

# Bochner
meta_train = make_ts(bochner_results, "meta_train_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.BOCHNER.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="IKML",
)

# Gaussian oracle
def prepare_oracle(results):
    errors = []
    for result in results:
        errors.append(result["holdout_meta_test_error"])
    errors = np.sqrt(np.array(errors)).squeeze()
    return errors


def plot_oracle(errors, fig, ax, alpha, alpha_fill, color, label):
    mean = errors.mean()
    std = errors.std()
    ax.axhline(mean, alpha=alpha, linestyle="--", color=color, label=label)
    ax.axhspan(mean - std, mean + std, alpha=alpha_fill, color=color)
    return fig, ax


errors = prepare_oracle(gaussian_oracle_results)
fig, ax = plot_oracle(
    errors,
    fig,
    ax,
    alpha,
    alpha_fill,
    color=ExperimentColormap.GAUSS_ORACLE.value,
    label="Gaussian Oracle",
)


leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(4.0)
ax.set_ylim((0.0, 40))
ax.set_ylabel("RMSE")
ax.set_xlabel("Iteration")
fig.savefig(FIGURES_DIR / "gas_sensor" / "learning_curves_train.pdf", format="pdf")
fig.savefig(FIGURES_DIR / "gas_sensor" / "learning_curves_train.png", format="png")

# Get meta-valid/test errors
fig, ax = plt.subplots(figsize=figsize)
lw = 2.0
alpha = 1.0
alpha_fill = 0.2
meta_val_every = mkl_good_results[0]["meta_val_every"]

# MKL GOOD
meta_train = make_ts(mkl_good_results, "meta_valid_error")
meta_val_every = mkl_good_results[0]["meta_val_every"]
t = meta_val_every * np.arange(len(meta_train.T))
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.MKL_GOOD.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="Gaussian MKL meta-KRR",
)

# LSQ bias
meta_train = make_ts(lsq_bias_results, "meta_valid_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.LSQ_BIAS.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="LS Biased Regularization",
)

# MAML
meta_train = make_ts(maml_results, "meta_valid_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.MAML.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="MAML",
)

# MAML
meta_train = make_ts(linear_results, "meta_valid_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.R2D2.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="R2D2",
)


# Gaussian
meta_train = make_ts(gaussian_results, "meta_valid_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.GAUSS.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="Gaussian meta-KRR",
)

# Bochner
meta_train = make_ts(bochner_results, "meta_valid_error")
fig, ax = plot_ci(
    meta_train,
    t,
    fig,
    ax,
    color=ExperimentColormap.BOCHNER.value,
    alpha=alpha,
    alpha_fill=alpha_fill,
    lw=lw,
    label="IKML",
)

errors = prepare_oracle(gaussian_oracle_results)
fig, ax = plot_oracle(
    errors,
    fig,
    ax,
    alpha,
    alpha_fill,
    color=ExperimentColormap.GAUSS_ORACLE.value,
    label="Gaussian Oracle",
)

leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(4.0)
ax.set_ylim((0.0, 40))
ax.set_ylabel("RMSE")
ax.set_xlabel("Iteration")
fig.savefig(FIGURES_DIR / "gas_sensor" / "learning_curves_valid.pdf", format="pdf")
fig.savefig(FIGURES_DIR / "gas_sensor" / "learning_curves_valid.png", format="png")

# Plot distribution of Schoenberg and MKL
# We take the first seed only
def load_csv(gid, name):
    complete_dir = glob(str(GUILD_RUNS_DIR / gid) + "*")[0]
    csv_path = os.path.join(complete_dir, "{}.csv".format(name))
    df = pd.read_csv(csv_path)
    return df


# MKL
def plot_mkl_dist(df, width=0.2):
    label_pos = [x for x in df["log_sigmas"]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(label_pos, df["coeff"], width=width)
    ax.set_xlabel("$-\log(2\sigma^{2})$")
    ax.set_ylabel("Density")
    return fig, ax


sns.set(font_scale=2.0)
# 1. Read in csvs: init and final
mkl_init_dist = load_csv(mkl_good_ids[0], "mkl_distribution-init")
mkl_final_dist = load_csv(mkl_good_ids[0], "mkl_distribution-final")

fig, ax = plot_mkl_dist(mkl_init_dist)
plt.tight_layout()
ax.set_xlim((-8.5, 0))
fig.savefig(FIGURES_DIR / "gas_sensor" / "good_mkl_init_distribution.pdf", format="pdf")
fig.savefig(FIGURES_DIR / "gas_sensor" / "good_mkl_init_distribution.png", format="png")
fig, ax = plot_mkl_dist(mkl_final_dist)
ax.set_xlim((-8.5, 0))
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "gas_sensor" / "good_mkl_final_distribution.pdf", format="pdf",
)
fig.savefig(
    FIGURES_DIR / "gas_sensor" / "good_mkl_final_distribution.png", format="png",
)


# Get mean +- std of test error

col = "holdout_meta_test_error"


def get_holdout_perf(results, col):
    new_list = []
    for result in results:
        new_list.append(result[col])
    arr = np.sqrt(np.array(new_list))
    print("{:.4f} +/- {:.4f} std".format(arr.mean(), arr.std()))


print("Holdout test RMSE")

print("MKL")
get_holdout_perf(mkl_good_results, col)

print("Least squares bias")
get_holdout_perf(lsq_bias_results, col)

print("MAML")
get_holdout_perf(maml_results, col)

print("Linear (R2D2)")
get_holdout_perf(linear_results, col)

print("Gaussian")
get_holdout_perf(gaussian_results, col)

print("Gaussian Oracle")
get_holdout_perf(gaussian_oracle_results, col)

print("Bochner")
get_holdout_perf(bochner_results, col)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from implicit_kernel_meta_learning.data_utils import GasSensorDataLoader
from implicit_kernel_meta_learning.parameters import FIGURES_DIR

fig_dir = FIGURES_DIR / "gas_sensor"
fig_dir.mkdir(exist_ok=True)

k_support = 20
k_query = 20

traindata = GasSensorDataLoader(k_support, k_query, split="train")

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))
ax = ax.ravel()
for axis in ax:
    batch = traindata.sample()
    df = batch["full"].reset_index().drop("time", axis=1)
    output_df = df[["r2"]]
    output_df.columns = ["y"]
    feature_df = df[traindata.feature_cols]
    feature_df.plot(ax=axis)
    axis.get_legend().remove()

    output_df.plot(
        color="black", linestyle="--", ax=axis,
    )

[axis.get_legend().remove() for axis in ax[1:]]

fig.savefig(fig_dir / "tasks_tsplot.pdf")

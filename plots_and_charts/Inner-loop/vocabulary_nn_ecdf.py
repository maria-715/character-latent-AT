#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# Assumes this plotting script is run from the same project directory
# in which embedding_nn_analysis/ was created.
DATA_DIR = Path("../../Continuous-AdvTrain/embedding_nn_analysis")

MODEL_DATA = {
    "Zephyr": {
        "csv": DATA_DIR
        / "zephyr-7b-beta"
        / "per_token_nn_distances.csv",
        "d_max": 0.04422,
    },
    "Llama 3": {
        "csv": DATA_DIR
        / "Meta-Llama-3-8B-Instruct"
        / "per_token_nn_distances.csv",
        "d_max": 0.13500,
    },
}


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# ECDF helper
# ---------------------------------------------------------------------
def ecdf(values):
    """Return x and y coordinates for an empirical CDF."""
    x = np.sort(np.asarray(values))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
loaded = {}

for model_name, info in MODEL_DATA.items():
    csv_path = info["csv"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n  {csv_path}\n\n"
            "Run analyse_embedding_nn.py first or update DATA_DIR."
        )

    df = pd.read_csv(csv_path)

    if "nn_distance" not in df.columns:
        raise ValueError(
            f"{csv_path} does not contain an 'nn_distance' column."
        )

    distances = df["nn_distance"].dropna().to_numpy()

    loaded[model_name] = {
        "distances": distances,
        "d_max": info["d_max"],
    }

    print(
        f"{model_name}: "
        f"n={len(distances):,}, "
        f"median={np.median(distances):.5f}, "
        f"D_max={info['d_max']:.5f}, "
        f"D_max / median={info['d_max'] / np.median(distances):.3f}"
    )


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.5, 3.5),
    sharey=True,
    constrained_layout=True,
)


for ax, (model_name, info) in zip(axes, loaded.items()):
    distances = info["distances"]
    d_max = info["d_max"]

    x, y = ecdf(distances)

    # Full nearest-neighbour distance distribution
    ax.plot(
        x,
        y,
        linewidth=1.7,
        label="Vocabulary nearest-neighbour distance",
    )

    # Largest realised CAT perturbation
    ax.axvline(
        d_max,
        linestyle="--",
        linewidth=1.4,
        color="0.25",
        label=rf"Largest realised $D$ = {d_max:.3f}",
    )

    # Median, useful for connecting the plot to the numerical discussion.
    median = np.median(distances)

    ax.axvline(
        median,
        linestyle=":",
        linewidth=1.2,
        color="0.45",
        label=rf"Median = {median:.3f}",
    )

    ax.set_title(model_name)
    ax.set_xlabel(r"Nearest-neighbour distance")

    ax.set_ylim(0, 1.01)
    ax.set_xlim(left=0)

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        loc="lower right",
    )


axes[0].set_ylabel("Cumulative fraction of vocabulary")


# ---------------------------------------------------------------------
# Save beside this plotting script
# ---------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "vocabulary_nn_ecdf.pdf"
png_path = output_dir / "vocabulary_nn_ecdf.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
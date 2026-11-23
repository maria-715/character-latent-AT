import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D

from or_n1000_data import models


# ============================================================
# OR_cross_config_n1000 (three configurations, stacked), with
# held-out OR-Bench success added as a third metric.
#
# Held-out OR-Bench = success on the OR-Bench prompts not used for
# training ("on 78k"; 80,359 - n prompts).
# ============================================================


# ============================================================
# Typography
# ============================================================

plt.rcParams.update({
    "font.size": 13,          # default text
    "axes.titlesize": 14,     # panel titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 11,    # x tick labels
    "ytick.labelsize": 11,    # y tick labels
    "legend.fontsize": 12,    # legend
})


# ============================================================
# Configuration
# ============================================================

# Baseline error bars are always shown.
# This controls only replacement-condition error bars where
# an SD is actually available.
SHOW_REPLACEMENT_ERROR_BARS = False

METRICS = ["Harmless", "OR held-out", "Soft-Prompt"]

METRIC_LABELS = {
    "Harmless": "Harmless",
    "OR held-out": "Held-out OR-Bench",
    "Soft-Prompt": "Soft-Prompt ASR",
}

# Metric encoding
COLORS = {
    "Harmless": "C0",
    "Soft-Prompt": "C1",
    "OR held-out": "C2",
}

MARKERS = {
    "Harmless": "o",
    "Soft-Prompt": "s",
    "OR held-out": "^",
}

# Condition encoding
REPLACEMENT_LINESTYLE = "-"
BASELINE_LINESTYLE = "--"

MARKER_SIZE = 6
LINE_WIDTH = 1.7
ERROR_LINE_WIDTH = 1.0
ERROR_CAPSIZE = 3

CONFIGURATION_ORDER = [
    "Zephyr",
    "Llama 3 (Llama 2)",
    "Llama 3 (MixAT)",
]


# ============================================================
# Plot one trajectory
# ============================================================

def plot_trajectory(
    ax,
    x,
    y,
    err,
    color,
    marker,
    linestyle,
    filled,
    show_error_bars,
    zorder,
):

    marker_face = color if filled else "white"

    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markerfacecolor=marker_face,
        markeredgecolor=color,
        markersize=MARKER_SIZE,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        zorder=zorder,
    )

    if show_error_bars:

        valid = ~np.isnan(err)

        if np.any(valid):

            ax.errorbar(
                x[valid],
                y[valid],
                yerr=err[valid],
                fmt="none",
                color=color,
                capsize=ERROR_CAPSIZE,
                linewidth=ERROR_LINE_WIDTH,
                zorder=zorder - 1,
            )


# ============================================================
# Plot one configuration
# ============================================================

def plot_configuration(ax, configuration, panel_label):

    data = models[configuration]
    epsilon = data["epsilon"]

    for metric in METRICS:

        # Baseline
        plot_trajectory(
            ax=ax,
            x=epsilon,
            y=data[metric]["baseline"]["y"],
            err=data[metric]["baseline"]["err"],
            color=COLORS[metric],
            marker=MARKERS[metric],
            linestyle=BASELINE_LINESTYLE,
            filled=False,
            show_error_bars=True,
            zorder=2,
        )

        # OR replacement
        plot_trajectory(
            ax=ax,
            x=epsilon,
            y=data[metric]["replacement"]["y"],
            err=data[metric]["replacement"]["err"],
            color=COLORS[metric],
            marker=MARKERS[metric],
            linestyle=REPLACEMENT_LINESTYLE,
            filled=True,
            show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
            zorder=4,
        )

    ax.set_ylim(0, 105)

    ax.set_xticks(epsilon)
    ax.set_xticklabels(data["epsilon_labels"])

    ax.set_ylabel("Score / ASR (%)")

    ax.set_title(rf"(${panel_label}$) {configuration}")

    ax.grid(axis="y", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Three-panel cross-configuration figure
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(7, 11),
)

for ax, configuration, panel_label in zip(
    axes,
    CONFIGURATION_ORDER,
    ["a", "b", "c"],
):
    plot_configuration(ax, configuration, panel_label)

axes[-1].set_xlabel(r"Configured radius $\epsilon$")


# ============================================================
# Shared legend
#
# Marker/colour = metric
# Line/fill     = training condition
# ============================================================

metric_handles = [
    Line2D(
        [0],
        [0],
        color=COLORS[metric],
        marker=MARKERS[metric],
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=METRIC_LABELS[metric],
    )
    for metric in METRICS
]

condition_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=LINE_WIDTH,
        label=r"$1{,}000$ OR examples",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Baseline",
    ),
]

fig.subplots_adjust(
    left=0.13,
    right=0.97,
    top=0.86,
    bottom=0.07,
    hspace=0.55,
)

legend_metrics = fig.legend(
    handles=metric_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
)

fig.legend(
    handles=condition_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=2,
)

# Ensure first legend is retained
fig.add_artist(legend_metrics)


# ============================================================
# Save
# ============================================================

output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "OR_cross_config_n1000_heldout.pdf"
png_path = output_dir / "OR_cross_config_n1000_heldout.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(
    "Saved cross-configuration figure to:\n"
    f"  {pdf_path}\n"
    f"  {png_path}\n\n"
    "Baseline error bars: always shown\n"
    f"Replacement error bars: {SHOW_REPLACEMENT_ERROR_BARS}"
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D


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

SHOW_ERROR_BARS = True

MAIN_COLOR = "C0"
PARTIAL_COLOR = "C1"
BASELINE_COLOR = "0.35"

LINE_WIDTH = 1.7
MARKER_SIZE = 6
ERROR_LINE_WIDTH = 1.0
ERROR_CAPSIZE = 3


# ============================================================
# Raw seed-level data
#
# Llama 3 (MixAT)
# s = 0.15
# n = 500 OR replacement examples
#
# Three main weights:
#   0.00371875, 0.014875, 0.0595
#   -> three seeds
#
# Partial-sweep weights:
#   0.238, 0.952
#   -> one seed
# ============================================================

raw = {

    0.00371875: {
        "Harmless": [77.50, 60.00, 65.00],
        "MMLU": [59.34, 59.59, 59.08],
        "ARC-E": [79.88, 80.30, 80.77],
        "ARC-C": [49.74, 50.17, 50.85],
        "GCG": [32.50, 27.50, 22.50],
        "AutoDAN": [2.50, 0.00, 0.00],
        "XSTest safe": [1.60, 76.40, 45.60],
        "XSTest unsafe": [99.50, 90.50, 99.00],
        "Soft-Prompt": [64.78, 59.37, 63.08],
    },

    0.014875: {
        "Harmless": [55.00, 70.00, 42.50],
        "MMLU": [58.65, 59.64, 59.54],
        "ARC-E": [77.74, 80.39, 80.51],
        "ARC-C": [48.21, 50.60, 50.09],
        "GCG": [20.00, 17.50, 10.00],
        "AutoDAN": [0.00, 0.00, 0.00],
        "XSTest safe": [80.00, 78.80, 59.20],
        "XSTest unsafe": [83.50, 88.00, 97.00],
        "Soft-Prompt": [60.31, 55.35, 66.04],
    },

    0.0595: {
        "Harmless": [57.50, 62.50, 70.00],
        "MMLU": [59.87, 59.54, 58.83],
        "ARC-E": [80.01, 80.01, 79.71],
        "ARC-C": [50.17, 49.66, 49.06],
        "GCG": [40.00, 35.00, 47.50],
        "AutoDAN": [0.00, 0.00, 0.00],
        "XSTest safe": [63.60, 72.00, 57.20],
        "XSTest unsafe": [95.50, 92.50, 97.00],
        "Soft-Prompt": [63.33, 60.57, 66.86],
    },

    # ========================================================
    # Partial sweep: single seed
    # ========================================================

    0.238: {
        "Harmless": [50.00],
        "MMLU": [59.08],
        "ARC-E": [79.08],
        "ARC-C": [49.40],
        "GCG": [42.50],
        "AutoDAN": [0.00],
        "XSTest safe": [60.40],
        "XSTest unsafe": [94.00],
        "Soft-Prompt": [66.86],
    },

    0.952: {
        "Harmless": [20.00],
        "MMLU": [59.60],
        "ARC-E": [79.21],
        "ARC-C": [48.98],
        "GCG": [35.00],
        "AutoDAN": [0.00],
        "XSTest safe": [61.60],
        "XSTest unsafe": [97.50],
        "Soft-Prompt": [71.95],
    },
}


# ============================================================
# No-OR baseline
#
# Ordinary Llama 3 (MixAT) sweep at s = 0.15.
#
# Displayed as horizontal dotted lines, not as alpha_OR = 0.
# ============================================================

baseline = {

    "Harmless": {
        "mean": 58.33,
        "sd": 14.65,
    },

    "MMLU": {
        "mean": 59.47,
        "sd": 0.14,
    },

    "ARC-E": {
        "mean": 79.44,
        "sd": 0.29,
    },

    "ARC-C": {
        "mean": 49.34,
        "sd": 0.65,
    },

    "GCG": {
        "mean": 12.50,
        "sd": 2.50,
    },

    "AutoDAN": {
        "mean": 0.00,
        "sd": 0.00,
    },

    "XSTest safe": {
        "mean": 51.07,
        "sd": 10.51,
    },

    "XSTest unsafe": {
        "mean": 98.50,
        "sd": 0.87,
    },

    "Soft-Prompt": {
        "mean": 43.23,
        "sd": 3.92,
    },
}


# ============================================================
# Weight groups
# ============================================================

MAIN_WEIGHTS = np.array([
    0.00371875,
    0.014875,
    0.0595,
])

PARTIAL_WEIGHTS = np.array([
    0.238,
    0.952,
])

ALL_WEIGHTS = np.concatenate([
    MAIN_WEIGHTS,
    PARTIAL_WEIGHTS,
])

X_TICKS = [
    0.00371875,
    0.014875,
    0.0595,
    0.238,
    0.952,
]

X_TICK_LABELS = [
    "0.00372",
    "0.0149",
    "0.0595",
    "0.238",
    "0.952",
]


# ============================================================
# Statistics
# ============================================================

def summarise(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    mean = np.mean(values)

    if len(values) > 1:

        sd = np.std(
            values,
            ddof=1,
        )

    else:

        sd = np.nan

    return mean, sd


def get_summary(metric, weights):

    means = []
    sds = []

    for weight in weights:

        mean, sd = summarise(
            raw[weight][metric]
        )

        means.append(mean)
        sds.append(sd)

    return (
        np.asarray(means),
        np.asarray(sds),
    )


# ============================================================
# Common x-axis styling
# ============================================================

def configure_x_axis(ax):

    ax.set_xscale(
        "log",
        base=10,
    )

    ax.set_xticks(
        X_TICKS
    )

    ax.set_xticklabels(
        X_TICK_LABELS
    )

    # Give a little room around first/last tested weights.
    ax.set_xlim(
        0.0027,
        1.25,
    )


# ============================================================
# Horizontal no-OR baseline
# ============================================================

def plot_baseline(ax, metric, color=BASELINE_COLOR):

    mean = baseline[metric]["mean"]
    sd = baseline[metric]["sd"]

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    ax.axhline(
        mean,
        color=color,
        linestyle=":",
        linewidth=1.7,
        zorder=1,
    )

    # --------------------------------------------------------
    # Baseline uncertainty
    #
    # A light band is more appropriate than an error bar for a
    # horizontal reference condition.
    # --------------------------------------------------------

    if SHOW_ERROR_BARS and sd > 0:

        ax.axhspan(
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.10,
            zorder=0,
        )


# ============================================================
# Main + partial weight sweep for one metric
# ============================================================

def plot_weight_metric(
    ax,
    metric,
    color=MAIN_COLOR,
):

    # ========================================================
    # No-OR baseline
    # ========================================================

    plot_baseline(
        ax,
        metric,
    )

    # ========================================================
    # Three-seed main sweep
    # ========================================================

    main_means, main_sds = get_summary(
        metric,
        MAIN_WEIGHTS,
    )

    ax.plot(
        MAIN_WEIGHTS,
        main_means,
        color=color,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        linestyle="-",
        zorder=3,
    )

    if SHOW_ERROR_BARS:

        ax.errorbar(
            MAIN_WEIGHTS,
            main_means,
            yerr=main_sds,
            fmt="none",
            color=color,
            capsize=ERROR_CAPSIZE,
            linewidth=ERROR_LINE_WIDTH,
            zorder=2,
        )

    # ========================================================
    # Partial sweep
    # ========================================================

    partial_means, _ = get_summary(
        metric,
        PARTIAL_WEIGHTS,
    )

    ax.scatter(
        PARTIAL_WEIGHTS,
        partial_means,
        marker="X",
        s=65,
        color=PARTIAL_COLOR,
        zorder=5,
    )

    # Connect the last replicated weight through the partial
    # sweep using a dotted line to distinguish its status.
    ax.plot(
        np.concatenate([
            [MAIN_WEIGHTS[-1]],
            PARTIAL_WEIGHTS,
        ]),
        np.concatenate([
            [main_means[-1]],
            partial_means,
        ]),
        color=PARTIAL_COLOR,
        linestyle=":",
        linewidth=1.3,
        zorder=2,
    )

    configure_x_axis(ax)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Common legend
# ============================================================

def get_standard_legend_handles():

    return [

        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle=":",
            linewidth=1.7,
            label="Baseline",
        ),

        Line2D(
            [0],
            [0],
            color=MAIN_COLOR,
            marker="o",
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Main sweep (3 seeds)",
        ),

        Line2D(
            [0],
            [0],
            color=PARTIAL_COLOR,
            marker="X",
            linestyle=":",
            linewidth=1.3,
            markersize=7,
            label="Partial sweep (1 seed)",
        ),
    ]


# ============================================================
# Figure 1
#
# Harmless / GCG / Soft-Prompt
# ============================================================

def plot_main_outcomes():

    metrics = [
        "Harmless",
        "GCG",
        "Soft-Prompt",
    ]

    titles = {
        "Harmless": "Harmless",
        "GCG": "GCG",
        "Soft-Prompt": "Soft-Prompt",
    }

    ylabels = {
        "Harmless": "Success (%)",
        "GCG": "ASR (%)",
        "Soft-Prompt": "ASR (%)",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 11),
        sharex=True,
    )

    for ax, metric in zip(
        axes,
        metrics,
    ):

        plot_weight_metric(
            ax,
            metric,
        )

        ax.set_title(
            titles[metric],
            # fontsize=11,
        )

        ax.set_ylabel(
            ylabels[metric],
            # fontsize=10,
        )

    axes[-1].set_xlabel(
        r"OR loss weight $\alpha_{\mathrm{OR}}$",
        # fontsize=10,
    )

    fig.legend(
        handles=get_standard_legend_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        # fontsize=9,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.89,
        bottom=0.07,
        hspace=0.48,
    )

    return fig


# ============================================================
# Figure 2
#
# Capability:
# MMLU / ARC-Easy / ARC-Challenge
# ============================================================

def plot_capability():

    metrics = [
        "MMLU",
        "ARC-E",
        "ARC-C",
    ]

    titles = {
        "MMLU": "MMLU",
        "ARC-E": "ARC-Easy",
        "ARC-C": "ARC-Challenge",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 11),
        sharex=True,
    )

    for ax, metric in zip(
        axes,
        metrics,
    ):

        plot_weight_metric(
            ax,
            metric,
        )

        ax.set_title(
            titles[metric],
            # fontsize=11,
        )

        ax.set_ylabel(
            "Accuracy (%)",
            # fontsize=10,
        )

    axes[-1].set_xlabel(
        r"OR loss weight $\alpha_{\mathrm{OR}}$",
        # fontsize=10,
    )

    fig.legend(
        handles=get_standard_legend_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        # fontsize=9,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.92,
        bottom=0.07,
        hspace=0.48,
    )

    return fig


# ============================================================
# Figure 3
#
# AutoDAN
# ============================================================

def plot_autodan():

    fig, ax = plt.subplots(
        figsize=(7, 3.8),
    )

    plot_weight_metric(
        ax,
        "AutoDAN",
    )

    ax.set_ylabel(
        "ASR (%)",
        # fontsize=10,
    )

    ax.set_xlabel(
        r"OR loss weight $\alpha_{\mathrm{OR}}$",
        # fontsize=10,
    )

    # AutoDAN is at/near the floor, so use a deliberately
    # narrow scale rather than wasting the full 0--100 range.
    ax.set_ylim(
        -0.2,
        3.2,
    )

    ax.legend(
        handles=get_standard_legend_handles(),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        # fontsize=9,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.82,
        bottom=0.16,
    )

    return fig


# ============================================================
# Figure 4
#
# XSTest:
# safe compliance + unsafe refusal
#
# Colour = XSTest component
# Marker/line treatment = sweep status
# ============================================================

def plot_xstest():

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5),
    )

    component_colors = {
        "XSTest safe": "C0",
        "XSTest unsafe": "C1",
    }

    component_labels = {
        "XSTest safe": "Safe compliance",
        "XSTest unsafe": "Unsafe refusal",
    }

    for metric in [
        "XSTest safe",
        "XSTest unsafe",
    ]:

        color = component_colors[metric]

        # ----------------------------------------------------
        # No-OR baseline
        # ----------------------------------------------------

        mean = baseline[metric]["mean"]
        sd = baseline[metric]["sd"]

        ax.axhline(
            mean,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=1,
        )

        if SHOW_ERROR_BARS and sd > 0:

            ax.axhspan(
                mean - sd,
                mean + sd,
                color=color,
                alpha=0.07,
                zorder=0,
            )

        # ----------------------------------------------------
        # Main sweep
        # ----------------------------------------------------

        main_means, main_sds = get_summary(
            metric,
            MAIN_WEIGHTS,
        )

        ax.plot(
            MAIN_WEIGHTS,
            main_means,
            color=color,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            linestyle="-",
            zorder=3,
        )

        if SHOW_ERROR_BARS:

            ax.errorbar(
                MAIN_WEIGHTS,
                main_means,
                yerr=main_sds,
                fmt="none",
                color=color,
                capsize=ERROR_CAPSIZE,
                linewidth=ERROR_LINE_WIDTH,
                zorder=2,
            )

        # ----------------------------------------------------
        # Partial sweep
        # ----------------------------------------------------

        partial_means, _ = get_summary(
            metric,
            PARTIAL_WEIGHTS,
        )

        ax.scatter(
            PARTIAL_WEIGHTS,
            partial_means,
            marker="X",
            s=65,
            color=color,
            zorder=5,
        )

        ax.plot(
            np.concatenate([
                [MAIN_WEIGHTS[-1]],
                PARTIAL_WEIGHTS,
            ]),
            np.concatenate([
                [main_means[-1]],
                partial_means,
            ]),
            color=color,
            linestyle=":",
            linewidth=1.3,
            zorder=2,
        )

    configure_x_axis(ax)

    ax.set_ylim(
        0,
        105,
    )

    ax.set_xlabel(
        r"OR loss weight $\alpha_{\mathrm{OR}}$",
        # fontsize=10,
    )

    ax.set_ylabel(
        "Score (%)",
        # fontsize=10,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Two legends:
    # component and sweep status
    # --------------------------------------------------------

    component_handles = [

        Line2D(
            [0],
            [0],
            color=component_colors["XSTest safe"],
            marker="o",
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Safe compliance",
        ),

        Line2D(
            [0],
            [0],
            color=component_colors["XSTest unsafe"],
            marker="o",
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Unsafe refusal",
        ),
    ]

    status_handles = [

        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="No OR term",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Main sweep (3 seeds)",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            marker="X",
            linestyle=":",
            linewidth=1.3,
            markersize=7,
            label="Partial sweep (1 seed)",
        ),
    ]

    all_handles = (
        component_handles
        + status_handles
    )

    fig.legend(
        handles=all_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        # fontsize=8.5,
        columnspacing=1.3,
        handlelength=2.0,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.80,
        bottom=0.15,
    )

    return fig


# ============================================================
# Generate all figures
# ============================================================

figures = {
    "OR_weight_sweep": plot_main_outcomes(),
    "OR_weight_sweep_capability": plot_capability(),
    "OR_weight_sweep_AutoDAN": plot_autodan(),
    "OR_weight_sweep_XSTest": plot_xstest(),
}


# ============================================================
# Output
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

for filename, figure in figures.items():

    pdf_path = (
        output_dir / f"{filename}.pdf"
    )

    png_path = (
        output_dir / f"{filename}.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved:\n"
        f"  {pdf_path}\n"
        f"  {png_path}\n"
    )


# ============================================================
# Display
# ============================================================

plt.show()
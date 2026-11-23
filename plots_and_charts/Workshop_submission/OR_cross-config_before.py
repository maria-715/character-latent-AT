import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D


# ============================================================
# Typography
# ============================================================

plt.rcParams.update({
    "font.size": 15,          # default text
    "axes.titlesize": 15,     # panel titles
    "axes.labelsize": 15,     # x/y labels
    "xtick.labelsize": 12,    # x tick labels
    "ytick.labelsize": 12,    # y tick labels
    "legend.fontsize": 12,    # legend
})


# ============================================================
# Configuration
# ============================================================

# Baseline error bars are always shown.
# This controls only replacement-condition error bars where
# an SD is actually available.
SHOW_REPLACEMENT_ERROR_BARS = False

# Metric encoding
COLORS = {
    "Harmless": "C0",
    "Soft-Prompt": "C1",
}

MARKERS = {
    "Harmless": "o",
    "Soft-Prompt": "s",
}

# Condition encoding
REPLACEMENT_LINESTYLE = "-"
BASELINE_LINESTYLE = "--"

MARKER_SIZE = 6
LINE_WIDTH = 1.7
ERROR_LINE_WIDTH = 1.0
ERROR_CAPSIZE = 3


# ============================================================
# Data
#
# Baseline:
#   ordinary UltraChat-only sweep
#
# Replacement:
#   CAT training with n = 1,000 OR examples
#
# np.nan means that no SD was reported.
# ============================================================

models = {

    # ========================================================
    # Zephyr
    # ========================================================

    "Zephyr": {

        "epsilon": np.array([
            0.00884,
            0.01768,
            0.03535,
            0.07070,
        ]),

        "epsilon_labels": [
            "0.00884",
            "0.01768",
            "0.03535",
            "0.07070",
        ],

        "Harmless": {

            "baseline": {
                "y": np.array([
                    98.33,
                    94.17,
                    89.17,
                    70.83,
                ]),
                "err": np.array([
                    1.44,
                    3.82,
                    1.44,
                    9.46,
                ]),
            },

            "replacement": {
                "y": np.array([
                    100.00,
                    95.00,
                    97.50,
                    75.00,
                ]),
                "err": np.array([
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]),
            },
        },

        "Soft-Prompt": {

            "baseline": {
                "y": np.array([
                    43.44,
                    43.40,
                    42.45,
                    51.13,
                ]),
                "err": np.array([
                    4.53,
                    5.42,
                    13.65,
                    3.21,
                ]),
            },

            "replacement": {
                "y": np.array([
                    63.71,
                    60.75,
                    55.22,
                    66.16,
                ]),
                "err": np.array([
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]),
            },
        },
    },

    # ========================================================
    # Llama 3 (Llama 2 recipe)
    # ========================================================

    "Llama 3 (Llama 2)": {

        "epsilon": np.array([
            0.04424,
            0.08848,
            0.17695,
            0.35391,
        ]),

        "epsilon_labels": [
            "0.04424",
            "0.08848",
            "0.17695",
            "0.35391",
        ],

        "Harmless": {

            "baseline": {
                "y": np.array([
                    90.83,
                    45.83,
                    7.50,
                    1.67,
                ]),
                "err": np.array([
                    7.22,
                    26.96,
                    12.99,
                    2.89,
                ]),
            },

            "replacement": {
                "y": np.array([
                    82.50,
                    55.00,
                    0.00,
                    0.00,
                ]),
                "err": np.array([
                    np.nan,
                    np.nan,
                    np.nan,
                    0.00,
                ]),
            },
        },

        "Soft-Prompt": {

            "baseline": {
                "y": np.array([
                    57.29,
                    59.25,
                    59.33,
                    64.46,
                ]),
                "err": np.array([
                    9.37,
                    16.08,
                    1.41,
                    2.30,
                ]),
            },

            "replacement": {
                "y": np.array([
                    79.18,
                    77.55,
                    82.26,
                    74.40,
                ]),
                "err": np.array([
                    np.nan,
                    np.nan,
                    np.nan,
                    4.92,
                ]),
            },
        },
    },

    # ========================================================
    # Llama 3 (MixAT)
    # ========================================================

    "Llama 3 (MixAT)": {

        "epsilon": np.array([
            0.04424,
            0.08848,
            0.17695,
            0.35391,
        ]),

        "epsilon_labels": [
            "0.04424",
            "0.08848",
            "0.17695",
            "0.35391",
        ],

        "Harmless": {

            "baseline": {
                "y": np.array([
                    82.50,
                    58.33,
                    20.83,
                    50.00,
                ]),
                "err": np.array([
                    13.92,
                    14.65,
                    15.07,
                    41.61,
                ]),
            },

            "replacement": {
                "y": np.array([
                    97.50,
                    45.00,
                    15.00,
                    0.00,
                ]),
                "err": np.array([
                    np.nan,
                    np.nan,
                    np.nan,
                    0.00,
                ]),
            },
        },

        "Soft-Prompt": {

            "baseline": {
                "y": np.array([
                    55.47,
                    43.23,
                    58.18,
                    63.21,
                ]),
                "err": np.array([
                    4.46,
                    3.92,
                    4.86,
                    3.20,
                ]),
            },

            "replacement": {
                "y": np.array([
                    74.72,
                    71.38,
                    83.21,
                    75.77,
                ]),
                "err": np.array([
                    np.nan,
                    np.nan,
                    np.nan,
                    5.12,
                ]),
            },
        },
    },
}


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

    # --------------------------------------------------------
    # Main trajectory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Error bars
    # --------------------------------------------------------

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

def plot_configuration(ax, configuration, panel_label=None):

    data = models[configuration]
    epsilon = data["epsilon"]

    # ========================================================
    # Harmless
    # ========================================================

    # Baseline
    plot_trajectory(
        ax=ax,
        x=epsilon,
        y=data["Harmless"]["baseline"]["y"],
        err=data["Harmless"]["baseline"]["err"],
        color=COLORS["Harmless"],
        marker=MARKERS["Harmless"],
        linestyle=BASELINE_LINESTYLE,
        filled=False,
        show_error_bars=True,
        zorder=2,
    )

    # OR replacement
    plot_trajectory(
        ax=ax,
        x=epsilon,
        y=data["Harmless"]["replacement"]["y"],
        err=data["Harmless"]["replacement"]["err"],
        color=COLORS["Harmless"],
        marker=MARKERS["Harmless"],
        linestyle=REPLACEMENT_LINESTYLE,
        filled=True,
        show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
        zorder=4,
    )

    # ========================================================
    # Soft-Prompt
    # ========================================================

    # Baseline
    plot_trajectory(
        ax=ax,
        x=epsilon,
        y=data["Soft-Prompt"]["baseline"]["y"],
        err=data["Soft-Prompt"]["baseline"]["err"],
        color=COLORS["Soft-Prompt"],
        marker=MARKERS["Soft-Prompt"],
        linestyle=BASELINE_LINESTYLE,
        filled=False,
        show_error_bars=True,
        zorder=2,
    )

    # OR replacement
    plot_trajectory(
        ax=ax,
        x=epsilon,
        y=data["Soft-Prompt"]["replacement"]["y"],
        err=data["Soft-Prompt"]["replacement"]["err"],
        color=COLORS["Soft-Prompt"],
        marker=MARKERS["Soft-Prompt"],
        linestyle=REPLACEMENT_LINESTYLE,
        filled=True,
        show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
        zorder=4,
    )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_ylim(
        0,
        105,
    )

    ax.set_xticks(
        epsilon
    )

    ax.set_xticklabels(
        data["epsilon_labels"],
    )

    ax.set_ylabel(
        "Score / ASR (%)",
    )

    ax.set_xlabel(
        r"Configured radius $\epsilon$",
    )

    if panel_label is None:
        ax.set_title(configuration)
    else:
        ax.set_title(
            rf"(${panel_label}$) " + configuration
        )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Legend handles
# ============================================================

def get_legend_handles():

    metric_handles = [

        Line2D(
            [0],
            [0],
            color=COLORS["Harmless"],
            marker=MARKERS["Harmless"],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Harmless",
        ),

        Line2D(
            [0],
            [0],
            color=COLORS["Soft-Prompt"],
            marker=MARKERS["Soft-Prompt"],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Soft-Prompt ASR",
        ),
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

    return metric_handles, condition_handles


# ============================================================
# Add shared legends
# ============================================================

def add_shared_legends(fig):

    metric_handles, condition_handles = get_legend_handles()

    # Metric legend
    legend_metrics = fig.legend(
        handles=metric_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
         fontsize=13,
    )

    # Condition legend
    fig.legend(
        handles=condition_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        fontsize=13,
    )

    # Ensure first legend is retained
    fig.add_artist(legend_metrics)


# ============================================================
# Figure 1
# Zephyr + Llama 3 (MixAT)
# ============================================================

def plot_zephyr_mixat():

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    plot_configuration(
        ax=axes[0],
        configuration="Zephyr",
        panel_label="a",
    )

    plot_configuration(
        ax=axes[1],
        configuration="Llama 3 (MixAT)",
        panel_label="b",
    )

    add_shared_legends(fig)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.78,
        bottom=0.16,
        wspace=0.25,
    )

    return fig


# ============================================================
# Figure 2
# Llama 3 (Llama 2)
# ============================================================

def plot_llama2():

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(6.5, 5),
    )

    plot_configuration(
        ax=ax,
        configuration="Llama 3 (Llama 2)",
        panel_label=None,
    )

    add_shared_legends(fig)

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.78,
        bottom=0.16,
    )

    return fig


# ============================================================
# Generate figures
# ============================================================

figure_zephyr_mixat = plot_zephyr_mixat()

figure_llama2 = plot_llama2()


# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent


# Figure 1
zephyr_mixat_pdf_path = (
    output_dir / "OR_Zephyr_MixAT_n1000.pdf"
)

zephyr_mixat_png_path = (
    output_dir / "OR_Zephyr_MixAT_n1000.png"
)


# Figure 2
llama2_pdf_path = (
    output_dir / "OR_Llama3_Llama2_n1000.pdf"
)

llama2_png_path = (
    output_dir / "OR_Llama3_Llama2_n1000.png"
)


# ============================================================
# Save Figure 1
# ============================================================

figure_zephyr_mixat.savefig(
    zephyr_mixat_pdf_path,
    bbox_inches="tight",
)

figure_zephyr_mixat.savefig(
    zephyr_mixat_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save Figure 2
# ============================================================

figure_llama2.savefig(
    llama2_pdf_path,
    bbox_inches="tight",
)

figure_llama2.savefig(
    llama2_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Report
# ============================================================

print(
    "Saved Zephyr + Llama 3 (MixAT) figure to:\n"
    f"  {zephyr_mixat_pdf_path}\n"
    f"  {zephyr_mixat_png_path}\n\n"
    "Saved Llama 3 (Llama 2) figure to:\n"
    f"  {llama2_pdf_path}\n"
    f"  {llama2_png_path}\n\n"
    "Baseline error bars: always shown\n"
    f"Replacement error bars: {SHOW_REPLACEMENT_ERROR_BARS}"
)


# ============================================================
# Display
# ============================================================

plt.show()
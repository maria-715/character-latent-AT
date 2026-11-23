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

        "s": np.array([
            0.05,
            0.10,
            0.20,
            0.40,
        ]),

        "s_labels": [
            "0.05",
            "0.1",
            "0.2",
            "0.4",
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

        "s": np.array([
            0.075,
            0.150,
            0.300,
            0.600,
        ]),

        "s_labels": [
            "0.075",
            "0.15",
            "0.3",
            "0.6",
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

        "s": np.array([
            0.075,
            0.150,
            0.300,
            0.600,
        ]),

        "s_labels": [
            "0.075",
            "0.15",
            "0.3",
            "0.6",
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
# Three-panel cross-configuration figure
# ============================================================

def plot_cross_configuration():

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7, 11),
        squeeze=False,
    )

    configuration_order = [
        "Zephyr",
        "Llama 3 (Llama 2)",
        "Llama 3 (MixAT)",
    ]

    panel_labels = [
        "a",
        "b",
        "c",
    ]

    panel_titles = {
        "Zephyr": "Zephyr",
        "Llama 3 (Llama 2)": "Llama 3 (Llama 2)",
        "Llama 3 (MixAT)": "Llama 3 (MixAT)",
    }

    # ========================================================
    # Panels
    # ========================================================

    for row_index, (
        configuration,
        panel_label,
    ) in enumerate(
        zip(
            configuration_order,
            panel_labels,
        )
    ):

        ax = axes[row_index, 0]
        data = models[configuration]

        s = data["s"]

        # ====================================================
        # Harmless
        # ====================================================

        # Baseline
        plot_trajectory(
            ax=ax,
            x=s,
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
            x=s,
            y=data["Harmless"]["replacement"]["y"],
            err=data["Harmless"]["replacement"]["err"],
            color=COLORS["Harmless"],
            marker=MARKERS["Harmless"],
            linestyle=REPLACEMENT_LINESTYLE,
            filled=True,
            show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
            zorder=4,
        )

        # ====================================================
        # Soft-Prompt
        # ====================================================

        # Baseline
        plot_trajectory(
            ax=ax,
            x=s,
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
            x=s,
            y=data["Soft-Prompt"]["replacement"]["y"],
            err=data["Soft-Prompt"]["replacement"]["err"],
            color=COLORS["Soft-Prompt"],
            marker=MARKERS["Soft-Prompt"],
            linestyle=REPLACEMENT_LINESTYLE,
            filled=True,
            show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
            zorder=4,
        )

        # ====================================================
        # Axes
        # ====================================================

        ax.set_ylim(
            0,
            105,
        )

        ax.set_xticks(
            s
        )

        ax.set_xticklabels(
            data["s_labels"],
        )

        ax.set_ylabel(
            "Score / ASR (%)",
            # fontsize=10,
        )

        # ax.set_xlabel(
        #     r"Configuration multiplier $s$",
        #     fontsize=10,
        # )

        ax.set_title(
            rf"(${panel_label}$) "
            + panel_titles[configuration],
            # fontsize=11,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel(
        r"Configuration multiplier $s$",
        # fontsize=10,
    )

    # ========================================================
    # Shared legend
    #
    # Two independent encodings:
    #
    # Marker/colour = metric
    # Line/fill     = training condition
    # ========================================================

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

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.86,
        bottom=0.07,
        hspace=0.55,
    )

    # Metric legend
    legend_metrics = fig.legend(
        handles=metric_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        # fontsize=10,
    )

    # Condition legend
    fig.legend(
        handles=condition_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        # fontsize=10,
    )

    # Ensure first legend is retained
    fig.add_artist(
        legend_metrics
    )

    return fig


# ============================================================
# Generate figure
# ============================================================

figure = plot_cross_configuration()


# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

pdf_path = (
    output_dir / "OR_cross_config_n1000.pdf"
)

png_path = (
    output_dir / "OR_cross_config_n1000.png"
)


# ============================================================
# Save
# ============================================================

figure.savefig(
    pdf_path,
    bbox_inches="tight",
)

figure.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Report
# ============================================================

print(
    "Saved cross-configuration figure to:\n"
    f"  {pdf_path}\n"
    f"  {png_path}\n\n"
    "Baseline error bars: always shown\n"
    f"Replacement error bars: {SHOW_REPLACEMENT_ERROR_BARS}"
)


# ============================================================
# Display
# ============================================================

plt.show()
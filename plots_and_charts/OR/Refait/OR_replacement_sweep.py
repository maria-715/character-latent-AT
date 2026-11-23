import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

# ============================================================
# Error-bar configuration
#
# Baseline error bars are always shown because SDs are
# available at every baseline point.
#
# This switch controls only replacement-condition error bars.
# ============================================================

SHOW_REPLACEMENT_ERROR_BARS = False

# Replacement condition and baseline deliberately use the same
# colour family. Line/marker style identifies the condition.
REPLACEMENT_COLOR = "C0"
BASELINE_COLOR = "0.35"

REPLACEMENT_MARKER = "o"
BASELINE_MARKER = "o"

REPLACEMENT_LINESTYLE = "-"
BASELINE_LINESTYLE = "--"

MARKER_SIZE = 5.5
LINE_WIDTH = 1.6
ERROR_LINE_WIDTH = 1.0
ERROR_CAPSIZE = 2.5


# ============================================================
# Multiplier grid
# ============================================================

S_VALUES = np.array([
    0.075,
    0.150,
    0.300,
    0.600,
])

S_LABELS = [
    "0.075",
    "0.15",
    "0.3",
    "0.6",
]


# ============================================================
# Baseline data
#
# Llama 3 (MixAT), UltraChat-only sweep.
#
# These baseline trajectories are repeated in every row so that
# each replacement size can be compared directly with the same
# reference condition.
# ============================================================

baseline = {

    "Harmless": {
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

    "XSTest safe": {
        "y": np.array([
            51.60,
            51.07,
            60.27,
            37.33,
        ]),
        "err": np.array([
            15.48,
            10.51,
            10.01,
            10.99,
        ]),
    },

    "XSTest unsafe": {
        "y": np.array([
            99.33,
            98.50,
            97.00,
            99.50,
        ]),
        "err": np.array([
            0.29,
            0.87,
            1.80,
            0.00,
        ]),
    },

    "MMLU": {
        "y": np.array([
            59.73,
            59.47,
            58.51,
            55.74,
        ]),
        "err": np.array([
            0.22,
            0.14,
            0.68,
            0.41,
        ]),
    },

    "ARC-E": {
        "y": np.array([
            80.46,
            79.44,
            75.45,
            41.29,
        ]),
        "err": np.array([
            0.43,
            0.29,
            2.54,
            19.91,
        ]),
    },

    "ARC-C": {
        "y": np.array([
            50.85,
            49.34,
            47.84,
            32.37,
        ]),
        "err": np.array([
            0.39,
            0.65,
            1.65,
            9.46,
        ]),
    },

    "GCG": {
        "y": np.array([
            25.83,
            12.50,
            17.50,
            25.00,
        ]),
        "err": np.array([
            1.44,
            2.50,
            6.61,
            10.90,
        ]),
    },

    "AutoDAN": {
        "y": np.array([
            0.83,
            0.00,
            0.00,
            0.00,
        ]),
        "err": np.array([
            1.44,
            0.00,
            0.00,
            0.00,
        ]),
    },

    "Soft-Prompt": {
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
}


# ============================================================
# Replacement data
#
# np.nan means that no standard deviation is reported for that
# point. It must NOT be interpreted as zero variance.
# ============================================================

replacement = {

    # ========================================================
    # n = 100
    # ========================================================

    100: {

        "Harmless": {
            "y": np.array([
                72.50,
                62.50,
                0.00,
                0.83,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                1.44,
            ]),
        },

        "GCG": {
            "y": np.array([
                37.50,
                47.50,
                57.50,
                50.00,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                12.99,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                63.02,
                73.71,
                56.48,
                70.73,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                5.29,
            ]),
        },

        "XSTest safe": {
            "y": np.array([
                6.40,
                68.00,
                6.40,
                4.67,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                4.31,
            ]),
        },

        "XSTest unsafe": {
            "y": np.array([
                99.50,
                91.50,
                99.00,
                98.50,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.50,
            ]),
        },

        "MMLU": {
            "y": np.array([
                60.03,
                59.26,
                56.65,
                56.13,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.74,
            ]),
        },

        "ARC-E": {
            "y": np.array([
                80.72,
                80.26,
                72.10,
                47.08,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                15.04,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                51.45,
                50.26,
                46.08,
                35.90,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                7.04,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                0.00,
                0.00,
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

    # ========================================================
    # n = 500
    # ========================================================

    500: {

        "Harmless": {
            "y": np.array([
                77.50,
                37.50,
                2.50,
                5.00,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                8.66,
            ]),
        },

        "GCG": {
            "y": np.array([
                37.50,
                34.17,
                50.00,
                61.67,
            ]),
            "err": np.array([
                np.nan,
                9.46,
                np.nan,
                16.07,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                65.16,
                65.56,
                76.04,
                78.06,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                3.23,
            ]),
        },

        "XSTest safe": {
            "y": np.array([
                4.80,
                90.80,
                22.40,
                2.00,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                1.06,
            ]),
        },

        "XSTest unsafe": {
            "y": np.array([
                99.50,
                77.00,
                95.00,
                82.17,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                30.46,
            ]),
        },

        "MMLU": {
            "y": np.array([
                60.54,
                60.11,
                57.84,
                56.70,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.22,
            ]),
        },

        "ARC-E": {
            "y": np.array([
                80.51,
                80.18,
                76.43,
                44.43,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                7.04,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                51.28,
                50.17,
                47.35,
                34.50,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                4.06,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                0.00,
                0.00,
                0.00,
                0.83,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                1.44,
            ]),
        },
    },

    # ========================================================
    # n = 1,000
    # ========================================================

    1000: {

        "Harmless": {
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

        "GCG": {
            "y": np.array([
                57.50,
                27.50,
                72.50,
                52.50,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                13.23,
            ]),
        },

        "Soft-Prompt": {
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

        "XSTest safe": {
            "y": np.array([
                78.40,
                86.00,
                53.60,
                0.53,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.61,
            ]),
        },

        "XSTest unsafe": {
            "y": np.array([
                91.50,
                91.00,
                87.00,
                100.00,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.00,
            ]),
        },

        "MMLU": {
            "y": np.array([
                60.61,
                58.36,
                58.20,
                56.09,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.43,
            ]),
        },

        "ARC-E": {
            "y": np.array([
                80.51,
                78.37,
                75.51,
                41.20,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                7.84,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                50.94,
                48.12,
                45.05,
                33.56,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                4.02,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                0.00,
                0.00,
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

    # ========================================================
    # n = 3,000
    # ========================================================

    3000: {

        "Harmless": {
            "y": np.array([
                77.50,
                52.50,
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

        "GCG": {
            "y": np.array([
                32.50,
                35.00,
                32.50,
                30.00,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                2.50,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                79.75,
                71.13,
                78.81,
                78.34,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                4.83,
            ]),
        },

        "XSTest safe": {
            "y": np.array([
                5.20,
                70.40,
                2.80,
                0.67,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.46,
            ]),
        },

        "XSTest unsafe": {
            "y": np.array([
                100.00,
                92.00,
                99.50,
                100.00,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.00,
            ]),
        },

        "MMLU": {
            "y": np.array([
                60.00,
                59.18,
                58.81,
                55.19,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                0.61,
            ]),
        },

        "ARC-E": {
            "y": np.array([
                80.64,
                79.12,
                75.00,
                39.70,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                4.76,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                50.51,
                48.38,
                45.39,
                32.14,
            ]),
            "err": np.array([
                np.nan,
                np.nan,
                np.nan,
                1.70,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                0.00,
                0.00,
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
}


# ============================================================
# Figure structure
# ============================================================

ROW_ORDER = [
    100,
    500,
    1000,
    3000,
]

UTILITY_PERCENTAGES = {
    100: 1.2,
    500: 6.0,
    1000: 11.9,
    3000: 35.7,
}


def format_row_label(n):

    if n == 1000:
        n_text = r"1{,}000"
    elif n == 3000:
        n_text = r"3{,}000"
    else:
        n_text = str(n)

    percentage = UTILITY_PERCENTAGES[n]

    return rf"$n={n_text}~({percentage:.1f}\%)$"

METRICS = [
    "Harmless",
    "MMLU",
    "ARC-E",
    "ARC-C",
    "GCG",
    "AutoDAN",
    "Soft-Prompt",
]

COLUMN_TITLES = {
    "Harmless": "Harmless",
    "MMLU": "MMLU",
    "ARC-E": "ARC-Easy",
    "ARC-C": "ARC-Challenge",
    "GCG": "GCG",
    "AutoDAN": "AutoDAN",
    "Soft-Prompt": "Soft-Prompt",
}

Y_LABELS = {
    "Harmless": "Success rate (%)",
    "MMLU": "Accuracy (%)",
    "ARC-E": "Accuracy (%)",
    "ARC-C": "Accuracy (%)",
    "GCG": "ASR (%)",
    "AutoDAN": "ASR (%)",
    "Soft-Prompt": "ASR (%)",
}


# ============================================================
# Calculate shared y-limits per metric column
#
# Every row within one metric uses exactly the same limits.
#
# Error bars are included in the limit calculation when they
# are displayed.
# ============================================================

def get_metric_limits(metric):

    values = []

    # --------------------------------------------------------
    # Baseline
    #
    # Baseline error bars are always displayed, so they always
    # contribute to the plotting limits.
    # --------------------------------------------------------

    baseline_y = baseline[metric]["y"]
    baseline_err = baseline[metric]["err"]

    values.extend(
        baseline_y - baseline_err
    )

    values.extend(
        baseline_y + baseline_err
    )

    # --------------------------------------------------------
    # Replacement sizes
    # --------------------------------------------------------

    for n in ROW_ORDER:

        y = replacement[n][metric]["y"]
        err = replacement[n][metric]["err"]

        if SHOW_REPLACEMENT_ERROR_BARS:

            lower = np.where(
                np.isnan(err),
                y,
                y - err,
            )

            upper = np.where(
                np.isnan(err),
                y,
                y + err,
            )

            values.extend(lower)
            values.extend(upper)

        else:

            values.extend(y)

    values = np.asarray(
        values,
        dtype=float,
    )

    ymin = np.nanmin(values)
    ymax = np.nanmax(values)

    padding = max(
        (ymax - ymin) * 0.07,
        2.0,
    )

    lower_limit = max(
        0.0,
        ymin - padding,
    )

    upper_limit = min(
        100.0,
        ymax + padding,
    )

    if metric == "Harmless":
        lower_limit = 0.0
        upper_limit = 100.0

    if metric == "AutoDAN":
        lower_limit = 0.0
        upper_limit = 3.0

    return lower_limit, upper_limit


# ============================================================
# Plot a trajectory
# ============================================================

def plot_trajectory(
    ax,
    y,
    err,
    color,
    marker,
    linestyle,
    filled=True,
    show_error_bars=True,
    zorder=3,
):

    marker_face = (
        color if filled else "white"
    )

    # --------------------------------------------------------
    # Main trajectory
    # --------------------------------------------------------

    ax.plot(
        S_VALUES,
        y,
        marker=marker,
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        linestyle=linestyle,
        color=color,
        markerfacecolor=marker_face,
        markeredgecolor=color,
        zorder=zorder,
    )

    # --------------------------------------------------------
    # Error bars
    #
    # NaN means that no SD was reported for that point.
    # --------------------------------------------------------

    if show_error_bars:

        valid = ~np.isnan(err)

        if np.any(valid):

            ax.errorbar(
                S_VALUES[valid],
                y[valid],
                yerr=err[valid],
                fmt="none",
                capsize=ERROR_CAPSIZE,
                linewidth=ERROR_LINE_WIDTH,
                color=color,
                zorder=zorder - 1,
            )


from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Create one figure per metric (4 stacked subplots, one per n)
# ============================================================

def plot_metric_figure(metric):

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(6, 10),
        sharex=True,
        squeeze=False,
    )

    lower_limit, upper_limit = get_metric_limits(metric)

    for row_index, n in enumerate(ROW_ORDER):

        ax = axes[row_index, 0]

        # ------------------------------------------------
        # Baseline
        # ------------------------------------------------

        baseline_y = baseline[metric]["y"]
        baseline_err = baseline[metric]["err"]

        plot_trajectory(
            ax=ax,
            y=baseline_y,
            err=baseline_err,
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            linestyle=BASELINE_LINESTYLE,
            filled=False,
            show_error_bars=True,
            zorder=2,
        )

        # ------------------------------------------------
        # Replacement condition
        # ------------------------------------------------

        replacement_y = replacement[n][metric]["y"]
        replacement_err = replacement[n][metric]["err"]

        plot_trajectory(
            ax=ax,
            y=replacement_y,
            err=replacement_err,
            color=REPLACEMENT_COLOR,
            marker=REPLACEMENT_MARKER,
            linestyle=REPLACEMENT_LINESTYLE,
            filled=True,
            show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
            zorder=4,
        )

        ax.set_ylim(lower_limit, upper_limit)

        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylabel(Y_LABELS[metric], fontsize=11)

        row_text = format_row_label(n)

        ax.text(
            -0.30,
            0.5,
            row_text,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xticks(S_VALUES)
        ax.set_xticklabels(S_LABELS, rotation=20, ha="right", fontsize=10)
        # ax.tick_params(
        #     axis="x",
        #     which="both",
        #     labelbottom=True,
        # )

        if row_index == len(ROW_ORDER) - 1:
            ax.set_xlabel(r"Configuration multiplier $s$", fontsize=11)

    legend_handles = [
        Line2D(
            [0], [0],
            color=REPLACEMENT_COLOR,
            marker=REPLACEMENT_MARKER,
            markerfacecolor=REPLACEMENT_COLOR,
            markeredgecolor=REPLACEMENT_COLOR,
            linestyle=REPLACEMENT_LINESTYLE,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="OR replacement",
        ),
        Line2D(
            [0], [0],
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            markerfacecolor="white",
            markeredgecolor=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Baseline (UltraChat only)",
        ),
    ]

    # ============================================================
    # Explicit, symmetric margins (instead of tight_layout)
    # ============================================================
    LEFT_MARGIN = 0.22   # room for row labels + y-axis label
    RIGHT_MARGIN = 1.0 - LEFT_MARGIN

    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=0.90,
        bottom=0.08,
        hspace=0.32,
    )

    # Center of the actual plot area, not of the whole figure
    center_x = (LEFT_MARGIN + RIGHT_MARGIN) / 2

    fig.suptitle(
        COLUMN_TITLES[metric],
        fontsize=14,
        x=center_x,
        y=0.97,
    )

    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(center_x, 0.945),
        ncol=2,
        fontsize=12,
    )

    return fig


# ============================================================
# Capability grid
#
# 4 rows:
#   n = 100, 500, 1,000, 3,000
#
# 3 columns:
#   MMLU, ARC-Easy, ARC-Challenge
#
# Each column uses a shared y-scale across all four rows.
# ============================================================

def plot_capability_grid():

    capability_metrics = [
        "MMLU",
        "ARC-E",
        "ARC-C",
    ]

    capability_titles = {
        "MMLU": "MMLU",
        "ARC-E": "ARC-Easy",
        "ARC-C": "ARC-Challenge",
    }

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(11, 10),
        sharex=True,
        squeeze=False,
    )

    # ========================================================
    # Shared y-limits within each column
    # ========================================================

    metric_limits = {
        metric: get_metric_limits(metric)
        for metric in capability_metrics
    }

    # ========================================================
    # Draw all 12 panels
    # ========================================================

    for row_index, n in enumerate(ROW_ORDER):

        for col_index, metric in enumerate(
            capability_metrics
        ):

            ax = axes[
                row_index,
                col_index,
            ]

            # =================================================
            # Baseline
            # =================================================

            baseline_y = (
                baseline[metric]["y"]
            )

            baseline_err = (
                baseline[metric]["err"]
            )

            plot_trajectory(
                ax=ax,
                y=baseline_y,
                err=baseline_err,
                color=BASELINE_COLOR,
                marker=BASELINE_MARKER,
                linestyle=BASELINE_LINESTYLE,
                filled=False,

                # Baseline error bars are always shown.
                show_error_bars=True,

                zorder=2,
            )

            # =================================================
            # OR replacement
            # =================================================

            replacement_y = (
                replacement[n][metric]["y"]
            )

            replacement_err = (
                replacement[n][metric]["err"]
            )

            plot_trajectory(
                ax=ax,
                y=replacement_y,
                err=replacement_err,
                color=REPLACEMENT_COLOR,
                marker=REPLACEMENT_MARKER,
                linestyle=REPLACEMENT_LINESTYLE,
                filled=True,

                # Optional replacement error bars.
                show_error_bars=(
                    SHOW_REPLACEMENT_ERROR_BARS
                ),

                zorder=4,
            )

            # =================================================
            # Y-axis
            # =================================================

            lower_limit, upper_limit = (
                metric_limits[metric]
            )

            ax.set_ylim(
                lower_limit,
                upper_limit,
            )

            # =================================================
            # Styling
            # =================================================

            ax.grid(
                axis="y",
                alpha=0.25,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # =================================================
            # Column titles
            #
            # Only the top row receives metric titles.
            # =================================================

            if row_index == 0:

                ax.set_title(
                    capability_titles[metric],
                    fontsize=12,
                )

            # =================================================
            # Row labels
            #
            # Only the left-hand panel receives n.
            # =================================================

            if col_index == 0:

                row_text = format_row_label(n)

                ax.text(
                    -0.26,
                    0.5,
                    row_text,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

            # =================================================
            # X-axis
            # =================================================

            ax.set_xticks(
                S_VALUES
            )

            if row_index == len(ROW_ORDER) - 1:

                ax.set_xticklabels(
                    S_LABELS,
                    fontsize=9,
                )

                ax.set_xlabel(
                    r"Configuration multiplier $s$",
                    fontsize=10,
                )

            else:

                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

    # ========================================================
    # Y labels
    #
    # Since all three metrics are accuracy percentages, a
    # single shared label is sufficient.
    # ========================================================

    fig.supylabel(
        "Accuracy (%)",
        fontsize=11,
        x=0.035,
    )

    # ========================================================
    # Legend
    # ========================================================

    legend_handles = [

        Line2D(
            [0],
            [0],
            color=REPLACEMENT_COLOR,
            marker=REPLACEMENT_MARKER,
            markerfacecolor=REPLACEMENT_COLOR,
            markeredgecolor=REPLACEMENT_COLOR,
            linestyle=REPLACEMENT_LINESTYLE,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="OR replacement",
        ),

        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            markerfacecolor="white",
            markeredgecolor=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Baseline (UltraChat only)",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        fontsize=11,
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.93,
        bottom=0.07,
        hspace=0.30,
        wspace=0.25,
    )

    return fig


# ============================================================
# XSTest figure
#
# Four stacked panels, one per replacement size.
#
# Colour/fill:
#   blue = OR replacement
#   grey = baseline
#
# Line style:
#   solid  = safe compliance
#   dashed = unsafe refusal
# ============================================================

def plot_xstest_figure():

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(6, 10),
        sharex=True,
        squeeze=False,
    )

    # XSTest always uses the full percentage range because the
    # two components occupy very different parts of the scale.
    lower_limit = 0.0
    upper_limit = 100.0

    for row_index, n in enumerate(ROW_ORDER):

        ax = axes[row_index, 0]

        # ====================================================
        # Baseline safe compliance
        # ====================================================

        plot_trajectory(
            ax=ax,
            y=baseline["XSTest safe"]["y"],
            err=baseline["XSTest safe"]["err"],
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            linestyle="-",
            filled=False,
            show_error_bars=True,
            zorder=2,
        )

        # ====================================================
        # Baseline unsafe refusal
        # ====================================================

        plot_trajectory(
            ax=ax,
            y=baseline["XSTest unsafe"]["y"],
            err=baseline["XSTest unsafe"]["err"],
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            linestyle="--",
            filled=False,
            show_error_bars=True,
            zorder=2,
        )

        # ====================================================
        # Replacement safe compliance
        # ====================================================

        plot_trajectory(
            ax=ax,
            y=replacement[n]["XSTest safe"]["y"],
            err=replacement[n]["XSTest safe"]["err"],
            color=REPLACEMENT_COLOR,
            marker=REPLACEMENT_MARKER,
            linestyle="-",
            filled=True,
            show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
            zorder=4,
        )

        # ====================================================
        # Replacement unsafe refusal
        # ====================================================

        plot_trajectory(
            ax=ax,
            y=replacement[n]["XSTest unsafe"]["y"],
            err=replacement[n]["XSTest unsafe"]["err"],
            color=REPLACEMENT_COLOR,
            marker=REPLACEMENT_MARKER,
            linestyle="--",
            filled=True,
            show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
            zorder=4,
        )

        # ====================================================
        # Axes
        # ====================================================

        ax.set_ylim(
            lower_limit,
            upper_limit,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylabel(
            "Score (%)",
            fontsize=11,
        )

        # ====================================================
        # Row label
        # ====================================================

        row_text = format_row_label(n)

        ax.text(
            -0.30,
            0.5,
            row_text,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        # ====================================================
        # X axis
        # ====================================================

        ax.set_xticks(S_VALUES)
        ax.set_xticklabels(
            S_LABELS,
            rotation=20,
            ha="right",
            fontsize=10,
        )
        # ax.tick_params(
        #     axis="x",
        #     which="both",
        #     labelbottom=True,
        # )

        if row_index == len(ROW_ORDER) - 1:
            ax.set_xlabel(
                r"Configuration multiplier $s$",
                fontsize=11,
            )

    # ========================================================
    # Legends
    #
    # First encoding: training condition
    # Second encoding: XSTest component
    # ========================================================

    condition_handles = [

        Line2D(
            [0],
            [0],
            color=REPLACEMENT_COLOR,
            marker=REPLACEMENT_MARKER,
            markerfacecolor=REPLACEMENT_COLOR,
            markeredgecolor=REPLACEMENT_COLOR,
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="OR replacement",
        ),

        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            markerfacecolor="white",
            markeredgecolor=BASELINE_COLOR,
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="Baseline (UltraChat only)",
        ),
    ]

    component_handles = [

        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=LINE_WIDTH,
            label="Safe compliance",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=LINE_WIDTH,
            label="Unsafe refusal",
        ),
    ]

    # ========================================================
    # Layout
    # ========================================================

    LEFT_MARGIN = 0.22
    RIGHT_MARGIN = 1.0 - LEFT_MARGIN

    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=0.87,
        bottom=0.08,
        hspace=0.32,
    )

    center_x = (
        LEFT_MARGIN + RIGHT_MARGIN
    ) / 2

    fig.suptitle(
        "XSTest",
        fontsize=14,
        x=center_x,
        y=0.97,
    )

    # Condition legend
    fig.legend(
        handles=condition_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(center_x, 0.945),
        ncol=2,
        fontsize=11,
    )

    # Component legend
    fig.legend(
        handles=component_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(center_x, 0.912),
        ncol=2,
        fontsize=11,
    )

    return fig


# ============================================================
# Generate and save all figures
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

pdf_path = (
    output_dir / "OR_replacement_sweep_all_metrics.pdf"
)


# ============================================================
# Ordered figure specification
# ============================================================

FIGURE_ORDER = [
    "Harmless",
    "XSTest",
    "Capability",
    "GCG",
    "AutoDAN",
    "Soft-Prompt",
]


# ============================================================
# Multi-page PDF
# ============================================================

with PdfPages(pdf_path) as pdf:

    for figure_name in FIGURE_ORDER:

        if figure_name == "XSTest":

            fig = plot_xstest_figure()

            file_stem = (
                "OR_replacement_sweep_XSTest"
            )

        elif figure_name == "Capability":

            fig = plot_capability_grid()

            file_stem = (
                "OR_replacement_sweep_capability"
            )

        else:

            fig = plot_metric_figure(
                figure_name
            )

            file_stem = (
                "OR_replacement_sweep_"
                + figure_name.replace(" ", "_")
            )

        # ----------------------------------------------------
        # Add page to multi-page PDF
        # ----------------------------------------------------

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        # ----------------------------------------------------
        # Individual PDF
        # ----------------------------------------------------

        fig.savefig(
            output_dir / f"{file_stem}.pdf",
        )

        # ----------------------------------------------------
        # Individual PNG
        # ----------------------------------------------------

        fig.savefig(
            output_dir / f"{file_stem}.png",
            dpi=300,
        )

        plt.close(fig)


# ============================================================
# Report
# ============================================================

print(
    "Saved multi-page PDF to:\n"
    f"  {pdf_path}\n\n"
    "Pages:\n"
    "  1. Harmless\n"
    "  2. XSTest\n"
    "  3. Capability: MMLU / ARC-Easy / ARC-Challenge\n"
    "  4. GCG\n"
    "  5. AutoDAN\n"
    "  6. Soft-Prompt\n\n"
    "Also saved individual PDF/PNG files."
)
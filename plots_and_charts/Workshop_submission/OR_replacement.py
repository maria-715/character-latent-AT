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
# Configured-radius grid
# ============================================================

EPSILON_VALUES = np.array([
    0.04424,
    0.08848,
    0.17695,
    0.35391,
])

EPSILON_LABELS = [
    "0.04424",
    "0.08848",
    "0.17695",
    "0.35391",
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
        EPSILON_VALUES,
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
                EPSILON_VALUES[valid],
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
# Harmless / GCG / Soft-Prompt -- side by side horizontally
#
# Rows: n = 100, 500, 1,000, 3,000
# Columns: Harmless, GCG, Soft-Prompt
# ============================================================

def plot_selected_metrics():

    metrics = ["Harmless", "GCG", "Soft-Prompt"]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 10),
        sharex=True,
        squeeze=False,
    )

    metric_limits = {
        metric: get_metric_limits(metric)
        for metric in metrics
    }

    for row_index, n in enumerate(ROW_ORDER):

        for col_index, metric in enumerate(metrics):

            ax = axes[row_index, col_index]

            # Baseline trajectory
            plot_trajectory(
                ax=ax,
                y=baseline[metric]["y"],
                err=baseline[metric]["err"],
                color=BASELINE_COLOR,
                marker=BASELINE_MARKER,
                linestyle=BASELINE_LINESTYLE,
                filled=False,
                show_error_bars=True,
                zorder=2,
            )

            # OR-replacement trajectory
            plot_trajectory(
                ax=ax,
                y=replacement[n][metric]["y"],
                err=replacement[n][metric]["err"],
                color=REPLACEMENT_COLOR,
                marker=REPLACEMENT_MARKER,
                linestyle=REPLACEMENT_LINESTYLE,
                filled=True,
                show_error_bars=SHOW_REPLACEMENT_ERROR_BARS,
                zorder=4,
            )

            lower_limit, upper_limit = metric_limits[metric]
            ax.set_ylim(lower_limit, upper_limit)

            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Column titles
            if row_index == 0:
                ax.set_title(
                    COLUMN_TITLES[metric],
                    fontsize=13,
                )

            # Row labels only on the far-left column
            if col_index == 0:
                ax.text(
                    -0.28,
                    0.5,
                    format_row_label(n),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

            # Metric y-label on each column
            ax.set_ylabel(
                Y_LABELS[metric],
                fontsize=10,
            )

            # Epsilon x-axis
            ax.set_xticks(EPSILON_VALUES)

            if row_index == len(ROW_ORDER) - 1:
                ax.set_xticklabels(
                    EPSILON_LABELS,
                    rotation=20,
                    ha="right",
                    fontsize=9,
                )
                ax.set_xlabel(
                    r"Configured radius $\epsilon$",
                    fontsize=10,
                )
            else:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

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

    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.09,
        hspace=0.30,
        wspace=0.32,
    )

    return fig


# ============================================================
# Generate and save ONLY this figure
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

fig = plot_selected_metrics()

pdf_path = output_dir / "OR_replacement_Harmless_GCG_SoftPrompt_epsilon.pdf"
png_path = output_dir / "OR_replacement_Harmless_GCG_SoftPrompt_epsilon.png"

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved Harmless / GCG / Soft-Prompt figure to:\n"
    f"  {pdf_path}\n"
    f"  {png_path}"
)

plt.show()

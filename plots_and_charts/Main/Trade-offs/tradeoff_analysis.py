import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
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

COLORS = {
    "Zephyr": "C0",
    "Llama 3 (Llama 2)": "C1",
    "Llama 3 (MixAT)": "C2",
}

MARKERS = {
    "Zephyr": "o",
    "Llama 3 (Llama 2)": "s",
    "Llama 3 (MixAT)": "^",
}


# ============================================================
# Data
#
# All values are mean +/- sample SD across three training seeds.
#
# Each configuration contains:
#   s
#   Harmless
#   MMLU
#   GCG
#   Soft-Prompt
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

        "Harmless": {
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

        "MMLU": {
            "y": np.array([
                55.05,
                54.98,
                55.17,
                54.92,
            ]),
            "err": np.array([
                0.30,
                0.26,
                0.17,
                0.17,
            ]),
        },

        "GCG": {
            "y": np.array([
                19.17,
                17.50,
                18.33,
                9.17,
            ]),
            "err": np.array([
                3.82,
                6.61,
                10.10,
                1.44,
            ]),
        },

        "Soft-Prompt": {
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
    },

    # ========================================================
    # Llama 3 (Llama 2)
    # ========================================================

    "Llama 3 (Llama 2)": {

        "s": np.array([
            0.05,
            0.075,
            0.10,
            0.150,
            0.20,
            0.300,
            0.40,
            0.600,
        ]),

        "Harmless": {
            "y": np.array([
                95.83,
                90.83,
                76.67,
                45.83,
                49.17,
                7.50,
                2.50,
                1.67,
            ]),
            "err": np.array([
                1.44,
                7.22,
                7.64,
                26.96,
                20.05,
                12.99,
                4.33,
                2.89,
            ]),
        },

        "MMLU": {
            "y": np.array([
                60.10,
                59.74,
                59.83,
                59.35,
                58.81,
                57.69,
                56.23,
                54.94,
            ]),
            "err": np.array([
                0.09,
                0.17,
                0.13,
                0.55,
                0.11,
                0.45,
                1.04,
                0.98,
            ]),
        },

        "GCG": {
            "y": np.array([
                22.50,
                20.83,
                16.67,
                22.50,
                19.17,
                19.17,
                22.50,
                33.33,
            ]),
            "err": np.array([
                6.61,
                3.82,
                10.41,
                12.99,
                7.22,
                11.55,
                5.00,
                9.46,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                52.62,
                57.29,
                48.32,
                59.25,
                51.26,
                59.33,
                62.53,
                64.46,
            ]),
            "err": np.array([
                2.55,
                9.37,
                1.89,
                16.08,
                3.17,
                1.41,
                5.64,
                2.30,
            ]),
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
    },
}

# ============================================================
# Annotation offsets
#
# Used only to prevent overlapping s labels.
# Values are offsets in display points: (dx, dy).
# ============================================================

ANNOTATION_OFFSETS = {
    ("MMLU", "Zephyr", 0.05): (6, 8),
    ("MMLU", "Zephyr", 0.10): (-8, -13),
    ("MMLU", "Zephyr", 0.20): (6, -2),
    ("MMLU", "Zephyr", 0.40): (6, 8),

    ("MMLU", "Llama 3 (Llama 2)", 0.075): (0, 10),
    ("MMLU", "Llama 3 (Llama 2)", 0.2): (-5, -15),
    ("MMLU", "Llama 3 (Llama 2)", 0.4): (-15, -15),

    ("MMLU", "Llama 3 (MixAT)", 0.075): (8, 5),
    ("MMLU", "Llama 3 (MixAT)", 0.3): (-15, -15),

    ("Harmless", "Zephyr", 0.1): (0, 6),
    ("Harmless", "Zephyr", 0.2): (-1, 4),
    ("Harmless", "Zephyr", 0.4): (-5, 6),

    ("Harmless", "Llama 3 (Llama 2)", 0.075): (-8, 8),
    ("Harmless", "Llama 3 (Llama 2)", 0.1): (-4, 7),
    ("Harmless", "Llama 3 (Llama 2)", 0.15): (5,3),

    ("Harmless", "Llama 3 (MixAT)", 0.15): (-8, 10),
    ("Harmless", "Llama 3 (MixAT)", 0.3): (-15, 0),
}

# ----------------------------------------------------
# Mean trade-off points
# Filled = non-dominated
# Hollow = dominated
# ----------------------------------------------------

def is_non_dominated(x_i, y_i, x, y):
    """
    Pareto dominance WITHIN one model.

    Higher x = better.
    Lower y = better.
    """

    for x_j, y_j in zip(x, y):

        if (
            x_j >= x_i
            and y_j <= y_i
            and (x_j > x_i or y_j < y_i)
        ):
            return False

    return True

# ============================================================
# Plot one trade-off panel
# ============================================================

def add_direction_annotation(ax):

    ax.text(
        0.98,
        1.0,
        "Utility improves \u2192\nRobustness improves \u2193",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
        linespacing=1.25,
        zorder=5,
    )

def plot_tradeoff_panel(
    ax,
    utility_metric,
    robustness_metric,
    panel_label,
    panel_title,
):

    # --------------------------------------------------------
# Collect all points for Pareto classification
# --------------------------------------------------------

    pareto_points = []

    for model_name, model_data in models.items():
        x_vals = model_data[utility_metric]["y"]
        y_vals = model_data[robustness_metric]["y"]

        for x_i, y_i in zip(x_vals, y_vals):
            pareto_points.append((x_i, y_i))

    all_x = []
    all_y = []

    # --------------------------------------------------------
    # Plot each configuration
    # --------------------------------------------------------

    for model_name, model_data in models.items():

        s = model_data["s"]

        x = model_data[utility_metric]["y"]
        y = model_data[robustness_metric]["y"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        # ----------------------------------------------------
        # Connect successive sweep settings
        # ----------------------------------------------------

        ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.4,
            color=color,
            alpha=0.75,
            zorder=1,
        )

        # ----------------------------------------------------
        # Mean trade-off points
        # ----------------------------------------------------

        # ax.scatter(
        #     x,
        #     y,
        #     marker=marker,
        #     s=42,
        #     color=color,
        #     zorder=3,
        # )
        for x_i, y_i in zip(x, y):

            non_dominated = is_non_dominated(
                x_i,
                y_i,
                x,
                y,
            )

            if non_dominated:
                # Filled marker
                ax.scatter(
                    x_i,
                    y_i,
                    marker=marker,
                    s=42,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=3,
                )

            else:
                # Empty / hollow marker
                ax.scatter(
                    x_i,
                    y_i,
                    marker=marker,
                    s=42,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=3,
                )


        # ----------------------------------------------------
        # Label each point by multiplier s
        # ----------------------------------------------------

        for x_i, y_i, s_i in zip(x, y, s):

            offset = ANNOTATION_OFFSETS.get(
                (utility_metric, model_name, float(s_i)),
                (5, 5),
            )

            ax.annotate(
                f"{s_i:g}",
                xy=(x_i, y_i),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                color=color,
                zorder=4,
            )

        all_x.extend(x)
        all_y.extend(y)

    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    xmin = min(all_x)
    xmax = max(all_x)

    ymin = min(all_y)
    ymax = max(all_y)

    xpadding = max(
        (xmax - xmin) * 0.07,
        0.5,
    )

    ypadding = max(
        (ymax - ymin) * 0.08,
        1.0,
    )

    ax.set_xlim(
        max(0, xmin - xpadding),
        min(100, xmax + xpadding),
    )

    ax.set_ylim(
        max(0, ymin - ypadding),
        min(100, ymax + ypadding),
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    if utility_metric == "Harmless":

        ax.set_xlabel(
            "Harmless successful response rate (%)"
        )

    elif utility_metric == "MMLU":

        ax.set_xlabel(
            "MMLU accuracy (%)"
        )

    ax.set_ylabel(
        f"{robustness_metric} ASR (%)"
    )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        rf"(${panel_label}$) {panel_title}",
        # fontsize=11,
    )

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.grid(
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # --------------------------------------------------------
    # Desirable trade-off direction
    # --------------------------------------------------------

    add_direction_annotation(ax)


# ============================================================
# Shared model legend
# ============================================================

def get_model_handles():

    return [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle="-",
            linewidth=1.4,
            markersize=6,
            label=name,
        )
        for name in models.keys()
    ]


def get_pareto_handles():

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=7,
            label="Non-dominated",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="0.25",
            markeredgewidth=1.2,
            markersize=7,
            label="Dominated",
        ),
    ]


# ============================================================
# Page 1:
# (a) Harmless x GCG
# (b) Harmless x Soft-Prompt
# ============================================================

def plot_tradeoff_page_1():

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
    )

    plot_tradeoff_panel(
        axes[0],
        utility_metric="Harmless",
        robustness_metric="GCG",
        panel_label="a",
        panel_title="Harmless vs GCG",
    )

    plot_tradeoff_panel(
        axes[1],
        utility_metric="Harmless",
        robustness_metric="Soft-Prompt",
        panel_label="b",
        panel_title="Harmless vs Soft-Prompt",
    )

    # First row: models
    fig.legend(
        handles=get_model_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
    )

    # Second row: Pareto status
    fig.legend(
        handles=get_pareto_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.89]
    )

    fig.subplots_adjust(
        hspace=0.45
    )

    return fig


# ============================================================
# Page 2:
# (c) MMLU x GCG
# (d) MMLU x Soft-Prompt
# ============================================================

def plot_tradeoff_page_2():

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
    )

    plot_tradeoff_panel(
        axes[0],
        utility_metric="MMLU",
        robustness_metric="GCG",
        panel_label="c",
        panel_title="MMLU vs GCG",
    )

    plot_tradeoff_panel(
        axes[1],
        utility_metric="MMLU",
        robustness_metric="Soft-Prompt",
        panel_label="d",
        panel_title="MMLU vs Soft-Prompt",
    )

    # First row: models
    fig.legend(
        handles=get_model_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
    )

    # Second row: Pareto status
    fig.legend(
        handles=get_pareto_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.89]
    )

    fig.subplots_adjust(
        hspace=0.45
    )

    return fig


# ============================================================
# Generate figures
# ============================================================

page_1 = plot_tradeoff_page_1()
page_2 = plot_tradeoff_page_2()


# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

pdf_path = (
    output_dir / "tradeoff_analysis.pdf"
)


# ============================================================
# Save two-page PDF
# ============================================================

with PdfPages(pdf_path) as pdf:

    pdf.savefig(
        page_1,
        bbox_inches="tight",
    )

    pdf.savefig(
        page_2,
        bbox_inches="tight",
    )


# ============================================================
# Report outputs
# ============================================================

print(
    "Saved trade-off analysis to:\n"
    f"  {pdf_path}\n\n"
)


# ============================================================
# Display
# ============================================================

plt.show()
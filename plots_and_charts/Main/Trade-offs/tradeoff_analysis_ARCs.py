import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


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
# Only means are used in these trade-off plots.
#
# Each configuration contains:
#   s
#   ARC-E
#   ARC-C
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

        "ARC-E": {
            "y": np.array([
                79.00,
                78.52,
                78.04,
                77.08,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                51.17,
                51.08,
                50.94,
                49.32,
            ]),
        },

        "GCG": {
            "y": np.array([
                19.17,
                17.50,
                18.33,
                9.17,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                43.44,
                43.40,
                42.45,
                51.13,
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

        "ARC-E": {
            "y": np.array([
                80.95,
                80.80,
                80.78,
                79.45,
                78.05,
                74.99,
                73.18,
                47.97,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                51.77,
                51.45,
                51.31,
                49.69,
                49.72,
                47.32,
                46.22,
                37.37,
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

        "ARC-E": {
            "y": np.array([
                80.46,
                79.44,
                75.45,
                41.29,
            ]),
        },

        "ARC-C": {
            "y": np.array([
                50.85,
                49.34,
                47.84,
                32.37,
            ]),
        },

        "GCG": {
            "y": np.array([
                25.83,
                12.50,
                17.50,
                25.00,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                55.47,
                43.23,
                58.18,
                63.21,
            ]),
        },
    },
}


# ============================================================
# Annotation offsets
#
# Used only to prevent overlapping s labels.
#
# Key:
#   (utility metric, robustness metric, model, s)
#
# Values:
#   (horizontal offset, vertical offset), in display points.
#
# Any point not listed here uses DEFAULT_ANNOTATION_OFFSET.
# ============================================================

DEFAULT_ANNOTATION_OFFSET = (5, 5)

ANNOTATION_OFFSETS = {

    # --------------------------------------------------------
    # ARC-Easy vs GCG
    # --------------------------------------------------------

    ("ARC-E", "GCG", "Zephyr", 0.05): (6, 8),
    ("ARC-E", "GCG", "Zephyr", 0.10): (6, -12),
    ("ARC-E", "GCG", "Zephyr", 0.20): (6, 5),
    ("ARC-E", "GCG", "Zephyr", 0.40): (6, 5),

    ("ARC-E", "GCG", "Llama 3 (Llama 2)", 0.05): (6, 7),
    ("ARC-E", "GCG", "Llama 3 (Llama 2)", 0.075): (-17, 8),
    ("ARC-E", "GCG", "Llama 3 (Llama 2)", 0.10): (6, -12),

    ("ARC-E", "GCG", "Llama 3 (MixAT)", 0.075): (7, 6),
    ("ARC-E", "GCG", "Llama 3 (MixAT)", 0.15): (-4, 8),

    # --------------------------------------------------------
    # ARC-Easy vs Soft-Prompt
    # --------------------------------------------------------

    ("ARC-E", "Soft-Prompt", "Zephyr", 0.05): (6, 8),
    ("ARC-E", "Soft-Prompt", "Zephyr", 0.10): (6, -12),
    ("ARC-E", "Soft-Prompt", "Zephyr", 0.20): (6, 5),
    ("ARC-E", "Soft-Prompt", "Zephyr", 0.40): (6, 5),

    ("ARC-E", "Soft-Prompt", "Llama 3 (Llama 2)", 0.05): (6, 7),
    ("ARC-E", "Soft-Prompt", "Llama 3 (Llama 2)", 0.075): (-17, 8),
    ("ARC-E", "Soft-Prompt", "Llama 3 (Llama 2)", 0.10): (6, -12),

    ("ARC-E", "Soft-Prompt", "Llama 3 (MixAT)", 0.075): (7, 6),
    ("ARC-E", "Soft-Prompt", "Llama 3 (MixAT)", 0.15): (-4, 8),

    # --------------------------------------------------------
    # ARC-Challenge vs GCG
    # --------------------------------------------------------

    ("ARC-C", "GCG", "Zephyr", 0.05): (6, 8),
    ("ARC-C", "GCG", "Zephyr", 0.10): (6, -12),
    ("ARC-C", "GCG", "Zephyr", 0.20): (6, 5),
    ("ARC-C", "GCG", "Zephyr", 0.40): (6, 5),

    ("ARC-C", "GCG", "Llama 3 (Llama 2)", 0.05): (6, 7),
    ("ARC-C", "GCG", "Llama 3 (Llama 2)", 0.075): (-17, 8),
    ("ARC-C", "GCG", "Llama 3 (Llama 2)", 0.10): (6, -12),

    ("ARC-C", "GCG", "Llama 3 (MixAT)", 0.075): (7, 6),
    ("ARC-C", "GCG", "Llama 3 (MixAT)", 0.15): (-4, 8),

    # --------------------------------------------------------
    # ARC-Challenge vs Soft-Prompt
    # --------------------------------------------------------

    ("ARC-C", "Soft-Prompt", "Zephyr", 0.05): (6, 8),
    ("ARC-C", "Soft-Prompt", "Zephyr", 0.10): (6, -12),
    ("ARC-C", "Soft-Prompt", "Zephyr", 0.20): (6, 5),
    ("ARC-C", "Soft-Prompt", "Zephyr", 0.40): (6, 5),

    ("ARC-C", "Soft-Prompt", "Llama 3 (Llama 2)", 0.05): (6, 7),
    ("ARC-C", "Soft-Prompt", "Llama 3 (Llama 2)", 0.075): (-17, 8),
    ("ARC-C", "Soft-Prompt", "Llama 3 (Llama 2)", 0.10): (6, -12),

    ("ARC-C", "Soft-Prompt", "Llama 3 (MixAT)", 0.075): (7, 6),
    ("ARC-C", "Soft-Prompt", "Llama 3 (MixAT)", 0.15): (-4, 8),
}


# ============================================================
# Desirable-direction annotation
# ============================================================

def add_direction_annotation(ax):

    ax.text(
        0.98,
        0.96,
        "Utility improves \u2192\nRobustness improves \u2193",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
        linespacing=1.25,
        zorder=5,
    )


# ============================================================
# Plot one trade-off panel
# ============================================================

def plot_tradeoff_panel(
    ax,
    utility_metric,
    robustness_metric,
    panel_label,
    panel_title,
):

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
        #
        # The line shows the trajectory through trade-off
        # space as s increases. It is a visual guide only.
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

        ax.scatter(
            x,
            y,
            marker=marker,
            s=42,
            color=color,
            zorder=3,
        )

        # ----------------------------------------------------
        # Label each point by multiplier s
        # ----------------------------------------------------

        for x_i, y_i, s_i in zip(x, y, s):

            key = (
                utility_metric,
                robustness_metric,
                model_name,
                float(s_i),
            )

            offset = ANNOTATION_OFFSETS.get(
                key,
                DEFAULT_ANNOTATION_OFFSET,
            )

            ax.annotate(
                f"{s_i:g}",
                xy=(x_i, y_i),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
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

    if utility_metric == "ARC-E":

        ax.set_xlabel(
            "ARC-Easy accuracy (%)"
        )

    elif utility_metric == "ARC-C":

        ax.set_xlabel(
            "ARC-Challenge accuracy (%)"
        )

    ax.set_ylabel(
        f"{robustness_metric} ASR (%)"
    )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        rf"(${panel_label}$) {panel_title}",
        fontsize=11,
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


# ============================================================
# Page 1:
# (a) ARC-Easy x GCG
# (b) ARC-Easy x Soft-Prompt
# ============================================================

def plot_tradeoff_page_1():

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
    )

    plot_tradeoff_panel(
        axes[0],
        utility_metric="ARC-E",
        robustness_metric="GCG",
        panel_label="a",
        panel_title="ARC-Easy vs GCG",
    )

    plot_tradeoff_panel(
        axes[1],
        utility_metric="ARC-E",
        robustness_metric="Soft-Prompt",
        panel_label="b",
        panel_title="ARC-Easy vs Soft-Prompt",
    )

    fig.legend(
        handles=get_model_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.94]
    )

    fig.subplots_adjust(
        hspace=0.45
    )

    return fig


# ============================================================
# Page 2:
# (c) ARC-Challenge x GCG
# (d) ARC-Challenge x Soft-Prompt
# ============================================================

def plot_tradeoff_page_2():

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
    )

    plot_tradeoff_panel(
        axes[0],
        utility_metric="ARC-C",
        robustness_metric="GCG",
        panel_label="c",
        panel_title="ARC-Challenge vs GCG",
    )

    plot_tradeoff_panel(
        axes[1],
        utility_metric="ARC-C",
        robustness_metric="Soft-Prompt",
        panel_label="d",
        panel_title="ARC-Challenge vs Soft-Prompt",
    )

    fig.legend(
        handles=get_model_handles(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.94]
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
    output_dir / "tradeoff_analysis_ARCs.pdf"
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
    "Saved ARC trade-off analysis to:\n"
    f"  {pdf_path}\n\n"
)


# ============================================================
# Display
# ============================================================

plt.show()
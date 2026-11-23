import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"

rows = [
    "only UltraChat",
    "OR (100)",
    "OR (500)",
    "OR (1000)",
    "trimmed OR (1000)",
    "OR (3000)",
]

# Same colors across ALL experiments/scripts
colors = {
    "only UltraChat":      "C0",
    "OR (100)":            "C1",
    "OR (500)":            "C2",
    "OR (1000)":           "C3",
    "trimmed OR (1000)":   "C4",
    "OR (3000)":           "C5",
}

labels = {
    "only UltraChat": "UltraChat",
    "OR (100)": "OR\n100",
    "OR (500)": "OR\n500",
    "OR (1000)": "OR\n1000",
    "trimmed OR (1000)": "Trimmed OR\n1000",
    "OR (3000)": "OR\n3000",
}

eps = np.array([0.075, 0.150, 0.300, 0.600])


# ============================================================
# SAFE COMPLIANCE
# ============================================================

safe_github = {
    "only UltraChat":      (49.87, 8.39),
    "OR (100)":            (58.27, 7.51),
    "OR (500)":            (50.13, 22.36),
    "OR (1000)":           (55.20, 7.23),
    "trimmed OR (1000)":   None,
    "OR (3000)":           (71.07, 19.59),
}

safe_references = {
    "Original Llama 3": 97.20,
    "HuggingFace CAT Llama 3": 28.00,
}

safe_relative = {
    "only UltraChat": {
        "y":   [51.60, 51.07, 60.27, 37.33],
        "err": [15.53, 10.50, 10.01, 10.92],
    },
    "OR (100)": {
        "y":   [6.40, 68.00, 6.40, 4.67],
        "err": [None, None, None, 4.31],
    },
    "OR (500)": {
        "y":   [4.80, 90.80, 22.40, 2.00],
        "err": [None, None, None, 1.06],
    },
    "OR (1000)": {
        "y":   [78.40, 86.00, 53.60, 0.53],
        "err": [None, None, None, 0.61],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 6.80, 9.07],
        "err": [None, None, None, 13.44],
    },
    "OR (3000)": {
        "y":   [5.20, 70.40, 2.80, 0.67],
        "err": [None, None, None, 0.46],
    },
}


# ============================================================
# UNSAFE REFUSAL
# ============================================================

unsafe_github = {
    "only UltraChat":      (99.50, 0.00),
    "OR (100)":            (82.50, 27.71),
    "OR (500)":            (94.33, 7.29),
    "OR (1000)":           (98.67, 0.29),
    "trimmed OR (1000)":   None,
    "OR (3000)":           (92.17, 12.29),
}

unsafe_references = {
    "Original Llama 3": 91.50,
    "HuggingFace CAT Llama 3": 99.50,
}

unsafe_relative = {
    "only UltraChat": {
        "y":   [99.50, 99.33, 98.50, 97.00],
        "err": [0.00, 0.29, 0.87, 1.80],
    },
    "OR (100)": {
        "y":   [99.50, 91.50, 99.00, 98.50],
        "err": [None, None, None, 0.50],
    },
    "OR (500)": {
        "y":   [99.50, 77.00, 95.00, 82.17],
        "err": [None, None, None, 30.46],
    },
    "OR (1000)": {
        "y":   [91.50, 91.00, 87.00, 100.00],
        "err": [None, None, None, 0.00],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 99.00, 97.83],
        "err": [None, None, None, 3.75],
    },
    "OR (3000)": {
        "y":   [100.00, 92.00, 99.50, 100.00],
        "err": [None, None, None, 0.00],
    },
}


# ============================================================
# Common styling
# ============================================================

def style_axis(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("XSTest score")
    ax.set_ylim(0, 110)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Representation plot
#
# Safe compliance + unsafe refusal on SAME plot.
# Two bars per training configuration.
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(11, 6))

    # Trimmed OR has no GitHub value for either metric
    available_rows = [
        row for row in rows
        if (
            safe_github.get(row) is not None
            or unsafe_github.get(row) is not None
        )
    ]

    x = np.arange(len(available_rows))

    width = 0.35

    # --------------------------------------------------------
    # Safe compliance
    # --------------------------------------------------------

    safe_values = [
        safe_github[row][0]
        if safe_github.get(row) is not None
        else np.nan
        for row in available_rows
    ]

    safe_errors = [
        safe_github[row][1]
        if safe_github.get(row) is not None
        and safe_github[row][1] is not None
        else 0
        for row in available_rows
    ]

    # --------------------------------------------------------
    # Unsafe refusal
    # --------------------------------------------------------

    unsafe_values = [
        unsafe_github[row][0]
        if unsafe_github.get(row) is not None
        else np.nan
        for row in available_rows
    ]

    unsafe_errors = [
        unsafe_github[row][1]
        if unsafe_github.get(row) is not None
        and unsafe_github[row][1] is not None
        else 0
        for row in available_rows
    ]

    # --------------------------------------------------------
    # Draw bars
    #
    # Color still represents the training configuration.
    # Hatching represents safe vs unsafe.
    # --------------------------------------------------------

    safe_bars = ax.bar(
        x - width / 2,
        safe_values,
        width,
        yerr=safe_errors,
        capsize=4,
        color=[colors[row] for row in available_rows],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )

    unsafe_bars = ax.bar(
        x + width / 2,
        unsafe_values,
        width,
        yerr=unsafe_errors,
        capsize=4,
        color=[colors[row] for row in available_rows],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
    )

    # --------------------------------------------------------
    # Numerical labels
    # --------------------------------------------------------

    for bar, value in zip(safe_bars, safe_values):
        if np.isnan(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar, value in zip(unsafe_bars, unsafe_values):
        if np.isnan(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # --------------------------------------------------------
    # Reference lines
    #
    # Safe references use solid/dashed black.
    # Unsafe references use dash-dot/dotted gray.
    # --------------------------------------------------------

    ax.axhline(
        safe_references["Original Llama 3"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Safe — Original Llama 3 "
            f"({safe_references['Original Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        safe_references["HuggingFace CAT Llama 3"],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Safe — HF CAT Llama 3 "
            f"({safe_references['HuggingFace CAT Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["Original Llama 3"],
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Unsafe — Original Llama 3 "
            f"({unsafe_references['Original Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["HuggingFace CAT Llama 3"],
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Unsafe — HF CAT Llama 3 "
            f"({unsafe_references['HuggingFace CAT Llama 3']:.2f})"
        ),
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    style_axis(
        ax,
        f"XSTest — {MODEL_NAME} representation",
    )

    # --------------------------------------------------------
    # Metric legend
    # --------------------------------------------------------

    metric_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=10,
            markerfacecolor="gray",
            markeredgecolor="black",
            label="Safe compliance",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=10,
            markerfacecolor="lightgray",
            markeredgecolor="black",
            label="Unsafe refusal",
        ),
    ]

    # First legend: safe/unsafe
    metric_legend = ax.legend(
        handles=metric_handles,
        frameon=False,
        loc="lower left",
    )

    ax.add_artist(metric_legend)

    # Second legend: references
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Relative learning-rate plot
#
# Safe compliance and unsafe refusal on SAME axes.
#
# Color       = training configuration
# solid/o     = safe compliance
# dashed/s    = unsafe refusal
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(11, 7))

    for row in rows:

        # ====================================================
        # Safe compliance
        # ====================================================

        safe_y = np.array([
            np.nan if v is None else float(v)
            for v in safe_relative[row]["y"]
        ])

        if not np.all(np.isnan(safe_y)):

            ax.plot(
                eps,
                safe_y,
                marker="o",
                markersize=7,
                linewidth=2,
                linestyle="-",
                color=colors[row],
                label=labels[row].replace("\n", " "),
            )

            for x_i, y_i, err_i in zip(
                eps,
                safe_relative[row]["y"],
                safe_relative[row]["err"],
            ):
                if y_i is None or err_i is None:
                    continue

                ax.errorbar(
                    x_i,
                    y_i,
                    yerr=err_i,
                    fmt="none",
                    capsize=4,
                    color=colors[row],
                )

        # ====================================================
        # Unsafe refusal
        # ====================================================

        unsafe_y = np.array([
            np.nan if v is None else float(v)
            for v in unsafe_relative[row]["y"]
        ])

        if not np.all(np.isnan(unsafe_y)):

            ax.plot(
                eps,
                unsafe_y,
                marker="s",
                markersize=6,
                linewidth=2,
                linestyle="--",
                color=colors[row],
                label="_nolegend_",
            )

            for x_i, y_i, err_i in zip(
                eps,
                unsafe_relative[row]["y"],
                unsafe_relative[row]["err"],
            ):
                if y_i is None or err_i is None:
                    continue

                ax.errorbar(
                    x_i,
                    y_i,
                    yerr=err_i,
                    fmt="none",
                    capsize=4,
                    color=colors[row],
                )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Self $\epsilon$ (relative lr)"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

    style_axis(
        ax,
        f"XSTest — {MODEL_NAME} relative learning rate",
    )

    # --------------------------------------------------------
    # First legend:
    # training configuration / colors
    # --------------------------------------------------------

    config_legend = ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Training data",
    )

    ax.add_artist(config_legend)

    # --------------------------------------------------------
    # Second legend:
    # safe vs unsafe
    # --------------------------------------------------------

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Safe compliance",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Unsafe refusal",
        ),
    ]

    ax.legend(
        handles=metric_handles,
        frameon=False,
        bbox_to_anchor=(1.02, 0.53),
        loc="upper left",
        title="Metric",
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Generate figures
# ============================================================

figures = []


# 1. GitHub representation
fig, ax = plot_representation()
figures.append(fig)


# 2. Relative learning rate
fig, ax = plot_relative_lr()
figures.append(fig)


# ============================================================
# Save ALL figures to ONE multipage PDF
#
# Example:
#
#     xstest_mixat.py
#
# creates:
#
#     xstest_mixat.pdf
#
# in the same directory as the script.
# ============================================================

script_path = Path(__file__).resolve()
pdf_path = script_path.with_suffix(".pdf")

with PdfPages(pdf_path) as pdf:
    for fig in figures:
        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

print(
    f"Saved {len(figures)} figures to:\n"
    f"{pdf_path}"
)


# ============================================================
# Display
# ============================================================

plt.show()
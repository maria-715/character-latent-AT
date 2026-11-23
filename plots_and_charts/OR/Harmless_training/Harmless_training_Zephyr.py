import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Zephyr"

rows = [
    "only UltraChat",
    "OR (1000)",
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


# ============================================================
# GitHub CAT Zephyr representation
# ============================================================

github = {
    "only UltraChat": (62.30, 3.61),
    "OR (1000)":      None,  # P
}

references = {
    "Original Zephyr": 98.40,
    "HuggingFace CAT Zephyr": 68.63,
}


# ============================================================
# Relative learning-rate representation
# ============================================================

eps = np.array([0.05, 0.10, 0.20, 0.40])

relative = {
    "only UltraChat": {
        "y":   [66.67, 64.64, 55.14, 61.97],
        "err": [6.35, 1.74, 7.06, 10.39],
    },

    "OR (1000)": {
        "y":   [98.80, 98.50, 98.30, 98.00],
        "err": [None, None, None, None],
    },
}


# ============================================================
# Styling
# ============================================================

def style_axis(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Harmless score")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Representation bar chart
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(8, 6))

    available_rows = [
        row for row in rows
        if github.get(row) is not None
    ]

    x = np.arange(len(available_rows))

    values = [
        github[row][0]
        for row in available_rows
    ]

    errors = [
        github[row][1] if github[row][1] is not None else 0
        for row in available_rows
    ]

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        width=0.6,
        color=[colors[row] for row in available_rows],
        alpha=0.85,
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    original = references["Original Zephyr"]
    hf = references["HuggingFace CAT Zephyr"]

    ax.axhline(
        original,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Original Zephyr ({original:.2f})",
    )

    ax.axhline(
        hf,
        color="black",
        linestyle=":",
        linewidth=2.5,
        label=f"HF CAT Zephyr ({hf:.2f})",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = {
        "only UltraChat": "UltraChat",
        "OR (1000)": "OR\n1000",
    }

    ax.set_xticks(x)
    ax.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    # Put numerical value above each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    style_axis(
        ax,
        f"Harmless — {MODEL_NAME} representation — training set",
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Relative-lr plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:

        y = np.array([
            np.nan if v is None else float(v)
            for v in relative[row]["y"]
        ])

        # Skip completely missing series
        if np.all(np.isnan(y)):
            continue

        # ----------------------------------------------------
        # Line + markers
        # ----------------------------------------------------

        ax.plot(
            eps,
            y,
            marker="o",
            markersize=7,
            linewidth=2,
            color=colors[row],
            label=row,
        )

        # ----------------------------------------------------
        # Error bars where reported
        # ----------------------------------------------------

        for x_i, y_i, err_i in zip(
            eps,
            relative[row]["y"],
            relative[row]["err"],
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

    ax.set_xlabel(r"Self $\epsilon$ (relative lr)")

    ax.set_xticks(eps)
    ax.set_xticklabels([
        "0.05",
        "0.10",
        "0.20",
        "0.40",
    ])

    style_axis(
        ax,
        f"Harmless — {MODEL_NAME} relative learning rate — training set",
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Generate plots
# ============================================================

figures = []


# 1. Representation
fig, ax = plot_representation()
figures.append(fig)


# 2. Relative learning rate
fig, ax = plot_relative_lr()
figures.append(fig)


# ============================================================
# Save ALL figures to ONE multipage PDF
#
# The PDF has the same name and location as this script.
#
# Example:
#
#     harmless_training_zephyr.py
#
# creates:
#
#     harmless_training_zephyr.pdf
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
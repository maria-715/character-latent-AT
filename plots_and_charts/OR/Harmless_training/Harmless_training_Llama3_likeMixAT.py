import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"

rows = [
    "only UltraChat",
    "OR (1000)",
    "trimmed OR (1000)",
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
# GitHub Llama 3 representation
# ============================================================

github = {
    "only UltraChat":      (66.90, 15.55),
    "OR (1000)":           None,  # P
    "trimmed OR (1000)":   None,  # -
}

references = {
    "Original Llama 3": 91.51,
    "HuggingFace CAT Llama 3": 41.46,
}


# ============================================================
# Relative learning-rate representation
# ============================================================

eps = np.array([0.075, 0.150, 0.300, 0.600])

relative = {
    "only UltraChat": {
        "y":   [70.73, 55.18, 50.18, 46.69],
        "err": [1.06, 7.95, 1.78, 11.14],
    },

    "OR (1000)": {
        "y":   [98.90, 97.10, 93.81, 40.99],
        "err": [None, None, None, 36.14],
    },

    "trimmed OR (1000)": {
        "y":   [None, None, 97.50, 53.15],
        "err": [None, None, None, 45.52],
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

    original = references["Original Llama 3"]
    hf = references["HuggingFace CAT Llama 3"]

    ax.axhline(
        original,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Original Llama 3 ({original:.2f})",
    )

    ax.axhline(
        hf,
        color="black",
        linestyle=":",
        linewidth=2.5,
        label=f"HF CAT Llama 3 ({hf:.2f})",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = {
        "only UltraChat": "UltraChat",
        "OR (1000)": "OR\n1000",
        "trimmed OR (1000)": "Trimmed OR\n1000",
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
        "0.075",
        "0.150",
        "0.300",
        "0.600",
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
# If the script is:
#
#     harmless_training_mixat.py
#
# the output is:
#
#     harmless_training_mixat.pdf
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
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


# ============================================================
# GitHub Llama 3 representation
# ============================================================

github_40 = {
    "only UltraChat":      (86.67, 7.64),
    "OR (100)":            (76.67, 6.29),
    "OR (500)":            (69.17, 8.78),
    "OR (1000)":           (71.67, 2.89),
    "trimmed OR (1000)":   None,
    "OR (3000)":           (76.67, 8.78),
}

github_78k = {
    "only UltraChat":      (55.24, 0.71),
    "OR (100)":            (91.99, 3.86),
    "OR (500)":            (96.32, 0.37),
    "OR (1000)":           (97.09, 0.36),
    "trimmed OR (1000)":   None,
    "OR (3000)":           None,  # P on 78k
}


# ============================================================
# Reference points
# ============================================================

references_40 = {
    "Original Llama 3": (100.00, None),
    "HuggingFace CAT Llama 3": (77.50, None),
}

references_78k = {
    "Original Llama 3": (93.16, None),
    "HuggingFace CAT Llama 3": (41.64, None),
}


# ============================================================
# Relative learning-rate representation
# ============================================================

eps = np.array([0.075, 0.150, 0.300, 0.600])

relative_40 = {
    "only UltraChat": {
        "y":   [82.50, 58.33, 21.67, 51.67],
        "err": [13.75, 14.55, 16.32, 38.80],
    },
    "OR (100)": {
        "y":   [72.50, 62.50, 5.00, 5.83],
        "err": [None, None, None, 1.44],
    },
    "OR (500)": {
        "y":   [77.50, 37.50, 7.50, 9.17],
        "err": [None, None, None, 7.22],
    },
    "OR (1000)": {
        "y":   [97.50, 45.00, 15.00, 5.00],
        "err": [None, None, None, 0.00],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 5.00, 5.00],
        "err": [None, None, None, 0.00],
    },
    "OR (3000)": {
        "y":   [77.50, 52.50, 5.00, 5.00],
        "err": [None, None, None, 0.00],
    },
}

relative_78k = {
    "only UltraChat": {
        "y":   [56.18, 46.74, 56.85, 14.60],
        "err": [7.43, 5.94, 8.20, 12.40],
    },
    "OR (100)": {
        "y":   [91.96, 95.02, 77.95, 44.80],
        "err": [None, None, None, 28.82],
    },
    "OR (500)": {
        "y":   [96.65, 96.06, 95.53, 55.16],
        "err": [None, None, None, 17.07],
    },
    "OR (1000)": {
        "y":   [98.40, 96.51, 93.47, 42.18],
        "err": [None, None, None, 36.15],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 96.73, 53.73],
        "err": [None, None, None, 45.49],
    },
    "OR (3000)": {
        "y":   [None, None, None, None],  # P on 78k
        "err": [None, None, None, None],
    },
}


# ============================================================
# Plotting helpers
# ============================================================

def style_axis(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Harmless score")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_github(data, references, title):
    """
    Bar plot for the GitHub Llama 3 representation.

    Each training configuration is a bar.
    Original Llama 3 and HuggingFace CAT Llama 3 are shown
    as horizontal reference lines.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Keep only rows with a numeric value
    available_rows = [
        row for row in rows
        if data.get(row) is not None
    ]

    x = np.arange(len(available_rows))

    values = [
        data[row][0]
        for row in available_rows
    ]

    errors = [
        data[row][1] if data[row][1] is not None else 0
        for row in available_rows
    ]

    bar_colors = [
        colors[row]
        for row in available_rows
    ]

    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        color=bar_colors,
        alpha=0.85,
        width=0.7,
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    original = references["Original Llama 3"][0]
    hf_cat = references["HuggingFace CAT Llama 3"][0]

    ax.axhline(
        original,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"Original Llama 3 ({original:.2f})",
    )

    ax.axhline(
        hf_cat,
        linestyle=":",
        linewidth=2.5,
        color="black",
        label=f"HF CAT Llama 3 ({hf_cat:.2f})",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = {
        "only UltraChat": "UltraChat",
        "OR (100)": "OR\n100",
        "OR (500)": "OR\n500",
        "OR (1000)": "OR\n1000",
        "trimmed OR (1000)": "Trimmed OR\n1000",
        "OR (3000)": "OR\n3000",
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
            bar.get_height() + 2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    style_axis(ax, title)

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()

    return fig, ax


def plot_relative_lr(data, title):
    """
    Plot score versus relative-learning-rate epsilon.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for row in rows:

        y_raw = data[row]["y"]
        err_raw = data[row]["err"]

        # Convert missing values to NaN
        y = np.array([
            np.nan if v is None else float(v)
            for v in y_raw
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
            y_raw,
            err_raw,
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
        f"{x:.3f}"
        for x in eps
    ])

    style_axis(ax, title)

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Generate plots
# ============================================================

figures = []


# 1. GitHub representation — on 40
fig, ax = plot_github(
    github_40,
    references_40,
    f"Harmless — {MODEL_NAME} representation — on 40",
)
figures.append(fig)


# 2. GitHub representation — on 78k
fig, ax = plot_github(
    github_78k,
    references_78k,
    f"Harmless — {MODEL_NAME} representation — on 78k",
)
figures.append(fig)


# 3. Relative learning rate — on 40
fig, ax = plot_relative_lr(
    relative_40,
    f"Harmless — {MODEL_NAME} relative learning rate — on 40",
)
figures.append(fig)


# 4. Relative learning rate — on 78k
fig, ax = plot_relative_lr(
    relative_78k,
    f"Harmless — {MODEL_NAME} relative learning rate — on 78k",
)
figures.append(fig)


# ============================================================
# Save ALL figures to ONE multipage PDF
#
# If this script is:
#
#     harmless_mixat.py
#
# the output will automatically be:
#
#     harmless_mixat.pdf
#
# in the same directory.
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
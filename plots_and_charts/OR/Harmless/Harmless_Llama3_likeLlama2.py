import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"

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

github_40 = {
    "only UltraChat":      None,          # P on 40
    "OR (1000)":           (92.50, 4.33),
    "trimmed OR (1000)":   None,          # -
}

github_78k = {
    "only UltraChat":      None,          # P on 78k
    "OR (1000)":           (97.50, 0.41),
    "trimmed OR (1000)":   None,          # -
}

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
        "y":   [90.83, 46.67, 10.83, 5.00],
        "err": [7.22, 25.24, 10.10, 0.00],
    },
    "OR (1000)": {
        "y":   [82.50, 55.00, 5.00, 5.00],
        "err": [None, None, None, 0.00],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 15.00, 39.17],
        "err": [None, None, None, 46.86],
    },
}

relative_78k = {
    "only UltraChat": {
        "y":   [61.09, 65.19, 59.01, 36.38],
        "err": [0.52, 28.54, 14.42, 15.54],
    },
    "OR (1000)": {
        "y":   [97.99, 98.25, 97.04, 82.61],
        "err": [None, None, None, 18.07],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 96.59, 64.39],
        "err": [None, None, None, 41.82],
    },
}


# ============================================================
# Plotting
# ============================================================

def style_axis(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Harmless score")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_representation(data, references, title):
    fig, ax = plt.subplots(figsize=(8, 6))

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

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        color=[colors[row] for row in available_rows],
        alpha=0.85,
        width=0.6,
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    original_name = "Original Llama 3"
    hf_name = "HuggingFace CAT Llama 3"

    original = references[original_name][0]
    hf_cat = references[hf_name][0]

    ax.axhline(
        original,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"{original_name} ({original:.2f})",
    )

    ax.axhline(
        hf_cat,
        linestyle=":",
        linewidth=2.5,
        color="black",
        label=f"{hf_name} ({hf_cat:.2f})",
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

    # Values above bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    style_axis(ax, title)

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()

    return fig, ax


def plot_relative_lr(data, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:

        y = np.array([
            np.nan if v is None else float(v)
            for v in data[row]["y"]
        ])

        # Skip completely missing series
        if np.all(np.isnan(y)):
            continue

        ax.plot(
            eps,
            y,
            marker="o",
            markersize=7,
            linewidth=2,
            color=colors[row],
            label=row,
        )

        # Error bars only where reported
        for x_i, y_i, err_i in zip(
            eps,
            data[row]["y"],
            data[row]["err"],
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
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Generate plots
# ============================================================

figures = []


# 1. Representation — on 40
fig, ax = plot_representation(
    github_40,
    references_40,
    f"Harmless — {MODEL_NAME} representation — on 40",
)
figures.append(fig)


# 2. Representation — on 78k
fig, ax = plot_representation(
    github_78k,
    references_78k,
    f"Harmless — {MODEL_NAME} representation — on 78k",
)
figures.append(fig)


# 3. Relative lr — on 40
fig, ax = plot_relative_lr(
    relative_40,
    f"Harmless — {MODEL_NAME} relative learning rate — on 40",
)
figures.append(fig)


# 4. Relative lr — on 78k
fig, ax = plot_relative_lr(
    relative_78k,
    f"Harmless — {MODEL_NAME} relative learning rate — on 78k",
)
figures.append(fig)


# ============================================================
# Save ALL plots to ONE multipage PDF
#
# Example:
#
#   harmless_llama2.py
#       ->
#   harmless_llama2.pdf
#
# The PDF is saved in the same directory as the Python script.
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
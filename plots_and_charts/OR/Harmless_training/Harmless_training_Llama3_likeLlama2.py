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
# Representation
# ============================================================

github = {
    "only UltraChat":      None,  # P
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
    # All P in the table
    "only UltraChat": {
        "y":   [None, None, None, None],
        "err": [None, None, None, None],
    },

    "OR (1000)": {
        "y":   [98.80, 98.40, 97.60, 82.52],
        "err": [None, None, None, 19.04],
    },

    "trimmed OR (1000)": {
        "y":   [None, None, 96.80, 64.83],
        "err": [None, None, None, 41.78],
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
# Relative-lr plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:

        y = np.array([
            np.nan if v is None else float(v)
            for v in relative[row]["y"]
        ])

        # Don't add a completely missing series
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

fig, ax = plot_relative_lr()
figures.append(fig)


# ============================================================
# Save ALL figures to ONE multipage PDF
#
# Example:
#
#     harmless_training_llama2.py
#
# creates:
#
#     harmless_training_llama2.pdf
#
# in the same directory as the Python script.
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
    f"Saved {len(figures)} figure(s) to:\n"
    f"{pdf_path}"
)


# ============================================================
# Display
# ============================================================

plt.show()
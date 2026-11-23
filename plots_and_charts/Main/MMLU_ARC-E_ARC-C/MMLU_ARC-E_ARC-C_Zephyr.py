import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Zephyr"
COLOR = "C0"

eps = np.array([0.05, 0.10, 0.20, 0.40])


# ============================================================
# MMLU
# ============================================================

mmlu_github = (55.15, 0.22)

mmlu_references = {
    "Original Zephyr": 56.02,
    "HuggingFace CAT Zephyr": 54.08,
}

mmlu_relative = {
    "y":   [55.05, 54.98, 55.17, 54.92],
    "err": [0.30, 0.26, 0.17, 0.17],
}


# ============================================================
# ARC-E
# ============================================================

arce_github = (78.22, 0.38)

arce_references = {
    "Original Zephyr": 81.00,
    "HuggingFace CAT Zephyr": 78.20,
}

arce_relative = {
    "y":   [79.00, 78.52, 78.04, 77.08],
    "err": [0.26, 0.36, 0.40, 0.88],
}


# ============================================================
# ARC-C
# ============================================================

arcc_github = (50.24, 0.61)

arcc_references = {
    "Original Zephyr": 55.20,
    "HuggingFace CAT Zephyr": 51.50,
}

arcc_relative = {
    "y":   [51.17, 51.08, 50.94, 49.32],
    "err": [0.35, 0.50, 0.76, 0.82],
}


# ============================================================
# Representation plot
# ============================================================

def plot_representation(github, references, metric):
    fig, ax = plt.subplots(figsize=(8, 6))

    value, error = github

    original = references["Original Zephyr"]
    hf = references["HuggingFace CAT Zephyr"]

    # --------------------------------------------------------
    # GitHub CAT Zephyr bar
    # --------------------------------------------------------

    bar = ax.bar(
        [0],
        [value],
        yerr=[error],
        capsize=5,
        width=0.55,
        color=COLOR,
        alpha=0.85,
    )[0]

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    ax.axhline(
        original,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"Original Zephyr ({original:.2f})",
    )

    ax.axhline(
        hf,
        linestyle=":",
        linewidth=2.5,
        color="black",
        label=f"HF CAT Zephyr ({hf:.2f})",
    )

    # --------------------------------------------------------
    # Numerical value
    # --------------------------------------------------------

    all_values = [
        value - error,
        value + error,
        original,
        hf,
    ]

    ymin = min(all_values)
    ymax = max(all_values)

    padding = max(
        (ymax - ymin) * 0.20,
        0.5,
    )

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + error + padding * 0.15,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["UltraChat"])

    ax.set_ylabel(f"{metric} score (%)")

    ax.set_title(
        f"{metric} — {MODEL_NAME} representation",
        fontsize=14,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    return fig


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr(relative, metric):
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.array(
        relative["y"],
        dtype=float,
    )

    err = np.array(
        relative["err"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Line
    # --------------------------------------------------------

    ax.plot(
        eps,
        y,
        marker="o",
        markersize=7,
        linewidth=2,
        color=COLOR,
        label="UltraChat",
    )

    # --------------------------------------------------------
    # Error bars
    # --------------------------------------------------------

    ax.errorbar(
        eps,
        y,
        yerr=err,
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    # --------------------------------------------------------
    # Adaptive y-axis, including error bars
    # --------------------------------------------------------

    ymin = np.min(y - err)
    ymax = np.max(y + err)

    padding = max(
        (ymax - ymin) * 0.20,
        0.25,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Configuration multiplier $s$"
    )

    ax.set_ylabel(
        f"{metric} score (%)"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.05",
        "0.10",
        "0.20",
        "0.40",
    ])

    ax.set_title(
        f"{metric} — {MODEL_NAME} relative learning rate",
        fontsize=14,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    return fig


# ============================================================
# Generate all six figures
# ============================================================

figures = []


# ------------------------------------------------------------
# MMLU
# ------------------------------------------------------------

figures.append(
    plot_representation(
        mmlu_github,
        mmlu_references,
        "MMLU",
    )
)

figures.append(
    plot_relative_lr(
        mmlu_relative,
        "MMLU",
    )
)


# ------------------------------------------------------------
# ARC-E
# ------------------------------------------------------------

figures.append(
    plot_representation(
        arce_github,
        arce_references,
        "ARC-E",
    )
)

figures.append(
    plot_relative_lr(
        arce_relative,
        "ARC-E",
    )
)


# ------------------------------------------------------------
# ARC-C
# ------------------------------------------------------------

figures.append(
    plot_representation(
        arcc_github,
        arcc_references,
        "ARC-C",
    )
)

figures.append(
    plot_relative_lr(
        arcc_relative,
        "ARC-C",
    )
)


# ============================================================
# Save ALL six figures to ONE multipage PDF
#
# Example:
#
#     capabilities_zephyr.py
#
# creates:
#
#     capabilities_zephyr.pdf
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
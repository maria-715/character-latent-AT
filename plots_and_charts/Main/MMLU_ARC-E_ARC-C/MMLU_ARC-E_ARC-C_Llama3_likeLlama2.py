import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
COLOR = "C0"

eps = np.array([
    0.05,
    0.075,
    0.10,
    0.150,
    0.20,
    0.300,
    0.40,
    0.600,
])


# ============================================================
# MMLU
# ============================================================

mmlu_github = (60.03, 0.41)

mmlu_references = {
    "Original Llama 3": 60.75,
    "HuggingFace CAT Llama 3": 59.24,
}

mmlu_relative = {
    "y": [
        60.10,
        59.74,
        59.83,
        59.35,
        58.81,
        57.69,
        56.23,
        54.94,
    ],
    "err": [
        0.09,
        0.17,
        0.13,
        0.55,
        0.11,
        0.45,
        1.04,
        0.98,
    ],
}


# ============================================================
# ARC-E
# ============================================================

arce_github = (80.95, 0.30)

arce_references = {
    "Original Llama 3": 79.12,
    "HuggingFace CAT Llama 3": 80.93,
}

arce_relative = {
    "y": [
        80.95,
        80.80,
        80.78,
        79.45,
        78.05,
        74.99,
        73.18,
        47.97,
    ],
    "err": [
        0.30,
        0.44,
        0.18,
        0.81,
        1.21,
        1.58,
        2.82,
        4.30,
    ],
}


# ============================================================
# ARC-C
# ============================================================

arcc_github = (51.88, 0.53)

arcc_references = {
    "Original Llama 3": 49.40,
    "HuggingFace CAT Llama 3": 50.77,
}

arcc_relative = {
    "y": [
        51.77,
        51.45,
        51.31,
        49.69,
        49.72,
        47.32,
        46.22,
        37.37,
    ],
    "err": [
        0.43,
        0.38,
        0.39,
        1.00,
        0.35,
        0.40,
        1.92,
        2.10,
    ],
}


# ============================================================
# Representation plot
# ============================================================

def plot_representation(github, references, metric):
    fig, ax = plt.subplots(figsize=(8, 6))

    value, error = github

    original = references["Original Llama 3"]
    hf = references["HuggingFace CAT Llama 3"]

    # --------------------------------------------------------
    # GitHub Llama 3 bar
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
        label=f"Original Llama 3 ({original:.2f})",
    )

    ax.axhline(
        hf,
        linestyle=":",
        linewidth=2.5,
        color="black",
        label=f"HF CAT Llama 3 ({hf:.2f})",
    )

    # --------------------------------------------------------
    # Adaptive y-axis
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

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
    )

    # --------------------------------------------------------
    # Numerical value
    # --------------------------------------------------------

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

    ax.set_xticks([0])

    ax.set_xticklabels([
        "UltraChat"
    ])

    ax.set_ylabel(
        f"{metric} score (%)"
    )

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
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ymin = np.min(
        y - err
    )

    ymax = np.max(
        y + err
    )

    padding = max(
        (ymax - ymin) * 0.20,
        0.25,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        min(100, ymax + padding),
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
        "0.075",
        "0.10",
        "0.150",
        "0.20",
        "0.300",
        "0.40",
        "0.600",
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
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"
METRIC_NAME = "GCG"
COLOR = "C0"

eps = np.array([
    0.075,
    0.150,
    0.300,
    0.600,
])


# ============================================================
# Representation
# ============================================================

github = (20.00, 15.21)

references = {
    "Original Llama 3": 28.33,
    "HuggingFace CAT Llama 3": 0.00,
}


# ============================================================
# Relative learning-rate representation
# ============================================================

relative = {
    "y": [
        25.83,
        12.50,
        17.50,
        25.00,
    ],
    "err": [
        1.44,
        2.50,
        6.61,
        10.90,
    ],
}


# ============================================================
# Representation plot
# ============================================================

def plot_representation():
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
    # Y axis
    # --------------------------------------------------------

    ymax = max(
        value + error,
        original,
        hf,
    ) * 1.15

    ax.set_ylim(
        0,
        ymax,
    )

    # --------------------------------------------------------
    # Numerical value
    # --------------------------------------------------------

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + error + ymax * 0.02,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks([0])

    ax.set_xticklabels([
        "UltraChat"
    ])

    # --------------------------------------------------------
    # Labels / styling
    # --------------------------------------------------------

    ax.set_ylabel(
        "GCG score (%)"
    )

    ax.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} representation",
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
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
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
    # Line + markers
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
    # Y axis
    # --------------------------------------------------------

    ymax = np.max(
        y + err
    ) * 1.15

    ax.set_ylim(
        0,
        ymax,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Configuration multiplier $s$"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

    # --------------------------------------------------------
    # Labels / styling
    # --------------------------------------------------------

    ax.set_ylabel(
        "GCG score (%)"
    )

    ax.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} relative learning rate",
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
# Generate
# ============================================================

figures = [
    plot_representation(),
    plot_relative_lr(),
]


# ============================================================
# Save to ONE multipage PDF
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
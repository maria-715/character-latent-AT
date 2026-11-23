import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "Harmless"

COLOR = "C0"


# ============================================================
# Representation
# ============================================================

github = (91.67, 8.04)

references = {
    "Original Llama 3": 100.00,
    "HuggingFace CAT Llama 3": 77.50,
}


# ============================================================
# Relative learning-rate representation
#
# All 8 epsilon values are now available.
# ============================================================

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

relative = {
    "y": [
        95.83,
        90.83,
        76.67,
        45.83,
        49.17,
        7.50,
        2.50,
        1.67,
    ],
    "err": [
        1.44,
        7.22,
        7.64,
        26.96,
        20.05,
        12.99,
        4.33,
        2.89,
    ],
}


# ============================================================
# Common styling
# ============================================================

def style_axis(ax, title):
    ax.set_title(
        title,
        fontsize=14,
    )

    ax.set_ylabel(
        f"{METRIC_NAME} score (%)"
    )

    ax.set_ylim(
        0,
        110,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Representation bar chart
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
    # Numerical value
    # --------------------------------------------------------

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + error + 1.5,
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
    # Styling
    # --------------------------------------------------------

    style_axis(
        ax,
        f"{METRIC_NAME} — {MODEL_NAME} representation",
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    return fig


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
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
    # X axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Configuration multiplier $s$"
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

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    style_axis(
        ax,
        f"{METRIC_NAME} — {MODEL_NAME} relative learning rate",
    )

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
# Save ALL figures to ONE multipage PDF
#
# The PDF gets the same filename and location as this script.
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
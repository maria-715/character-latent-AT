import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"
METRIC_NAME = "Inpainting"
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

github = (89.00, 2.65)

references = {
    "Original Llama 3": 97.00,
    "HuggingFace CAT Llama 3": 79.00,
}


# ============================================================
# Relative learning-rate representation
# ============================================================

relative = {
    "y": [
        87.67,
        89.67,
        94.00,
        89.67,
    ],
    "err": [
        6.51,
        4.04,
        1.73,
        1.15,
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
        (ymax - ymin) * 0.15,
        1,
    )

    ax.set_ylim(
        ymin - padding,
        ymax + padding,
    )

    # --------------------------------------------------------
    # Numerical value
    # --------------------------------------------------------

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + error + 0.5,
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
        "Inpainting score (%)"
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
    # Adaptive y-axis
    # --------------------------------------------------------

    ymin = np.min(
        y - err
    )

    ymax = np.max(
        y + err
    )

    padding = max(
        (ymax - ymin) * 0.15,
        1,
    )

    ax.set_ylim(
        ymin - padding,
        min(100, ymax + padding),
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
        "Inpainting score (%)"
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
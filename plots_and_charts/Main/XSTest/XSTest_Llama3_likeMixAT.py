import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"
COLOR = "C0"

eps = np.array([
    0.075,
    0.150,
    0.300,
    0.600,
])


# ============================================================
# Safe compliance
# ============================================================

safe_github = (49.87, 8.39)

safe_references = {
    "Original Llama 3": 97.20,
    "HuggingFace CAT Llama 3": 28.00,
}

safe_relative = {
    "y": [
        51.60,
        51.07,
        60.27,
        37.33,
    ],
    "err": [
        15.48,
        10.51,
        10.01,
        10.99,
    ],
}


# ============================================================
# Unsafe refusal
# ============================================================

unsafe_github = (99.50, 0.00)

unsafe_references = {
    "Original Llama 3": 91.50,
    "HuggingFace CAT Llama 3": 99.50,
}

unsafe_relative = {
    "y": [
        99.33,
        98.50,
        97.00,
        99.50,
    ],
    "err": [
        0.29,
        0.87,
        1.80,
        0.50,
    ],
}


# ============================================================
# Representation
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(9, 6))

    width = 0.30

    safe_value, safe_error = safe_github
    unsafe_value, unsafe_error = unsafe_github

    # --------------------------------------------------------
    # GitHub bars
    # --------------------------------------------------------

    ax.bar(
        -width / 2,
        safe_value,
        width,
        yerr=safe_error,
        capsize=5,
        color=COLOR,
        alpha=0.85,
        edgecolor="black",
        label="Safe compliance",
    )

    ax.bar(
        width / 2,
        unsafe_value,
        width,
        yerr=unsafe_error,
        capsize=5,
        color=COLOR,
        alpha=0.45,
        edgecolor="black",
        hatch="//",
        label="Unsafe refusal",
    )

    # --------------------------------------------------------
    # Safe references
    # --------------------------------------------------------

    ax.axhline(
        safe_references["Original Llama 3"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Safe — Original "
            f"({safe_references['Original Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        safe_references["HuggingFace CAT Llama 3"],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Safe — HF CAT "
            f"({safe_references['HuggingFace CAT Llama 3']:.2f})"
        ),
    )

    # --------------------------------------------------------
    # Unsafe references
    # --------------------------------------------------------

    ax.axhline(
        unsafe_references["Original Llama 3"],
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Unsafe — Original "
            f"({unsafe_references['Original Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["HuggingFace CAT Llama 3"],
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Unsafe — HF CAT "
            f"({unsafe_references['HuggingFace CAT Llama 3']:.2f})"
        ),
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xticks([0])
    ax.set_xticklabels(["UltraChat"])

    ax.set_ylim(0, 110)

    ax.set_ylabel("XSTest score (%)")

    ax.set_title(
        f"XSTest — {MODEL_NAME} representation",
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
        fontsize=8,
        loc="lower right",
    )

    fig.tight_layout()

    return fig


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(9, 6))

    # --------------------------------------------------------
    # Safe compliance
    # --------------------------------------------------------

    ax.plot(
        eps,
        safe_relative["y"],
        marker="o",
        markersize=7,
        linewidth=2,
        linestyle="-",
        color=COLOR,
    )

    ax.errorbar(
        eps,
        safe_relative["y"],
        yerr=safe_relative["err"],
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    # --------------------------------------------------------
    # Unsafe refusal
    # --------------------------------------------------------

    ax.plot(
        eps,
        unsafe_relative["y"],
        marker="s",
        markersize=6,
        linewidth=2,
        linestyle="--",
        color=COLOR,
    )

    ax.errorbar(
        eps,
        unsafe_relative["y"],
        yerr=unsafe_relative["err"],
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_ylim(
        0,
        110,
    )

    ax.set_xlabel(
        r"Configuration multiplier $s$"
    )

    ax.set_ylabel(
        "XSTest score (%)"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

    ax.set_title(
        f"XSTest — {MODEL_NAME} relative learning rate",
        fontsize=14,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=COLOR,
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Safe compliance",
        ),

        Line2D(
            [0],
            [0],
            color=COLOR,
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Unsafe refusal",
        ),
    ]

    ax.legend(
        handles=metric_handles,
        frameon=False,
    )

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
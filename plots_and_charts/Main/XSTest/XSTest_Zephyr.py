import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


MODEL_NAME = "Zephyr"
METRIC_NAME = "XSTest"
COLOR = "C0"

eps = np.array([0.05, 0.10, 0.20, 0.40])


# ============================================================
# Safe compliance
# ============================================================

safe_github = (55.52, 6.06)

safe_references = {
    "Original Zephyr": 94.80,
    "HuggingFace CAT Zephyr": 54.40,
}

safe_relative = {
    "y":   [62.40, 54.67, 53.47, 70.93],
    "err": [7.34, 6.47, 2.95, 13.05],
}


# ============================================================
# Unsafe refusal
# ============================================================

unsafe_github = (98.70, 1.10)

unsafe_references = {
    "Original Zephyr": 9.50,
    "HuggingFace CAT Zephyr": 99.50,
}

unsafe_relative = {
    "y":   [98.83, 98.83, 99.00, 89.00],
    "err": [0.58, 0.76, 0.50, 9.37],
}


# ============================================================
# Representation
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(8, 6))

    width = 0.30

    safe_value, safe_error = safe_github
    unsafe_value, unsafe_error = unsafe_github

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

    # Safe references
    ax.axhline(
        safe_references["Original Zephyr"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Safe — Original "
            f"({safe_references['Original Zephyr']:.2f})"
        ),
    )

    ax.axhline(
        safe_references["HuggingFace CAT Zephyr"],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Safe — HF CAT "
            f"({safe_references['HuggingFace CAT Zephyr']:.2f})"
        ),
    )

    # Unsafe references
    ax.axhline(
        unsafe_references["Original Zephyr"],
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Unsafe — Original "
            f"({unsafe_references['Original Zephyr']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["HuggingFace CAT Zephyr"],
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Unsafe — HF CAT "
            f"({unsafe_references['HuggingFace CAT Zephyr']:.2f})"
        ),
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["UltraChat"])

    ax.set_ylim(0, 110)
    ax.set_ylabel("XSTest score (%)")
    ax.set_title(f"XSTest — {MODEL_NAME} representation")

    ax.grid(axis="y", alpha=0.25)
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
# Relative LR
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Safe
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

    # Unsafe
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

    ax.set_ylim(0, 110)

    ax.set_xlabel(r"Configuration multiplier $s$")
    ax.set_ylabel("XSTest score (%)")

    ax.set_xticks(eps)
    ax.set_xticklabels(["0.05", "0.10", "0.20", "0.40"])

    ax.set_title(
        f"XSTest — {MODEL_NAME} relative learning rate"
    )

    metric_handles = [
        Line2D(
            [0], [0],
            color=COLOR,
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Safe compliance",
        ),
        Line2D(
            [0], [0],
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

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


figures = [
    plot_representation(),
    plot_relative_lr(),
]

pdf_path = Path(__file__).resolve().with_suffix(".pdf")

with PdfPages(pdf_path) as pdf:
    for fig in figures:
        pdf.savefig(fig, bbox_inches="tight")

print(f"Saved to:\n{pdf_path}")

plt.show()
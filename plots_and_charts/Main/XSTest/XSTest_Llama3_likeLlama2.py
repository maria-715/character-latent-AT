import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "XSTest"

COLOR = "C0"


# ============================================================
# Representation
#
# Safe compliance and unsafe refusal are shown together.
# ============================================================

github_safe = (41.73, 25.47)
github_unsafe = (99.67, 0.29)


safe_references = {
    "Original Llama 3": 97.20,
    "HuggingFace CAT Llama 3": 28.00,
}

unsafe_references = {
    "Original Llama 3": 91.50,
    "HuggingFace CAT Llama 3": 99.50,
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


# ------------------------------------------------------------
# Safe compliance
# ------------------------------------------------------------

safe_relative = {
    "y": [
        61.87,
        48.13,
        47.47,
        52.13,
        32.40,
        20.67,
        32.00,
        0.80,
    ],
    "err": [
        5.72,
        18.36,
        14.14,
        29.31,
        26.89,
        22.10,
        28.93,
        0.40,
    ],
}


# ------------------------------------------------------------
# Unsafe refusal
# ------------------------------------------------------------

unsafe_relative = {
    "y": [
        99.33,
        92.67,
        99.67,
        96.83,
        99.17,
        98.50,
        98.83,
        100.00,
    ],
    "err": [
        0.29,
        10.97,
        0.29,
        3.51,
        0.29,
        0.87,
        1.26,
        0.00,
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
        "XSTest score (%)"
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
# Representation plot
#
# Safe compliance and unsafe refusal on the SAME plot.
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(9, 6))

    safe_value, safe_error = github_safe
    unsafe_value, unsafe_error = github_unsafe

    x = np.array([
        0,
        1,
    ])

    # --------------------------------------------------------
    # GitHub Llama 3 bars
    #
    # Same base color, different visual treatment for the
    # two XSTest metrics.
    # --------------------------------------------------------

    safe_bar = ax.bar(
        x[0],
        safe_value,
        yerr=safe_error,
        capsize=5,
        width=0.55,
        color=COLOR,
        alpha=0.85,
        edgecolor="black",
        label="Safe compliance",
    )

    unsafe_bar = ax.bar(
        x[1],
        unsafe_value,
        yerr=unsafe_error,
        capsize=5,
        width=0.55,
        color=COLOR,
        alpha=0.45,
        edgecolor="black",
        hatch="//",
        label="Unsafe refusal",
    )

    # --------------------------------------------------------
    # Safe-compliance references
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
    # Unsafe-refusal references
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
    # Numerical values above bars
    # --------------------------------------------------------

    ax.text(
        x[0],
        safe_value + safe_error + 2,
        f"{safe_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.text(
        x[1],
        unsafe_value + unsafe_error + 2,
        f"{unsafe_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels([
        "Safe compliance",
        "Unsafe refusal",
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
        fontsize=8,
        loc="lower right",
    )

    fig.tight_layout()

    return fig


# ============================================================
# Relative learning-rate plot
#
# Safe compliance and unsafe refusal remain on SAME plot.
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(10, 6))

    safe_y = np.array(
        safe_relative["y"],
        dtype=float,
    )

    safe_err = np.array(
        safe_relative["err"],
        dtype=float,
    )

    unsafe_y = np.array(
        unsafe_relative["y"],
        dtype=float,
    )

    unsafe_err = np.array(
        unsafe_relative["err"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Safe compliance
    # --------------------------------------------------------

    ax.plot(
        eps,
        safe_y,
        marker="o",
        markersize=7,
        linewidth=2,
        linestyle="-",
        color=COLOR,
        label="Safe compliance",
    )

    ax.errorbar(
        eps,
        safe_y,
        yerr=safe_err,
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    # --------------------------------------------------------
    # Unsafe refusal
    # --------------------------------------------------------

    ax.plot(
        eps,
        unsafe_y,
        marker="s",
        markersize=7,
        linewidth=2,
        linestyle="--",
        color=COLOR,
        label="Unsafe refusal",
    )

    ax.errorbar(
        eps,
        unsafe_y,
        yerr=unsafe_err,
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
        loc="best",
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
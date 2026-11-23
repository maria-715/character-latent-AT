import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "Inpainting"

COLOR = "C0"


# ============================================================
# Representation
# ============================================================

github = (94.33, 2.08)

references = {
    "Original Llama 3": 97.00,
    "HuggingFace CAT Llama 3": 79.00,
}


# ============================================================
# Relative learning-rate representation
#
# Results are reported only at:
#
# eps = 0.05, 0.10, 0.20, 0.40
#
# The intermediate epsilon values:
#
# 0.075, 0.150, 0.300, 0.600
#
# have no Inpainting result.
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

relative_y = np.array([
    94.00,
    np.nan,
    89.67,
    np.nan,
    88.67,
    np.nan,
    93.67,
    np.nan,
])

relative_err = np.array([
    5.29,
    np.nan,
    3.06,
    np.nan,
    2.52,
    np.nan,
    3.21,
    np.nan,
])


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
    # Axis range
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
        1.0,
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
        value + error + padding * 0.15,
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
        f"{METRIC_NAME} score (%)"
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
        loc="best",
    )

    fig.tight_layout()

    return fig


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------------
    # Plot
    #
    # NaN values ensure that missing measurements are not
    # accidentally treated as real observations.
    # --------------------------------------------------------

    ax.plot(
        eps,
        relative_y,
        marker="o",
        markersize=7,
        linewidth=2,
        color=COLOR,
        label="UltraChat",
    )

    # --------------------------------------------------------
    # Error bars only for measured points
    # --------------------------------------------------------

    valid = (
        ~np.isnan(relative_y)
        & ~np.isnan(relative_err)
    )

    ax.errorbar(
        eps[valid],
        relative_y[valid],
        yerr=relative_err[valid],
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    # --------------------------------------------------------
    # Adaptive y-axis
    # --------------------------------------------------------

    lower = (
        relative_y[valid]
        - relative_err[valid]
    )

    upper = (
        relative_y[valid]
        + relative_err[valid]
    )

    ymin = np.min(lower)
    ymax = np.max(upper)

    padding = max(
        (ymax - ymin) * 0.15,
        1.0,
    )

    ax.set_ylim(
        max(0, ymin - padding),
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
    # Labels / styling
    # --------------------------------------------------------

    ax.set_ylabel(
        f"{METRIC_NAME} score (%)"
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
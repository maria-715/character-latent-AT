import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "GCG"
COLOR = "C0"


# ============================================================
# Representation
# ============================================================

github = (22.50, 4.33)

references = {
    "Original Llama 3": 28.33,
    "HuggingFace CAT Llama 3": 0.00,
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
        22.50,
        20.83,
        16.67,
        22.50,
        19.17,
        19.17,
        22.50,
        33.33,
    ],
    "err": [
        6.61,
        3.82,
        10.41,
        12.99,
        7.22,
        11.55,
        5.00,
        9.46,
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
        (ymax - ymin) * 0.15,
        2,
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
        0,
        ymax + padding,
    )

    ax.set_xticks([0])
    ax.set_xticklabels([
        "UltraChat"
    ])

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
        (ymax - ymin) * 0.10,
        2,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
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
    # Y axis
    # --------------------------------------------------------

    ax.set_ylabel(
        f"{METRIC_NAME} score (%)"
    )

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

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
# Save ALL figures to ONE multipage PDF
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
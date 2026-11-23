import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"
METRIC_NAME = "AutoDAN"
COLOR = "C0"

eps = np.array([
    0.075,
    0.150,
    0.300,
    0.600,
])


# ============================================================
# Representation
#
# AutoDAN values are stored directly on the 0-100 scale.
#
# Original Llama 3:
#     10
#
# HuggingFace CAT Llama 3:
#     0
#
# GitHub Llama 3:
#     0 ± 0
# ============================================================

github = (0.00, 0.00)

references = {
    "Original Llama 3": 10.00,
    "HuggingFace CAT Llama 3": 0.00,
}


# ============================================================
# Relative learning-rate representation
#
# Values are stored directly on the 0-100 percentage scale.
#
# eps = 0.075 -> 0.83 ± 1.44
# eps = 0.150 -> 0 ± 0
# eps = 0.300 -> 0 ± 0
# eps = 0.600 -> 0 ± 0
# ============================================================

relative = {
    "y": [
        0.83,
        0.00,
        0.00,
        0.00,
    ],
    "err": [
        1.44,
        0.00,
        0.00,
        0.00,
    ],
}


# ============================================================
# Broken-axis marks
# ============================================================

def add_break_marks(ax_top, ax_bottom):
    d = 0.008

    kwargs = dict(
        color="black",
        clip_on=False,
        linewidth=1,
    )

    ax_top.plot(
        (-d, +d),
        (-d, +d),
        transform=ax_top.transAxes,
        **kwargs,
    )

    ax_top.plot(
        (1 - d, 1 + d),
        (-d, +d),
        transform=ax_top.transAxes,
        **kwargs,
    )

    ax_bottom.plot(
        (-d, +d),
        (1 - d, 1 + d),
        transform=ax_bottom.transAxes,
        **kwargs,
    )

    ax_bottom.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        transform=ax_bottom.transAxes,
        **kwargs,
    )


# ============================================================
# Representation plot
# ============================================================

def plot_representation():
    value, error = github

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 7),
        gridspec_kw={
            "height_ratios": [1, 3],
            "hspace": 0.06,
        },
    )

    original = references["Original Llama 3"]
    hf = references["HuggingFace CAT Llama 3"]

    # --------------------------------------------------------
    # GitHub bar + references on both parts of broken axis
    # --------------------------------------------------------

    for ax in (ax_top, ax_bottom):

        ax.bar(
            [0],
            [value],
            yerr=[error],
            capsize=5,
            width=0.55,
            color=COLOR,
            alpha=0.85,
        )

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
    # Broken y-axis
    #
    # Lower section contains GitHub/HF values near 0.
    # Upper section contains Original Llama 3 at 10.
    # --------------------------------------------------------

    ax_bottom.set_ylim(
        0,
        2.5,
    )

    ax_top.set_ylim(
        9,
        11,
    )

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    ax_top.tick_params(
        bottom=False,
        labelbottom=False,
    )

    add_break_marks(
        ax_top,
        ax_bottom,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax_bottom.set_xticks([0])

    ax_bottom.set_xticklabels([
        "UltraChat"
    ])

    # --------------------------------------------------------
    # Numerical value
    # --------------------------------------------------------

    ax_bottom.text(
        0,
        0.15,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Labels / styling
    # --------------------------------------------------------

    ax_bottom.set_ylabel(
        "AutoDAN score (%)"
    )

    ax_top.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} representation",
        fontsize=14,
    )

    for ax in (ax_top, ax_bottom):
        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["right"].set_visible(False)

    ax_top.legend(
        frameon=False,
        loc="best",
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.12,
        top=0.92,
    )

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
    #
    # Maximum:
    # 0.83 + 1.44 = 2.27
    # --------------------------------------------------------

    ymax = max(
        np.max(y + err) * 1.15,
        1.0,
    )

    ax.set_ylim(
        0,
        ymax,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Configuration multiplier $s$"
    )

    ax.set_ylabel(
        "AutoDAN score (%)"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

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
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "AutoDAN"
COLOR = "C0"


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
#     1.67 ± 2.89
# ============================================================

references = {
    "Original Llama 3": 10.00,
    "HuggingFace CAT Llama 3": 0.00,
}

github = (1.67, 2.89)


# ============================================================
# Relative learning-rate representation
#
# All 8 epsilon values are available.
# Values are stored directly on the 0-100 percentage scale.
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
    1.67,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
])

relative_err = np.array([
    2.89,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
])


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
#
# - GitHub Llama 3 = bar
# - Original Llama 3 = dashed reference
# - HF CAT Llama 3 = dotted reference
#
# Broken y-axis because Original = 10 while GitHub/HF
# are near zero.
# ============================================================

def plot_representation():
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

    value, error = github

    original = references["Original Llama 3"]
    hf = references["HuggingFace CAT Llama 3"]

    # --------------------------------------------------------
    # Bar + references on both parts of the broken axis
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

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Broken-axis limits
    #
    # GitHub upper error:
    #
    # 1.67 + 2.89 = 4.56
    #
    # So the lower region extends slightly beyond that.
    # --------------------------------------------------------

    ax_bottom.set_ylim(
        0,
        5.5,
    )

    ax_top.set_ylim(
        9,
        11,
    )

    # --------------------------------------------------------
    # Broken-axis appearance
    # --------------------------------------------------------

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
        value + error + 0.2,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax_bottom.set_ylabel(
        "AutoDAN score (%)"
    )

    ax_top.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} representation",
        fontsize=14,
    )

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
#
# All 8 epsilon values are plotted together.
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------------
    # Line + markers
    # --------------------------------------------------------

    ax.plot(
        eps,
        relative_y,
        marker="o",
        markersize=8,
        linewidth=2,
        color=COLOR,
        label="UltraChat",
    )

    # --------------------------------------------------------
    # Error bars
    # --------------------------------------------------------

    ax.errorbar(
        eps,
        relative_y,
        yerr=relative_err,
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    # --------------------------------------------------------
    # Adaptive y-axis
    #
    # Largest upper error:
    #
    # 1.67 + 2.89 = 4.56
    # --------------------------------------------------------

    ymax = np.max(
        relative_y + relative_err
    )

    padding = max(
        ymax * 0.15,
        0.5,
    )

    ax.set_ylim(
        0,
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
        "AutoDAN score (%)"
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
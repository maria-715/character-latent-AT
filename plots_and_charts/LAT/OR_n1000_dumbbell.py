import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.lines import Line2D

from or_n1000_data import models


# ============================================================
# Dumbbell plots: baseline (UltraChat only) -> 1,000 OR examples
#
# One panel per (configuration, radius); each row is a metric,
# with an arrow from the baseline mean to the replacement value
# and the change annotated as Delta.
#
# Baseline values are means across three seeds. Replacement
# conditions are mostly single-seed, so no confidence interval on
# Delta is available. Error bars are off by default (see below).
# ============================================================


# ============================================================
# Typography
#
# The figure is drawn at FIGSIZE, close to its printed width, so it is
# scaled down only ~2x at \textwidth (5.5in) and fonts stay legible
# (~7pt in print). Enlarge fonts rather than the canvas if needed.
# ============================================================

FIGSIZE = (11.0, 4.2)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Configuration
# ============================================================

# (configuration, radius index) per panel, left to right.
# Index 2 = the radius below the largest; index 3 = the largest.
PANELS = [
    ("Zephyr", 2),
    ("Zephyr", 3),
    ("Llama 3 (MixAT)", 2),
    ("Llama 3 (MixAT)", 3),
]

# Utility metrics (higher is better) above the separator,
# robustness (lower ASR is better) below it.
ROWS = [
    ("Harmless", "Harmless"),
    ("OR held-out", "Held-out\nOR-Bench"),
    ("Soft-Prompt", "Soft-Prompt\nASR"),
]

ROW_Y = {
    "Harmless": 2.5,
    "OR held-out": 1.4,
    "Soft-Prompt": 0.0,
}

SEPARATOR_Y = 0.8

BASELINE_COLOR = "#4C72B0"
REPLACEMENT_COLOR = "#DD8452"

# Error bars (one sample SD) where an SD is available.
SHOW_BASELINE_ERROR_BARS = False
SHOW_REPLACEMENT_ERROR_BARS = False

# Arrows shorter than this (in % points) are omitted; the markers
# already overlap.
MIN_ARROW = 1.5


# ============================================================
# One dumbbell panel
# ============================================================

def plot_panel(ax, configuration, idx, show_ylabels):

    data = models[configuration]

    for metric, _ in ROWS:

        y = ROW_Y[metric]

        base = data[metric]["baseline"]["y"][idx]
        base_err = data[metric]["baseline"]["err"][idx]

        rep = data[metric]["replacement"]["y"][idx]
        rep_err = data[metric]["replacement"]["err"][idx]

        delta = rep - base

        # ----------------------------------------------------
        # Arrow baseline -> replacement
        # ----------------------------------------------------

        if abs(delta) >= MIN_ARROW:
            ax.annotate(
                "",
                xy=(rep, y),
                xytext=(base, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="black",
                    linewidth=1.6,
                    shrinkA=5,
                    shrinkB=7,
                    mutation_scale=12,
                ),
                zorder=2,
            )

        # ----------------------------------------------------
        # Error bars
        # ----------------------------------------------------

        if SHOW_BASELINE_ERROR_BARS and not np.isnan(base_err):
            ax.errorbar(
                base,
                y,
                xerr=base_err,
                fmt="none",
                color=BASELINE_COLOR,
                capsize=3,
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )

        if SHOW_REPLACEMENT_ERROR_BARS and not np.isnan(rep_err):
            ax.errorbar(
                rep,
                y,
                xerr=rep_err,
                fmt="none",
                color=REPLACEMENT_COLOR,
                capsize=3,
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )

        # ----------------------------------------------------
        # Markers
        # ----------------------------------------------------

        ax.scatter(
            base,
            y,
            marker="o",
            s=90,
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        ax.scatter(
            rep,
            y,
            marker="D",
            s=80,
            color=REPLACEMENT_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

        # ----------------------------------------------------
        # Delta annotation, centred above the dumbbell
        # ----------------------------------------------------

        ax.text(
            np.clip((base + rep) / 2, 22, 83),
            y + 0.2,
            rf"$\Delta$ {delta:+.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    # --------------------------------------------------------
    # Utility / robustness separator
    # --------------------------------------------------------

    ax.axhline(
        SEPARATOR_Y,
        color="0.4",
        linestyle="-.",
        linewidth=1.2,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(-3, 105)
    ax.set_ylim(-0.35, 3.05)

    # Shared y-axis: set labels once, hide them on later panels
    # (set_yticklabels([]) would clear them on every shared axis).
    ax.set_yticks([ROW_Y[m] for m, _ in ROWS])
    ax.set_yticklabels([label for _, label in ROWS])

    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xticks([0, 50, 100])

    eps_label = data["epsilon_labels"][idx]
    ax.set_title(rf"{configuration}" "\n" rf"$\epsilon = {eps_label}$")

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Figure: four panels in a row
# ============================================================

fig, axes = plt.subplots(
    1,
    len(PANELS),
    figsize=FIGSIZE,
    sharey=True,
)

for i, (ax, (configuration, idx)) in enumerate(zip(axes, PANELS)):
    plot_panel(ax, configuration, idx, show_ylabels=(i == 0))

# One shared x-axis label instead of one per panel
fig.supxlabel("Score / ASR (%)", x=0.56, y=0.01)


# ============================================================
# Legend
# ============================================================

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=BASELINE_COLOR,
        markeredgecolor="white",
        markersize=10,
        label="Baseline (UltraChat only)",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor=REPLACEMENT_COLOR,
        markeredgecolor="white",
        markersize=9,
        label=r"$1{,}000$ OR examples",
    ),
]

fig.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
)

fig.subplots_adjust(
    left=0.13,
    right=0.99,
    bottom=0.19,
    top=0.74,
    wspace=0.10,
)


# ============================================================
# Save
# ============================================================

output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "OR_n1000_dumbbell.pdf"
png_path = output_dir / "OR_n1000_dumbbell.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()

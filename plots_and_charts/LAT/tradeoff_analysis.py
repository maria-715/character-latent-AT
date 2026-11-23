import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from lat_data import (
    COLORS,
    MARKERS,
    SERIES_LABELS,
    models,
    style_axis,
)


# ============================================================
# Typography
# ============================================================

plt.rcParams.update({
    "font.size": 13,          # default text
    "axes.titlesize": 14,     # panel titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 11,    # x tick labels
    "ytick.labelsize": 11,    # y tick labels
    "legend.fontsize": 12,    # legend
})


# ============================================================
# Annotation offsets
#
# Used only to prevent overlapping epsilon labels.
# Values are offsets in display points: (dx, dy).
# Key: (utility_metric, robustness_metric, series, eps)
# ============================================================

ANNOTATION_OFFSETS = {
    # Harmless: the whole "10pct" sweep sits at 100%, so label
    # it on the left to keep it clear of the "base" labels.
    **{
        ("Harmless", robustness, "10pct", eps): (-20, -4)
        for robustness in ("GCG", "Soft-Prompts")
        for eps in (6.0, 12.0, 24.0, 48.0)
    },
    ("Harmless", "Soft-Prompts", "10pct", 24.0): (-20, 2),
    ("Harmless", "Soft-Prompts", "10pct", 48.0): (-20, -10),

    ("MMLU", "GCG", "10pct", 24.0): (-8, -16),
    ("MMLU", "Soft-Prompts", "base", 48.0): (5, -14),
}


# ============================================================
# Pareto dominance WITHIN one sweep.
#
# Higher x (utility) = better, lower y (ASR) = better.
# ============================================================

def is_non_dominated(x_i, y_i, x, y):

    for x_j, y_j in zip(x, y):

        if (
            x_j >= x_i
            and y_j <= y_i
            and (x_j > x_i or y_j < y_i)
        ):
            return False

    return True


def add_direction_annotation(ax):

    ax.text(
        0.98,
        1.0,
        "Utility improves →\nRobustness improves ↓",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
        linespacing=1.25,
        zorder=5,
    )


# ============================================================
# One trade-off panel
#
# Only positive-radius checkpoints are used, as in the paper.
# ============================================================

def plot_tradeoff_panel(
    ax,
    utility_metric,
    robustness_metric,
    panel_label,
    panel_title,
):

    all_x = []
    all_y = []

    for name, model_data in models.items():

        positive = model_data["eps"] > 0

        eps = model_data["eps"][positive]
        x = model_data[utility_metric]["y"][positive]
        y = model_data[robustness_metric]["y"][positive]

        color = COLORS[name]
        marker = MARKERS[name]

        # Connect successive sweep settings
        ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.4,
            color=color,
            alpha=0.75,
            zorder=1,
        )

        # Filled = non-dominated, hollow = dominated
        for x_i, y_i in zip(x, y):

            filled = is_non_dominated(x_i, y_i, x, y)

            ax.scatter(
                x_i,
                y_i,
                marker=marker,
                s=42,
                facecolor=color if filled else "none",
                edgecolor=color,
                linewidth=1.2,
                zorder=3,
            )

        # Label each point by configured radius
        for x_i, y_i, eps_i in zip(x, y, eps):

            offset = ANNOTATION_OFFSETS.get(
                (utility_metric, robustness_metric, name, float(eps_i)),
                (5, 5),
            )

            ax.annotate(
                f"{eps_i:g}",
                xy=(x_i, y_i),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                color=color,
                zorder=4,
            )

        all_x.extend(x)
        all_y.extend(y)

    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    xpadding = max((xmax - xmin) * 0.07, 0.5)
    ypadding = max((ymax - ymin) * 0.08, 1.0)

    ax.set_xlim(
        max(0, xmin - xpadding),
        min(100.5, xmax + xpadding),
    )

    ax.set_ylim(
        max(0, ymin - ypadding),
        min(100, ymax + ypadding),
    )

    # --------------------------------------------------------
    # Labels and styling
    # --------------------------------------------------------

    if utility_metric == "Harmless":
        ax.set_xlabel("Harmless successful response rate (%)")
    elif utility_metric == "MMLU":
        ax.set_xlabel("MMLU accuracy (%)")

    ax.set_ylabel(f"{robustness_metric} ASR (%)")

    ax.set_title(rf"(${panel_label}$) {panel_title}")

    style_axis(ax, grid_axis="both")

    add_direction_annotation(ax)


# ============================================================
# Legends
# ============================================================

def add_legends(fig):

    series_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle="-",
            linewidth=1.4,
            markersize=6,
            label=SERIES_LABELS[name],
        )
        for name in models.keys()
    ]

    pareto_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=7,
            label="Non-dominated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="0.25",
            markeredgewidth=1.2,
            markersize=7,
            label="Dominated",
        ),
    ]

    fig.legend(
        handles=series_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
    )

    fig.legend(
        handles=pareto_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )


def plot_tradeoff_page(utility_metric, labels):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
    )

    for ax, robustness_metric, panel_label in zip(
        axes,
        ["GCG", "Soft-Prompts"],
        labels,
    ):
        plot_tradeoff_panel(
            ax,
            utility_metric=utility_metric,
            robustness_metric=robustness_metric,
            panel_label=panel_label,
            panel_title=f"{utility_metric} vs {robustness_metric}",
        )

    add_legends(fig)

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.89]
    )

    fig.subplots_adjust(
        hspace=0.45
    )

    return fig


# ============================================================
# Page 1: Harmless x {GCG, Soft-Prompts}
# Page 2: MMLU x {GCG, Soft-Prompts}
# ============================================================

page_1 = plot_tradeoff_page("Harmless", ["a", "b"])
page_2 = plot_tradeoff_page("MMLU", ["c", "d"])


# ============================================================
# Save
# ============================================================

output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "tradeoff_analysis.pdf"

with PdfPages(pdf_path) as pdf:
    pdf.savefig(page_1, bbox_inches="tight")
    pdf.savefig(page_2, bbox_inches="tight")

page_1.savefig(output_dir / "tradeoff_harmless.png", dpi=300, bbox_inches="tight")
page_2.savefig(output_dir / "tradeoff_mmlu.png", dpi=300, bbox_inches="tight")

print(
    "Saved trade-off analysis to:\n"
    f"  {pdf_path}"
)

plt.show()

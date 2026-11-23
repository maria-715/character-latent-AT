import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Zephyr"

rows = [
    "only UltraChat",
    "OR (1000)",
]

# Same colors across ALL experiments/scripts
colors = {
    "only UltraChat":      "C0",
    "OR (100)":            "C1",
    "OR (500)":            "C2",
    "OR (1000)":           "C3",
    "trimmed OR (1000)":   "C4",
    "OR (3000)":           "C5",
}

labels = {
    "only UltraChat": "UltraChat",
    "OR (100)": "OR\n100",
    "OR (500)": "OR\n500",
    "OR (1000)": "OR\n1000",
    "trimmed OR (1000)": "Trimmed OR\n1000",
    "OR (3000)": "OR\n3000",
}

eps = np.array([0.05, 0.10, 0.20, 0.40])


# ============================================================
# SAFE COMPLIANCE
# ============================================================

safe_github = {
    "only UltraChat": (55.52, 6.16),
    "OR (1000)":      (80.40, None),
}

safe_references = {
    "Original Zephyr": 94.80,
    "HuggingFace CAT Zephyr": 54.40,
}

safe_relative = {
    "only UltraChat": {
        "y":   [62.40, 54.70, 53.50, 70.90],
        "err": [7.30, 6.50, 2.90, 13.00],
    },
    "OR (1000)": {
        "y":   [85.20, 80.80, 83.60, 89.20],
        "err": [None, None, None, None],
    },
}


# ============================================================
# UNSAFE REFUSAL
# ============================================================

unsafe_github = {
    "only UltraChat": (98.70, 1.15),
    "OR (1000)":      (83.00, None),
}

unsafe_references = {
    "Original Zephyr": 9.50,
    "HuggingFace CAT Zephyr": 99.50,
}

unsafe_relative = {
    "only UltraChat": {
        "y":   [98.83, 98.83, 99.00, 89.00],
        "err": [0.58, 0.76, 0.50, 9.29],
    },
    "OR (1000)": {
        "y":   [81.50, 85.50, 78.50, 63.00],
        "err": [None, None, None, None],
    },
}


# ============================================================
# Common styling
# ============================================================

def style_axis(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("XSTest score")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Representation plot
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(10, 6))

    available_rows = [
        row for row in rows
        if (
            safe_github.get(row) is not None
            or unsafe_github.get(row) is not None
        )
    ]

    x = np.arange(len(available_rows))
    width = 0.35

    safe_values = [
        safe_github[row][0]
        if safe_github.get(row) is not None
        else np.nan
        for row in available_rows
    ]

    safe_errors = [
        safe_github[row][1]
        if safe_github.get(row) is not None
        and safe_github[row][1] is not None
        else 0
        for row in available_rows
    ]

    unsafe_values = [
        unsafe_github[row][0]
        if unsafe_github.get(row) is not None
        else np.nan
        for row in available_rows
    ]

    unsafe_errors = [
        unsafe_github[row][1]
        if unsafe_github.get(row) is not None
        and unsafe_github[row][1] is not None
        else 0
        for row in available_rows
    ]

    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    safe_bars = ax.bar(
        x - width / 2,
        safe_values,
        width,
        yerr=safe_errors,
        capsize=4,
        color=[colors[row] for row in available_rows],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )

    unsafe_bars = ax.bar(
        x + width / 2,
        unsafe_values,
        width,
        yerr=unsafe_errors,
        capsize=4,
        color=[colors[row] for row in available_rows],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
    )

    # --------------------------------------------------------
    # Numerical values
    # --------------------------------------------------------

    for bar, value in zip(safe_bars, safe_values):
        if np.isnan(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar, value in zip(unsafe_bars, unsafe_values):
        if np.isnan(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    ax.axhline(
        safe_references["Original Zephyr"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Safe — Original Zephyr "
            f"({safe_references['Original Zephyr']:.2f})"
        ),
    )

    ax.axhline(
        safe_references["HuggingFace CAT Zephyr"],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Safe — HF CAT Zephyr "
            f"({safe_references['HuggingFace CAT Zephyr']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["Original Zephyr"],
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Unsafe — Original Zephyr "
            f"({unsafe_references['Original Zephyr']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["HuggingFace CAT Zephyr"],
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Unsafe — HF CAT Zephyr "
            f"({unsafe_references['HuggingFace CAT Zephyr']:.2f})"
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    style_axis(
        ax,
        f"XSTest — {MODEL_NAME} representation",
    )

    metric_handles = [
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markersize=10,
            markerfacecolor="gray",
            markeredgecolor="black",
            label="Safe compliance",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markersize=10,
            markerfacecolor="lightgray",
            markeredgecolor="black",
            label="Unsafe refusal",
        ),
    ]

    metric_legend = ax.legend(
        handles=metric_handles,
        frameon=False,
        loc="lower left",
    )

    ax.add_artist(metric_legend)

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout()
    return fig, ax


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(11, 7))

    for row in rows:

        # ----------------------------------------------------
        # Safe compliance
        # ----------------------------------------------------

        safe_y = np.array([
            np.nan if v is None else float(v)
            for v in safe_relative[row]["y"]
        ])

        if not np.all(np.isnan(safe_y)):
            ax.plot(
                eps,
                safe_y,
                marker="o",
                markersize=7,
                linewidth=2,
                linestyle="-",
                color=colors[row],
                label=labels[row].replace("\n", " "),
            )

            for x_i, y_i, err_i in zip(
                eps,
                safe_relative[row]["y"],
                safe_relative[row]["err"],
            ):
                if y_i is None or err_i is None:
                    continue

                ax.errorbar(
                    x_i,
                    y_i,
                    yerr=err_i,
                    fmt="none",
                    capsize=4,
                    color=colors[row],
                )

        # ----------------------------------------------------
        # Unsafe refusal
        # ----------------------------------------------------

        unsafe_y = np.array([
            np.nan if v is None else float(v)
            for v in unsafe_relative[row]["y"]
        ])

        if not np.all(np.isnan(unsafe_y)):
            ax.plot(
                eps,
                unsafe_y,
                marker="s",
                markersize=6,
                linewidth=2,
                linestyle="--",
                color=colors[row],
                label="_nolegend_",
            )

            for x_i, y_i, err_i in zip(
                eps,
                unsafe_relative[row]["y"],
                unsafe_relative[row]["err"],
            ):
                if y_i is None or err_i is None:
                    continue

                ax.errorbar(
                    x_i,
                    y_i,
                    yerr=err_i,
                    fmt="none",
                    capsize=4,
                    color=colors[row],
                )

    ax.set_xlabel(r"Self $\epsilon$ (relative lr)")

    ax.set_xticks(eps)
    ax.set_xticklabels([
        "0.05",
        "0.10",
        "0.20",
        "0.40",
    ])

    style_axis(
        ax,
        f"XSTest — {MODEL_NAME} relative learning rate",
    )

    # --------------------------------------------------------
    # Training configuration legend
    # --------------------------------------------------------

    config_legend = ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Training data",
    )

    ax.add_artist(config_legend)

    # --------------------------------------------------------
    # Safe / unsafe legend
    # --------------------------------------------------------

    metric_handles = [
        Line2D(
            [0], [0],
            color="black",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Safe compliance",
        ),
        Line2D(
            [0], [0],
            color="black",
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Unsafe refusal",
        ),
    ]

    ax.legend(
        handles=metric_handles,
        frameon=False,
        bbox_to_anchor=(1.02, 0.53),
        loc="upper left",
        title="Metric",
    )

    fig.tight_layout()
    return fig, ax


# ============================================================
# Generate
# ============================================================

figures = []

fig, ax = plot_representation()
figures.append(fig)

fig, ax = plot_relative_lr()
figures.append(fig)


# ============================================================
# Save to ONE multipage PDF
# ============================================================

script_path = Path(__file__).resolve()
pdf_path = script_path.with_suffix(".pdf")

with PdfPages(pdf_path) as pdf:
    for fig in figures:
        pdf.savefig(fig, bbox_inches="tight")

print(f"Saved {len(figures)} figures to:\n{pdf_path}")

plt.show()
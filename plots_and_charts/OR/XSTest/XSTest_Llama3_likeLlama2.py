import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"

rows = [
    "only UltraChat",
    "OR (1000)",
    "trimmed OR (1000)",
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

eps = np.array([0.075, 0.150, 0.300, 0.600])


# ============================================================
# SAFE COMPLIANCE
# ============================================================

safe_github = {
    "only UltraChat":      None,
    "OR (1000)":           (42.93, 34.00),
    "trimmed OR (1000)":   None,
}

safe_references = {
    "Original Llama 3": 97.20,
    "HuggingFace CAT Llama 3": 28.00,
}

safe_relative = {
    "only UltraChat": {
        "y":   [48.13, 52.13, 20.67, 0.80],
        "err": [18.36, 29.31, 22.10, 0.40],
    },
    "OR (1000)": {
        "y":   [32.00, 81.60, 95.20, 11.33],
        "err": [None, None, None, 14.78],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 2.40, 2.00],
        "err": [None, None, None, 1.44],
    },
}


# ============================================================
# UNSAFE REFUSAL
# ============================================================

unsafe_github = {
    "only UltraChat":      None,
    "OR (1000)":           (98.00, 2.29),
    "trimmed OR (1000)":   None,
}

unsafe_references = {
    "Original Llama 3": 91.50,
    "HuggingFace CAT Llama 3": 99.50,
}

unsafe_relative = {
    "only UltraChat": {
        "y":   [92.67, 96.83, 98.50, 100.00],
        "err": [10.97, 3.51, 0.87, 0.00],
    },
    "OR (1000)": {
        "y":   [100.00, 87.00, 62.50, 96.50],
        "err": [None, None, None, 3.04],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 98.50, 99.17],
        "err": [None, None, None, 0.76],
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
#
# Safe compliance + unsafe refusal on SAME plot
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
    # Values above bars
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
        safe_references["Original Llama 3"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Safe — Original Llama 3 "
            f"({safe_references['Original Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        safe_references["HuggingFace CAT Llama 3"],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Safe — HF CAT Llama 3 "
            f"({safe_references['HuggingFace CAT Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["Original Llama 3"],
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Unsafe — Original Llama 3 "
            f"({unsafe_references['Original Llama 3']:.2f})"
        ),
    )

    ax.axhline(
        unsafe_references["HuggingFace CAT Llama 3"],
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=(
            "Unsafe — HF CAT Llama 3 "
            f"({unsafe_references['HuggingFace CAT Llama 3']:.2f})"
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
#
# Color    = training configuration
# solid/o  = safe compliance
# dashed/s = unsafe refusal
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
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

    style_axis(
        ax,
        f"XSTest — {MODEL_NAME} relative learning rate",
    )

    # Training configuration legend
    config_legend = ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Training data",
    )

    ax.add_artist(config_legend)

    # Safe/unsafe legend
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
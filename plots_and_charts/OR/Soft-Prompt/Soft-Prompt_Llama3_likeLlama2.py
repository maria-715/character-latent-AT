import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "Soft-Prompts"

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
    "only UltraChat":      "UltraChat",
    "OR (100)":            "OR\n100",
    "OR (500)":            "OR\n500",
    "OR (1000)":           "OR\n1000",
    "trimmed OR (1000)":   "Trimmed OR\n1000",
    "OR (3000)":           "OR\n3000",
}


# ============================================================
# GitHub Llama 3 representation
# ============================================================

github = {
    "only UltraChat":      None,          # P
    "OR (1000)":           (72.91, 1.03),
    "trimmed OR (1000)":   None,          # -
}

references = {
    "Original Llama 3": 75.41,
    "HuggingFace CAT Llama 3": 70.00,
}


# ============================================================
# Relative learning-rate representation
# ============================================================

eps = np.array([0.075, 0.150, 0.300, 0.600])

relative = {
    "only UltraChat": {
        "y":   [57.29, 59.25, 59.33, 64.46],
        "err": [9.37, 16.08, 1.41, 2.30],
    },

    "OR (1000)": {
        "y":   [79.18, 77.55, 82.26, 74.40],
        "err": [None, None, None, 4.91],
    },

    "trimmed OR (1000)": {
        "y":   [None, None, 74.65, 66.00],
        "err": [None, None, None, 8.70],
    },
}


# ============================================================
# Representation bar chart
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(9, 6))

    available_rows = [
        row for row in rows
        if github.get(row) is not None
    ]

    x = np.arange(len(available_rows))

    values = [
        github[row][0]
        for row in available_rows
    ]

    errors = [
        github[row][1]
        if github[row][1] is not None
        else 0
        for row in available_rows
    ]

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        width=0.60,
        color=[
            colors[row]
            for row in available_rows
        ],
        alpha=0.85,
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    original = references["Original Llama 3"]
    hf = references["HuggingFace CAT Llama 3"]

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
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    # Values above bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    all_values = values + [original, hf]

    ymin = min(all_values)
    ymax = max(all_values)

    padding = max(
        (ymax - ymin) * 0.25,
        1,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
    )

    ax.set_ylabel(
        f"{METRIC_NAME} score"
    )

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

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

    return fig, ax


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(9, 6))

    all_values = []

    for row in rows:

        y_raw = relative[row]["y"]
        err_raw = relative[row]["err"]

        y = np.array([
            np.nan if value is None else float(value)
            for value in y_raw
        ])

        if np.all(np.isnan(y)):
            continue

        all_values.extend(
            y[~np.isnan(y)].tolist()
        )

        # ----------------------------------------------------
        # Line + markers
        # ----------------------------------------------------

        ax.plot(
            eps,
            y,
            marker="o",
            markersize=7,
            linewidth=2,
            color=colors[row],
            label=row,
        )

        # ----------------------------------------------------
        # Error bars where reported
        # ----------------------------------------------------

        for x_i, y_i, err_i in zip(
            eps,
            y_raw,
            err_raw,
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

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Self $\epsilon$ (relative lr)"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_ylabel(
        f"{METRIC_NAME} score"
    )

    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)

        padding = max(
            (ymax - ymin) * 0.10,
            2,
        )

        ax.set_ylim(
            max(0, ymin - padding),
            ymax + padding,
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

    return fig, ax


# ============================================================
# Generate figures
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
        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

print(
    f"Saved {len(figures)} figures to:\n"
    f"{pdf_path}"
)

plt.show()
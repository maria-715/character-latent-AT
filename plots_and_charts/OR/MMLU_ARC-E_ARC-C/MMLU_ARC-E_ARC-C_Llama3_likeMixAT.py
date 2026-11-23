import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"

rows = [
    "only UltraChat",
    "OR (100)",
    "OR (500)",
    "OR (1000)",
    "trimmed OR (1000)",
    "OR (3000)",
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

eps = np.array([0.075, 0.150, 0.300, 0.600])


# ============================================================
# MMLU
# ============================================================

mmlu_github = {
    "only UltraChat":    (59.64, 0.26),
    "OR (100)":          (59.50, 0.41),
    "OR (500)":          (60.14, 0.25),
    "OR (1000)":         (59.96, 0.39),
    "trimmed OR (1000)": None,
    "OR (3000)":         (60.13, 0.13),
}

mmlu_references = {
    "Original Llama 3": 60.75,
    "HuggingFace CAT Llama 3": 59.24,
}

mmlu_relative = {
    "only UltraChat": {
        "y":   [59.73, 59.47, 58.51, 55.74],
        "err": [0.22, 0.14, 0.68, 0.41],
    },
    "OR (100)": {
        "y":   [60.03, 59.26, 56.65, 56.13],
        "err": [None, None, None, 0.74],
    },
    "OR (500)": {
        "y":   [60.54, 60.11, 57.84, 56.70],
        "err": [None, None, None, 0.22],
    },
    "OR (1000)": {
        "y":   [60.61, 58.36, 58.20, 56.09],
        "err": [None, None, None, 0.43],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 57.77, 53.68],
        "err": [None, None, None, 4.85],
    },
    "OR (3000)": {
        "y":   [60.00, 59.18, 58.81, 55.19],
        "err": [None, None, None, 0.61],
    },
}


# ============================================================
# ARC-E
# ============================================================

arce_github = {
    "only UltraChat":    (80.29, 0.38),
    "OR (100)":          (80.33, 0.28),
    "OR (500)":          (80.22, 0.61),
    "OR (1000)":         (80.57, 0.81),
    "trimmed OR (1000)": None,
    "OR (3000)":         (80.09, 0.04),
}

arce_references = {
    "Original Llama 3": 79.12,
    "HuggingFace CAT Llama 3": 80.93,
}

arce_relative = {
    "only UltraChat": {
        "y":   [80.46, 79.44, 75.45, 41.29],
        "err": [0.43, 0.29, 2.57, 19.88],
    },
    "OR (100)": {
        "y":   [80.72, 80.26, 72.10, 47.08],
        "err": [None, None, None, 15.04],
    },
    "OR (500)": {
        "y":   [80.51, 80.18, 76.43, 44.43],
        "err": [None, None, None, 7.04],
    },
    "OR (1000)": {
        "y":   [80.51, 78.37, 75.51, 41.20],
        "err": [None, None, None, 7.77],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 73.57, 39.56],
        "err": [None, None, None, 14.17],
    },
    "OR (3000)": {
        "y":   [80.64, 79.12, 75.00, 39.70],
        "err": [None, None, None, 4.76],
    },
}


# ============================================================
# ARC-C
# ============================================================

arcc_github = {
    "only UltraChat":    (51.05, 0.18),
    "OR (100)":          (50.20, 0.56),
    "OR (500)":          (50.60, 0.90),
    "OR (1000)":         (50.43, 0.09),
    "trimmed OR (1000)": None,
    "OR (3000)":         (50.03, 0.13),
}

arcc_references = {
    "Original Llama 3": 49.40,
    "HuggingFace CAT Llama 3": 50.77,
}

arcc_relative = {
    "only UltraChat": {
        "y":   [50.85, 49.34, 47.84, 32.37],
        "err": [0.39, 0.65, 1.67, 9.49],
    },
    "OR (100)": {
        "y":   [51.45, 50.26, 46.08, 35.90],
        "err": [None, None, None, 7.04],
    },
    "OR (500)": {
        "y":   [51.28, 50.17, 47.35, 34.50],
        "err": [None, None, None, 4.06],
    },
    "OR (1000)": {
        "y":   [50.94, 48.12, 45.05, 33.56],
        "err": [None, None, None, 4.02],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 45.73, 32.05],
        "err": [None, None, None, 8.57],
    },
    "OR (3000)": {
        "y":   [50.51, 48.38, 45.39, 32.14],
        "err": [None, None, None, 1.70],
    },
}


# ============================================================
# Common labels
# ============================================================

labels = {
    "only UltraChat": "UltraChat",
    "OR (100)": "OR\n100",
    "OR (500)": "OR\n500",
    "OR (1000)": "OR\n1000",
    "trimmed OR (1000)": "Trimmed OR\n1000",
    "OR (3000)": "OR\n3000",
}


# ============================================================
# Representation bar chart
# ============================================================

def plot_representation(data, references, metric):
    fig, ax = plt.subplots(figsize=(10, 6))

    available_rows = [
        row for row in rows
        if data.get(row) is not None
    ]

    x = np.arange(len(available_rows))

    values = [
        data[row][0]
        for row in available_rows
    ]

    errors = [
        data[row][1] if data[row][1] is not None else 0
        for row in available_rows
    ]

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        width=0.65,
        color=[colors[row] for row in available_rows],
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
    # Axis labels
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
            bar.get_height() + 0.5,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Adaptive y-axis
    # --------------------------------------------------------

    all_values = values + [original, hf]

    ymin = min(all_values)
    ymax = max(all_values)

    padding = max(
        (ymax - ymin) * 0.25,
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
        f"{metric} — {MODEL_NAME} representation",
        fontsize=14,
    )

    ax.set_ylabel(
        f"{metric} score"
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

def plot_relative_lr(data, metric):
    fig, ax = plt.subplots(figsize=(10, 6))

    all_numeric_values = []

    for row in rows:

        y = np.array([
            np.nan if v is None else float(v)
            for v in data[row]["y"]
        ])

        # Skip completely missing series
        if np.all(np.isnan(y)):
            continue

        all_numeric_values.extend(
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
            data[row]["y"],
            data[row]["err"],
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
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Self $\epsilon$ (relative lr)"
    )

    ax.set_ylabel(
        f"{metric} score"
    )

    ax.set_xticks(eps)

    ax.set_xticklabels([
        "0.075",
        "0.150",
        "0.300",
        "0.600",
    ])

    # --------------------------------------------------------
    # Adaptive y-axis
    # --------------------------------------------------------

    if all_numeric_values:

        ymin = min(all_numeric_values)
        ymax = max(all_numeric_values)

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
        f"{metric} — {MODEL_NAME} relative learning rate",
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
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Generate all six figures
# ============================================================

figures = []


# ------------------------------------------------------------
# MMLU
# ------------------------------------------------------------

fig, ax = plot_representation(
    mmlu_github,
    mmlu_references,
    "MMLU",
)
figures.append(fig)

fig, ax = plot_relative_lr(
    mmlu_relative,
    "MMLU",
)
figures.append(fig)


# ------------------------------------------------------------
# ARC-E
# ------------------------------------------------------------

fig, ax = plot_representation(
    arce_github,
    arce_references,
    "ARC-E",
)
figures.append(fig)

fig, ax = plot_relative_lr(
    arce_relative,
    "ARC-E",
)
figures.append(fig)


# ------------------------------------------------------------
# ARC-C
# ------------------------------------------------------------

fig, ax = plot_representation(
    arcc_github,
    arcc_references,
    "ARC-C",
)
figures.append(fig)

fig, ax = plot_relative_lr(
    arcc_relative,
    "ARC-C",
)
figures.append(fig)


# ============================================================
# Save ALL six figures to ONE multipage PDF
#
# The PDF automatically gets the same filename and directory
# as this Python script.
#
# Example:
#
#     capabilities_mixat.py
#
# creates:
#
#     capabilities_mixat.pdf
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
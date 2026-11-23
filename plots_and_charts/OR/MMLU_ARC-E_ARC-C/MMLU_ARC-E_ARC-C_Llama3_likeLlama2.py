import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


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

eps = np.array([0.075, 0.150, 0.300, 0.600])


# ============================================================
# MMLU
# ============================================================

mmlu_github = {
    "only UltraChat":      None,
    "OR (1000)":           (60.27, 0.29),
    "trimmed OR (1000)":   None,
}

mmlu_references = {
    "Original Llama 3": 60.75,
    "HuggingFace CAT Llama 3": 59.24,
}

mmlu_relative = {
    "only UltraChat": {
        "y":   [59.74, 59.35, 57.69, 54.94],
        "err": [0.17, 0.55, 0.45, 0.98],
    },
    "OR (1000)": {
        "y":   [60.08, 60.17, 58.33, 55.07],
        "err": [None, None, None, 1.59],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 58.51, 55.15],
        "err": [None, None, None, 1.57],
    },
}


# ============================================================
# ARC-E
# ============================================================

arce_github = {
    "only UltraChat":      None,
    "OR (1000)":           (80.60, 0.48),
    "trimmed OR (1000)":   None,
}

arce_references = {
    "Original Llama 3": 79.12,
    "HuggingFace CAT Llama 3": 80.93,
}

arce_relative = {
    "only UltraChat": {
        "y":   [80.80, 79.45, 74.99, 47.97],
        "err": [0.44, 0.81, 1.58, 4.30],
    },
    "OR (1000)": {
        "y":   [80.85, 80.13, 73.65, 42.34],
        "err": [None, None, None, 5.56],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 73.23, 36.97],
        "err": [None, None, None, 3.74],
    },
}


# ============================================================
# ARC-C
# ============================================================

arcc_github = {
    "only UltraChat":      None,
    "OR (1000)":           (50.88, 0.71),
    "trimmed OR (1000)":   None,
}

arcc_references = {
    "Original Llama 3": 49.40,
    "HuggingFace CAT Llama 3": 50.77,
}

arcc_relative = {
    "only UltraChat": {
        "y":   [51.45, 49.69, 47.32, 37.37],
        "err": [0.38, 1.00, 0.40, 2.10],
    },
    "OR (1000)": {
        "y":   [51.45, 49.66, 45.48, 32.85],
        "err": [None, None, None, 1.95],
    },
    "trimmed OR (1000)": {
        "y":   [None, None, 45.99, 32.20],
        "err": [None, None, None, 2.46],
    },
}


# ============================================================
# Labels
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
# Representation plot
# ============================================================

def plot_representation(data, references, metric):
    fig, ax = plt.subplots(figsize=(9, 6))

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
        width=0.60,
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
    # Labels
    # --------------------------------------------------------

    ax.set_xticks(x)
    ax.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    # Numerical value above each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
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
        1,
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

    ax.set_ylabel(f"{metric} score")

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
# Relative-lr plot
# ============================================================

def plot_relative_lr(data, metric):
    fig, ax = plt.subplots(figsize=(9, 6))

    all_values = []

    for row in rows:

        y = np.array([
            np.nan if v is None else float(v)
            for v in data[row]["y"]
        ])

        # Skip completely missing series
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
        # Error bars
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
    # Adaptive y-axis
    # --------------------------------------------------------

    ymin = min(all_values)
    ymax = max(all_values)

    padding = max(
        (ymax - ymin) * 0.10,
        1,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
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
# Example:
#
#     capabilities_llama2.py
#
# creates:
#
#     capabilities_llama2.pdf
#
# in the same directory.
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
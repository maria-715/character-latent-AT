import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


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

eps = np.array([0.05, 0.10, 0.20, 0.40])


# ============================================================
# MMLU
# ============================================================

mmlu_github = {
    "only UltraChat": (55.15, 0.22),
    "OR (1000)":      (55.44, None),
}

mmlu_references = {
    "Original Zephyr": 56.02,
    "HuggingFace CAT Zephyr": 54.08,
}

mmlu_relative = {
    "only UltraChat": {
        "y":   [55.05, 54.98, 55.17, 54.92],
        "err": [0.30, 0.26, 0.17, 0.17],
    },
    "OR (1000)": {
        "y":   [55.71, 55.38, 55.80, 55.16],
        "err": [None, None, None, None],
    },
}


# ============================================================
# ARC-E
# ============================================================

arce_github = {
    "only UltraChat": (78.22, 0.38),
    "OR (1000)":      (77.23, None),
}

arce_references = {
    "Original Zephyr": 81.00,
    "HuggingFace CAT Zephyr": 78.20,
}

arce_relative = {
    "only UltraChat": {
        "y":   [79.00, 78.52, 78.04, 77.08],
        "err": [0.26, 0.35, 0.40, 0.87],
    },
    "OR (1000)": {
        "y":   [77.95, 77.95, 77.61, 77.65],
        "err": [None, None, None, None],
    },
}


# ============================================================
# ARC-C
# ============================================================

arcc_github = {
    "only UltraChat": (50.24, 0.61),
    "OR (1000)":      (51.37, None),
}

arcc_references = {
    "Original Zephyr": 55.20,
    "HuggingFace CAT Zephyr": 51.50,
}

arcc_relative = {
    "only UltraChat": {
        "y":   [51.17, 51.08, 50.94, 49.32],
        "err": [0.35, 0.51, 0.76, 0.82],
    },
    "OR (1000)": {
        "y":   [51.62, 51.45, 52.30, 50.60],
        "err": [None, None, None, None],
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
        width=0.65,
        color=[colors[row] for row in available_rows],
        alpha=0.85,
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    original = references["Original Zephyr"]
    hf = references["HuggingFace CAT Zephyr"]

    ax.axhline(
        original,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Original Zephyr ({original:.2f})",
    )

    ax.axhline(
        hf,
        color="black",
        linestyle=":",
        linewidth=2.5,
        label=f"HF CAT Zephyr ({hf:.2f})",
    )

    # --------------------------------------------------------
    # Labels
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

    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)

        padding = max(
            (ymax - ymin) * 0.15,
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
        "0.05",
        "0.10",
        "0.20",
        "0.40",
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
# The PDF gets the same name and location as this .py file.
#
# Example:
#
#     capabilities_zephyr.py
#
# creates:
#
#     capabilities_zephyr.pdf
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
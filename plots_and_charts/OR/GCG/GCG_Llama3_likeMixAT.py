import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (MixAT)"
METRIC_NAME = "GCG"

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
    "only UltraChat":      (20.00, 15.21),
    "OR (100)":            (30.83, 1.44),
    "OR (500)":            (36.67, 14.22),
    "OR (1000)":           (40.83, 5.77),
    "trimmed OR (1000)":   None,  # -
    "OR (3000)":           None,  # P
}

references = {
    "Original Llama 3": 28.33,
    "HuggingFace CAT Llama 3": 0.00,
}


# ============================================================
# Relative learning-rate representation
# ============================================================

eps = np.array([0.075, 0.150, 0.300, 0.600])

relative = {
    "only UltraChat": {
        "y":   [25.83, 12.50, 17.50, 25.00],
        "err": [1.44, 2.50, 6.61, 10.90],
    },

    "OR (100)": {
        "y":   [37.50, 47.50, 57.50, 50.00],
        "err": [None, None, None, 12.99],
    },

    "OR (500)": {
        "y":   [37.50, 27.50, 50.00, 61.67],
        "err": [None, None, None, 16.07],
    },

    "OR (1000)": {
        "y":   [57.50, 27.50, 72.50, 52.50],
        "err": [None, None, None, 13.23],
    },

    "trimmed OR (1000)": {
        "y":   [None, None, 47.50, 54.17],
        "err": [None, None, None, 6.29],
    },

    "OR (3000)": {
        "y":   [None, None, None, None],  # P
        "err": [None, None, None, None],
    },
}


# ============================================================
# Representation bar chart
# ============================================================

def plot_representation():
    fig, ax = plt.subplots(figsize=(10, 6))

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
        width=0.65,
        color=[
            colors[row]
            for row in available_rows
        ],
        alpha=0.85,
    )

    # --------------------------------------------------------
    # References
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

    for bar, value, error in zip(
        bars,
        values,
        errors,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 1.5,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    max_value = max([
        value + error
        for value, error in zip(values, errors)
    ] + [original, hf])

    ax.set_ylim(
        0,
        min(100, max_value * 1.15 + 2),
    )

    ax.set_ylabel(
        f"{METRIC_NAME} score"
    )

    ax.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} representation",
        fontsize=14,
    )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    return fig


# ============================================================
# Relative learning-rate plot
# ============================================================

def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(10, 6))

    upper_values = []

    for row in rows:

        y_raw = relative[row]["y"]
        err_raw = relative[row]["err"]

        y = np.array([
            np.nan if value is None else float(value)
            for value in y_raw
        ])

        if np.all(np.isnan(y)):
            continue

        ax.plot(
            eps,
            y,
            marker="o",
            markersize=7,
            linewidth=2,
            color=colors[row],
            label=row,
        )

        for x_i, y_i, err_i in zip(
            eps,
            y_raw,
            err_raw,
        ):
            if y_i is None:
                continue

            error = 0 if err_i is None else err_i

            upper_values.append(
                y_i + error
            )

            if err_i is not None:
                ax.errorbar(
                    x_i,
                    y_i,
                    yerr=err_i,
                    fmt="none",
                    capsize=4,
                    color=colors[row],
                )

    # --------------------------------------------------------
    # Dynamic y-axis
    # --------------------------------------------------------

    data_max = max(upper_values)

    ax.set_ylim(
        0,
        min(100, data_max * 1.10 + 2),
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Self $\epsilon$ (relative lr)"
    )

    ax.set_ylabel(
        f"{METRIC_NAME} score"
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

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
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

plt.show()
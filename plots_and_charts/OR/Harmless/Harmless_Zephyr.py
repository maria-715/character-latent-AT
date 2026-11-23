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


# ============================================================
# GitHub CAT Zephyr representation
# ============================================================

github_40 = {
    "only UltraChat": (96.00, 4.18),
    "OR (1000)":      (92.50, None),
}

github_78k = {
    "only UltraChat": (62.03, 3.51),
    "OR (1000)":      (96.46, None),
}

references_40 = {
    "Original Zephyr": (100.00, None),
    "HuggingFace CAT Zephyr": (97.50, None),
}

references_78k = {
    "Original Zephyr": (98.32, None),
    "HuggingFace CAT Zephyr": (67.48, None),
}


# ============================================================
# Relative learning-rate representation
# ============================================================

eps = np.array([0.05, 0.1, 0.2, 0.4])

relative_40 = {
    "only UltraChat": {
        "y":   [98.33, 94.17, 89.17, 70.83],
        "err": [1.44, 3.11, 1.44, 9.29],
    },
    "OR (1000)": {
        "y":   [100.00, 95.00, 97.50, 75.00],
        "err": [None, None, None, None],
    },
}

relative_78k = {
    "only UltraChat": {
        "y":   [67.15, 64.84, 55.39, 61.55],
        "err": [5.99, 1.89, 6.83, 9.98],
    },
    "OR (1000)": {
        "y":   [98.39, 98.47, 98.19, 97.85],
        "err": [None, None, None, None],
    },
}


# ============================================================
# Plotting
# ============================================================

def style_axis(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Harmless score")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_representation(data, references, title):
    fig, ax = plt.subplots(figsize=(8, 6))

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
        color=[colors[row] for row in available_rows],
        alpha=0.85,
        width=0.65,
    )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    original_name = "Original Zephyr"
    hf_name = "HuggingFace CAT Zephyr"

    original = references[original_name][0]
    hf_cat = references[hf_name][0]

    ax.axhline(
        original,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"{original_name} ({original:.2f})",
    )

    ax.axhline(
        hf_cat,
        linestyle=":",
        linewidth=2.5,
        color="black",
        label=f"{hf_name} ({hf_cat:.2f})",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = {
        "only UltraChat": "UltraChat",
        "OR (1000)": "OR\n1000",
    }

    ax.set_xticks(x)
    ax.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    # Put numerical value above each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    style_axis(ax, title)

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()

    return fig, ax


def plot_relative_lr(data, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:

        y_raw = data[row]["y"]
        err_raw = data[row]["err"]

        y = np.array([
            np.nan if v is None else float(v)
            for v in y_raw
        ])

        # Skip completely missing series
        if np.all(np.isnan(y)):
            continue

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
        # Error bars only where explicitly reported
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

    ax.set_xlabel(r"Self $\epsilon$ (relative lr)")

    ax.set_xticks(eps)
    ax.set_xticklabels([
        f"{x:g}"
        for x in eps
    ])

    style_axis(ax, title)

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# Generate plots
# ============================================================

figures = []


# 1. Representation — on 40
fig, ax = plot_representation(
    github_40,
    references_40,
    f"Harmless — {MODEL_NAME} representation — on 40",
)
figures.append(fig)


# 2. Representation — on 78k
fig, ax = plot_representation(
    github_78k,
    references_78k,
    f"Harmless — {MODEL_NAME} representation — on 78k",
)
figures.append(fig)


# 3. Relative learning rate — on 40
fig, ax = plot_relative_lr(
    relative_40,
    f"Harmless — {MODEL_NAME} relative learning rate — on 40",
)
figures.append(fig)


# 4. Relative learning rate — on 78k
fig, ax = plot_relative_lr(
    relative_78k,
    f"Harmless — {MODEL_NAME} relative learning rate — on 78k",
)
figures.append(fig)


# ============================================================
# Save ALL figures to ONE multipage PDF
#
# Example:
#
#     harmless_zephyr.py
#
# creates:
#
#     harmless_zephyr.pdf
#
# in the same directory as the Python script.
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
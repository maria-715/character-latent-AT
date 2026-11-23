import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Llama 3 (Llama 2)"
METRIC_NAME = "AutoDAN"

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
# Data
# All values are directly stored on the 0-1 scale.
# ============================================================

github = {
    "only UltraChat":      None,  # P
    "OR (1000)":           (0.0083, 0.0144),
    "trimmed OR (1000)":   None,  # -
}

references = {
    "Original Llama 3": 0.1000,
    "HuggingFace CAT Llama 3": 0.0000,
}


eps = np.array([0.075, 0.150, 0.300, 0.600])

relative = {
    "only UltraChat": {
        "y":   [0.0000, 0.0000, 0.0000, 0.0000],
        "err": [0.0000, 0.0000, 0.0000, 0.0000],
    },

    "OR (1000)": {
        "y":   [0.0000, 0.0000, 0.0000, 0.0083],
        "err": [None, None, None, 0.0144],
    },

    "trimmed OR (1000)": {
        "y":   [None, None, 0.0000, 0.0000],
        "err": [None, None, None, 0.0000],
    },
}


# ============================================================
# Helpers
# ============================================================

def clean_axis(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines["right"].set_visible(False)


def add_break_marks(ax_top, ax_bottom):
    d = 0.008

    kwargs = dict(
        color="black",
        clip_on=False,
        linewidth=1,
    )

    ax_top.plot(
        (-d, +d),
        (-d, +d),
        transform=ax_top.transAxes,
        **kwargs,
    )
    ax_top.plot(
        (1 - d, 1 + d),
        (-d, +d),
        transform=ax_top.transAxes,
        **kwargs,
    )

    ax_bottom.plot(
        (-d, +d),
        (1 - d, 1 + d),
        transform=ax_bottom.transAxes,
        **kwargs,
    )
    ax_bottom.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        transform=ax_bottom.transAxes,
        **kwargs,
    )


# ============================================================
# Representation — broken y-axis
# ============================================================

def plot_representation():

    available_rows = [
        row for row in rows
        if github.get(row) is not None
    ]

    x = np.arange(len(available_rows))

    values = np.array([
        github[row][0]
        for row in available_rows
    ])

    errors = np.array([
        github[row][1]
        if github[row][1] is not None
        else 0
        for row in available_rows
    ])

    low_reference = references[
        "HuggingFace CAT Llama 3"
    ]

    high_reference = references[
        "Original Llama 3"
    ]

    experimental_max = np.max(
        values + errors
    )

    lower_max = max(
        experimental_max,
        low_reference,
        0.01,
    )

    lower_max *= 1.25

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9, 7),
        gridspec_kw={
            "height_ratios": [1, 3],
            "hspace": 0.06,
        },
    )

    bar_colors = [
        colors[row]
        for row in available_rows
    ]

    for ax in (ax_top, ax_bottom):

        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=5,
            width=0.60,
            color=bar_colors,
            alpha=0.85,
        )

        ax.axhline(
            high_reference,
            color="black",
            linestyle="--",
            linewidth=2,
            label=(
                f"Original Llama 3 "
                f"({high_reference:.3f})"
            ),
        )

        ax.axhline(
            low_reference,
            color="black",
            linestyle=":",
            linewidth=2.5,
            label=(
                f"HF CAT Llama 3 "
                f"({low_reference:.3f})"
            ),
        )

    # --------------------------------------------------------
    # Limits
    # --------------------------------------------------------

    ax_bottom.set_ylim(
        0,
        lower_max,
    )

    ax_top.set_ylim(
        0.09,
        0.11,
    )

    # --------------------------------------------------------
    # Broken-axis styling
    # --------------------------------------------------------

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    ax_top.tick_params(
        bottom=False,
        labelbottom=False,
    )

    add_break_marks(
        ax_top,
        ax_bottom,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax_bottom.set_xticks(x)

    ax_bottom.set_xticklabels([
        labels[row]
        for row in available_rows
    ])

    ax_bottom.set_ylabel(
        f"{METRIC_NAME} score"
    )

    for x_i, value, error in zip(
        x,
        values,
        errors,
    ):
        ax_bottom.text(
            x_i,
            value + error + lower_max * 0.05,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_top.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} representation",
        fontsize=14,
    )

    clean_axis(ax_top)
    clean_axis(ax_bottom)

    ax_top.legend(
        frameon=False,
        loc="upper right",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.12,
        top=0.92,
    )

    return fig


# ============================================================
# Relative LR — zoomed linear axis
# ============================================================

def plot_relative_lr():

    fig, ax = plt.subplots(figsize=(9, 6))

    upper_values = []

    for row in rows:

        y_raw = relative[row]["y"]
        err_raw = relative[row]["err"]

        y = np.array([
            np.nan if value is None else value
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

    y_max = max(
        data_max * 1.15,
        0.01,
    )

    ax.set_ylim(
        0,
        y_max,
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

    clean_axis(ax)
    ax.spines["top"].set_visible(False)

    ax.legend(
        frameon=False,
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
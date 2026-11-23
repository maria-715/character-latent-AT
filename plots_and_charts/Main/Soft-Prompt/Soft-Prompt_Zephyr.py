import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


MODEL_NAME = "Zephyr"
METRIC_NAME = "Soft-Prompts"
COLOR = "C0"

eps = np.array([0.05, 0.10, 0.20, 0.40])

github = (45.94, 6.08)

references = {
    "Original Zephyr": 74.09,
    "HuggingFace CAT Zephyr": 33.77,
}

relative = {
    "y":   [43.44, 43.40, 42.45, 51.13],
    "err": [4.53, 5.42, 13.65, 3.21],
}


def plot_representation():
    fig, ax = plt.subplots(figsize=(8, 6))

    value, error = github
    original = references["Original Zephyr"]
    hf = references["HuggingFace CAT Zephyr"]

    bar = ax.bar(
        [0], [value],
        yerr=[error],
        capsize=5,
        width=0.55,
        color=COLOR,
        alpha=0.85,
    )[0]

    ax.axhline(
        original, linestyle="--", linewidth=2, color="black",
        label=f"Original Zephyr ({original:.2f})",
    )
    ax.axhline(
        hf, linestyle=":", linewidth=2.5, color="black",
        label=f"HF CAT Zephyr ({hf:.2f})",
    )

    ax.text(
        0, value + error + 1,
        f"{value:.2f}",
        ha="center",
    )

    all_values = [value + error, original, hf]
    ymin, ymax = min(all_values), max(all_values)
    padding = max((ymax - ymin) * 0.15, 2)

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["UltraChat"])
    ax.set_ylabel("Soft-Prompts score (%)")
    ax.set_title(f"{METRIC_NAME} — {MODEL_NAME} representation")

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.array(relative["y"])
    err = np.array(relative["err"])

    ax.plot(
        eps, y,
        marker="o",
        markersize=7,
        linewidth=2,
        color=COLOR,
    )

    ax.errorbar(
        eps, y,
        yerr=err,
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    ymin = np.min(y - err)
    ymax = np.max(y + err)
    padding = max((ymax - ymin) * 0.10, 2)

    ax.set_ylim(
        max(0, ymin - padding),
        ymax + padding,
    )

    ax.set_xlabel(r"Configuration multiplier $s$")
    ax.set_ylabel("Soft-Prompts score (%)")

    ax.set_xticks(eps)
    ax.set_xticklabels(["0.05", "0.10", "0.20", "0.40"])

    ax.set_title(
        f"{METRIC_NAME} — {MODEL_NAME} relative learning rate"
    )

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


figures = [
    plot_representation(),
    plot_relative_lr(),
]

pdf_path = Path(__file__).resolve().with_suffix(".pdf")

with PdfPages(pdf_path) as pdf:
    for fig in figures:
        pdf.savefig(fig, bbox_inches="tight")

print(f"Saved to:\n{pdf_path}")

plt.show()
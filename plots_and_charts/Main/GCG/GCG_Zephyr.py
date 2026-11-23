import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


MODEL_NAME = "Zephyr"
METRIC_NAME = "GCG"
COLOR = "C0"

eps = np.array([0.05, 0.10, 0.20, 0.40])

github = (12.50, 5.86)

references = {
    "Original Zephyr": 92.50,
    "HuggingFace CAT Zephyr": 10.00,
}

relative = {
    "y":   [19.17, 17.50, 18.33, 9.17],
    "err": [3.82, 6.61, 10.10, 1.44],
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
        0, value + error + 2,
        f"{value:.2f}",
        ha="center",
    )

    ax.set_ylim(0, 100)
    ax.set_xticks([0])
    ax.set_xticklabels(["UltraChat"])

    ax.set_ylabel("GCG score (%)")
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

    ymax = np.max(y + err) * 1.15

    ax.set_ylim(0, ymax)

    ax.set_xlabel(r"Configuration multiplier $s$")
    ax.set_ylabel("GCG score (%)")

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
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


MODEL_NAME = "Zephyr"
METRIC_NAME = "Harmless"

COLOR = "C0"

eps = np.array([0.05, 0.10, 0.20, 0.40])

github = (96.00, 4.18)

references = {
    "Original Zephyr": 100.00,
    "HuggingFace CAT Zephyr": 97.50,
}

relative = {
    "y":   [98.33, 94.17, 89.17, 70.83],
    "err": [1.44, 3.82, 1.44, 9.46],
}


def plot_representation():
    fig, ax = plt.subplots(figsize=(8, 6))

    value, error = github

    bar = ax.bar(
        [0],
        [value],
        yerr=[error],
        capsize=5,
        width=0.55,
        color=COLOR,
        alpha=0.85,
    )[0]

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

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + error + 1,
        f"{value:.2f}",
        ha="center",
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["UltraChat"])

    ax.set_ylim(0, 110)
    ax.set_ylabel("Harmless score (%)")
    ax.set_title(f"{METRIC_NAME} — {MODEL_NAME} representation")

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_relative_lr():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        eps,
        relative["y"],
        marker="o",
        markersize=7,
        linewidth=2,
        color=COLOR,
        label="UltraChat",
    )

    ax.errorbar(
        eps,
        relative["y"],
        yerr=relative["err"],
        fmt="none",
        capsize=4,
        color=COLOR,
    )

    ax.set_xlabel(r"Configuration multiplier $s$")
    ax.set_ylabel("Harmless score (%)")

    ax.set_xticks(eps)
    ax.set_xticklabels(["0.05", "0.10", "0.20", "0.40"])

    ax.set_ylim(0, 110)

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

script_path = Path(__file__).resolve()
pdf_path = script_path.with_suffix(".pdf")

with PdfPages(pdf_path) as pdf:
    for fig in figures:
        pdf.savefig(fig, bbox_inches="tight")

print(f"Saved to:\n{pdf_path}")

plt.show()
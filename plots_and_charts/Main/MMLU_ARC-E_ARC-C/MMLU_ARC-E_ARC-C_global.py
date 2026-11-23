import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Typography
# ============================================================

plt.rcParams.update({
    "font.size": 13,          # default text
    "axes.titlesize": 14,     # panel titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 12,    # x tick labels
    "ytick.labelsize": 12,    # y tick labels
    "legend.fontsize": 12,    # legend
})


# ============================================================
# Configuration
# ============================================================

COLORS = {
    "Zephyr": "C0",               # blue
    "Llama 3 (Llama 2)": "C1",   # orange
    "Llama 3 (MixAT)": "C2",     # green
}

MARKERS = {
    "Zephyr": "o",
    "Llama 3 (Llama 2)": "s",
    "Llama 3 (MixAT)": "^",
}


# ============================================================
# Data
# ============================================================

models = {

    # ========================================================
    # Zephyr
    # ========================================================

    "Zephyr": {
        "eps": np.array([
            0.0,
            0.05,
            0.1,
            0.2,
            0.4,
        ]),

        "MMLU": {
            "github": (55.15, 0.22),
            "original": 56.02,
            "hf": 54.08,
            "y": np.array([
                56.02,
                55.05,
                54.98,
                55.17,
                54.92,
            ]),
            "err": np.array([
                0.0,
                0.30,
                0.26,
                0.17,
                0.17,
            ]),
        },

        "ARC-E": {
            "github": (78.22, 0.38),
            "original": 81.00,
            "hf": 78.20,
            "y": np.array([
                81.00,
                79.00,
                78.52,
                78.04,
                77.08,
            ]),
            "err": np.array([
                0.0,
                0.26,
                0.36,
                0.40,
                0.88,
            ]),
        },

        "ARC-C": {
            "github": (50.24, 0.61),
            "original": 55.20,
            "hf": 51.50,
            "y": np.array([
                55.20,
                51.17,
                51.08,
                50.94,
                49.32,
            ]),
            "err": np.array([
                0.0,
                0.35,
                0.50,
                0.76,
                0.82,
            ]),
        },
    },

    # ========================================================
    # Llama 3 (Llama 2)
    # ========================================================

    "Llama 3 (Llama 2)": {
        "eps": np.array([
            0.0,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.6,
        ]),

        "MMLU": {
            "github": (60.03, 0.41),
            "original": 60.75,
            "hf": 59.24,
            "y": np.array([
                60.75,
                60.10,
                59.74,
                59.83,
                59.35,
                58.81,
                57.69,
                56.23,
                54.94,
            ]),
            "err": np.array([
                0.0,
                0.09,
                0.17,
                0.13,
                0.55,
                0.11,
                0.45,
                1.04,
                0.98,
            ]),
        },

        "ARC-E": {
            "github": (80.95, 0.30),
            "original": 79.12,
            "hf": 80.93,
            "y": np.array([
                79.12,
                80.95,
                80.80,
                80.78,
                79.45,
                78.05,
                74.99,
                73.18,
                47.97,
            ]),
            "err": np.array([
                0.0,
                0.30,
                0.44,
                0.18,
                0.81,
                1.21,
                1.58,
                2.82,
                4.30,
            ]),
        },

        "ARC-C": {
            "github": (51.88, 0.53),
            "original": 49.40,
            "hf": 50.77,
            "y": np.array([
                49.40,
                51.77,
                51.45,
                51.31,
                49.69,
                49.72,
                47.32,
                46.22,
                37.37,
            ]),
            "err": np.array([
                0.0,
                0.43,
                0.38,
                0.39,
                1.00,
                0.35,
                0.40,
                1.92,
                2.10,
            ]),
        },
    },

    # ========================================================
    # Llama 3 (MixAT)
    # ========================================================

    "Llama 3 (MixAT)": {
        "eps": np.array([
            0.0,
            0.075,
            0.15,
            0.3,
            0.6,
        ]),

        "MMLU": {
            "github": (59.64, 0.26),
            "original": 60.75,
            "hf": 59.24,
            "y": np.array([
                60.75,
                59.73,
                59.47,
                58.51,
                55.74,
            ]),
            "err": np.array([
                0.0,
                0.22,
                0.14,
                0.68,
                0.41,
            ]),
        },

        "ARC-E": {
            "github": (80.29, 0.38),
            "original": 79.12,
            "hf": 80.93,
            "y": np.array([
                79.12,
                80.46,
                79.44,
                75.45,
                41.29,
            ]),
            "err": np.array([
                0.0,
                0.43,
                0.29,
                2.54,
                19.91,
            ]),
        },

        "ARC-C": {
            "github": (51.05, 0.18),
            "original": 49.40,
            "hf": 50.77,
            "y": np.array([
                49.40,
                50.85,
                49.34,
                47.84,
                32.37,
            ]),
            "err": np.array([
                0.0,
                0.39,
                0.65,
                1.65,
                9.46,
            ]),
        },
    },
}


# ============================================================
# Combined representation plot
# ============================================================

def plot_representation(metric):

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(models.keys())
    x = np.arange(len(model_names))

    all_values = []

    # --------------------------------------------------------
    # GitHub bars
    # --------------------------------------------------------

    for i, model_name in enumerate(model_names):

        data = models[model_name][metric]

        value, error = data["github"]
        color = COLORS[model_name]

        bar = ax.bar(
            i,
            value,
            yerr=error,
            capsize=5,
            width=0.60,
            color=color,
            alpha=0.85,
            label=model_name,
            zorder=3,
        )[0]

        all_values.extend([
            value - error,
            value + error,
        ])

        # Numerical value above bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.15,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            # fontsize=10,
        )

    # --------------------------------------------------------
    # Reference lines
    #
    # Short horizontal segments are used instead of full-width
    # lines so each reference clearly belongs to its model.
    #
    # Dashed = Original
    # Dotted = HuggingFace CAT
    # --------------------------------------------------------

    reference_half_width = 0.42

    for i, model_name in enumerate(model_names):

        data = models[model_name][metric]

        original = data["original"]
        hf = data["hf"]

        color = COLORS[model_name]

        # Original model
        ax.hlines(
            original,
            i - reference_half_width,
            i + reference_half_width,
            color=color,
            linestyle="--",
            linewidth=2,
            zorder=4,
        )

        # HuggingFace CAT
        ax.hlines(
            hf,
            i - reference_half_width,
            i + reference_half_width,
            color=color,
            linestyle=":",
            linewidth=2.5,
            zorder=4,
        )

        all_values.extend([
            original,
            hf,
        ])

    # --------------------------------------------------------
    # Adaptive y-axis
    # --------------------------------------------------------

    ymin = min(all_values)
    ymax = max(all_values)

    padding = max(
        (ymax - ymin) * 0.15,
        0.75,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        min(100, ymax + padding),
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        model_names
    )

    ax.set_ylabel(
        f"{metric} score (%)"
    )

    ax.set_title(
        f"{metric} — Representation Comparison",
        # fontsize=14,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Legend
    #
    # First legend = models/colors
    # Second legend = meaning of reference line styles
    # --------------------------------------------------------

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=10,
            color=COLORS[name],
            label=name,
        )
        for name in model_names
    ]

    reference_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Original model",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=2.5,
            label="HuggingFace CAT",
        ),
    ]

    handles = model_handles + reference_handles

    ax.legend(
        handles=handles,
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    return fig


# ============================================================
# Combined configuration multiplier plot
# ============================================================

# ============================================================
# Three-panel configuration multiplier plot
# ============================================================

def plot_capability_sweeps():

    metrics = [
        ("MMLU", "MMLU"),
        ("ARC-E", "ARC-Easy"),
        ("ARC-C", "ARC-Challenge"),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 11),
        sharex=True,
    )

    # Combined multiplier grid
    all_eps = np.unique(
        np.concatenate([
            model_data["eps"]
            for model_data in models.values()
        ])
    )

    labels = []

    for value in all_eps:
        if np.isclose(value, 0.0):
            labels.append("0")
        elif np.isclose(value, 0.05):
            labels.append("0.05")
        elif np.isclose(value, 0.075):
            labels.append("0.075")
        elif np.isclose(value, 0.1):
            labels.append("0.1")
        elif np.isclose(value, 0.150):
            labels.append("0.15")
        elif np.isclose(value, 0.2):
            labels.append("0.2")
        elif np.isclose(value, 0.300):
            labels.append("0.3")
        elif np.isclose(value, 0.4):
            labels.append("0.4")
        elif np.isclose(value, 0.600):
            labels.append("0.6")
        else:
            labels.append(f"{value:.2f}")

    # --------------------------------------------------------
    # Plot each capability metric
    # --------------------------------------------------------

    for ax, (metric, panel_title) in zip(axes, metrics):

        all_lower = []
        all_upper = []

        for model_name, model_data in models.items():

            eps = model_data["eps"]
            data = model_data[metric]

            y = data["y"]
            err = data["err"]

            color = COLORS[model_name]
            marker = MARKERS[model_name]

            # Mean trajectory
            ax.plot(
                eps,
                y,
                marker=marker,
                markersize=6,
                linewidth=1.8,
                color=color,
                label=model_name,
                zorder=3,
            )

            # Standard deviation
            ax.errorbar(
                eps,
                y,
                yerr=err,
                fmt="none",
                capsize=3,
                linewidth=1.2,
                color=color,
                zorder=2,
            )

            all_lower.extend(y - err)
            all_upper.extend(y + err)

        # ----------------------------------------------------
        # Adaptive y-axis for this panel
        # ----------------------------------------------------

        ymin = min(all_lower)
        ymax = max(all_upper)

        padding = max(
            (ymax - ymin) * 0.10,
            0.5,
        )

        ax.set_ylim(
            max(0, ymin - padding),
            min(100, ymax + padding),
        )

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        ax.set_xticks(all_eps)
        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )

        ax.set_xlim(left=0)
        # ax.margins(x=0)

        ax.tick_params(
            axis="x",
            labelbottom=True,
        )

        ax.set_title(
            panel_title,
            # fontsize=11,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Shared labels
    # --------------------------------------------------------

    # fig.supxlabel(
    #     r"Configuration multiplier $s$",
    #     # fontsize=11,
    # )

    axes[-1].set_xlabel(
        r"Configuration multiplier $s$",
        labelpad=3.5,
    )

    fig.supylabel(
        "Score (%)",
        # fontsize=11,
    )

    # --------------------------------------------------------
    # One shared legend
    # --------------------------------------------------------

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    # Leave room for shared legend and labels
    fig.tight_layout(
        rect=[0.04, 0.04, 1.0, 0.94]
    )
    fig.subplots_adjust(hspace=0.45)  # room between panels

    return fig


# ============================================================
# Generate figures
# ============================================================

figures = []


# ------------------------------------------------------------
# Representation comparisons
# ------------------------------------------------------------

for metric in ["MMLU", "ARC-E", "ARC-C"]:
    figures.append(
        plot_representation(metric)
    )


# ------------------------------------------------------------
# Capability sweeps
#
# One figure containing three panels:
# MMLU | ARC-Easy | ARC-Challenge
# ------------------------------------------------------------

capability_figure = plot_capability_sweeps()

figures.append(
    capability_figure
)


# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent


# ============================================================
# Save thesis capability figure separately
# ============================================================

capability_pdf_path = (
    output_dir / "task_capability_sweeps.pdf"
)

capability_png_path = (
    output_dir / "task_capability_sweeps.png"
)

capability_figure.savefig(
    capability_pdf_path,
    bbox_inches="tight",
)

capability_figure.savefig(
    capability_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save all figures to one diagnostic PDF
# ============================================================

combined_pdf_path = (
    output_dir / f"{script_path.stem}.pdf"
)

with PdfPages(combined_pdf_path) as pdf:

    for fig in figures:

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )


# ============================================================
# Report outputs
# ============================================================

print(
    "Saved capability figure to:\n"
    f"  {capability_pdf_path}\n"
    f"  {capability_png_path}\n\n"
    "Saved all figures to:\n"
    f"  {combined_pdf_path}"
)


# ============================================================
# Display
# ============================================================

plt.show()
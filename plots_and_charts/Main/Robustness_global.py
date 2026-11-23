import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Configuration
# ============================================================

COLORS = {
    "Zephyr": "C0",
    "Llama 3 (Llama 2)": "C1",
    "Llama 3 (MixAT)": "C2",
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
            0.10,
            0.20,
            0.40,
        ]),

        "GCG": {
            "y": np.array([
                92.50,
                19.17,
                17.50,
                18.33,
                9.17,
            ]),
            "err": np.array([
                0.0,
                3.82,
                6.61,
                10.10,
                1.44,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                90.0,
                1.67,
                2.50,
                0.83,
                0.00,
            ]),
            "err": np.array([
                0.0,
                1.44,
                0.00,
                1.44,
                0.00,
            ]),
        },

        "Soft-Prompts": {
            "y": np.array([
                74.09,
                43.44,
                43.40,
                42.45,
                51.13,
            ]),
            "err": np.array([
                0.0,
                4.53,
                5.42,
                13.65,
                3.21,
            ]),
        },

        "Inpainting": {
            "y": np.array([
                100.0,
                91.67,
                95.67,
                95.67,
                96.00,
            ]),
            "err": np.array([
                0.0,
                4.93,
                2.52,
                1.15,
                3.46,
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
            0.10,
            0.150,
            0.20,
            0.300,
            0.40,
            0.600,
        ]),

        "GCG": {
            "y": np.array([
                28.33,
                22.50,
                20.83,
                16.67,
                22.50,
                19.17,
                19.17,
                22.50,
                33.33,
            ]),
            "err": np.array([
                0.0,
                6.61,
                3.82,
                10.41,
                12.99,
                7.22,
                11.55,
                5.00,
                9.46,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                10.0,
                1.67,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
            ]),
            "err": np.array([
                0.0,
                2.89,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
            ]),
        },

        "Soft-Prompts": {
            "y": np.array([
                75.41,
                52.62,
                57.29,
                48.32,
                59.25,
                51.26,
                59.33,
                62.53,
                64.46,
            ]),
            "err": np.array([
                0.0,
                2.55,
                9.37,
                1.89,
                16.08,
                3.17,
                1.41,
                5.64,
                2.30,
            ]),
        },

        # Inpainting is only measured at:
        # 0.05, 0.10, 0.20, 0.40.
        #
        # NaN values preserve the missing measurements.
        "Inpainting": {
            "y": np.array([
                97.00,
                94.00,
                np.nan,
                89.67,
                np.nan,
                88.67,
                np.nan,
                93.67,
                np.nan,
            ]),
            "err": np.array([
                0.0,
                5.29,
                np.nan,
                3.06,
                np.nan,
                2.52,
                np.nan,
                3.21,
                np.nan,
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
            0.150,
            0.300,
            0.600,
        ]),

        "GCG": {
            "y": np.array([
                28.33,
                25.83,
                12.50,
                17.50,
                25.00,
            ]),
            "err": np.array([
                0.0,
                1.44,
                2.50,
                6.61,
                10.90,
            ]),
        },

        "AutoDAN": {
            "y": np.array([
                10.0,
                0.83,
                0.00,
                0.00,
                0.00,
            ]),
            "err": np.array([
                0.0,
                1.44,
                0.00,
                0.00,
                0.00,
            ]),
        },

        "Soft-Prompts": {
            "y": np.array([
                75.41,
                55.47,
                43.23,
                58.18,
                63.21,
            ]),
            "err": np.array([
                0.0,
                4.46,
                3.92,
                4.86,
                3.20,
            ]),
        },

        "Inpainting": {
            "y": np.array([
                97.00,
                87.67,
                89.67,
                94.00,
                89.67,
            ]),
            "err": np.array([
                0.0,
                6.51,
                4.04,
                1.73,
                1.15,
            ]),
        },
    },
}

# ============================================================
# Global scale factor — bump this to resize everything
# proportionally (figure, fonts, markers, lines)
# ============================================================

SCALE = 1.0  # e.g. 1.0 = current size, 1.4 = 40% bigger

plt.rcParams.update({
    "font.size":        13 * SCALE,
    "axes.titlesize":   14 * SCALE,
    "axes.labelsize":   13 * SCALE,
    "xtick.labelsize":  11 * SCALE,
    "ytick.labelsize":  11 * SCALE,
    "legend.fontsize":  11 * SCALE,
    "lines.linewidth":  1.8 * SCALE,
    "lines.markersize": 6 * SCALE,
})


# ============================================================
# Four-panel attack sweep plot
# ============================================================

def plot_attack_sweeps(metrics, figsize=(8, 8.5), show_legend=True):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
    )

    # ========================================================
    # Combined multiplier grid
    # ========================================================

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
        elif np.isclose(value, 0.10):
            labels.append("0.1")
        elif np.isclose(value, 0.150):
            labels.append("0.15")
        elif np.isclose(value, 0.20):
            labels.append("0.2")
        elif np.isclose(value, 0.300):
            labels.append("0.3")
        elif np.isclose(value, 0.40):
            labels.append("0.4")
        elif np.isclose(value, 0.600):
            labels.append("0.6")
        else:
            labels.append(f"{value:.2f}")

    # ========================================================
    # Plot each attack
    # ========================================================

    for ax, (metric, panel_title) in zip(
        axes,
        metrics,
    ):

        all_lower = []
        all_upper = []

        for model_name, model_data in models.items():

            eps = model_data["eps"]
            data = model_data[metric]

            y = data["y"]
            err = data["err"]

            color = COLORS[model_name]
            marker = MARKERS[model_name]

            # ------------------------------------------------
            # Keep only actual measurements
            # ------------------------------------------------

            valid = (
                ~np.isnan(y)
                & ~np.isnan(err)
            )

            valid_eps = eps[valid]
            valid_y = y[valid]
            valid_err = err[valid]

            # ------------------------------------------------
            # Mean trajectory
            # ------------------------------------------------

            ax.plot(
                valid_eps,
                valid_y,
                marker=marker,
                markersize=6 * SCALE,
                linewidth=1.8 * SCALE,
                color=color,
                label=model_name,
                zorder=3,
            )

            # ------------------------------------------------
            # Standard deviation
            # ------------------------------------------------

            ax.errorbar(
                valid_eps,
                valid_y,
                yerr=valid_err,
                fmt="none",
                capsize=3 * SCALE,
                linewidth=1.2 * SCALE,
                color=color,
                zorder=2,
            )

            all_lower.extend(
                valid_y - valid_err
            )

            all_upper.extend(
                valid_y + valid_err
            )

        # ====================================================
        # Adaptive y-axis
        # ====================================================

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

        # ====================================================
        # X-axis
        # ====================================================

        ax.set_xticks(all_eps)

        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )

        ax.set_xlim(left=0)

        # Show x labels on both panels
        ax.tick_params(
            axis="x",
            labelbottom=True,
        )

        # ====================================================
        # Styling
        # ====================================================

        ax.set_title(
            panel_title,
            # fontsize=11 * SCALE,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ========================================================
    # Shared labels
    # ========================================================

    # fig.supxlabel(
    #     r"Configuration multiplier $s$",
    #     # fontsize=11 * SCALE,
    # )

    axes[-1].set_xlabel(
        r"Configuration multiplier $s$",
        labelpad=3.5,
    )

    fig.supylabel(
        "ASR (%)",
        # fontsize=11 * SCALE,
        x=0.08,
    )

    # ========================================================
    # Shared legend
    # ========================================================

    if show_legend:

        handles, legend_labels = (
            axes[0].get_legend_handles_labels()
        )

        fig.legend(
            handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
        )

        top = 0.93

    else:

        top = 0.97

    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout(
        rect=[
            0.04,
            0.04,
            1.0,
            top,
        ]
    )

    fig.subplots_adjust(
        hspace=0.45,
    )

    return fig


def plot_attack_sweeps_broken(figsize=(8, 8.5), show_legend=True):

    fig = plt.figure(figsize=figsize)

    # ========================================================
    # Two equal-height panels, exactly like the original figure
    # ========================================================

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1, 1],
        hspace=0.45,
    )

    # --------------------------------------------------------
    # GCG broken axis
    # --------------------------------------------------------

    gs_gcg = outer[0].subgridspec(
        2,
        1,
        height_ratios=[1, 3],
        hspace=0.05,
    )

    gcg_top = fig.add_subplot(gs_gcg[0])
    gcg_bottom = fig.add_subplot(
        gs_gcg[1],
        sharex=gcg_top,
    )

    # --------------------------------------------------------
    # AutoDAN broken axis
    # --------------------------------------------------------

    gs_autodan = outer[1].subgridspec(
        2,
        1,
        height_ratios=[1, 3],
        hspace=0.05,
    )

    autod_top = fig.add_subplot(
        gs_autodan[0],
        sharex=gcg_top,
    )

    autod_bottom = fig.add_subplot(
        gs_autodan[1],
        sharex=gcg_top,
    )

    metric_axes = [
        (
            "GCG",
            "GCG",
            gcg_top,
            gcg_bottom,
            (80, 100),
            (0, 45),
        ),
        (
            "AutoDAN",
            "AutoDAN",
            autod_top,
            autod_bottom,
            (80, 100),
            (0, 10.5),
        ),
    ]

    # ========================================================
    # Combined multiplier grid
    # ========================================================

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
        elif np.isclose(value, 0.10):
            labels.append("0.1")
        elif np.isclose(value, 0.150):
            labels.append("0.15")
        elif np.isclose(value, 0.20):
            labels.append("0.2")
        elif np.isclose(value, 0.300):
            labels.append("0.3")
        elif np.isclose(value, 0.40):
            labels.append("0.4")
        elif np.isclose(value, 0.600):
            labels.append("0.6")
        else:
            labels.append(f"{value:.2f}")

    # ========================================================
    # Plot
    # ========================================================

    for (
        metric,
        title,
        ax_top,
        ax_bottom,
        top_ylim,
        bottom_ylim,
    ) in metric_axes:

        for model_name, model_data in models.items():

            eps = model_data["eps"]
            data = model_data[metric]

            y = data["y"]
            err = data["err"]

            color = COLORS[model_name]
            marker = MARKERS[model_name]

            valid = (
                ~np.isnan(y)
                & ~np.isnan(err)
            )

            valid_eps = eps[valid]
            valid_y = y[valid]
            valid_err = err[valid]

            # ------------------------------------------------
            # Plot same data on both halves
            # ------------------------------------------------

            for ax in (ax_top, ax_bottom):

                ax.plot(
                    valid_eps,
                    valid_y,
                    marker=marker,
                    markersize=6 * SCALE,
                    linewidth=1.8 * SCALE,
                    color=color,
                    label=model_name,
                    zorder=3,
                )

                ax.errorbar(
                    valid_eps,
                    valid_y,
                    yerr=valid_err,
                    fmt="none",
                    capsize=3 * SCALE,
                    linewidth=1.2 * SCALE,
                    color=color,
                    zorder=2,
                )

        # ====================================================
        # Y limits
        # ====================================================

        ax_top.set_ylim(*top_ylim)
        ax_bottom.set_ylim(*bottom_ylim)

        # s = 0 exactly on y-axis
        ax_top.set_xlim(left=0)
        ax_bottom.set_xlim(left=0)

        # ====================================================
        # Broken-axis appearance
        # ====================================================

        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)

        ax_top.spines["right"].set_visible(False)
        ax_bottom.spines["right"].set_visible(False)

        ax_top.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False,
        )

        # ----------------------------------------------------
        # Diagonal break marks
        # ----------------------------------------------------

        d = 0.008

        kwargs = dict(
            color="black",
            clip_on=False,
            linewidth=1.0 * SCALE,
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

        # ====================================================
        # Grid
        # ====================================================

        ax_top.grid(
            axis="y",
            alpha=0.25,
        )

        ax_bottom.grid(
            axis="y",
            alpha=0.25,
        )

        # ====================================================
        # Title
        # ====================================================

        ax_top.set_title(
            title,
            pad=5,
        )

        # ====================================================
        # X ticks
        # ====================================================

        ax_bottom.set_xticks(all_eps)

        ax_bottom.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )

    # ========================================================
    # Only show x labels on AutoDAN
    # ========================================================

    gcg_bottom.tick_params(
        axis="x",
        labelbottom=False,
    )

    autod_bottom.tick_params(
        axis="x",
        labelbottom=True,
    )

    # ========================================================
    # Shared labels
    # ========================================================

    autod_bottom.set_xlabel(
        r"Configuration multiplier $s$",
        labelpad=3.5,
    )

    fig.supylabel(
        "ASR (%)"
    )

    # ========================================================
    # Legend
    # ========================================================

    if show_legend:

        handles, legend_labels = (
            gcg_top.get_legend_handles_labels()
        )

        fig.legend(
            handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
        )

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.90,
    )

    return fig


# ============================================================
# Generate figures
# ============================================================

# Page 1
# attack_figure_part1 = plot_attack_sweeps(
#     metrics=[
#         ("GCG", "GCG"),
#         ("AutoDAN", "AutoDAN"),
#     ],
#     figsize=(8 * SCALE, 8.5 * SCALE),
#     show_legend=True,
# )
attack_figure_part1 = plot_attack_sweeps_broken(
    figsize=(8 * SCALE, 8.5 * SCALE),
    show_legend=True,
)

# Page 2
attack_figure_part2 = plot_attack_sweeps(
    metrics=[
        ("Soft-Prompts", "Soft-Prompts"),
        ("Inpainting", "Inpainting"),
    ],
    figsize=(8 * SCALE, 8.5 * SCALE),
    show_legend=False,
)

# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

# ============================================================
# Individual PDF / PNG files
# ============================================================

part1_pdf_path = (
    output_dir / "Robustness_global_part1.pdf"
)

part1_png_path = (
    output_dir / "Robustness_global_part1.png"
)

part2_pdf_path = (
    output_dir / "Robustness_global_part2.pdf"
)

part2_png_path = (
    output_dir / "Robustness_global_part2.png"
)


# ============================================================
# Save page 1
# ============================================================

attack_figure_part1.savefig(
    part1_pdf_path,
    bbox_inches="tight",
)

attack_figure_part1.savefig(
    part1_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save page 2
# ============================================================

attack_figure_part2.savefig(
    part2_pdf_path,
    bbox_inches="tight",
)

attack_figure_part2.savefig(
    part2_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save both pages into one two-page PDF
# ============================================================

combined_pdf_path = (
    output_dir / "Robustness_global_combined.pdf"
)

with PdfPages(combined_pdf_path) as pdf:

    pdf.savefig(
        attack_figure_part1,
        bbox_inches="tight",
    )

    pdf.savefig(
        attack_figure_part2,
        bbox_inches="tight",
    )

# ============================================================
# Report outputs
# ============================================================

print(
    "Saved attack figures to:\n"
    f"  {part1_pdf_path}\n"
    f"  {part1_png_path}\n"
    f"  {part2_pdf_path}\n"
    f"  {part2_png_path}\n\n"
    "Saved combined two-page PDF to:\n"
    f"  {combined_pdf_path}"
)

# ============================================================
# Display
# ============================================================

plt.show()
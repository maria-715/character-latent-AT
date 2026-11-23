import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ============================================================
# Typography
# ============================================================

plt.rcParams.update({
    "font.size": 13,          # default text
    "axes.titlesize": 14,     # panel titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 11,    # x tick labels
    "ytick.labelsize": 11,    # y tick labels
    "legend.fontsize": 12,    # legend
})

XMIN = -0.015
XMAX = 0.375


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
#
# Only controlled sweep checkpoints are included.
#
# D:
#   Realised CAT training perturbation distance.
#   No SD is plotted because the table reports these values
#   without standard deviations.
#
# D_SP:
#   Realised Soft-Prompt perturbation distance.
#
# SP_ASR:
#   Soft-Prompt attack success rate.
# ============================================================

models = {

    # ========================================================
    # Zephyr
    # ========================================================

    "Zephyr": {
        "epsilon": np.array([
            0.0,
            0.00884,
            0.01768,
            0.03535,
            0.07070,
        ]),

        "D": np.array([
            0.0,
            0.00734,
            0.01466,
            0.02688,
            0.04422,
        ]),

        "D_SP": {
            "y": np.array([
                0.5467,
                0.6065,
                0.5860,
                0.5803,
                0.5288,
            ]),
            "err": np.array([
                0.0,
                0.0157,
                0.0232,
                0.0252,
                0.0071,
            ]),
        },

        "SP_ASR": {
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
    },

    # ========================================================
    # Llama 3 (Llama 2)
    # ========================================================

    "Llama 3 (Llama 2)": {
        "epsilon": np.array([
            0.0,
            0.02949,
            0.04424,
            0.05898,
            0.08848,
            0.11797,
            0.17695,
            0.23594,
            0.35391,
        ]),

        "D": np.array([
            0.0,
            0.02492,
            0.03491,
            0.04517,
            0.06035,
            0.07039,
            0.09350,
            0.10172,
            0.13176,
        ]),

        "D_SP": {
            "y": np.array([
                0.5436,
                0.6416,
                0.6058,
                0.5828,
                0.5415,
                0.5865,
                0.5555,
                0.5551,
                0.5573,
            ]),
            "err": np.array([
                0.0,
                0.0367,
                0.0575,
                0.0104,
                0.0087,
                0.0272,
                0.0128,
                0.0054,
                0.0162,
            ]),
        },

        "SP_ASR": {
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
    },

    # ========================================================
    # Llama 3 (MixAT)
    # ========================================================

    "Llama 3 (MixAT)": {
        "epsilon": np.array([
            0.0,
            0.04424,
            0.08848,
            0.17695,
            0.35391,
        ]),

        "D": np.array([
            0.0,
            0.03539,
            0.05884,
            0.09078,
            0.13500,
        ]),

        "D_SP": {
            "y": np.array([
                0.5436,
                0.5833,
                0.5715,
                0.5645,
                0.5487,
            ]),
            "err": np.array([
                0.0,
                0.0282,
                0.0248,
                0.0089,
                0.0158,
            ]),
        },

        "SP_ASR": {
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
    },
}


# ============================================================
# Two-panel Soft-Prompt geometry figure
# ============================================================

def plot_softprompt_geometry():

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    distance_ax = axes[0]
    relationship_ax = axes[1]

    # # ========================================================
    # # Combined configured-radius grid
    # # ========================================================

    # all_epsilon = np.unique(
    #     np.concatenate([
    #         model_data["epsilon"]
    #         for model_data in models.values()
    #     ])
    # )

    # labels = [
    #     "0" if np.isclose(value, 0.0)
    #     else f"{value:.5f}".rstrip("0").rstrip(".")
    #     for value in all_epsilon
    # ]

    # ========================================================
    # Panel 1:
    # CAT realised distance D vs Soft-Prompt distance D_SP
    #
    # Color/marker = configuration
    # Solid line   = D
    # Dashed line  = D_SP
    # ========================================================

    for model_name, model_data in models.items():
        epsilon = model_data["epsilon"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        D = model_data["D"]

        D_sp = model_data["D_SP"]["y"]
        D_sp_err = model_data["D_SP"]["err"]

        # ----------------------------------------------------
        # CAT realised distance D
        # ----------------------------------------------------

        distance_ax.plot(
            epsilon,
            D,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            linestyle="-",
            color=color,
            zorder=3,
        )

        # ----------------------------------------------------
        # Soft-Prompt realised distance D_SP
        # ----------------------------------------------------

        distance_ax.plot(
            epsilon,
            D_sp,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            linestyle="--",
            color=color,
            zorder=3,
        )

        distance_ax.errorbar(
            epsilon,
            D_sp,
            yerr=D_sp_err,
            fmt="none",
            capsize=3,
            linewidth=1.2,
            color=color,
            zorder=2,
        )

    # --------------------------------------------------------
    # Axes
    #
    # Deliberately keep a single linear scale. The large gap
    # between D and D_SP is part of the result.
    # --------------------------------------------------------

    distance_ax.set_ylim(
        0.0,
        0.70,
    )

    # distance_ax.set_xticks(
    #     all_epsilon
    # )

    # distance_ax.set_xticklabels(
    #     labels,
    #     rotation=45,
    #     ha="right",
    # )

    distance_ax.set_xlim(0, 0.37)

    distance_ax.set_xticks([
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
    ])

    distance_ax.set_xticklabels([
        "0",
        "0.05",
        "0.10",
        "0.15",
        "0.20",
        "0.25",
        "0.30",
        "0.35",
    ])

    distance_ax.set_xlim(XMIN, XMAX)

    distance_ax.set_xlabel(
        r"Configured radius $\epsilon$"
    )

    distance_ax.set_ylabel(
        r"Mean per-token $L_2$ distance",
        labelpad=10,
    )

    distance_ax.set_title(
        "($a$) Realized training and attack perturbation distance",
        # fontsize=11,
    )

    distance_ax.grid(
        axis="y",
        alpha=0.25,
    )

    distance_ax.spines["top"].set_visible(False)
    distance_ax.spines["right"].set_visible(False)

    # ========================================================
    # Panel 2:
    # Soft-Prompt ASR vs D_SP
    #
    # No connecting lines: this is a relationship plot rather
    # than a trajectory over the sweep.
    # ========================================================

    all_asr_lower = []
    all_asr_upper = []
    all_dsp_lower = []
    all_dsp_upper = []

    for model_name, model_data in models.items():

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        D_sp = model_data["D_SP"]["y"]
        D_sp_err = model_data["D_SP"]["err"]

        asr = model_data["SP_ASR"]["y"]
        asr_err = model_data["SP_ASR"]["err"]

        relationship_ax.errorbar(
            D_sp,
            asr,
            xerr=D_sp_err,
            yerr=asr_err,
            fmt=marker,
            markersize=6,
            linestyle="none",
            capsize=3,
            linewidth=1.2,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            zorder=3,
        )

        all_dsp_lower.extend(
            D_sp - D_sp_err
        )

        all_dsp_upper.extend(
            D_sp + D_sp_err
        )

        all_asr_lower.extend(
            asr - asr_err
        )

        all_asr_upper.extend(
            asr + asr_err
        )

    # --------------------------------------------------------
    # Adaptive x-axis
    # --------------------------------------------------------

    xmin = min(all_dsp_lower)
    xmax = max(all_dsp_upper)

    xpadding = max(
        (xmax - xmin) * 0.08,
        0.005,
    )

    relationship_ax.set_xlim(
        xmin - xpadding,
        xmax + xpadding,
    )

    # --------------------------------------------------------
    # Adaptive y-axis
    # --------------------------------------------------------

    ymin = min(all_asr_lower)
    ymax = max(all_asr_upper)

    ypadding = max(
        (ymax - ymin) * 0.10,
        1.0,
    )

    relationship_ax.set_ylim(
        max(0, ymin - ypadding),
        min(100, ymax + ypadding),
    )

    relationship_ax.set_xlabel(
        r"Soft-Prompt distance $D_{\mathrm{SP}}$",
        labelpad=12,
    )

    relationship_ax.set_ylabel(
        r"Soft-Prompt ASR (\%)",
        labelpad=10,
    )

    relationship_ax.set_title(
        "$(b)$ Soft-Prompt success versus realized attack distance",
        # fontsize=11,
    )

    relationship_ax.grid(
        alpha=0.25,
    )

    relationship_ax.spines["top"].set_visible(False)
    relationship_ax.spines["right"].set_visible(False)

    # ========================================================
    # Legends
    # ========================================================

    # --------------------------------------------------------
    # Model/configuration legend
    # --------------------------------------------------------

    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            label=name,
        )
        for name in models.keys()
    ]

    fig.legend(
        handles=model_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    # --------------------------------------------------------
    # Meaning-of-line-style legend for panel 1
    # --------------------------------------------------------

    quantity_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.8,
            label=r"CAT training distance $D$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=r"Soft-Prompt distance $D_{\mathrm{SP}}$",
        ),
    ]

    distance_ax.legend(
        handles=quantity_handles,
        frameon=False,
        loc="center left",
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.94]
    )

    fig.subplots_adjust(
        wspace=0.30
    )

    return fig


# ============================================================
# Generate figure
# ============================================================

softprompt_geometry_figure = plot_softprompt_geometry()


# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent


# ============================================================
# Save separately
# ============================================================

figure_pdf_path = (
    output_dir / "Soft_Prompt_distances.pdf"
)

figure_png_path = (
    output_dir / "Soft_Prompt_distances.png"
)

softprompt_geometry_figure.savefig(
    figure_pdf_path,
    bbox_inches="tight",
)

softprompt_geometry_figure.savefig(
    figure_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save to diagnostic PDF
# ============================================================

combined_pdf_path = (
    output_dir / f"{script_path.stem}.pdf"
)

with PdfPages(combined_pdf_path) as pdf:

    pdf.savefig(
        softprompt_geometry_figure,
        bbox_inches="tight",
    )


# ============================================================
# Report outputs
# ============================================================

print(
    "Saved Soft-Prompt geometry figure to:\n"
    f"  {figure_pdf_path}\n"
    f"  {figure_png_path}\n\n"
    "Saved diagnostic PDF to:\n"
    f"  {combined_pdf_path}"
)


# ============================================================
# Display
# ============================================================

plt.show()
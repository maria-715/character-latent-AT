from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Configuration
# =====================================================================

COLORS = {
    "Zephyr": "C0",
    "Llama 3 (Llama 2 recipe)": "C1",
    "Llama 3 (MixAT recipe)": "C2",
}

MARKERS = {
    "Zephyr": "o",
    "Llama 3 (Llama 2 recipe)": "s",
    "Llama 3 (MixAT recipe)": "^",
}

XMIN = -0.015
XMAX = 0.375

# =====================================================================
# Data
# =====================================================================
#
# Configured radii epsilon and robustness results for the controlled
# sweeps. GitHub reproduction configurations are excluded.
#
# Attack values are means across three independently trained
# checkpoints. Error bars show one sample standard deviation across
# the three training seeds.
#
# Arrays are ordered by increasing configured radius epsilon.
# =====================================================================

models = {

    # -----------------------------------------------------------------
    # Zephyr
    # -----------------------------------------------------------------
    "Zephyr": {

        "epsilon": np.array([
            0.0,
            0.00884,
            0.01768,
            0.03535,
            0.07070,
        ]),

        "GCG": {
            "y": np.array([
                92.5,
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

        "Soft-Prompt": {
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

    # -----------------------------------------------------------------
    # Llama 3, trained with Llama 2 recipe
    # -----------------------------------------------------------------
    "Llama 3 (Llama 2 recipe)": {

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

        "Soft-Prompt": {
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

    # -----------------------------------------------------------------
    # Llama 3, trained with MixAT recipe
    # -----------------------------------------------------------------
    "Llama 3 (MixAT recipe)": {

        "epsilon": np.array([
            0.0,
            0.04424,
            0.08848,
            0.17695,
            0.35391,
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

        "Soft-Prompt": {
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


# =====================================================================
# Plot style
# =====================================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =====================================================================
# Plot one robustness panel
# =====================================================================

def plot_robustness_panel(
    ax,
    metric,
    panel_label,
    panel_title,
):

    all_lower = []
    all_upper = []

    for model_name, model_data in models.items():

        epsilon = model_data["epsilon"]
        y = model_data[metric]["y"]
        err = model_data[metric]["err"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        # -------------------------------------------------------------
        # Mean trajectory
        #
        # Lines connect successive sampled radii within each sweep only
        # as a visual guide; they do not imply interpolation.
        # -------------------------------------------------------------

        ax.plot(
            epsilon,
            y,
            marker=marker,
            markersize=5,
            linewidth=1.4,
            color=color,
            label=model_name,
            zorder=3,
        )

        # -------------------------------------------------------------
        # One sample SD across the three training seeds
        # -------------------------------------------------------------

        ax.errorbar(
            epsilon,
            y,
            yerr=err,
            fmt="none",
            capsize=2.5,
            linewidth=1.1,
            color=color,
            zorder=2,
        )

        all_lower.extend(y - err)
        all_upper.extend(y + err)

    # -----------------------------------------------------------------
    # Linear epsilon axis
    # -----------------------------------------------------------------

    ax.set_xlim(XMIN, XMAX)

    # -----------------------------------------------------------------
    # Adaptive y-axis including error bars
    # -----------------------------------------------------------------

    ymin = min(all_lower)
    ymax = max(all_upper)

    padding = max(
        (ymax - ymin) * 0.08,
        1.0,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        min(100, ymax + padding),
    )

    # -----------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------

    ax.set_xlabel(
        r"Configured radius $\epsilon$"
    )

    ax.set_ylabel(
        f"{panel_title} (%)"
    )

    ax.set_title(
        rf"({panel_label}) {panel_title} vs configured radius $\epsilon$"
    )

    # -----------------------------------------------------------------
    # Styling
    # -----------------------------------------------------------------

    ax.grid(
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



# =====================================================================
# Figure
# =====================================================================

fig = plt.figure(
    figsize=(10.0, 4.2),
)

gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1, 1],
    height_ratios=[0.7, 3.0],
    hspace=0.05,
    wspace=0.27,
)

# GCG gets two axes -> broken y-axis
ax_gcg_top = fig.add_subplot(gs[0, 0])
ax_gcg_bottom = fig.add_subplot(
    gs[1, 0],
    sharex=ax_gcg_top,
)

# Soft-Prompt occupies the full right column
ax_soft = fig.add_subplot(gs[:, 1])


# =====================================================================
# Helper: plot data without automatically changing y limits
# =====================================================================

def plot_metric(ax, metric):

    for model_name, model_data in models.items():

        epsilon = model_data["epsilon"]
        y = model_data[metric]["y"]
        err = model_data[metric]["err"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        # Mean trajectory
        ax.plot(
            epsilon,
            y,
            marker=marker,
            markersize=5,
            linewidth=1.4,
            color=color,
            label=model_name,
            zorder=3,
        )

        # Error bars
        ax.errorbar(
            epsilon,
            y,
            yerr=err,
            fmt="none",
            capsize=2.5,
            linewidth=1.1,
            color=color,
            zorder=2,
        )

    ax.set_xlim(XMIN, XMAX)

    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["right"].set_visible(False)


# =====================================================================
# (a) GCG -- broken y-axis
# =====================================================================

plot_metric(
    ax_gcg_top,
    "GCG",
)

plot_metric(
    ax_gcg_bottom,
    "GCG",
)


# ---------------------------------------------------------------------
# Broken-axis ranges
#
# Values from 45 to 80 are omitted.
# ---------------------------------------------------------------------

ax_gcg_bottom.set_ylim(
    0,
    44.5,
)

ax_gcg_top.set_ylim(
    85,
    100,
)


# ---------------------------------------------------------------------
# Hide touching spines
# ---------------------------------------------------------------------

ax_gcg_top.spines["bottom"].set_visible(False)
ax_gcg_bottom.spines["top"].set_visible(False)

ax_gcg_top.spines["top"].set_visible(False)


# No x ticks/labels on upper GCG axis
ax_gcg_top.tick_params(
    axis="x",
    which="both",
    bottom=False,
    labelbottom=False,
)


# ---------------------------------------------------------------------
# Diagonal break marks
# ---------------------------------------------------------------------

d = 0.012

kwargs = dict(
    color="k",
    clip_on=False,
    linewidth=1.0,
)

# Bottom of upper axis
ax_gcg_top.plot(
    (-d, +d),
    (-d, +d),
    transform=ax_gcg_top.transAxes,
    **kwargs,
)

# ax_gcg_top.plot(
#     (1 - d, 1 + d),
#     (-d, +d),
#     transform=ax_gcg_top.transAxes,
#     **kwargs,
# )

# Top of lower axis
ax_gcg_bottom.plot(
    (-d, +d),
    (1 - d, 1 + d),
    transform=ax_gcg_bottom.transAxes,
    **kwargs,
)

# ax_gcg_bottom.plot(
#     (1 - d, 1 + d),
#     (1 - d, 1 + d),
#     transform=ax_gcg_bottom.transAxes,
#     **kwargs,
# )


# ---------------------------------------------------------------------
# GCG labels
# ---------------------------------------------------------------------

ax_gcg_bottom.set_xlabel(
    r"Configured radius $\epsilon$"
)

ax_gcg_bottom.set_ylabel(
    "GCG ASR (%)"
)

# Move ylabel vertically so it represents both broken axes
ax_gcg_bottom.yaxis.set_label_coords(
    -0.12,
    0.78,
)

ax_gcg_top.set_title(
    r"(a) GCG ASR vs configured radius $\epsilon$"
)


# =====================================================================
# (b) Soft-Prompt
# =====================================================================

plot_robustness_panel(
    ax_soft,
    metric="Soft-Prompt",
    panel_label="b",
    panel_title="Soft-Prompt ASR",
)


# =====================================================================
# Shared legend
# =====================================================================

handles, labels = ax_gcg_bottom.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
)


# =====================================================================
# Layout
# =====================================================================

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.14,
    top=0.84,
    wspace=0.27,
    hspace=0.05,
)


# =====================================================================
# Save beside this script
# =====================================================================

output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "robustness_vs_epsilon.pdf"
png_path = output_dir / "robustness_vs_epsilon.png"

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# =====================================================================
# Typography
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
# epsilon values come from thesis Table 18.
#
# Utility and attack values are means across three training seeds,
# with one sample standard deviation.
#
# GitHub reproduction configurations are excluded throughout.
#
# Arrays within each configuration are ordered by increasing epsilon.
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

        "Harmless": {
            "y": np.array([
                100,
                98.33,
                94.17,
                89.17,
                70.83,
            ]),
            "err": np.array([
                0.0,
                1.44,
                3.82,
                1.44,
                9.46,
            ]),
        },

        "MMLU": {
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

        "GCG": {
            "y": np.array([
                19.17,
                17.50,
                18.33,
                9.17,
            ]),
            "err": np.array([
                3.82,
                6.61,
                10.10,
                1.44,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                43.44,
                43.40,
                42.45,
                51.13,
            ]),
            "err": np.array([
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

        "Harmless": {
            "y": np.array([
                100,
                95.83,
                90.83,
                76.67,
                45.83,
                49.17,
                7.50,
                2.50,
                1.67,
            ]),
            "err": np.array([
                0.0,
                1.44,
                7.22,
                7.64,
                26.96,
                20.05,
                12.99,
                4.33,
                2.89,
            ]),
        },

        "MMLU": {
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

        "GCG": {
            "y": np.array([
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

        "Harmless": {
            "y": np.array([
                100.0,
                82.50,
                58.33,
                20.83,
                50.00,
            ]),
            "err": np.array([
                0.0,
                13.92,
                14.65,
                15.07,
                41.61,
            ]),
        },

        "MMLU": {
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

        "GCG": {
            "y": np.array([
                25.83,
                12.50,
                17.50,
                25.00,
            ]),
            "err": np.array([
                1.44,
                2.50,
                6.61,
                10.90,
            ]),
        },

        "Soft-Prompt": {
            "y": np.array([
                55.47,
                43.23,
                58.18,
                63.21,
            ]),
            "err": np.array([
                4.46,
                3.92,
                4.86,
                3.20,
            ]),
        },
    },
}


# =====================================================================
# Pareto dominance
# =====================================================================
#
# For the bottom panels:
#
#   x = Harmless successful-response rate  -> higher is better
#   y = attack success rate                -> lower is better
#
# Point i is dominated if another observed point j from the SAME
# configuration has:
#
#   harmless_j >= harmless_i
#   ASR_j      <= ASR_i
#
# with at least one strict inequality.
#
# Dominance is therefore computed from the three-seed means.
# =====================================================================

def non_dominated_mask(x, y):
    n = len(x)
    non_dominated = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            at_least_as_good = (
                x[j] >= x[i]
                and y[j] <= y[i]
            )

            strictly_better = (
                x[j] > x[i]
                or y[j] < y[i]
            )

            if at_least_as_good and strictly_better:
                non_dominated[i] = False
                break

    return non_dominated


# =====================================================================
# Annotation offsets
# =====================================================================
#
# Defaults work for most points. Add panel/model/epsilon-specific
# overrides here if labels overlap after rendering.
#
# Values are offsets in display points: (dx, dy).
# =====================================================================

ANNOTATION_OFFSETS = {
    # =========================================================
    # GCG
    # =========================================================

    # Zephyr
    ("GCG", "Zephyr", 0.00884): (-5, 5),
    ("GCG", "Zephyr", 0.01768): (-10, -12),
    ("GCG", "Zephyr", 0.03535): (-10, 5),
    ("GCG", "Zephyr", 0.07070): (7, -5),

    # Llama 3 (Llama 2 recipe)
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.02949): (-15, 7),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.04424): (-25, 7),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.05898): (-20, -17),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.08848): (7, 3),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.11797): (5, -5),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.17695): (7, 5),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.23594): (7, 5),
    ("GCG", "Llama 3 (Llama 2 recipe)", 0.35391): (7, 5),

    # Llama 3 (MixAT recipe)
    ("GCG", "Llama 3 (MixAT recipe)", 0.04424): (5, 5),
    ("GCG", "Llama 3 (MixAT recipe)", 0.08848): (7, -10),
    ("GCG", "Llama 3 (MixAT recipe)", 0.17695): (7, 0),
    ("GCG", "Llama 3 (MixAT recipe)", 0.35391): (5, 5),


    # =========================================================
    # Soft-Prompt
    # =========================================================

    # Zephyr
    ("Soft-Prompt", "Zephyr", 0.00884): (2, 5),
    ("Soft-Prompt", "Zephyr", 0.01768): (-20, 7),
    ("Soft-Prompt", "Zephyr", 0.03535): (-20, -10),
    ("Soft-Prompt", "Zephyr", 0.07070): (-10, 7),

    # Llama 3 (Llama 2 recipe)
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.02949): (-20, -15),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.04424): (-15, 7),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.05898): (7, -7),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.08848): (7, 7),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.11797): (7, -10),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.17695): (7, 5),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.23594): (7, 5),
    ("Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.35391): (7, 5),

    # Llama 3 (MixAT recipe)
    ("Soft-Prompt", "Llama 3 (MixAT recipe)", 0.04424): (-20, 5),
    ("Soft-Prompt", "Llama 3 (MixAT recipe)", 0.08848): (7, -5),
    ("Soft-Prompt", "Llama 3 (MixAT recipe)", 0.17695): (7, -5),
    ("Soft-Prompt", "Llama 3 (MixAT recipe)", 0.35391): (5, 5),
}


def epsilon_label(value):
    """Compact but unambiguous epsilon label."""
    return f"{value:.5f}".rstrip("0").rstrip(".")


# =====================================================================
# Top-row utility panels
# =====================================================================

def plot_utility_panel(ax, metric, panel_label, panel_title):

    all_lower = []
    all_upper = []

    for model_name, model_data in models.items():

        epsilon = model_data["epsilon"]
        y = model_data[metric]["y"]
        err = model_data[metric]["err"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        ax.errorbar(
            epsilon,
            y,
            yerr=err,
            marker=marker,
            markersize=5,
            linewidth=1.4,
            capsize=2.5,
            color=color,
            label=model_name,
            zorder=3,
        )

        all_lower.extend(y - err)
        all_upper.extend(y + err)

    # Linear epsilon axis.
    ax.set_xlim(XMIN, XMAX)

    ymin = min(all_lower)
    ymax = max(all_upper)

    padding = max(
        0.08 * (ymax - ymin),
        0.5,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        min(100, ymax + padding),
    )

    ax.set_xlabel(r"Configured radius $\epsilon$")

    if metric == "MMLU":
        ax.set_ylabel("MMLU accuracy (%)")
    elif metric == "Harmless":
        ax.set_ylabel(
            "Harmless successful-response rate (%)"
        )

    ax.set_title(
        rf"({panel_label}) {panel_title}"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =====================================================================
# Bottom-row trade-off panels
# =====================================================================

def plot_tradeoff_panel(
    ax,
    robustness_metric,
    panel_label,
    panel_title,
):

    all_x = []
    all_y = []

    for model_name, model_data in models.items():

        epsilon = model_data["epsilon"][1:]

        x = model_data["Harmless"]["y"][1:]
        y = model_data[robustness_metric]["y"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        # -------------------------------------------------------------
        # Connect successive configured radii lightly
        # -------------------------------------------------------------
        ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.0,
            color=color,
            alpha=0.45,
            zorder=1,
        )

        # -------------------------------------------------------------
        # Determine observed Pareto status
        # -------------------------------------------------------------
        nd = non_dominated_mask(x, y)
        dominated = ~nd

        # -------------------------------------------------------------
        # Dominated points: hollow markers
        # -------------------------------------------------------------
        ax.scatter(
            x[dominated],
            y[dominated],
            marker=marker,
            s=42,
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            zorder=3,
        )

        # -------------------------------------------------------------
        # Non-dominated points: filled markers
        # -------------------------------------------------------------
        ax.scatter(
            x[nd],
            y[nd],
            marker=marker,
            s=42,
            facecolors=color,
            edgecolors=color,
            linewidths=1.0,
            zorder=4,
        )

        # -------------------------------------------------------------
        # Label every point by epsilon
        # -------------------------------------------------------------
        for x_i, y_i, eps_i in zip(x, y, epsilon):

            offset = ANNOTATION_OFFSETS.get(
                (
                    robustness_metric,
                    model_name,
                    float(eps_i),
                ),
                (5, 5),
            )

            ax.annotate(
                epsilon_label(eps_i),
                xy=(x_i, y_i),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                zorder=5,
            )

        all_x.extend(x)
        all_y.extend(y)

    # -----------------------------------------------------------------
    # Axis limits
    # -----------------------------------------------------------------

    xmin = min(all_x)
    xmax = max(all_x)
    ymin = min(all_y)
    ymax = max(all_y)

    xpadding = max(
        0.07 * (xmax - xmin),
        0.5,
    )

    ypadding = max(
        0.08 * (ymax - ymin),
        1.0,
    )

    ax.set_xlim(
        max(0, xmin - xpadding),
        min(100, xmax + xpadding),
    )

    ax.set_ylim(
        max(0, ymin - ypadding),
        min(100, ymax + ypadding),
    )

    # Higher utility is better, so x increases normally to the right.
    # Lower ASR is better, so the desirable direction is downwards.
    ax.set_xlabel(
        "Harmless successful-response rate (%)"
    )

    ax.set_ylabel(
        f"{robustness_metric} ASR (%)"
    )

    ax.set_title(
        rf"({panel_label}) {panel_title}"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Small orientation cue.
    ax.text(
        0.98,
        0.97,
        "better utility \u2192\nbetter robustness \u2193",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
        linespacing=1.2,
    )


# =====================================================================
# Figure
# =====================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.0, 7.8),
)


# ---------------------------------------------------------------------
# Top row: utility costs
# ---------------------------------------------------------------------

plot_utility_panel(
    axes[0, 0],
    metric="MMLU",
    panel_label="a",
    panel_title=r"MMLU vs configured radius $\epsilon$",
)

plot_utility_panel(
    axes[0, 1],
    metric="Harmless",
    panel_label="b",
    panel_title=r"Harmless vs configured radius $\epsilon$",
)


# ---------------------------------------------------------------------
# Bottom row: are utility costs compensated by robustness?
# ---------------------------------------------------------------------

plot_tradeoff_panel(
    axes[1, 0],
    robustness_metric="GCG",
    panel_label="c",
    panel_title="Harmless vs GCG ASR",
)

plot_tradeoff_panel(
    axes[1, 1],
    robustness_metric="Soft-Prompt",
    panel_label="d",
    panel_title="Harmless vs Soft-Prompt ASR",
)


# =====================================================================
# Shared legends
# =====================================================================

model_handles = [
    Line2D(
        [0],
        [0],
        color=COLORS[name],
        marker=MARKERS[name],
        linestyle="-",
        linewidth=1.4,
        markersize=5,
        label=name,
    )
    for name in models.keys()
]

dominance_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="0.25",
        markeredgecolor="0.25",
        markersize=5,
        label="Non-dominated",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="0.25",
        markersize=5,
        label="Dominated",
    ),
]


# Model/configuration legend across the top.
fig.legend(
    handles=model_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    fontsize=12,
)

# Dominance legend below the figure.
fig.legend(
    handles=dominance_handles,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=2,
    fontsize=12,
)


# =====================================================================
# Layout
# =====================================================================

fig.tight_layout(
    rect=[
        0.02,
        0.055,
        1.0,
        0.93,
    ]
)

fig.subplots_adjust(
    wspace=0.27,
    hspace=0.34,
)


# =====================================================================
# Save beside this script
# =====================================================================

output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "utility_and_dominance.pdf"
png_path = output_dir / "utility_and_dominance.png"

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
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
# Columns:
#   epsilon,
#   D mean, D std,
#   Rank-2 mean, Rank-2 std,
#   Changed mean, Changed std
#
# Error bars are standard deviations across training seeds.

data = {
    "Zephyr": np.array([
        [0.00884, 0.00734, 0.00003, 0.13535, 0.00001, 0.00000, 0.00000],
        [0.01768, 0.01466, 0.00006, 0.13574, 0.00006, 0.00000, 0.00000],
        [0.03535, 0.02688, 0.00068, 0.13710, 0.00017, 0.00000, 0.00000],
        [0.07070, 0.04422, 0.00423, 0.14057, 0.00067, 0.00000, 0.00000],
    ]),

    "Llama 3 (Llama 2)": np.array([
        [0.02949, 0.02492, 0.00050, 0.36474, 0.00022, 0.00000, 0.00000],
        [0.04424, 0.03491, 0.00107, 0.36621, 0.00027, 0.00000, 0.00000],
        [0.05898, 0.04517, 0.00030, 0.36817, 0.00001, 0.00000, 0.00000],
        [0.08848, 0.06035, 0.00039, 0.37127, 0.00028, 0.00000, 0.00000],
        [0.11797, 0.07039, 0.00285, 0.37407, 0.00093, 0.00004, 0.00005],
        [0.17695, 0.09350, 0.00361, 0.38030, 0.00063, 0.00100, 0.00000],
        [0.23594, 0.10172, 0.00553, 0.38384, 0.00141, 0.00300, 0.00200],
        [0.35391, 0.13176, 0.00496, 0.39605, 0.00196, 0.01000, 0.00200],
    ]),

    "Llama 3 (MixAT)": np.array([
        [0.04424, 0.03539, 0.00045, 0.36637, 0.00017, 0.00000, 0.00000],
        [0.08848, 0.05884, 0.00232, 0.37101, 0.00080, 0.00000, 0.00000],
        [0.17695, 0.09078, 0.00334, 0.37950, 0.00068, 0.00100, 0.00100],
        [0.35391, 0.13500, 0.00616, 0.39738, 0.00174, 0.00900, 0.00100],
    ]),
}


# ---------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------
# Configuration:
#   colour + marker
#
# Quantity in panel (a):
#   solid  = realised perturbation distance D
#   dashed = nearest-alternative distance

styles = {
    "Zephyr": {
        "color": "tab:blue",
        "marker": "o",
    },
    "Llama 3 (Llama 2)": {
        "color": "tab:orange",
        "marker": "s",
    },
    "Llama 3 (MixAT)": {
        "color": "tab:green",
        "marker": "^",
    },
}


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, (ax_dist, ax_remain) = plt.subplots(
    1,
    2,
    figsize=(9.4, 3.8),
    constrained_layout=True,
)


# =====================================================================
# Panel (a): perturbation and nearest-alternative distances
# =====================================================================
for label, values in data.items():

    epsilon = values[:, 0]

    D = values[:, 1]
    D_std = values[:, 2]

    rank2 = values[:, 3]
    rank2_std = values[:, 4]

    color = styles[label]["color"]
    marker = styles[label]["marker"]

    # -------------------------------------------------------------
    # Realised perturbation distance D
    # -------------------------------------------------------------
    ax_dist.errorbar(
        epsilon,
        D,
        yerr=D_std,
        color=color,
        marker=marker,
        linestyle="-",
        markersize=5,
        linewidth=1.5,
        elinewidth=1.0,
        capsize=2.5,
    )

    # -------------------------------------------------------------
    # Distance from perturbed point to nearest alternative
    # vocabulary embedding
    # -------------------------------------------------------------
    ax_dist.errorbar(
        epsilon,
        rank2,
        yerr=rank2_std,
        color=color,
        marker=marker,
        linestyle="--",
        markersize=5,
        linewidth=1.5,
        elinewidth=1.0,
        capsize=2.5,
    )


# ---------------------------------------------------------------------
# Panel (a): separate legends
# ---------------------------------------------------------------------

# Legend 1: configuration
configuration_handles = [
    Line2D(
        [0],
        [0],
        color=styles[label]["color"],
        marker=styles[label]["marker"],
        linestyle="-",
        linewidth=1.5,
        markersize=5,
        label=label,
    )
    for label in data
]

configuration_legend = ax_dist.legend(
    handles=configuration_handles,
    title="Configuration",
    frameon=False,
    loc="center right",
    bbox_to_anchor=(0.98, 0.5),
)

# Keep the first legend when adding the second.
ax_dist.add_artist(configuration_legend)


# Legend 2: quantity
quantity_handles = [
    Line2D(
        [0],
        [0],
        color="0.25",
        linestyle="-",
        linewidth=1.5,
        label=r"Perturbation distance $D$",
    ),
    Line2D(
        [0],
        [0],
        color="0.25",
        linestyle="--",
        linewidth=1.5,
        label="Second-nearest distance",
    ),
]

ax_dist.legend(
    handles=quantity_handles,
    title="Quantity",
    frameon=False,
    loc="center right",
    bbox_to_anchor=(0.98, 0.73),
)


# =====================================================================
# Panel (b): probability original token remains nearest
# =====================================================================
for label, values in data.items():

    epsilon = values[:, 0]

    changed = values[:, 5]
    changed_std = values[:, 6]

    # Convert Changed to percentage of positions for which the
    # original token remains the nearest vocabulary entry.
    remain = 100.0 * (1.0 - changed)
    remain_std = 100.0 * changed_std

    color = styles[label]["color"]
    marker = styles[label]["marker"]

    ax_remain.errorbar(
        epsilon,
        remain,
        yerr=remain_std,
        color=color,
        marker=marker,
        linestyle="-",
        markersize=5,
        linewidth=1.5,
        elinewidth=1.0,
        capsize=2.5,
        label=label,
    )


# ---------------------------------------------------------------------
# Full-retention reference
# ---------------------------------------------------------------------
ax_remain.axhline(
    100.0,
    linestyle="--",
    linewidth=1.0,
    color="0.35",
    zorder=0,
)


# =====================================================================
# Axis formatting
# =====================================================================

# Panel (a)
ax_dist.set_xlabel(r"Configured radius $\epsilon$")
ax_dist.set_ylabel("Distance")
ax_dist.set_title(
    "(a) Distance to nearest vocabulary embeddings"
)

ax_dist.set_xlim(left=0)
ax_dist.set_ylim(bottom=0)


# Panel (b)
ax_remain.set_xlabel(r"Configured radius $\epsilon$")
ax_remain.set_ylabel("Original token remains nearest (%)")
ax_remain.set_title(
    "(b) Original token remains nearest"
)

ax_remain.set_xlim(left=0)

# Deliberately truncated because the quantity of interest is a
# rare departure from 100%.
ax_remain.set_ylim(98.5, 100.1)


# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
for ax in (ax_dist, ax_remain):
    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------
# Panel (b) legend
# ---------------------------------------------------------------------
ax_remain.legend(
    frameon=False,
    loc="lower left",
)


# ---------------------------------------------------------------------
# Save beside script
# ---------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "perturbation_position.pdf"
png_path = output_dir / "perturbation_position.png"

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
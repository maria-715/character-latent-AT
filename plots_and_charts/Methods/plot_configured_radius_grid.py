import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
})

# Configured radii from the Methods chapter
zephyr = np.array([0.00884, 0.01768, 0.03535, 0.07070])

# Llama 3 (Llama 2 recipe): two interleaved four-point sweeps
llama_l2_a = np.array([0.02949, 0.05898, 0.11797, 0.23594])
llama_l2_b = np.array([0.04424, 0.08848, 0.17695, 0.35391])

# Llama 3 (MixAT recipe): same radii as the second Llama-2-recipe sub-sweep
llama_mixat = np.array([0.04424, 0.08848, 0.17695, 0.35391])

fig, ax = plt.subplots(figsize=(6.2, 2.35), constrained_layout=True)

y_zephyr, y_l2, y_mixat = 1.4, 0.7, 0.0

# Row guides
for y in [y_zephyr, y_l2, y_mixat]:
    ax.axhline(y, color="0.88", linewidth=0.7, zorder=0)

# IMPORTANT: force every marker to black; shape/fill, not colour, carries meaning.
# Zephyr: filled circles
ax.scatter(
    zephyr, np.full(zephyr.shape, y_zephyr),
    s=42, marker="o", facecolors="black", edgecolors="black", zorder=3
)

# Llama 3 (Llama 2): first sub-sweep filled, second hollow
ax.scatter(
    llama_l2_a, np.full(llama_l2_a.shape, y_l2),
    s=42, marker="o", facecolors="black", edgecolors="black", zorder=3
)
ax.scatter(
    llama_l2_b, np.full(llama_l2_b.shape, y_l2),
    s=42, marker="o", facecolors="white", edgecolors="black",
    linewidths=1.0, zorder=3
)

# MixAT uses the same s values/radii as the second Llama-2-recipe
# sub-sweep, so give it the same visual encoding (hollow markers).
ax.scatter(
    llama_mixat, np.full(llama_mixat.shape, y_mixat),
    s=42, marker="o", facecolors="white", edgecolors="black",
    linewidths=1.0, zorder=3
)

ax.set_xscale("log")
ax.set_xlim(0.007, 0.45)
ax.set_ylim(-0.25, 1.65)

ax.set_yticks([y_zephyr, y_l2, y_mixat])
ax.set_yticklabels([
    "Zephyr",
    "Llama 3 (Llama 2)",
    "Llama 3 (MixAT)",
])

# Put the model names on the RIGHT
ax.yaxis.tick_left()
ax.tick_params(axis="y", labelright=False, labelleft=True, length=0, pad=8)

# Powers-of-two spacing is natural for these geometric sweeps.
ticks = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
ax.set_xticks(ticks)
ax.set_xticklabels(["0.01", "0.02", "0.04", "0.08", "0.16", "0.32"])
ax.set_xlabel(r"Configured radius $\epsilon$ (log scale)")

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor="black", markeredgecolor="black", markersize=6,
           label=r"$s=\{0.05,0.1,0.2,0.4\}$"),
    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor="white", markeredgecolor="black", markersize=6,
           label=r"$s=\{0.075,0.15,0.3,0.6\}$"),
]
ax.legend(
    handles=legend_handles,
    title="Llama 3 (Llama 2) sub-sweeps",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.4),
    ncol=2,
    frameon=False,
    # fontsize=8,
    # title_fontsize=8,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# fig.subplots_adjust(left=0.25, right=0.98, top=0.96, bottom=0.40)

from pathlib import Path
from matplotlib.transforms import Bbox

output_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Make exported PDF symmetric around the plotting axes
# ---------------------------------------------------------

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

# Bounding box of everything Matplotlib would normally save
full_bbox = fig.get_tightbbox(renderer)

# Position of the actual plotting axes, in inches
ax_bbox = ax.get_window_extent(renderer).transformed(
    fig.dpi_scale_trans.inverted()
)

# Distance from centre of axes to each edge of current tight bbox
ax_center = (ax_bbox.x0 + ax_bbox.x1) / 2

left_extent = ax_center - full_bbox.x0
right_extent = full_bbox.x1 - ax_center

# Use whichever side is larger on BOTH sides
half_width = max(left_extent, right_extent)

balanced_bbox = Bbox.from_extents(
    ax_center - half_width,
    full_bbox.y0,
    ax_center + half_width,
    full_bbox.y1,
)

# Small equal padding
balanced_bbox = balanced_bbox.expanded(
    1.01,
    1.02,
)

fig.savefig(
    output_dir / "configured_radius_grid.pdf",
    bbox_inches=balanced_bbox,
)

fig.savefig(
    output_dir / "configured_radius_grid.png",
    dpi=300,
    bbox_inches=balanced_bbox,
)

plt.show()
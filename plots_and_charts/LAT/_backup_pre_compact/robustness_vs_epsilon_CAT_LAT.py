from pathlib import Path

import matplotlib.pyplot as plt

from cat_lat_common import (
    add_row_legends,
    make_figure,
    plot_metric,
    plot_sweep_panel,
    save_figure,
)


# =====================================================================
# Figure 1 -- Robustness vs configured radius
#
# Top row    : CAT (identical to Workshop_submission/robustness.py)
# Bottom row : LAT inner-LR sweeps (no HuggingFace reference)
# =====================================================================

fig, outer = make_figure(hspace=0.50)

# (a) CAT GCG gets two axes -> broken y-axis
gs_gcg = outer[0, 0].subgridspec(
    2,
    1,
    height_ratios=[0.7, 3.0],
    hspace=0.05,
)

ax_gcg_top = fig.add_subplot(gs_gcg[0])
ax_gcg_bottom = fig.add_subplot(gs_gcg[1], sharex=ax_gcg_top)

ax_cat_soft = fig.add_subplot(outer[0, 1])
ax_lat_gcg = fig.add_subplot(outer[1, 0])
ax_lat_soft = fig.add_subplot(outer[1, 1])


# =====================================================================
# (a) CAT GCG -- broken y-axis
# =====================================================================

for ax in (ax_gcg_top, ax_gcg_bottom):
    plot_metric(ax, "CAT", "GCG")

# Values from 45 to 80 are omitted.
ax_gcg_bottom.set_ylim(0, 44.5)
ax_gcg_top.set_ylim(85, 100)

ax_gcg_top.spines["bottom"].set_visible(False)
ax_gcg_bottom.spines["top"].set_visible(False)

ax_gcg_top.tick_params(
    axis="x",
    which="both",
    bottom=False,
    labelbottom=False,
)

# Diagonal break marks (left spine only, as in the CAT figure)
d = 0.012

kwargs = dict(
    color="k",
    clip_on=False,
    linewidth=1.0,
)

ax_gcg_top.plot(
    (-d, +d),
    (-d, +d),
    transform=ax_gcg_top.transAxes,
    **kwargs,
)

ax_gcg_bottom.plot(
    (-d, +d),
    (1 - d, 1 + d),
    transform=ax_gcg_bottom.transAxes,
    **kwargs,
)

ax_gcg_bottom.set_xlabel(r"Configured radius $\epsilon$")
ax_gcg_bottom.set_ylabel("GCG ASR (%)")

# Move ylabel vertically so it represents both broken axes
ax_gcg_bottom.yaxis.set_label_coords(-0.12, 0.78)

ax_gcg_top.set_title(r"(a) GCG ASR vs configured radius $\epsilon$")


# =====================================================================
# (b) CAT Soft-Prompt, (c) LAT GCG, (d) LAT Soft-Prompt
# =====================================================================

plot_sweep_panel(
    ax_cat_soft, "CAT", "Soft-Prompt", "b",
    r"Soft-Prompt ASR vs configured radius $\epsilon$",
)

plot_sweep_panel(
    ax_lat_gcg, "LAT", "GCG", "c",
    r"GCG ASR vs configured radius $\epsilon$",
)

plot_sweep_panel(
    ax_lat_soft, "LAT", "Soft-Prompt", "d",
    r"Soft-Prompt ASR vs configured radius $\epsilon$",
)

# Align LAT GCG ylabel with the CAT GCG ylabel above
ax_lat_gcg.yaxis.set_label_coords(-0.12, 0.5)


# =====================================================================
# Legends and save
# =====================================================================

add_row_legends(fig, ax_lat_gcg)

save_figure(
    fig,
    Path(__file__).resolve().parent,
    "robustness_vs_epsilon_CAT_LAT",
)

plt.show()

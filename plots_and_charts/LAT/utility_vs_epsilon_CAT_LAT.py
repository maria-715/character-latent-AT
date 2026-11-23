from pathlib import Path

import matplotlib.pyplot as plt

from cat_lat_common import (
    add_compact_row_legends,
    compact_axes_style,
    make_compact_figure,
    plot_sweep_panel,
    save_compact_figure,
)


# =====================================================================
# Figure 2 -- Utility vs configured radius
#
# Top row    : CAT MMLU, CAT Harmless
#              (as panels (a)-(b) of Workshop_submission/utility-dominance.py)
# Bottom row : LAT MMLU, LAT Harmless
# =====================================================================

fig, outer = make_compact_figure()

ax_cat_mmlu = fig.add_subplot(outer[0, 0])
ax_cat_harmless = fig.add_subplot(outer[0, 1])
ax_lat_mmlu = fig.add_subplot(outer[1, 0])
ax_lat_harmless = fig.add_subplot(outer[1, 1])

PANELS = [
    (ax_cat_mmlu, "CAT", "MMLU", "a"),
    (ax_cat_harmless, "CAT", "Harmless", "b"),
    (ax_lat_mmlu, "LAT", "MMLU", "c"),
    (ax_lat_harmless, "LAT", "Harmless", "d"),
]

for ax, method, metric, label in PANELS:
    plot_sweep_panel(
        ax, method, metric, label,
        rf"{metric} vs configured radius $\epsilon$",
        min_padding=0.5,
        # Slight headroom so markers at exactly 100% are not clipped
        ymax_cap=101.5 if metric == "Harmless" else 100,
    )

    # Same label text, broken over two lines so it fits the
    # shorter compact panels.
    if metric == "Harmless":
        ax.set_ylabel("Harmless successful-\nresponse rate (%)")


# =====================================================================
# Legends and save
# =====================================================================

cat_axes = [ax_cat_mmlu, ax_cat_harmless]
lat_axes = [ax_lat_mmlu, ax_lat_harmless]

compact_axes_style(cat_axes + lat_axes)

add_compact_row_legends(fig, cat_axes, lat_axes)

save_compact_figure(
    fig,
    Path(__file__).resolve().parent,
    "utility_vs_epsilon_CAT_LAT",
)

plt.show()

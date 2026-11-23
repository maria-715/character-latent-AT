from pathlib import Path

import matplotlib.pyplot as plt

from cat_lat_common import (
    add_row_legends,
    make_figure,
    plot_sweep_panel,
    save_figure,
)


# =====================================================================
# Figure 2 -- Utility vs configured radius
#
# Top row    : CAT MMLU, CAT Harmless
#              (as panels (a)-(b) of Workshop_submission/utility-dominance.py)
# Bottom row : LAT MMLU, LAT Harmless
# =====================================================================

fig, outer = make_figure(hspace=0.50)

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


# =====================================================================
# Legends and save
# =====================================================================

add_row_legends(fig, ax_lat_mmlu)

save_figure(
    fig,
    Path(__file__).resolve().parent,
    "utility_vs_epsilon_CAT_LAT",
)

plt.show()

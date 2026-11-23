from pathlib import Path

import matplotlib.pyplot as plt

from cat_lat_common import (
    add_row_legends,
    dominance_handles,
    make_figure,
    plot_tradeoff_panel,
    save_figure,
)


# =====================================================================
# Figure 3 -- Joint robustness-utility (MMLU for both methods)
#
#   (a) CAT MMLU vs GCG     (b) CAT MMLU vs Soft-Prompt
#   (c) LAT MMLU vs GCG     (d) LAT MMLU vs Soft-Prompt
# =====================================================================


# =====================================================================
# Annotation offsets
#
# Values are offsets in display points: (dx, dy).
# Key: (method, robustness_metric, series, epsilon)
# =====================================================================

ANNOTATION_OFFSETS = {
    # =========================================================
    # CAT -- GCG
    # =========================================================

    ("CAT", "GCG", "Zephyr", 0.00884): (-4, 7),
    ("CAT", "GCG", "Zephyr", 0.01768): (-12, -13),
    ("CAT", "GCG", "Zephyr", 0.03535): (6, -4),

    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.04424): (6, -12),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.11797): (-10, -14),

    ("CAT", "GCG", "Llama 3 (MixAT recipe)", 0.17695): (-10, -14),

    # =========================================================
    # CAT -- Soft-Prompt
    # =========================================================

    ("CAT", "Soft-Prompt", "Zephyr", 0.00884): (0, 8),
    ("CAT", "Soft-Prompt", "Zephyr", 0.01768): (-25, -14),
    ("CAT", "Soft-Prompt", "Zephyr", 0.03535): (8, -3),

    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.23594): (6, -12),

    ("CAT", "Soft-Prompt", "Llama 3 (MixAT recipe)", 0.17695): (-10, -14),

    # =========================================================
    # LAT
    # =========================================================

    ("LAT", "GCG", "10%-step scaling", 24.0): (-16, -8),

    ("LAT", "Soft-Prompt", "Original-LR scaling", 48.0): (-20, -4),
}

# Where the "better utility / better robustness" cue goes, so it does
# not cover points: CAT's upper-left is occupied, LAT's upper-right is.
CUE_POSITION = {
    "CAT": (0.98, "right"),
    "LAT": (0.02, "left"),
}


# =====================================================================
# Figure
# =====================================================================

fig, outer = make_figure(figsize=(10.0, 9.2), bottom=0.10)

axes = {
    (row, col): fig.add_subplot(outer[row, col])
    for row in range(2)
    for col in range(2)
}

PANELS = [
    ((0, 0), "CAT", "GCG", "a"),
    ((0, 1), "CAT", "Soft-Prompt", "b"),
    ((1, 0), "LAT", "GCG", "c"),
    ((1, 1), "LAT", "Soft-Prompt", "d"),
]

for pos, method, metric, label in PANELS:
    plot_tradeoff_panel(
        axes[pos],
        method,
        utility_metric="MMLU",
        robustness_metric=metric,
        panel_label=label,
        offsets=ANNOTATION_OFFSETS,
        cue=CUE_POSITION[method],
    )


# =====================================================================
# Legends
# =====================================================================

add_row_legends(fig, axes[(1, 0)])

fig.legend(
    handles=dominance_handles(),
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
)


# =====================================================================
# Save
# =====================================================================

save_figure(
    fig,
    Path(__file__).resolve().parent,
    "tradeoff_MMLU_CAT_LAT",
)

plt.show()

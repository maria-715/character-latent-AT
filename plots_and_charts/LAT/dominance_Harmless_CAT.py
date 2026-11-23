from pathlib import Path

import matplotlib.pyplot as plt

from cat_lat_common import (
    dominance_handles,
    method_handles,
    plot_tradeoff_panel,
    save_figure,
)


# =====================================================================
# CAT dominance analysis only
#
# The bottom row of Workshop_submission/utility_and_dominance.pdf:
#
#   (a) Harmless vs GCG ASR     (b) Harmless vs Soft-Prompt ASR
# =====================================================================


# =====================================================================
# Annotation offsets (copied from utility-dominance.py)
#
# Values are offsets in display points: (dx, dy).
# Key: (method, robustness_metric, series, epsilon)
# =====================================================================

ANNOTATION_OFFSETS = {
    # =========================================================
    # GCG
    # =========================================================

    # Zephyr
    ("CAT", "GCG", "Zephyr", 0.00884): (-5, 5),
    ("CAT", "GCG", "Zephyr", 0.01768): (-10, -12),
    ("CAT", "GCG", "Zephyr", 0.03535): (-10, 5),
    ("CAT", "GCG", "Zephyr", 0.07070): (7, -5),

    # Llama 3 (Llama 2 recipe)
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.02949): (-15, 7),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.04424): (-25, 7),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.05898): (-20, -17),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.08848): (7, 3),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.11797): (5, -5),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.17695): (7, 5),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.23594): (7, 5),
    ("CAT", "GCG", "Llama 3 (Llama 2 recipe)", 0.35391): (7, 5),

    # Llama 3 (MixAT recipe)
    ("CAT", "GCG", "Llama 3 (MixAT recipe)", 0.04424): (5, 5),
    ("CAT", "GCG", "Llama 3 (MixAT recipe)", 0.08848): (7, -10),
    ("CAT", "GCG", "Llama 3 (MixAT recipe)", 0.17695): (7, 0),
    ("CAT", "GCG", "Llama 3 (MixAT recipe)", 0.35391): (5, 5),

    # =========================================================
    # Soft-Prompt
    # =========================================================

    # Zephyr
    ("CAT", "Soft-Prompt", "Zephyr", 0.00884): (2, 5),
    ("CAT", "Soft-Prompt", "Zephyr", 0.01768): (-20, 7),
    ("CAT", "Soft-Prompt", "Zephyr", 0.03535): (-20, -10),
    ("CAT", "Soft-Prompt", "Zephyr", 0.07070): (-10, 7),

    # Llama 3 (Llama 2 recipe)
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.02949): (-20, -15),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.04424): (-15, 7),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.05898): (7, -7),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.08848): (7, 7),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.11797): (7, -10),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.17695): (7, 5),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.23594): (7, 5),
    ("CAT", "Soft-Prompt", "Llama 3 (Llama 2 recipe)", 0.35391): (7, 5),

    # Llama 3 (MixAT recipe)
    ("CAT", "Soft-Prompt", "Llama 3 (MixAT recipe)", 0.04424): (-20, 5),
    ("CAT", "Soft-Prompt", "Llama 3 (MixAT recipe)", 0.08848): (7, -5),
    ("CAT", "Soft-Prompt", "Llama 3 (MixAT recipe)", 0.17695): (7, -5),
    ("CAT", "Soft-Prompt", "Llama 3 (MixAT recipe)", 0.35391): (5, 5),
}


# =====================================================================
# Figure
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.0, 4.6),
)

for ax, metric, label in (
    (axes[0], "GCG", "a"),
    (axes[1], "Soft-Prompt", "b"),
):
    plot_tradeoff_panel(
        ax,
        "CAT",
        utility_metric="Harmless",
        robustness_metric=metric,
        panel_label=label,
        offsets=ANNOTATION_OFFSETS,
    )


# =====================================================================
# Legends
# =====================================================================

# Model/configuration legend across the top.
fig.legend(
    handles=method_handles("CAT"),
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    fontsize=12,
)

# Dominance legend below the figure.
fig.legend(
    handles=dominance_handles(),
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=2,
    fontsize=12,
)


# =====================================================================
# Layout and save
# =====================================================================

fig.tight_layout(
    rect=[0.02, 0.09, 1.0, 0.89]
)

fig.subplots_adjust(
    wspace=0.27,
)

save_figure(
    fig,
    Path(__file__).resolve().parent,
    "dominance_Harmless_CAT",
)

plt.show()

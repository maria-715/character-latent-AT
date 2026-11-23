#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Typography
# ============================================================

plt.rcParams.update({
    "font.size": 13,          # default text
    "axes.titlesize": 14,     # panel titles
    "axes.labelsize": 13,     # x/y labels
    "xtick.labelsize": 11,    # x tick labels
    "ytick.labelsize": 11,    # y tick labels
    "legend.fontsize": 11,    # legend
})

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
# Columns:
#   D_selected mean,
#   D_selected std,
#   D_final mean,
#   D_final std

data = {
    "Zephyr": np.array([
        [0.00734, 0.00003, 0.00785, 0.00009],
        [0.01466, 0.00006, 0.01512, 0.00016],
        [0.02688, 0.00068, 0.02746, 0.00071],
        [0.04422, 0.00423, 0.04514, 0.00424],
    ]),

    "Llama 3 (Llama 2)": np.array([
        [0.02492, 0.00050, 0.02506, 0.00049],
        [0.03491, 0.00107, 0.03501, 0.00102],
        [0.04517, 0.00030, 0.04522, 0.00032],
        [0.06035, 0.00039, 0.06046, 0.00037],
        [0.07039, 0.00285, 0.07100, 0.00248],
        [0.09350, 0.00361, 0.09417, 0.00336],
        [0.10172, 0.00553, 0.10467, 0.00470],
        [0.13176, 0.00496, 0.13789, 0.00295],
    ]),

    "Llama 3 (MixAT)": np.array([
        [0.03539, 0.00045, 0.03544, 0.00044],
        [0.05884, 0.00232, 0.05903, 0.00225],
        [0.09078, 0.00334, 0.09179, 0.00277],
        [0.13500, 0.00616, 0.14001, 0.00442],
    ]),
}


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
markers = {
    "Zephyr": "o",
    "Llama 3 (Llama 2)": "s",
    "Llama 3 (MixAT)": "^",
}

# plt.rcParams.update({
#     "font.size": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 9,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(5.0, 4.5),
    constrained_layout=True,
)


# ---------------------------------------------------------------------
# Data points
# ---------------------------------------------------------------------
for label, values in data.items():

    d_selected = values[:, 0]
    d_selected_std = values[:, 1]

    d_final = values[:, 2]
    d_final_std = values[:, 3]

    ax.errorbar(
        d_selected,
        d_final,
        xerr=d_selected_std,
        yerr=d_final_std,
        fmt=markers[label],
        markersize=5.5,
        capsize=2.5,
        linewidth=1.0,
        elinewidth=1.0,
        label=label,
    )


# ---------------------------------------------------------------------
# y = x parity reference
# ---------------------------------------------------------------------
# Use identical limits on both axes so the diagonal is genuinely 45°.
all_values = np.concatenate([
    values[:, [0, 2]].ravel()
    for values in data.values()
])

upper = all_values.max() * 1.08

ax.plot(
    [0, upper],
    [0, upper],
    linestyle="--",
    linewidth=1.0,
    color="0.35",
    label=r"Parity ($y=x$)",
)


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------
ax.set_xlabel(r"Selected-iterate distance $D_{\mathrm{selected}}$")
ax.set_ylabel(r"Final-iterate distance $D_{\mathrm{final}}$")

ax.set_xlim(0, upper)
ax.set_ylim(0, upper)

# Important for a parity plot: one unit on x should have the same
# physical length as one unit on y.
ax.set_aspect("equal", adjustable="box")

ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5,
)
ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    loc="upper left",
)


# ---------------------------------------------------------------------
# Save beside this script
# ---------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "selected_vs_final_distance.pdf"
png_path = output_dir / "selected_vs_final_distance.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
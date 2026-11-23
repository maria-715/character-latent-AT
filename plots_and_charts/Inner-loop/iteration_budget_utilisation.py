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
epsilon = np.array([
    0.00884,
    0.01768,
    0.03535,
    0.07070,
])

rho_k10 = np.array([
    0.831,
    0.829,
    0.760,
    0.625,
])

rho_k100 = np.array([
    1.000,
    0.999,
    0.997,
    0.999,
])


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
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
    figsize=(5.2, 3.6),
    constrained_layout=True,
)


# K = 10
ax.plot(
    epsilon,
    rho_k10,
    marker="o",
    markersize=5,
    linewidth=1.6,
    label=r"$K=10$",
)


# K = 100
ax.plot(
    epsilon,
    rho_k100,
    marker="s",
    markersize=5,
    linewidth=1.6,
    label=r"$K=100$",
)


# ---------------------------------------------------------------------
# Full-utilisation reference
# ---------------------------------------------------------------------
ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    color="0.35",
    label=r"Full utilisation ($\rho=1$)",
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_xlabel(r"Configured radius $\epsilon$")
ax.set_ylabel(r"Utilisation $\rho = D/\epsilon$")

ax.set_xlim(left=0)
ax.set_ylim(0.58, 1.025)

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
    loc="lower left",
)


# ---------------------------------------------------------------------
# Save beside this script
# ---------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "iteration_budget_utilisation.pdf"
png_path = output_dir / "iteration_budget_utilisation.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Data from thesis Table 18
# ---------------------------------------------------------------------
# Each row is:
#   epsilon, mean realised perturbation distance D, std. dev. of D
#
# Sweep values are means across three training seeds, with error bars
# showing one sample standard deviation across those seeds.
#
# GitHub reproduction points are intentionally excluded because they are
# not part of the controlled sweeps.

data = {
    "Zephyr": np.array([
        [0.00884, 0.00734, 0.00003],
        [0.01768, 0.01466, 0.00006],
        [0.03535, 0.02688, 0.00068],
        [0.07070, 0.04422, 0.00423],
    ]),
    "Llama 3 (Llama 2 recipe)": np.array([
        [0.02949, 0.02492, 0.00050],
        [0.04424, 0.03491, 0.00107],
        [0.05898, 0.04517, 0.00030],
        [0.08848, 0.06035, 0.00039],
        [0.11797, 0.07039, 0.00285],
        [0.17695, 0.09350, 0.00361],
        [0.23594, 0.10172, 0.00553],
        [0.35391, 0.13176, 0.00496],
    ]),
    "Llama 3 (MixAT recipe)": np.array([
        [0.04424, 0.03539, 0.00045],
        [0.08848, 0.05884, 0.00232],
        [0.17695, 0.09078, 0.00334],
        [0.35391, 0.13500, 0.00616],
    ]),
}


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
# Distinct markers supplement colour differences so that the three
# configurations remain distinguishable in print.

markers = ["o", "s", "^"]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
# Use a linear-linear plot, as in thesis Figure 5. This preserves the
# direct geometric interpretation of the D = epsilon reference line:
# increasing distance below the diagonal indicates declining utilisation.

fig, ax = plt.subplots(
    figsize=(5.5, 4.2),
    constrained_layout=True,
)


for (label, values), marker in zip(data.items(), markers):
    epsilon = values[:, 0]
    D = values[:, 1]
    D_std = values[:, 2]

    ax.errorbar(
        epsilon,
        D,
        yerr=D_std,
        marker=marker,
        markersize=5,
        linewidth=1.5,
        capsize=2.5,
        label=label,
    )


# ---------------------------------------------------------------------
# D = epsilon reference
# ---------------------------------------------------------------------
max_epsilon = max(values[:, 0].max() for values in data.values())

reference_x = np.linspace(0, max_epsilon, 500)

ax.plot(
    reference_x,
    reference_x,
    linestyle="--",
    linewidth=1.0,
    color="0.35",
    label=r"$D=\epsilon$",
)


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------
ax.set_xlabel(r"Configured radius $\epsilon$")
ax.set_ylabel(r"Realised perturbation distance $D$")

# Start both axes at zero so that the geometry of D = epsilon is
# immediately interpretable.
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Light grid to aid comparison across configured radii.
ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5,
)
ax.set_axisbelow(True)


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------
ax.legend(frameon=False)


# ---------------------------------------------------------------------
# Save beside this script
# ---------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "budget_utilisation.pdf"
png_path = output_dir / "budget_utilisation.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()

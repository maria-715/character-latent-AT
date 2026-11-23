from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
# Each row is:
#   epsilon, mean realised distance D, standard deviation of D
#
# GitHub reproduction points are intentionally excluded because they
# are orientation/reference configurations rather than part of the sweep.

data = {
    "Zephyr": np.array([
        [0.00884, 0.00734, 0.00003],
        [0.01768, 0.01466, 0.00006],
        [0.03535, 0.02688, 0.00068],
        [0.07070, 0.04422, 0.00423],
    ]),
    "Llama 3 (Llama 2)": np.array([
        [0.02949, 0.02492, 0.00050],
        [0.04424, 0.03491, 0.00107],
        [0.05898, 0.04517, 0.00030],
        [0.08848, 0.06035, 0.00039],
        [0.11797, 0.07039, 0.00285],
        [0.17695, 0.09350, 0.00361],
        [0.23594, 0.10172, 0.00553],
        [0.35391, 0.13176, 0.00496],
    ]),
    "Llama 3 (MixAT)": np.array([
        [0.04424, 0.03539, 0.00045],
        [0.08848, 0.05884, 0.00232],
        [0.17695, 0.09078, 0.00334],
        [0.35391, 0.13500, 0.00616],
    ]),
}


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
# Use distinct markers as well as colours so the curves remain
# distinguishable in print and for colour-impaired readers.
markers = ["o", "s", "^"]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, (ax_d, ax_rho) = plt.subplots(
    1,
    2,
    figsize=(9.0, 3.7),
    constrained_layout=True,
)


for (label, values), marker in zip(data.items(), markers):
    epsilon = values[:, 0]
    D = values[:, 1]
    D_std = values[:, 2]

    rho = D / epsilon
    rho_std = D_std / epsilon

    # -------------------------------------------------------------
    # (a) Realised distance
    # -------------------------------------------------------------
    ax_d.errorbar(
        epsilon,
        D,
        yerr=D_std,
        marker=marker,
        markersize=5,
        linewidth=1.5,
        capsize=2.5,
        label=label,
    )

    # -------------------------------------------------------------
    # (b) Radius utilisation
    # -------------------------------------------------------------
    ax_rho.errorbar(
        epsilon,
        rho,
        yerr=rho_std,
        marker=marker,
        markersize=5,
        linewidth=1.5,
        capsize=2.5,
        label=label,
    )


# ---------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------
max_epsilon = max(values[:, 0].max() for values in data.values())

# D = epsilon: full radius utilisation
reference_x = np.linspace(0, max_epsilon, 500)

ax_d.plot(
    reference_x,
    reference_x,
    linestyle="--",
    linewidth=1.0,
    color="0.35",
    label=r"$D=\epsilon$",
)

# rho = 1: full radius utilisation
ax_rho.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    color="0.35",
    label=r"$\rho=1$",
)


# ---------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------
ax_d.set_xlabel(r"Configured radius $\epsilon$")
ax_d.set_ylabel(r"Realised distance $D$")
ax_d.set_title("(a) Realised perturbation distance")

ax_rho.set_xlabel(r"Configured radius $\epsilon$")
ax_rho.set_ylabel(r"Utilisation $\rho = D/\epsilon$")
ax_rho.set_title("(b) Radius utilisation")

# Start at zero to make the geometric interpretation of D = epsilon clear.
ax_d.set_xlim(left=0)
ax_d.set_ylim(bottom=0)

ax_rho.set_xlim(left=0)
ax_rho.set_ylim(0.3, 1.05)

# Light grid, primarily to help compare values across radii.
for ax in (ax_d, ax_rho):
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
# A single legend is sufficient because both panels contain the same
# configurations. Put it in panel (a), including the D=epsilon reference.
ax_d.legend(frameon=False)

# The rho=1 line is self-explanatory from its position, so no second
# legend is necessary.


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
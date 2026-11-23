import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


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

# ============================================================
# Configuration
# ============================================================

COLORS = {
    "Zephyr": "C0",
    "Llama 3 (Llama 2)": "C1",
    "Llama 3 (MixAT)": "C2",
}

MARKERS = {
    "Zephyr": "o",
    "Llama 3 (Llama 2)": "s",
    "Llama 3 (MixAT)": "^",
}


# ============================================================
# Data
# ============================================================

models = {

    # ========================================================
    # Zephyr
    # ========================================================

    "Zephyr": {
        "eps": np.array([
            0.0,
            0.05,
            0.10,
            0.20,
            0.40,
        ]),

        "Harmless": {
            "y": np.array([
                100.0,
                98.33,
                94.17,
                89.17,
                70.83,
            ]),
            "err": np.array([
                0.0,
                1.44,
                3.82,
                1.44,
                9.46,
            ]),
        },

        "XSTest": {
            "safe": {
                "y": np.array([
                    94.80,
                    62.40,
                    54.67,
                    53.47,
                    70.93,
                ]),
                "err": np.array([
                    0.0,
                    7.34,
                    6.47,
                    2.95,
                    13.05,
                ]),
            },

            "unsafe": {
                "y": np.array([
                    9.50,
                    98.83,
                    98.83,
                    99.00,
                    89.00,
                ]),
                "err": np.array([
                    0.0,
                    0.58,
                    0.76,
                    0.50,
                    9.37,
                ]),
            },
        },
    },

    # ========================================================
    # Llama 3 (Llama 2)
    # ========================================================

    "Llama 3 (Llama 2)": {
        "eps": np.array([
            0.0,
            0.05,
            0.075,
            0.10,
            0.150,
            0.20,
            0.300,
            0.40,
            0.600,
        ]),

        "Harmless": {
            "y": np.array([
                100.0,
                95.83,
                90.83,
                76.67,
                45.83,
                49.17,
                7.50,
                2.50,
                1.67,
            ]),
            "err": np.array([
                0.0,
                1.44,
                7.22,
                7.64,
                26.96,
                20.05,
                12.99,
                4.33,
                2.89,
            ]),
        },

        "XSTest": {
            "safe": {
                "y": np.array([
                    97.20,
                    61.87,
                    48.13,
                    47.47,
                    52.13,
                    32.40,
                    20.67,
                    32.00,
                    0.80,
                ]),
                "err": np.array([
                    0.0,
                    5.72,
                    18.36,
                    14.14,
                    29.31,
                    26.89,
                    22.10,
                    28.93,
                    0.40,
                ]),
            },

            "unsafe": {
                "y": np.array([
                    91.50,
                    99.33,
                    92.67,
                    99.67,
                    96.83,
                    99.17,
                    98.50,
                    98.83,
                    100.00,
                ]),
                "err": np.array([
                    0.0,
                    0.29,
                    10.97,
                    0.29,
                    3.51,
                    0.29,
                    0.87,
                    1.26,
                    0.00,
                ]),
            },
        },
    },

    # ========================================================
    # Llama 3 (MixAT)
    # ========================================================

    "Llama 3 (MixAT)": {
        "eps": np.array([
            0.0,
            0.075,
            0.150,
            0.300,
            0.600,
        ]),

        "Harmless": {
            "y": np.array([
                100.0,
                82.50,
                58.33,
                20.83,
                50.00,
            ]),
            "err": np.array([
                0.0,
                13.92,
                14.65,
                15.07,
                41.61,
            ]),
        },

        "XSTest": {
            "safe": {
                "y": np.array([
                    97.20,
                    51.60,
                    51.07,
                    60.27,
                    37.33,
                ]),
                "err": np.array([
                    0.0,
                    15.48,
                    10.51,
                    10.01,
                    10.99,
                ]),
            },

            "unsafe": {
                "y": np.array([
                    91.50,
                    99.33,
                    98.50,
                    97.00,
                    99.50,
                ]),
                "err": np.array([
                    0.0,
                    0.29,
                    0.87,
                    1.80,
                    0.50,
                ]),
            },
        },
    },
}


# ============================================================
# Two-panel safety sweep
# ============================================================

def plot_safety_sweeps():

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
    )

    harmless_ax = axes[0]
    xstest_ax = axes[1]

    # ========================================================
    # Combined multiplier grid
    # ========================================================

    all_eps = np.unique(
        np.concatenate([
            model_data["eps"]
            for model_data in models.values()
        ])
    )

    labels = []

    for value in all_eps:
        if np.isclose(value, 0.0):
            labels.append("0")
        elif np.isclose(value, 0.05):
            labels.append("0.05")
        elif np.isclose(value, 0.075):
            labels.append("0.075")
        elif np.isclose(value, 0.10):
            labels.append("0.1")
        elif np.isclose(value, 0.150):
            labels.append("0.15")
        elif np.isclose(value, 0.20):
            labels.append("0.2")
        elif np.isclose(value, 0.300):
            labels.append("0.3")
        elif np.isclose(value, 0.40):
            labels.append("0.4")
        elif np.isclose(value, 0.600):
            labels.append("0.6")
        else:
            labels.append(f"{value:.2f}")

    # ========================================================
    # Panel 1: Harmless
    # ========================================================

    harmless_lower = []
    harmless_upper = []

    for model_name, model_data in models.items():

        eps = model_data["eps"]
        data = model_data["Harmless"]

        y = data["y"]
        err = data["err"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        harmless_ax.plot(
            eps,
            y,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            color=color,
            label=model_name,
            zorder=3,
        )

        harmless_ax.errorbar(
            eps,
            y,
            yerr=err,
            fmt="none",
            capsize=3,
            linewidth=1.2,
            color=color,
            zorder=2,
        )

        harmless_lower.extend(y - err)
        harmless_upper.extend(y + err)

    # Adaptive y-axis
    ymin = min(harmless_lower)
    ymax = max(harmless_upper)

    padding = max(
        (ymax - ymin) * 0.10,
        0.5,
    )

    harmless_ax.set_ylim(
        max(0, ymin - padding),
        min(110, ymax + padding),
    )

    harmless_ax.set_title(
        "Harmless",
        # fontsize=11,
    )

    harmless_ax.grid(
        axis="y",
        alpha=0.25,
    )

    harmless_ax.spines["top"].set_visible(False)
    harmless_ax.spines["right"].set_visible(False)

    # ========================================================
    # Panel 2: XSTest
    #
    # Color/marker = model
    # Solid line     = Safe compliance
    # Dashed line    = Unsafe refusal
    # ========================================================

    xstest_lower = []
    xstest_upper = []

    for model_name, model_data in models.items():

        eps = model_data["eps"]

        color = COLORS[model_name]
        marker = MARKERS[model_name]

        # ----------------------------------------------------
        # Safe compliance
        # ----------------------------------------------------

        safe_y = model_data["XSTest"]["safe"]["y"]
        safe_err = model_data["XSTest"]["safe"]["err"]

        xstest_ax.plot(
            eps,
            safe_y,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            linestyle="-",
            color=color,
            zorder=3,
        )

        xstest_ax.errorbar(
            eps,
            safe_y,
            yerr=safe_err,
            fmt="none",
            capsize=3,
            linewidth=1.2,
            color=color,
            zorder=2,
        )

        # ----------------------------------------------------
        # Unsafe refusal
        # ----------------------------------------------------

        unsafe_y = model_data["XSTest"]["unsafe"]["y"]
        unsafe_err = model_data["XSTest"]["unsafe"]["err"]

        xstest_ax.plot(
            eps,
            unsafe_y,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            linestyle="--",
            color=color,
            zorder=3,
        )

        xstest_ax.errorbar(
            eps,
            unsafe_y,
            yerr=unsafe_err,
            fmt="none",
            capsize=3,
            linewidth=1.2,
            color=color,
            zorder=2,
        )

        xstest_lower.extend(safe_y - safe_err)
        xstest_lower.extend(unsafe_y - unsafe_err)

        xstest_upper.extend(safe_y + safe_err)
        xstest_upper.extend(unsafe_y + unsafe_err)

    # Adaptive y-axis
    ymin = min(xstest_lower)
    ymax = max(xstest_upper)

    padding = max(
        (ymax - ymin) * 0.10,
        0.5,
    )

    xstest_ax.set_ylim(
        max(0, ymin - padding),
        min(110, ymax + padding),
    )

    xstest_ax.set_title(
        "XSTest",
        # fontsize=11,
    )

    xstest_ax.grid(
        axis="y",
        alpha=0.25,
    )

    xstest_ax.spines["top"].set_visible(False)
    xstest_ax.spines["right"].set_visible(False)

    # ========================================================
    # X-axis
    #
    # Keep labels on both panels, exactly like your current
    # capability figure.
    # ========================================================

    for ax in axes:

        ax.set_xticks(all_eps)

        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )
        ax.set_xlim(left=0)

        ax.tick_params(
            axis="x",
            labelbottom=True,
        )

    # ========================================================
    # Shared labels
    # ========================================================

    # fig.supxlabel(
    #     r"Configuration multiplier $s$",
    #     fontsize=11,
    # )

    axes[-1].set_xlabel(
        r"Configuration multiplier $s$",
        labelpad=3.5,
    )

    fig.supylabel(
        "Score (%)",
        # fontsize=11,
    )

    # ========================================================
    # Legends
    # ========================================================

    # Model/color legend
    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            label=name,
        )
        for name in models.keys()
    ]

    fig.legend(
        handles=model_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    # XSTest meaning-of-line-style legend
    metric_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.8,
            label="Safe compliance",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Unsafe refusal",
        ),
    ]

    xstest_ax.legend(
        handles=metric_handles,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.70, 0.60),
        borderaxespad=0,
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout(
        rect=[0.04, 0.04, 1.0, 0.94]
    )

    fig.subplots_adjust(
        hspace=0.45
    )

    return fig


# ============================================================
# Generate figure
# ============================================================

safety_figure = plot_safety_sweeps()


# ============================================================
# Output paths
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent


# ============================================================
# Save separately
# ============================================================

safety_pdf_path = (
    output_dir / "safety_sweeps.pdf"
)

safety_png_path = (
    output_dir / "safety_sweeps.png"
)

safety_figure.savefig(
    safety_pdf_path,
    bbox_inches="tight",
)

safety_figure.savefig(
    safety_png_path,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save to diagnostic PDF
# ============================================================

combined_pdf_path = (
    output_dir / f"{script_path.stem}.pdf"
)

with PdfPages(combined_pdf_path) as pdf:

    pdf.savefig(
        safety_figure,
        bbox_inches="tight",
    )


# ============================================================
# Report outputs
# ============================================================

print(
    "Saved safety figure to:\n"
    f"  {safety_pdf_path}\n"
    f"  {safety_png_path}\n\n"
    "Saved diagnostic PDF to:\n"
    f"  {combined_pdf_path}"
)


# ============================================================
# Display
# ============================================================

plt.show()
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from lat_data import (
    COLORS,
    EPS,
    HF_LAT,
    MARKERS,
    SERIES_LABELS,
    eps_tick_labels,
    models,
    style_axis,
)


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

HF_STYLE = dict(
    color="0.35",
    linestyle=":",
    linewidth=2.0,
    zorder=1,
)


# ============================================================
# One series (mean trajectory + SD error bars)
# ============================================================

def plot_series(ax, eps, y, err, color, marker, linestyle="-", label=None):

    ax.plot(
        eps,
        y,
        marker=marker,
        markersize=6,
        linewidth=1.8,
        linestyle=linestyle,
        color=color,
        label=label,
        zorder=3,
    )

    ax.errorbar(
        eps,
        y,
        yerr=err,
        fmt="none",
        capsize=3,
        linewidth=1.2,
        color=color,
        zorder=2,
    )

    return list(y - err), list(y + err)


# ============================================================
# Multi-panel epsilon sweep
#
# panels: list of (metric, title). metric "XSTest" draws safe
# compliance (solid) and unsafe refusal (dashed) together.
# ============================================================

def plot_sweeps(panels, ylabel, figsize=(8, 8.5), ymax_cap=100):

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=figsize,
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    for ax, (metric, panel_title) in zip(axes, panels):

        all_lower = []
        all_upper = []

        for name, model_data in models.items():

            eps = model_data["eps"]
            color = COLORS[name]
            marker = MARKERS[name]

            if metric == "XSTest":
                parts = [
                    ("safe", "-"),
                    ("unsafe", "--"),
                ]
            else:
                parts = [(None, "-")]

            for part, linestyle in parts:

                data = (
                    model_data[metric][part]
                    if part is not None
                    else model_data[metric]
                )

                lower, upper = plot_series(
                    ax,
                    eps,
                    data["y"],
                    data["err"],
                    color,
                    marker,
                    linestyle=linestyle,
                    label=SERIES_LABELS[name] if linestyle == "-" else None,
                )

                all_lower.extend(lower)
                all_upper.extend(upper)

        # ----------------------------------------------------
        # HuggingFace LAT reference
        # ----------------------------------------------------

        if metric == "XSTest":
            hf_values = [HF_LAT["XSTest-safe"], HF_LAT["XSTest-unsafe"]]
        else:
            hf_values = [HF_LAT[metric]]

        for value in hf_values:
            ax.axhline(value, **HF_STYLE)

        all_lower.extend(hf_values)
        all_upper.extend(hf_values)

        # ----------------------------------------------------
        # Adaptive y-axis
        # ----------------------------------------------------

        ymin = min(all_lower)
        ymax = max(all_upper)

        padding = max(
            (ymax - ymin) * 0.10,
            0.5,
        )

        ax.set_ylim(
            max(0, ymin - padding),
            min(ymax_cap, ymax + padding),
        )

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        ax.set_xticks(EPS)
        ax.set_xticklabels(eps_tick_labels(EPS))
        # Small left margin so eps = 0 sits off the y-axis
        ax.set_xlim(left=-0.03 * EPS.max())

        ax.tick_params(
            axis="x",
            labelbottom=True,
        )

        ax.set_title(panel_title)

        style_axis(ax)

        if metric == "XSTest":
            ax.legend(
                handles=[
                    Line2D([0], [0], color="black", linestyle="-",
                           linewidth=1.8, label="Safe compliance"),
                    Line2D([0], [0], color="black", linestyle="--",
                           linewidth=1.8, label="Unsafe refusal"),
                ],
                frameon=False,
                loc="center right",
            )

    # ========================================================
    # Shared labels
    # ========================================================

    axes[-1].set_xlabel(
        r"Configured radius $\epsilon$",
        labelpad=3.5,
    )

    fig.supylabel(ylabel)

    # ========================================================
    # Shared legend
    # ========================================================

    handles, legend_labels = axes[0].get_legend_handles_labels()

    handles.append(Line2D([0], [0], **HF_STYLE))
    legend_labels.append("HuggingFace LAT Llama 3")

    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    fig.tight_layout(
        rect=[0.04, 0.02, 1.0, 0.94]
    )

    fig.subplots_adjust(
        hspace=0.35,
    )

    return fig

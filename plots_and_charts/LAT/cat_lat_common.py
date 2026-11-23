import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from lat_data import (
    SERIES_LABELS as LAT_LABELS,
    models as lat_models,
)


# =====================================================================
# Shared data and styling for the CAT (top row) / LAT (bottom row)
# 2x2 figures.
#
# CAT values are copied from Workshop_submission/robustness.py and
# utility-dominance.py (those scripts draw on import, so they cannot
# be imported here). GitHub reproduction configurations are excluded.
#
# Every array includes epsilon = 0 (unmodified reference model, no
# error bar) at index 0 and is ordered by increasing epsilon.
# =====================================================================


# =====================================================================
# Typography
# =====================================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =====================================================================
# Colours / markers
# =====================================================================

CAT_COLORS = {
    "Zephyr": "C0",
    "Llama 3 (Llama 2 recipe)": "C1",
    "Llama 3 (MixAT recipe)": "C2",
}

CAT_MARKERS = {
    "Zephyr": "o",
    "Llama 3 (Llama 2 recipe)": "s",
    "Llama 3 (MixAT recipe)": "^",
}

# Distinct from every CAT colour / marker
LAT_COLORS = {
    LAT_LABELS["base"]: "C3",
    LAT_LABELS["10pct"]: "C4",
}

LAT_MARKERS = {
    LAT_LABELS["base"]: "D",
    LAT_LABELS["10pct"]: "v",
}

CAT_XLIM = (-0.015, 0.375)

# Same relative left margin as the CAT axis
LAT_XLIM = (-2.0, 50.0)


# =====================================================================
# CAT data
# =====================================================================

cat_models = {

    "Zephyr": {
        "epsilon": np.array([0.0, 0.00884, 0.01768, 0.03535, 0.07070]),
        "GCG": {
            "y": np.array([92.5, 19.17, 17.50, 18.33, 9.17]),
            "err": np.array([0.0, 3.82, 6.61, 10.10, 1.44]),
        },
        "Soft-Prompt": {
            "y": np.array([74.09, 43.44, 43.40, 42.45, 51.13]),
            "err": np.array([0.0, 4.53, 5.42, 13.65, 3.21]),
        },
        "MMLU": {
            "y": np.array([56.02, 55.05, 54.98, 55.17, 54.92]),
            "err": np.array([0.0, 0.30, 0.26, 0.17, 0.17]),
        },
        "Harmless": {
            "y": np.array([100.0, 98.33, 94.17, 89.17, 70.83]),
            "err": np.array([0.0, 1.44, 3.82, 1.44, 9.46]),
        },
    },

    "Llama 3 (Llama 2 recipe)": {
        "epsilon": np.array([
            0.0, 0.02949, 0.04424, 0.05898, 0.08848,
            0.11797, 0.17695, 0.23594, 0.35391,
        ]),
        "GCG": {
            "y": np.array([
                28.33, 22.50, 20.83, 16.67, 22.50,
                19.17, 19.17, 22.50, 33.33,
            ]),
            "err": np.array([
                0.0, 6.61, 3.82, 10.41, 12.99,
                7.22, 11.55, 5.00, 9.46,
            ]),
        },
        "Soft-Prompt": {
            "y": np.array([
                75.41, 52.62, 57.29, 48.32, 59.25,
                51.26, 59.33, 62.53, 64.46,
            ]),
            "err": np.array([
                0.0, 2.55, 9.37, 1.89, 16.08,
                3.17, 1.41, 5.64, 2.30,
            ]),
        },
        "MMLU": {
            "y": np.array([
                60.75, 60.10, 59.74, 59.83, 59.35,
                58.81, 57.69, 56.23, 54.94,
            ]),
            "err": np.array([
                0.0, 0.09, 0.17, 0.13, 0.55,
                0.11, 0.45, 1.04, 0.98,
            ]),
        },
        "Harmless": {
            "y": np.array([
                100.0, 95.83, 90.83, 76.67, 45.83,
                49.17, 7.50, 2.50, 1.67,
            ]),
            "err": np.array([
                0.0, 1.44, 7.22, 7.64, 26.96,
                20.05, 12.99, 4.33, 2.89,
            ]),
        },
    },

    "Llama 3 (MixAT recipe)": {
        "epsilon": np.array([0.0, 0.04424, 0.08848, 0.17695, 0.35391]),
        "GCG": {
            "y": np.array([28.33, 25.83, 12.50, 17.50, 25.00]),
            "err": np.array([0.0, 1.44, 2.50, 6.61, 10.90]),
        },
        "Soft-Prompt": {
            "y": np.array([75.41, 55.47, 43.23, 58.18, 63.21]),
            "err": np.array([0.0, 4.46, 3.92, 4.86, 3.20]),
        },
        "MMLU": {
            "y": np.array([60.75, 59.73, 59.47, 58.51, 55.74]),
            "err": np.array([0.0, 0.22, 0.14, 0.68, 0.41]),
        },
        "Harmless": {
            "y": np.array([100.0, 82.50, 58.33, 20.83, 50.00]),
            "err": np.array([0.0, 13.92, 14.65, 15.07, 41.61]),
        },
    },
}


# =====================================================================
# LAT data, reshaped to the same layout as cat_models
# =====================================================================

lat_series = {
    LAT_LABELS[key]: {
        "epsilon": data["eps"],
        "GCG": data["GCG"],
        "Soft-Prompt": data["Soft-Prompts"],
        "MMLU": data["MMLU"],
        "Harmless": data["Harmless"],
    }
    for key, data in lat_models.items()
}

LAT_EPS = lat_models["base"]["eps"]


# =====================================================================
# Method bundles: everything a panel needs to draw one method
# =====================================================================

METHODS = {
    "CAT": {
        "series": cat_models,
        "colors": CAT_COLORS,
        "markers": CAT_MARKERS,
        "xlim": CAT_XLIM,
        "xticks": None,
    },
    "LAT": {
        "series": lat_series,
        "colors": LAT_COLORS,
        "markers": LAT_MARKERS,
        "xlim": LAT_XLIM,
        "xticks": LAT_EPS,
    },
}

YLABELS = {
    "GCG": "GCG ASR (%)",
    "Soft-Prompt": "Soft-Prompt ASR (%)",
    "MMLU": "MMLU accuracy (%)",
    "Harmless": "Harmless successful-response rate (%)",
}


def style_axis(ax):
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_adaptive_ylim(ax, lower, upper, min_padding=1.0, ymax_cap=100):

    ymin = min(lower)
    ymax = max(upper)

    padding = max(
        (ymax - ymin) * 0.08,
        min_padding,
    )

    ax.set_ylim(
        max(0, ymin - padding),
        min(ymax_cap, ymax + padding),
    )


# =====================================================================
# Metric vs configured radius (no y-limit changes)
# =====================================================================

def plot_metric(ax, method, metric):

    m = METHODS[method]

    all_lower = []
    all_upper = []

    for name, data in m["series"].items():

        epsilon = data["epsilon"]
        y = data[metric]["y"]
        err = data[metric]["err"]

        # Lines connect successive sampled radii only as a visual guide.
        ax.plot(
            epsilon,
            y,
            marker=m["markers"][name],
            markersize=5,
            linewidth=1.4,
            color=m["colors"][name],
            label=name,
            zorder=3,
        )

        ax.errorbar(
            epsilon,
            y,
            yerr=err,
            fmt="none",
            capsize=2.5,
            linewidth=1.1,
            color=m["colors"][name],
            zorder=2,
        )

        all_lower.extend(y - err)
        all_upper.extend(y + err)

    ax.set_xlim(*m["xlim"])

    if m["xticks"] is not None:
        ax.set_xticks(m["xticks"])
        ax.set_xticklabels([f"{v:g}" for v in m["xticks"]])

    style_axis(ax)

    return all_lower, all_upper


def plot_sweep_panel(ax, method, metric, panel_label, panel_title,
                     min_padding=1.0, ymax_cap=100):

    lower, upper = plot_metric(ax, method, metric)

    set_adaptive_ylim(
        ax, lower, upper,
        min_padding=min_padding,
        ymax_cap=ymax_cap,
    )

    ax.set_xlabel(r"Configured radius $\epsilon$")
    ax.set_ylabel(YLABELS[metric])
    ax.set_title(rf"({panel_label}) {panel_title}")


# =====================================================================
# Utility vs ASR trade-off panels
#
# Only positive-radius checkpoints are used. Point i is dominated if
# another point j from the SAME sweep has utility_j >= utility_i and
# ASR_j <= ASR_i, with at least one strict inequality (three-seed
# means).
# =====================================================================

def non_dominated_mask(x, y):

    n = len(x)
    non_dominated = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            at_least_as_good = x[j] >= x[i] and y[j] <= y[i]
            strictly_better = x[j] > x[i] or y[j] < y[i]

            if at_least_as_good and strictly_better:
                non_dominated[i] = False
                break

    return non_dominated


def epsilon_label(value):
    """Compact but unambiguous epsilon label."""
    return f"{value:.5f}".rstrip("0").rstrip(".")


def plot_tradeoff_panel(
    ax,
    method,
    utility_metric,
    robustness_metric,
    panel_label,
    offsets=None,
    cue=(0.98, "right"),
):
    """
    offsets: {(method, robustness_metric, series, epsilon): (dx, dy)}
             in display points, to prevent overlapping epsilon labels.
    cue:     (x in axes coords, horizontal alignment) of the
             "better utility / better robustness" note.
    """

    m = METHODS[method]
    offsets = offsets or {}

    all_x = []
    all_y = []

    for name, data in m["series"].items():

        epsilon = data["epsilon"][1:]
        x = data[utility_metric]["y"][1:]
        y = data[robustness_metric]["y"][1:]

        color = m["colors"][name]
        marker = m["markers"][name]

        # Connect successive configured radii lightly
        ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.0,
            color=color,
            alpha=0.45,
            zorder=1,
        )

        nd = non_dominated_mask(x, y)

        # Dominated points: hollow markers
        ax.scatter(
            x[~nd],
            y[~nd],
            marker=marker,
            s=42,
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            zorder=3,
        )

        # Non-dominated points: filled markers
        ax.scatter(
            x[nd],
            y[nd],
            marker=marker,
            s=42,
            facecolors=color,
            edgecolors=color,
            linewidths=1.0,
            zorder=4,
        )

        # Label every point by epsilon
        for x_i, y_i, eps_i in zip(x, y, epsilon):

            offset = offsets.get(
                (method, robustness_metric, name, float(eps_i)),
                (5, 5),
            )

            ax.annotate(
                epsilon_label(eps_i),
                xy=(x_i, y_i),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                zorder=5,
            )

        all_x.extend(x)
        all_y.extend(y)

    # -----------------------------------------------------------------
    # Axis limits
    # -----------------------------------------------------------------

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    xpadding = max(0.07 * (xmax - xmin), 0.5)
    ypadding = max(0.08 * (ymax - ymin), 1.0)

    ax.set_xlim(
        max(0, xmin - xpadding),
        min(100, xmax + xpadding),
    )

    ax.set_ylim(
        max(0, ymin - ypadding),
        min(100, ymax + ypadding),
    )

    ax.set_xlabel(YLABELS[utility_metric])
    ax.set_ylabel(YLABELS[robustness_metric])
    ax.set_title(
        rf"({panel_label}) {utility_metric} vs {robustness_metric} ASR"
    )

    style_axis(ax)

    # Small orientation cue.
    cue_x, cue_ha = cue

    ax.text(
        cue_x,
        0.97,
        "better utility →\nbetter robustness ↓",
        transform=ax.transAxes,
        ha=cue_ha,
        va="top",
        fontsize=8,
        color="0.35",
        linespacing=1.2,
    )


def dominance_handles():

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=5,
            label="Non-dominated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="0.25",
            markersize=5,
            label="Dominated",
        ),
    ]


# =====================================================================
# Figure scaffolding: 2x2 grid with one legend per row
# =====================================================================

def make_figure(figsize=(10.0, 8.8), bottom=0.07, hspace=0.62):

    fig = plt.figure(figsize=figsize)

    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        hspace=hspace,
        wspace=0.27,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=bottom,
        top=0.89,
    )

    return fig, outer


def method_handles(method):

    m = METHODS[method]

    return [
        Line2D(
            [0],
            [0],
            color=m["colors"][name],
            marker=m["markers"][name],
            linestyle="-",
            linewidth=1.4,
            markersize=5,
            label=name,
        )
        for name in m["series"]
    ]


def add_row_legends(fig, lat_ax):

    # CAT legend: above the top row
    fig.legend(
        handles=method_handles("CAT"),
        title="CAT",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )

    # LAT legend: in the gap between the rows, above the LAT titles
    lat_top = lat_ax.get_position().y1

    fig.legend(
        handles=method_handles("LAT"),
        title="LAT",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, lat_top + 0.03),
        ncol=2,
    )


# =====================================================================
# Compact 2x2 layout (Figures 1 and 2)
#
# Shared by the robustness and utility sweeps so the two figures are
# a matched pair: identical page size, margins, spacing and legend
# placement. Saved WITHOUT tight cropping so both PDFs have exactly
# COMPACT_FIGSIZE.
#
# Each row's legend sits on one line with its bold CAT / LAT label to
# the left; its vertical position is computed from the rendered
# titles and x-labels so the bands around it stay minimal.
# =====================================================================

COMPACT_FIGSIZE = (10.0, 5.9)

COMPACT_MARGINS = dict(
    left=0.085,
    right=0.99,
    bottom=0.09,
    top=0.90,
    hspace=0.58,
    wspace=0.24,
)

COMPACT_TITLE_PAD = 4
COMPACT_XLABEL_PAD = 2

# Gap (figure fraction) between a legend and the elements around it
COMPACT_LEGEND_GAP = 0.008


def make_compact_figure():

    fig = plt.figure(figsize=COMPACT_FIGSIZE)

    margins = dict(COMPACT_MARGINS)

    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        hspace=margins.pop("hspace"),
        wspace=margins.pop("wspace"),
    )

    fig.subplots_adjust(**margins)

    return fig, outer


def compact_axes_style(axes):
    """Tighter title / x-label padding; call after panels are drawn."""

    for ax in axes:
        if ax.get_title():
            ax.set_title(ax.get_title(), pad=COMPACT_TITLE_PAD)
        ax.xaxis.labelpad = COMPACT_XLABEL_PAD

        # Shorter panels make matplotlib pick coarser y ticks; keep
        # them on round 1/2/5 steps (no 2.5 increments). Limits are
        # unchanged.
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins="auto", steps=[1, 2, 5, 10])
        )


def _fig_bbox(fig, artists):
    """Union bounding box of artists, in figure-fraction coordinates."""

    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    boxes = [
        a.get_window_extent(renderer).transformed(inv)
        for a in artists
        if a.get_visible() and a.get_window_extent(renderer).width > 0
    ]

    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)

    return x0, y0, x1, y1


def _labelled_legend(fig, method, y_center, ncol):
    """One-line legend with a bold method label to its left."""

    legend = fig.legend(
        handles=method_handles(method),
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, y_center),
        ncol=ncol,
        borderaxespad=0.0,
        borderpad=0.2,
        handlelength=1.8,
        columnspacing=1.4,
    )

    label = fig.text(
        0.0,
        y_center,
        method,
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=plt.rcParams["legend.fontsize"] + 1,
    )

    # Centre (label + legend) as one unit over the panels
    fig.canvas.draw()
    lx0, _, lx1, _ = _fig_bbox(fig, [legend])
    tx0, _, tx1, _ = _fig_bbox(fig, [label])

    gap = 0.012
    total = (tx1 - tx0) + gap + (lx1 - lx0)
    panels_center = (COMPACT_MARGINS["left"] + COMPACT_MARGINS["right"]) / 2
    start = panels_center - total / 2

    label.set_x(start + (tx1 - tx0))
    legend.set_bbox_to_anchor((start + (tx1 - tx0) + gap + (lx1 - lx0) / 2,
                               y_center))

    return legend


def add_compact_row_legends(fig, cat_axes, lat_axes):
    """
    cat_axes / lat_axes: every axes in that row (including both
    halves of a broken axis), used to measure titles and x-labels.
    """

    fig.canvas.draw()

    # --- CAT: just above the top-row titles ---------------------
    _, _, _, cat_titles_top = _fig_bbox(fig, [ax.title for ax in cat_axes])

    cat_legend = _labelled_legend(fig, "CAT", 0.5, ncol=3)
    fig.canvas.draw()
    _, ly0, _, ly1 = _fig_bbox(fig, [cat_legend])
    half = (ly1 - ly0) / 2

    y_cat = cat_titles_top + COMPACT_LEGEND_GAP + half
    cat_legend.remove()
    fig.texts.pop()
    _labelled_legend(fig, "CAT", y_cat, ncol=3)

    # --- LAT: centred between CAT x-labels and LAT titles --------
    _, cat_bottom, _, _ = _fig_bbox(
        fig,
        [ax.xaxis.label for ax in cat_axes]
        + [t for ax in cat_axes for t in ax.get_xticklabels()],
    )
    _, _, _, lat_titles_top = _fig_bbox(fig, [ax.title for ax in lat_axes])

    y_lat = (cat_bottom + lat_titles_top) / 2
    _labelled_legend(fig, "LAT", y_lat, ncol=2)


def save_compact_figure(fig, output_dir, stem):
    """Save at exactly COMPACT_FIGSIZE (no tight cropping)."""

    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def save_figure(fig, output_dir, stem):

    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

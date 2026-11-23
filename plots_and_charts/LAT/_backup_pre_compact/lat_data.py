import numpy as np


# ============================================================
# LAT (Llama 3) perturbation-radius sweeps
#
# Two inner-LR schedules, both scaling the inner LR linearly
# with epsilon:
#
#   "base"  : inner LR = 1e-3 at epsilon = 6
#             (1e-3, 2e-3, 4e-3, 8e-3)
#   "10pct" : first inner step = 10% of epsilon
#             (0.009375, 0.01875, 0.0375, 0.075)
#
# Index 0 (epsilon = 0) is the unmodified Llama 3 reference,
# shared by both sweeps and without error bars.
#
# All positive-radius values are mean +/- sample SD across
# three training seeds unless stated otherwise.
# ============================================================

SERIES_LABELS = {
    "base": "Original-LR scaling",
    "10pct": "10%-step scaling",
}

COLORS = {
    "base": "C1",
    "10pct": "C2",
}

MARKERS = {
    "base": "s",
    "10pct": "^",
}

EPS = np.array([0.0, 6.0, 12.0, 24.0, 48.0])


# ============================================================
# Reference: HuggingFace LAT Llama 3 (single checkpoint)
# ============================================================

HF_LAT = {
    "Harmless": 97.50,
    "MMLU": 58.23,
    "ARC-E": 73.19,
    "ARC-C": 42.66,
    "GCG": 7.50,
    "AutoDAN": 0.0,
    "Soft-Prompts": 62.58,
    "XSTest-safe": 86.0,
    "XSTest-unsafe": 7.0,
}


# ============================================================
# Sweeps
# ============================================================

models = {

    # ========================================================
    # Inner LR = 1e-3 at epsilon = 6
    # ========================================================

    "base": {
        "eps": EPS,

        "Harmless": {
            "y": np.array([100.0, 91.67, 97.50, 100.00, 100.00]),
            "err": np.array([0.0, 2.89, 0.00, 0.00, 0.00]),
        },

        "MMLU": {
            "y": np.array([60.75, 48.05, 54.72, 59.59, 60.15]),
            "err": np.array([0.0, 2.25, 5.55, 0.53, 0.09]),
        },

        "ARC-E": {
            "y": np.array([79.12, 48.87, 52.16, 57.45, 78.33]),
            "err": np.array([0.0, 0.61, 18.83, 6.01, 1.36]),
        },

        "ARC-C": {
            "y": np.array([49.40, 28.10, 32.88, 37.09, 48.04]),
            "err": np.array([0.0, 0.44, 9.92, 2.18, 0.72]),
        },

        # Original Llama 3 GCG has its own attack-seed SD (5.20);
        # it is kept at 0 here so eps = 0 has no error bar, as in
        # the paper figures.
        "GCG": {
            "y": np.array([28.33, 7.50, 15.83, 21.67, 12.50]),
            "err": np.array([0.0, 6.61, 6.29, 3.82, 5.00]),
        },

        "AutoDAN": {
            "y": np.array([10.0, 0.00, 0.00, 2.50, 1.67]),
            "err": np.array([0.0, 0.00, 0.00, 4.33, 1.44]),
        },

        "Soft-Prompts": {
            "y": np.array([75.41, 38.15, 82.79, 84.76, 76.67]),
            "err": np.array([0.0, 7.79, 1.16, 1.26, 1.51]),
        },

        "XSTest": {
            "safe": {
                "y": np.array([97.20, 96.13, 96.80, 99.20, 99.47]),
                "err": np.array([0.0, 0.46, 0.69, 0.40, 0.23]),
            },
            "unsafe": {
                "y": np.array([91.50, 4.33, 4.50, 3.67, 3.50]),
                "err": np.array([0.0, 0.58, 0.87, 0.29, 0.00]),
            },
        },
    },

    # ========================================================
    # First inner step = 10% of epsilon
    # ========================================================

    "10pct": {
        "eps": EPS,

        "Harmless": {
            "y": np.array([100.0, 100.00, 100.00, 100.00, 100.00]),
            "err": np.array([0.0, 0.00, 0.00, 0.00, 0.00]),
        },

        "MMLU": {
            "y": np.array([60.75, 60.34, 60.21, 59.61, 60.68]),
            "err": np.array([0.0, 0.13, 0.15, 1.60, 0.14]),
        },

        "ARC-E": {
            "y": np.array([79.12, 78.88, 79.10, 75.44, 79.22]),
            "err": np.array([0.0, 0.49, 0.11, 6.21, 0.14]),
        },

        "ARC-C": {
            "y": np.array([49.40, 47.81, 48.78, 47.52, 49.29]),
            "err": np.array([0.0, 0.27, 0.39, 3.63, 0.22]),
        },

        # eps = 48: only two of three seeds available so far.
        "GCG": {
            "y": np.array([28.33, 16.67, 17.50, 15.83, 23.75]),
            "err": np.array([0.0, 1.44, 5.00, 17.01, 1.77]),
        },

        "AutoDAN": {
            "y": np.array([10.0, 5.00, 5.00, 6.67, 6.67]),
            "err": np.array([0.0, 4.33, 0.00, 6.29, 1.44]),
        },

        "Soft-Prompts": {
            "y": np.array([75.41, 79.98, 63.92, 52.17, 50.44]),
            "err": np.array([0.0, 1.68, 8.91, 5.70, 7.60]),
        },

        "XSTest": {
            "safe": {
                "y": np.array([97.20, 99.33, 99.20, 99.07, 99.20]),
                "err": np.array([0.0, 0.23, 0.00, 0.23, 0.00]),
            },
            "unsafe": {
                "y": np.array([91.50, 3.33, 3.33, 2.83, 3.17]),
                "err": np.array([0.0, 0.29, 0.29, 0.76, 0.29]),
            },
        },
    },
}


# ============================================================
# Shared helpers
# ============================================================

def eps_tick_labels(values):
    return [f"{v:g}" for v in values]


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

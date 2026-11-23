import numpy as np


# ============================================================
# OR-Bench replacement (n = 1,000) vs UltraChat-only baseline
#
# Baseline:
#   ordinary UltraChat-only sweep (mean +/- sample SD, three seeds)
#
# Replacement:
#   CAT training with n = 1,000 OR examples (single seed unless an
#   SD is given)
#
# "OR held-out" = success on the OR-Bench prompts not used for
# training ("on 78k"; 80,359 - n prompts).
#
# np.nan means that no SD was reported.
# ============================================================

models = {

    # ========================================================
    # Zephyr
    # ========================================================

    "Zephyr": {

        "epsilon": np.array([0.00884, 0.01768, 0.03535, 0.07070]),

        "epsilon_labels": ["0.00884", "0.01768", "0.03535", "0.07070"],

        "Harmless": {
            "baseline": {
                "y": np.array([98.33, 94.17, 89.17, 70.83]),
                "err": np.array([1.44, 3.82, 1.44, 9.46]),
            },
            "replacement": {
                "y": np.array([100.00, 95.00, 97.50, 75.00]),
                "err": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
        },

        "OR held-out": {
            "baseline": {
                "y": np.array([66.98, 64.62, 55.08, 61.30]),
                "err": np.array([6.04, 1.90, 6.84, 10.13]),
            },
            "replacement": {
                "y": np.array([98.38, 98.47, 98.19, 97.84]),
                "err": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
        },

        "Soft-Prompt": {
            "baseline": {
                "y": np.array([43.44, 43.40, 42.45, 51.13]),
                "err": np.array([4.53, 5.42, 13.65, 3.21]),
            },
            "replacement": {
                "y": np.array([63.71, 60.75, 55.22, 66.16]),
                "err": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
        },
    },

    # ========================================================
    # Llama 3 (Llama 2 recipe)
    #
    # 1,000-OR sweep uses s = 0.075, 0.15, 0.3, 0.6.
    # ========================================================

    "Llama 3 (Llama 2)": {

        "epsilon": np.array([0.04424, 0.08848, 0.17695, 0.35391]),

        "epsilon_labels": ["0.04424", "0.08848", "0.17695", "0.35391"],

        "Harmless": {
            "baseline": {
                "y": np.array([90.83, 45.83, 7.50, 1.67]),
                "err": np.array([7.22, 26.96, 12.99, 2.89]),
            },
            "replacement": {
                "y": np.array([82.50, 55.00, 0.00, 0.00]),
                "err": np.array([np.nan, np.nan, np.nan, 0.00]),
            },
        },

        "OR held-out": {
            "baseline": {
                "y": np.array([60.75, 64.89, 58.68, 35.78]),
                "err": np.array([0.52, 28.82, 14.61, 15.77]),
            },
            "replacement": {
                "y": np.array([97.99, 98.24, 97.02, 82.41]),
                "err": np.array([np.nan, np.nan, np.nan, 18.27]),
            },
        },

        "Soft-Prompt": {
            "baseline": {
                "y": np.array([57.29, 59.25, 59.33, 64.46]),
                "err": np.array([9.37, 16.08, 1.41, 2.30]),
            },
            "replacement": {
                "y": np.array([79.18, 77.55, 82.26, 74.40]),
                "err": np.array([np.nan, np.nan, np.nan, 4.92]),
            },
        },
    },

    # ========================================================
    # Llama 3 (MixAT)
    # ========================================================

    "Llama 3 (MixAT)": {

        "epsilon": np.array([0.04424, 0.08848, 0.17695, 0.35391]),

        "epsilon_labels": ["0.04424", "0.08848", "0.17695", "0.35391"],

        "Harmless": {
            "baseline": {
                "y": np.array([82.50, 58.33, 20.83, 50.00]),
                "err": np.array([13.92, 14.65, 15.07, 41.61]),
            },
            "replacement": {
                "y": np.array([97.50, 45.00, 15.00, 0.00]),
                "err": np.array([np.nan, np.nan, np.nan, 0.00]),
            },
        },

        # eps = 0.35391 replacement: three seeds [51.62, 0.41, 72.28]
        "OR held-out": {
            "baseline": {
                "y": np.array([55.84, 46.20, 56.47, 13.65]),
                "err": np.array([7.53, 6.03, 8.42, 12.61]),
            },
            "replacement": {
                "y": np.array([98.40, 96.50, 93.42, 41.44]),
                "err": np.array([np.nan, np.nan, np.nan, 37.00]),
            },
        },

        "Soft-Prompt": {
            "baseline": {
                "y": np.array([55.47, 43.23, 58.18, 63.21]),
                "err": np.array([4.46, 3.92, 4.86, 3.20]),
            },
            "replacement": {
                "y": np.array([74.72, 71.38, 83.21, 75.77]),
                "err": np.array([np.nan, np.nan, np.nan, 5.12]),
            },
        },
    },
}

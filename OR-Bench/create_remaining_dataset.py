#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

# ===========================
# Paths
# ===========================

ORIGINAL = Path("OR-Bench/datasets/or-bench_datasets/or-bench-80k.csv")
TRAIN = Path("OR-Bench/datasets/or-bench_train_and_eval/or-bench-from80kto1k_train.csv")
EVAL = Path("OR-Bench/datasets/or-bench_train_and_eval/or-bench-from80kto1k_eval.csv")

OUTPUT = Path("OR-Bench/datasets/or-bench_train_and_eval/or-bench-remaining.csv")

# ===========================
# Load
# ===========================

original = pd.read_csv(ORIGINAL)
original = original.drop(columns=["category"])
train = pd.read_csv(TRAIN)
eval = pd.read_csv(EVAL)

# Remove duplicates just in case
to_remove = (
    pd.concat([train, eval], ignore_index=True)
    .drop_duplicates()
)

# ===========================
# Remove train+eval rows
# ===========================

remaining = (
    original.merge(
        to_remove,
        how="left",
        indicator=True,
    )
    .query("_merge == 'left_only'")
    .drop(columns="_merge")
)

# ===========================
# Save
# ===========================

remaining.to_csv(OUTPUT, index=False)

print(f"Original size : {len(original)}")
print(f"Train size    : {len(train)}")
print(f"Eval size     : {len(eval)}")
print(f"Remaining     : {len(remaining)}")
print(f"Removed       : {len(original) - len(remaining)}")
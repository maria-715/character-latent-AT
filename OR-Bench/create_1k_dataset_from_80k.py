#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


CSV_FILE = Path("OR-Bench") / "or-bench-80k.csv"
TRAIN_FILE = Path("OR-Bench") / "or-bench-from80kto1k_train.csv"
EVAL_FILE = Path("OR-Bench") / "or-bench-from80kto1k_eval.csv"

TRAIN_SIZE = 1000
EVAL_SIZE = 200


def allocate_counts(category_counts: pd.Series, target_size: int) -> pd.Series:
    """
    Allocate target_size across categories proportionally using
    floor + largest remainder so the total is exactly target_size.
    """
    if target_size > category_counts.sum():
        raise ValueError(
            f"target_size={target_size} is larger than the available rows ({category_counts.sum()})."
        )

    proportions = category_counts / category_counts.sum()
    exact = proportions * target_size
    base = exact.astype(int)
    remainder = target_size - base.sum()

    frac = (exact - base).sort_values(ascending=False)
    for cat in frac.index[:remainder]:
        base[cat] += 1

    return base


def build_split(df: pd.DataFrame, target_size: int):
    """
    Build one subset by taking the first N rows per category in dataset order.
    """
    category_counts = df["category"].value_counts()
    target_counts = allocate_counts(category_counts, target_size)

    selected_indices = []
    selected_counts = {}

    for category in category_counts.index:
        n = int(target_counts[category])
        rows = df[df["category"] == category].head(n)
        selected_indices.extend(rows.index.tolist())
        selected_counts[category] = len(rows)

    # Keep rows in original dataset order
    split_df = df.loc[sorted(selected_indices)].copy()
    return split_df, selected_counts


def print_counts(title: str, counts: dict):
    print(f"\n{title}")
    for category, count in counts.items():
        print(f"{category}: {count}")
    print(f"Total: {sum(counts.values())}")


def main():
    df = pd.read_csv(CSV_FILE)

    required = {"prompt", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if TRAIN_SIZE + EVAL_SIZE > len(df):
        raise ValueError(
            f"Requested {TRAIN_SIZE + EVAL_SIZE} rows, but dataset only has {len(df)}."
        )

    # Build train first, then eval from the remaining rows so they are disjoint.
    train_df, train_counts = build_split(df, TRAIN_SIZE)
    remaining_df = df.drop(index=train_df.index)
    eval_df, eval_counts = build_split(remaining_df, EVAL_SIZE)

    # Save prompt-only CSVs
    train_df[["prompt"]].to_csv(TRAIN_FILE, index=False)
    eval_df[["prompt"]].to_csv(EVAL_FILE, index=False)

    print(f"Saved train subset to: {TRAIN_FILE}")
    print(f"Saved eval subset to:  {EVAL_FILE}")

    print_counts("\nTrain prompts per category:", train_counts)
    print_counts("\nEval prompts per category:", eval_counts)

    print(f"\nTrain total: {len(train_df)}")
    print(f"Eval total:  {len(eval_df)}")
    print(f"Disjoint check: {set(train_df.index).isdisjoint(set(eval_df.index))}")


if __name__ == "__main__":
    main()
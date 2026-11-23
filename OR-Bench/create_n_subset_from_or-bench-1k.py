#!/usr/bin/env python3
"""
Given n, build category-stratified n-row subsets of the existing 1k
train/eval OR-Bench splits, both as prompt-only files and as
prompt+response files (one per answering model).

Category information was not kept in the 1k train/eval files, so it is
recovered by joining prompts back against the original 80k dataset
(which does have a "category" column).
"""

import argparse
from pathlib import Path

import pandas as pd

FULL_80K_FILE = Path("OR-Bench/datasets/or-bench_datasets") / "or-bench-80k.csv"
TRAIN_EVAL_DIR = Path("OR-Bench/datasets/or-bench_train_and_eval")
WITH_ANSWERS_DIR = Path("OR-Bench/datasets/or-bench_with_answers")

SPLITS = ["train", "eval"]
MODELS = ["llama3_8b_instruct", "zephyr_7b_beta"]


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


def select_n_subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select n rows from df (which must have a "category" column),
    taking the first n rows per category in existing order, with
    per-category counts allocated proportionally to df's own
    category distribution.
    """
    category_counts = df["category"].value_counts()
    target_counts = allocate_counts(category_counts, n)

    selected_indices = []
    for category in category_counts.index:
        count = int(target_counts[category])
        rows = df[df["category"] == category].head(count)
        selected_indices.extend(rows.index.tolist())

    return df.loc[sorted(selected_indices)].copy()


def print_counts(title: str, df: pd.DataFrame):
    print(f"\n{title}")
    counts = df["category"].value_counts()
    for category, count in counts.items():
        print(f"{category}: {count}")
    print(f"Total: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int, help="Number of rows to keep per split")
    args = parser.parse_args()
    n = args.n

    category_lookup = pd.read_csv(FULL_80K_FILE)[["prompt", "category"]]

    for split in SPLITS:
        prompt_file = TRAIN_EVAL_DIR / f"or-bench-from80kto1k_{split}.csv"
        prompts_df = pd.read_csv(prompt_file)

        merged = prompts_df.merge(category_lookup, on="prompt", how="left")
        missing = merged["category"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} prompts in {prompt_file} could not be matched to a category."
            )

        subset = select_n_subset(merged, n)
        print_counts(f"{split} subset categories (n={n}):", subset)

        prompt_only_out = TRAIN_EVAL_DIR / f"or-bench-from80kto1k_{split}_{n}.csv"
        subset[["prompt"]].to_csv(prompt_only_out, index=False)
        print(f"Saved: {prompt_only_out}")

        for model in MODELS:
            answers_file = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}.csv"
            answers_df = pd.read_csv(answers_file)

            merged_answers = subset[["prompt"]].merge(answers_df, on="prompt", how="left")
            missing_answers = merged_answers["response"].isna().sum()
            if missing_answers:
                raise ValueError(
                    f"{missing_answers} prompts from the {split} subset had no response in {answers_file}."
                )

            answers_out = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}_{n}.csv"
            merged_answers[["prompt", "response"]].to_csv(answers_out, index=False)
            print(f"Saved: {answers_out}")


if __name__ == "__main__":
    main()

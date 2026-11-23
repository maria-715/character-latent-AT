#!/usr/bin/env python3
"""
Expand the existing 1000-row train/eval OR-Bench splits to 3000 rows
each by keeping all existing prompts and adding 2000 new ones per
split, drawn from the 80k source so that the final 3000-row split
matches the full 80k category distribution.

Train is expanded first, then eval, so the two splits' added prompts
never overlap (and never overlap the other split's existing prompts).

New prompts have no model response yet, so the response column is
left blank for them in the prompt+response outputs.
"""

from pathlib import Path

import pandas as pd

FULL_80K_FILE = Path("OR-Bench/datasets/or-bench_datasets") / "or-bench-80k.csv"
TRAIN_EVAL_DIR = Path("OR-Bench/datasets/or-bench_train_and_eval")
WITH_ANSWERS_DIR = Path("OR-Bench/datasets/or-bench_with_answers")

SPLITS = ["train", "eval"]
MODELS = ["llama3_8b_instruct", "zephyr_7b_beta"]

ADD_SIZE = 2000
TARGET_SIZE = 3000


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


def print_counts(title: str, df: pd.DataFrame):
    print(f"\n{title}")
    counts = df["category"].value_counts()
    for category, count in counts.items():
        print(f"{category}: {count}")
    print(f"Total: {len(df)}")


def main():
    full_df = pd.read_csv(FULL_80K_FILE)
    full_category_counts = full_df["category"].value_counts()
    target_counts_3000 = allocate_counts(full_category_counts, TARGET_SIZE)

    existing_by_split = {}
    used_prompts = set()
    for split in SPLITS:
        prompts_df = pd.read_csv(TRAIN_EVAL_DIR / f"or-bench-from80kto1k_{split}.csv")
        existing_df = prompts_df.merge(full_df, on="prompt", how="left")
        missing = existing_df["category"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} prompts in the current {split} split could not be matched to a category."
            )
        existing_by_split[split] = existing_df
        used_prompts.update(existing_df["prompt"])

    if len(used_prompts) != sum(len(df) for df in existing_by_split.values()):
        raise ValueError("Train and eval prompts are not disjoint; refusing to proceed.")

    for split in SPLITS:
        existing_df = existing_by_split[split]
        existing_counts = existing_df["category"].value_counts()

        new_rows = []
        for category in full_category_counts.index:
            target = int(target_counts_3000.get(category, 0))
            have = int(existing_counts.get(category, 0))
            additional = target - have
            if additional < 0:
                raise ValueError(
                    f"{split}/{category}: already has {have} rows, more than the "
                    f"target of {target} for {TARGET_SIZE} total; cannot proceed."
                )

            pool = full_df[
                (full_df["category"] == category) & (~full_df["prompt"].isin(used_prompts))
            ]
            chosen = pool.head(additional)
            if len(chosen) < additional:
                raise ValueError(
                    f"{split}/{category}: needed {additional} new rows but only "
                    f"{len(chosen)} remain unused in the 80k source."
                )

            new_rows.append(chosen)
            used_prompts.update(chosen["prompt"])

        new_df = pd.concat(new_rows, ignore_index=False)
        full_split_df = pd.concat([existing_df, new_df], ignore_index=False).sort_index()

        print_counts(f"{split} full distribution (n={TARGET_SIZE}):", full_split_df)

        prompt_only_out = TRAIN_EVAL_DIR / f"or-bench-from80kto1k_{split}_{TARGET_SIZE}.csv"
        full_split_df[["prompt"]].to_csv(prompt_only_out, index=False)
        print(f"Saved: {prompt_only_out}")

        new_prompt_set = set(new_df["prompt"])
        for model in MODELS:
            answers_file = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}.csv"
            answers_df = pd.read_csv(answers_file)

            merged_answers = full_split_df[["prompt"]].merge(answers_df, on="prompt", how="left")
            merged_answers.loc[merged_answers["prompt"].isin(new_prompt_set), "response"] = ""

            missing_existing = merged_answers.loc[
                ~merged_answers["prompt"].isin(new_prompt_set), "response"
            ].isna().sum()
            if missing_existing:
                raise ValueError(
                    f"{missing_existing} existing {split} prompts had no response in {answers_file}."
                )

            answers_out = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}_{TARGET_SIZE}.csv"
            merged_answers[["prompt", "response"]].to_csv(answers_out, index=False)
            print(f"Saved: {answers_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract the still-blank-response prompts from the _3000 answer files
into small prompts-only CSVs (one per split, shared across models,
since both models are missing the same newly-added 2000 prompts).

Feed the resulting file into generate_answers_to_or-bench.sh as
--input to generate just the missing answers, then use
fill_missing_responses.py to merge them back in.
"""

from pathlib import Path

import pandas as pd

TRAIN_EVAL_DIR = Path("OR-Bench/datasets/or-bench_train_and_eval")
WITH_ANSWERS_DIR = Path("OR-Bench/datasets/or-bench_with_answers")

SPLITS = ["train", "eval"]
MODELS = ["llama3_8b_instruct", "zephyr_7b_beta"]
TARGET_SIZE = 3000


def main():
    for split in SPLITS:
        missing_sets = {}
        for model in MODELS:
            answers_file = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}_{TARGET_SIZE}.csv"
            answers_df = pd.read_csv(answers_file)
            is_blank = answers_df["response"].isna() | (answers_df["response"] == "")
            missing_sets[model] = set(answers_df.loc[is_blank, "prompt"])

        sets = list(missing_sets.values())
        if any(s != sets[0] for s in sets[1:]):
            details = ", ".join(f"{m}={len(s)}" for m, s in missing_sets.items())
            raise ValueError(
                f"{split}: missing-prompt sets differ across models ({details}); "
                "expected them to be identical. Resolve before extracting."
            )

        missing_prompts = sets[0]
        print(f"{split}: {len(missing_prompts)} prompts with blank responses")

        # Preserve dataset order using the canonical _3000 prompt-only file.
        ordered_prompts = pd.read_csv(TRAIN_EVAL_DIR / f"or-bench-from80kto1k_{split}_{TARGET_SIZE}.csv")
        missing_df = ordered_prompts[ordered_prompts["prompt"].isin(missing_prompts)]

        out_file = TRAIN_EVAL_DIR / f"or-bench-from80kto1k_{split}_{TARGET_SIZE}_missing.csv"
        missing_df[["prompt"]].to_csv(out_file, index=False)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

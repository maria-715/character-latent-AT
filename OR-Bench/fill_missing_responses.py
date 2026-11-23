#!/usr/bin/env python3
"""
Fill blank responses in the _3000 answer files using freshly generated
answers for just the missing prompts (produced by running
generate_answers_to_or-bench.sh on the *_missing.csv files from
extract_missing_prompts.py).

Existing non-blank responses are left untouched, row order is
preserved, and each base file is overwritten in place. A model/split
is skipped (with a message) if its patch file doesn't exist yet, so
this can be re-run as generation jobs finish one at a time.
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
        for model in MODELS:
            base_file = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}_{TARGET_SIZE}.csv"
            model_patch_file = WITH_ANSWERS_DIR / f"utility_{model}_from80kto1k_{split}_{TARGET_SIZE}_missing.csv"

            if not model_patch_file.exists():
                print(f"Skipping {model}/{split}: patch file not found ({model_patch_file})")
                continue

            base_df = pd.read_csv(base_file)
            patch_df = pd.read_csv(model_patch_file)

            is_blank = base_df["response"].isna() | (base_df["response"] == "")
            blank_prompts = set(base_df.loc[is_blank, "prompt"])

            patch_prompts = set(patch_df["prompt"])
            missing_from_patch = blank_prompts - patch_prompts
            if missing_from_patch:
                raise ValueError(
                    f"{model}/{split}: {len(missing_from_patch)} blank prompts have no "
                    f"answer in {model_patch_file}."
                )

            patch_lookup = patch_df.drop_duplicates(subset="prompt", keep="last").set_index("prompt")["response"]
            base_df.loc[is_blank, "response"] = base_df.loc[is_blank, "prompt"].map(patch_lookup)

            still_blank = base_df["response"].isna() | (base_df["response"] == "")
            if still_blank.any():
                raise ValueError(
                    f"{model}/{split}: {still_blank.sum()} responses still blank after merge."
                )

            base_df.to_csv(base_file, index=False)
            print(f"Filled {is_blank.sum()} responses in {base_file}")


if __name__ == "__main__":
    main()

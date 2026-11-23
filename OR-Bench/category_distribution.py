#!/usr/bin/env python3

import pandas as pd


CSV_FILE = r"OR-Bench\or-bench-80k.csv"


def main():
    # Read the CSV
    df = pd.read_csv(CSV_FILE)

    # Check required columns
    required = {"prompt", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Count prompts per category
    counts = (
        df["category"]
        .value_counts()
        .sort_index()
        .rename_axis("category")
        .reset_index(name="num_prompts")
    )

    print("\nPrompt counts by category:\n")
    print(counts.to_string(index=False))

    print(f"\nTotal prompts: {len(df)}")
    print(f"Total categories: {counts.shape[0]}")


if __name__ == "__main__":
    main()
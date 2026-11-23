#!/usr/bin/env python3

import pandas as pd

CSV1 = "OR-Bench\datasets\or-bench_train_and_eval\or-bench-from80kto1k_train.csv"
CSV2 = "OR-Bench\or-bench-from80kto1k_eval.csv"
COMPARE_COLUMNS = None  # e.g. ["prompt"] or None for all columns


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    return df.reindex(sorted(df.columns), axis=1)


def df_to_row_tuples(df: pd.DataFrame) -> set[tuple]:
    df = df.copy().fillna("<NA>")
    return set(map(tuple, df.itertuples(index=False, name=None)))


def main() -> None:
    df1 = pd.read_csv(CSV1)
    df2 = pd.read_csv(CSV2)

    df1 = normalize_df(df1)
    df2 = normalize_df(df2)

    if COMPARE_COLUMNS is not None:
        missing1 = [c for c in COMPARE_COLUMNS if c not in df1.columns]
        missing2 = [c for c in COMPARE_COLUMNS if c not in df2.columns]
        if missing1 or missing2:
            print("ERROR: missing columns in one or both files.")
            if missing1:
                print(f"Missing in {CSV1}: {missing1}")
            if missing2:
                print(f"Missing in {CSV2}: {missing2}")
            return

        df1 = df1[COMPARE_COLUMNS]
        df2 = df2[COMPARE_COLUMNS]

    rows1 = df_to_row_tuples(df1)
    rows2 = df_to_row_tuples(df2)

    common = rows1 & rows2

    print(f"{CSV1}: {len(rows1)} unique rows")
    print(f"{CSV2}: {len(rows2)} unique rows")
    print(f"Common rows: {len(common)}")

    if common:
        print("\nExample common rows:")
        for i, row in enumerate(list(common)[:10], start=1):
            print(f"{i}: {row}")
        print("\nResult: NOT mutually exclusive")
    else:
        print("\nResult: mutually exclusive")


if __name__ == "__main__":
    main()
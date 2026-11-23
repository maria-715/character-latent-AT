#!/usr/bin/env python3
"""Analyse CAT data files and mirror the training preprocessing as closely as possible.

This script focuses on:
- raw adversarial behavior files
- JSON target lists
- safe-response files
- Hugging Face utility datasets (ultrachat / ultrachat_200k)
- the interleaved training view used by CAT

Run examples:
    python analyse_cat_data.py --data-path ./data --utility ultrachat_200k
    python analyse_cat_data.py --data-path ../Reachable-Continuous-AdvTrain/data --utility ultrachat_200k --save-json summary.json

Notes:
- The script uses Hugging Face datasets for utility data, exactly like the training code.
- The default mix is the one you provided: 12.5% adversarial and 87.5% utility.
- The final number of rows after interleaving is determined by the datasets and the stopping strategy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk


DEFAULT_PROBABILITIES = [0.125, 0.875]  # [adv, utility]
DEFAULT_STOPPING_STRATEGY = "first_exhausted"


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def load_json_targets(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(obj)}")

    normalized: Dict[str, List[str]] = {}
    for behavior_id, value in obj.items():
        if isinstance(value, list):
            normalized[str(behavior_id)] = [str(x) for x in value]
        else:
            normalized[str(behavior_id)] = [str(value)]
    return normalized


def behavior_target_paths(data_path: Path) -> Dict[str, Path]:
    return {
        "adv_training_behaviors": data_path / "behavior_datasets" / "extra_behavior_datasets" / "adv_training_behaviors.csv",
        "adv_training_targets": data_path / "optimizer_targets" / "extra_targets" / "adv_training_targets.json",
        "adv_val_behaviors": data_path / "behavior_datasets" / "harmbench_behaviors_text_val.csv",
        "adv_val_targets": data_path / "optimizer_targets" / "extra_targets" / "adv_training_val_targets.json",
        "adv_test_behaviors": data_path / "behavior_datasets" / "harmbench_behaviors_text_all.csv",
        "adv_test_targets": data_path / "optimizer_targets" / "harmbench_targets_text.json",
        "safe_responses": data_path / "safe_responses" / "adv_training_behaviors_safe_responses.csv",
        "safe_responses_longer": data_path / "safe_responses" / "adv_training_behaviors_safe_responses_longer.csv",
    }


def build_expanded_behavior_target_df(behavior_csv: Path, target_json: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw_behavior_df, merged_df, per_behavior_stats_df)."""
    raw_df = safe_read_csv(behavior_csv)
    if "BehaviorID" not in raw_df.columns or "Behavior" not in raw_df.columns:
        raise ValueError(f"{behavior_csv} does not look like a CAT/HarmBench behavior file")

    targets = load_json_targets(target_json)
    target_rows = [
        {"BehaviorID": behavior_id, "Target": target}
        for behavior_id, target_list in targets.items()
        for target in target_list
    ]
    target_df = pd.DataFrame(target_rows)
    merged_df = raw_df.merge(target_df, on="BehaviorID", how="inner")

    stats_rows = []
    for behavior_id, group in merged_df.groupby("BehaviorID", sort=False):
        row = {
            "BehaviorID": behavior_id,
            "Behavior": group["Behavior"].iloc[0],
            "TargetCount": len(group),
        }
        if "Category" in group.columns:
            row["Category"] = group["Category"].iloc[0]
        if "Tags" in group.columns:
            row["Tags"] = group["Tags"].iloc[0]
        if "ContextString" in group.columns:
            row["ContextString"] = group["ContextString"].iloc[0]
        stats_rows.append(row)

    per_behavior_df = pd.DataFrame(stats_rows).sort_values(["TargetCount", "BehaviorID"], ascending=[False, True])
    return raw_df, merged_df, per_behavior_df


def attach_safe_responses(expanded_df: pd.DataFrame, safe_csv: Path) -> pd.DataFrame:
    if not safe_csv.exists():
        return expanded_df.copy()

    safe_df = safe_read_csv(safe_csv)
    if "Behavior" not in safe_df.columns:
        return expanded_df.copy()

    merged = expanded_df.merge(safe_df, on="Behavior", how="left", suffixes=("", "_safefile"))
    return merged


def print_dataset_summary(name: str, raw_df: pd.DataFrame, merged_df: pd.DataFrame, per_behavior_df: pd.DataFrame) -> None:
    print_section(f"{name}")
    print(f"Raw CSV rows: {len(raw_df)}")
    print(f"Unique BehaviorIDs in CSV: {raw_df['BehaviorID'].nunique()}")
    print(f"BehaviorIDs matched in JSON: {merged_df['BehaviorID'].nunique()}")
    print(f"Expanded rows (one per target): {len(merged_df)}")
    print(f"Average targets per matched behavior: {per_behavior_df['TargetCount'].mean():.3f}")
    print(f"Min targets per behavior: {per_behavior_df['TargetCount'].min()}")
    print(f"Max targets per behavior: {per_behavior_df['TargetCount'].max()}")

    missing_ids = sorted(set(raw_df["BehaviorID"]) - set(merged_df["BehaviorID"]))
    extra_ids = sorted(set(merged_df["BehaviorID"]) - set(raw_df["BehaviorID"]))
    print(f"Behaviors in CSV without targets: {len(missing_ids)}")
    if missing_ids:
        print("  first few:", ", ".join(missing_ids[:10]))
    print(f"Target IDs not found in CSV: {len(extra_ids)}")
    if extra_ids:
        print("  first few:", ", ".join(extra_ids[:10]))

    if "Category" in raw_df.columns:
        print("\nRaw CSV rows by category:")
        print(raw_df["Category"].fillna("UNKNOWN").value_counts().to_string())
        print("\nExpanded rows by category:")
        print(merged_df["Category"].fillna("UNKNOWN").value_counts().to_string())

    print("\nTarget-count distribution per behavior:")
    print(per_behavior_df["TargetCount"].describe().to_string())


def show_examples(df: pd.DataFrame, title: str, columns: Sequence[str], n: int = 5) -> None:
    print_section(title)
    existing = [c for c in columns if c in df.columns]
    if not existing:
        print("No requested columns are present.")
        return
    print(df[existing].head(n).to_string(index=False))


def print_category_examples(raw_df: pd.DataFrame, merged_df: pd.DataFrame, per_behavior_df: pd.DataFrame, n_categories: int = 5, n_examples: int = 2) -> None:
    if "Category" not in raw_df.columns:
        return

    print_section("Example behaviors by category")
    categories = raw_df["Category"].fillna("UNKNOWN").value_counts().head(n_categories).index.tolist()
    for category in categories:
        print(f"\nCategory: {category}")
        subset = raw_df[raw_df["Category"].fillna("UNKNOWN") == category].drop_duplicates(subset=["BehaviorID"]).head(n_examples)
        for _, row in subset.iterrows():
            bid = row["BehaviorID"]
            target_count = int(per_behavior_df.loc[per_behavior_df["BehaviorID"] == bid, "TargetCount"].iloc[0])
            print("-" * 88)
            print(f"BehaviorID: {bid}")
            print(f"Target count: {target_count}")
            print(f"Behavior: {row['Behavior']}")
            targets = merged_df.loc[merged_df["BehaviorID"] == bid, "Target"].tolist()
            print("Targets (first few):")
            for t in targets[:3]:
                print(f"  - {t}")


def text_length_stats(label: str, series: pd.Series) -> None:
    text = series.fillna("").astype(str)
    char_lengths = text.map(len)
    word_lengths = text.map(lambda s: len(s.split()))
    print_section(f"{label} length statistics")
    print("Characters:")
    print(char_lengths.describe().to_string())
    print("\nWords:")
    print(word_lengths.describe().to_string())


def inspect_safe_response_file(safe_csv: Path, n: int = 5) -> None:
    print_section("Safe-response file preview")
    if not safe_csv.exists():
        print(f"File not found: {safe_csv}")
        return
    df = safe_read_csv(safe_csv)
    print(f"Rows: {len(df)}")
    cols = [c for c in ["BehaviorID", "Behavior", "Target", "Safe_Response"] if c in df.columns]
    if cols:
        print(df[cols].head(n).to_string(index=False))
    else:
        print(df.head(n).to_string(index=False))


def cache_dir_has_contents(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def load_utility_dataset(data_path: Path, utility_name: str) -> Tuple[Any, Any]:
    """Load utility data exactly like training code does."""
    utility_root = data_path / "utility"
    utility_root.mkdir(parents=True, exist_ok=True)

    if utility_name == "ultrachat":
        cache_path = utility_root / "ultrachat"
        if cache_dir_has_contents(cache_path):
            dataset = load_from_disk(str(cache_path))
        else:
            dataset = load_dataset("stingning/ultrachat", split="train")
            dataset.save_to_disk(str(cache_path))

        dataset = dataset.map(
            lambda example: {
                "User": example["data"][::2],
                "Model": [ans for ans in example["data"][1::2]],
            }
        )
        frac = int(0.9 * len(dataset))
        return dataset.select(range(frac)), dataset.select(range(frac, len(dataset)))

    if utility_name == "ultrachat_200k":
        train_cache = utility_root / "ultrachat_200k_train"
        eval_cache = utility_root / "ultrachat_200k_eval"

        if cache_dir_has_contents(train_cache):
            train_ds = load_from_disk(str(train_cache))
        else:
            train_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
            train_ds.save_to_disk(str(train_cache))

        if cache_dir_has_contents(eval_cache):
            eval_ds = load_from_disk(str(eval_cache))
        else:
            eval_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
            eval_ds.save_to_disk(str(eval_cache))

        train_ds = train_ds.map(
            lambda example: {
                "User": [msg["content"] for msg in example["messages"][::2]],
                "Model": [msg["content"] for msg in example["messages"][1::2]],
            }
        )
        eval_ds = eval_ds.map(
            lambda example: {
                "User": [msg["content"] for msg in example["messages"][::2]],
                "Model": [msg["content"] for msg in example["messages"][1::2]],
            }
        )

        # Same filter as training code: keep only prompts shorter than ~768 chars.
        train_ds = train_ds.filter(lambda example: len(example["User"][0]) < 768)
        eval_ds = eval_ds.filter(lambda example: len(example["User"][0]) < 768)
        return train_ds, eval_ds

    raise ValueError(f"Unsupported utility dataset: {utility_name}")


def preview_utility_examples(ds: Any, split_name: str, n: int = 3) -> None:
    print_section(f"Utility examples: {split_name}")
    print(f"Rows: {len(ds)}")
    for i in range(min(n, len(ds))):
        ex = ds[i]
        print("\n" + "-" * 88)
        print(f"Example {i}")
        print(f"Keys: {list(ex.keys())}")
        user = ex.get("User")
        model = ex.get("Model")
        if isinstance(user, list) and isinstance(model, list):
            print(f"Turns: {min(len(user), len(model))}")
            for j, (u, m) in enumerate(zip(user, model), start=1):
                print(f"Turn {j}")
                print(f"USER: {u}")
                print(f"ASSISTANT: {m}")
                print()
        else:
            print("USER:")
            print(user)
            print("ASSISTANT:")
            print(model)


def build_training_view(adv_train_ds: Dataset, utility_train_ds: Optional[Dataset], probabilities: Sequence[float], stopping_strategy: str) -> Optional[Dataset]:
    if utility_train_ds is None:
        return adv_train_ds
    return interleave_datasets(
        [adv_train_ds, utility_train_ds],
        probabilities=list(probabilities),
        stopping_strategy=stopping_strategy,
    )


def make_adv_training_dataset(raw_df: pd.DataFrame, merged_df: pd.DataFrame, safe_df: Optional[pd.DataFrame]) -> Dataset:
    """Create the adversarial training dataset in the same shape the trainer uses before tokenization."""
    df = merged_df.copy()
    if safe_df is not None and not safe_df.empty and "Behavior" in safe_df.columns:
        safe_cols = [c for c in ["Behavior", "Safe_Response"] if c in safe_df.columns]
        if len(safe_cols) >= 2:
            df = df.merge(safe_df[safe_cols], on="Behavior", how="left")

    if "Safe_Response" not in df.columns:
        df["Safe_Response"] = None

    # Match the structure used in training: one row per target, each row holds a single prompt/target pair.
    df = df.rename(columns={"Behavior": "User", "Target": "Model", "Safe_Response": "Safe_Model"})
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Expand to one row per target exactly as the training code does.
    dataset = dataset.map(
        lambda batch: {
            "dataset_id": [0 for _ in range(len(batch["Model"][0]))],
            "User": [[batch["User"][0]] for _ in range(len(batch["Model"][0]))],
            "Model": [[target] for target in batch["Model"][0]],
            "Safe_Model": [[batch["Safe_Model"][0]] for _ in range(len(batch["Model"][0]))],
        },
        batched=True,
        batch_size=1,
    )
    return dataset


def make_utility_dataset_for_training_view(utility_ds: Any) -> Dataset:
    # Add the same columns the training code expects.
    utility_ds = utility_ds.map(lambda example: {"dataset_id": 2, "Safe_Model": None, **example}, num_proc=16)
    return utility_ds


def compute_interleaving_report(adv_train_ds: Dataset, utility_train_ds: Optional[Dataset], probabilities: Sequence[float], stopping_strategy: str) -> None:
    print_section("Interleaving / training mix")
    print(f"Adversarial rows before interleaving: {len(adv_train_ds)}")
    if utility_train_ds is None:
        print("Utility rows before interleaving: none")
        print("Final training rows: same as adversarial rows")
        return

    print(f"Utility rows before interleaving: {len(utility_train_ds)}")
    mixed = interleave_datasets(
        [adv_train_ds, utility_train_ds],
        probabilities=list(probabilities),
        stopping_strategy=stopping_strategy,
    )
    print(f"Probabilities: adv={probabilities[0]}, utility={probabilities[1]}")
    print(f"Stopping strategy: {stopping_strategy}")
    print(f"Final mixed training rows: {len(mixed)}")

    # Peek at the dataset-id mix in the first few rows.
    counts = {0: 0, 1: 0, 2: 0}
    for i in range(min(1000, len(mixed))):
        ex = mixed[i]
        did = ex.get("dataset_id", None)
        if did is None:
            continue
        try:
            counts[int(did)] += 1
        except Exception:
            pass
    print("Approximate dataset_id counts in first 1000 mixed rows:")
    print(counts)


def save_summary_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse CAT data files and utility datasets.")
    parser.add_argument("--data-path", type=str, default="../Reachable-Continuous-AdvTrain/data/", help="Path to the CAT data directory")
    parser.add_argument("--utility", type=str, default="ultrachat_200k", choices=["None", "none", "ultrachat", "ultrachat_200k"], help="Utility dataset to inspect")
    parser.add_argument("--prob-adv", type=float, default=DEFAULT_PROBABILITIES[0], help="Interleaving probability for adversarial data")
    parser.add_argument("--prob-utility", type=float, default=DEFAULT_PROBABILITIES[1], help="Interleaving probability for utility data")
    parser.add_argument("--stopping-strategy", type=str, default=DEFAULT_STOPPING_STRATEGY, help="Hugging Face interleave_datasets stopping strategy")
    parser.add_argument("--max-behavior-examples", type=int, default=5, help="How many raw behavior examples to print")
    parser.add_argument("--max-category-examples", type=int, default=2, help="How many examples per category to print")
    parser.add_argument("--max-utility-examples", type=int, default=3, help="How many utility examples to print per split")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save a machine-readable summary")
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    paths = behavior_target_paths(data_path)

    print_section("Resolved inputs")
    print(f"Data path: {data_path}")
    print(f"Utility dataset: {args.utility}")
    print(f"Probabilities: [adv={args.prob_adv}, utility={args.prob_utility}]")
    print(f"Stopping strategy: {args.stopping_strategy}")
    for name, path in paths.items():
        print(f"{name}: {path}")

    # --- Adversarial training data ---
    raw_train_df, merged_train_df, per_behavior_train_df = build_expanded_behavior_target_df(
        paths["adv_training_behaviors"], paths["adv_training_targets"]
    )
    print_dataset_summary("Adversarial training set", raw_train_df, merged_train_df, per_behavior_train_df)
    text_length_stats("Adversarial training behaviors", raw_train_df["Behavior"])
    show_examples(raw_train_df, "Raw adversarial training preview", ["BehaviorID", "Category", "Behavior", "Tags", "ContextString"], n=args.max_behavior_examples)
    print_category_examples(raw_train_df, merged_train_df, per_behavior_train_df, n_categories=5, n_examples=args.max_category_examples)

    safe_train_df = safe_read_csv(paths["safe_responses"]) if paths["safe_responses"].exists() else None
    inspect_safe_response_file(paths["safe_responses"], n=args.max_behavior_examples)
    if paths["safe_responses_longer"].exists():
        print_section("Longer safe-response file preview")
        long_safe_df = safe_read_csv(paths["safe_responses_longer"])
        print(f"Rows: {len(long_safe_df)}")
        cols = [c for c in ["BehaviorID", "Behavior", "Target", "Safe_Response"] if c in long_safe_df.columns]
        if cols:
            print(long_safe_df[cols].head(args.max_behavior_examples).to_string(index=False))
        else:
            print(long_safe_df.head(args.max_behavior_examples).to_string(index=False))

    adv_train_dataset = make_adv_training_dataset(raw_train_df, merged_train_df, safe_train_df)
    print_section("Expanded adversarial training rows")
    print(f"Rows after target expansion: {len(adv_train_dataset)}")
    print(f"Columns: {adv_train_dataset.column_names}")
    for i in range(min(args.max_behavior_examples, len(adv_train_dataset))):
        ex = adv_train_dataset[i]
        print("\n" + "-" * 88)
        print(f"Example {i}")
        print(f"dataset_id: {ex.get('dataset_id')}")
        print(f"User: {ex.get('User')}")
        print(f"Model: {ex.get('Model')}")
        print(f"Safe_Model: {ex.get('Safe_Model')}")

    # --- Validation and test adversarial data ---
    raw_val_df, merged_val_df, per_behavior_val_df = build_expanded_behavior_target_df(
        paths["adv_val_behaviors"], paths["adv_val_targets"]
    )
    print_dataset_summary("Adversarial validation set", raw_val_df, merged_val_df, per_behavior_val_df)
    raw_test_df, merged_test_df, per_behavior_test_df = build_expanded_behavior_target_df(
        paths["adv_test_behaviors"], paths["adv_test_targets"]
    )
    print_dataset_summary("Adversarial test set", raw_test_df, merged_test_df, per_behavior_test_df)

    # --- Utility data (Hugging Face) ---
    utility_train_ds = None
    utility_eval_ds = None
    if args.utility not in {"None", "none"}:
        utility_train_ds, utility_eval_ds = load_utility_dataset(data_path, args.utility)
        print_section(f"Utility dataset: {args.utility}")
        print(f"Train rows after preprocessing/filtering: {len(utility_train_ds)}")
        print(f"Eval rows after preprocessing/filtering: {len(utility_eval_ds)}")
        text_length_stats(f"{args.utility} train first-user-message", pd.Series([utility_train_ds[i]["User"][0] for i in range(min(5000, len(utility_train_ds)))]))
        preview_utility_examples(utility_train_ds, "train", n=args.max_utility_examples)
        preview_utility_examples(utility_eval_ds, "eval", n=args.max_utility_examples)

    # --- Interleaving report ---
    if utility_train_ds is not None:
        utility_train_for_mix = make_utility_dataset_for_training_view(utility_train_ds)
        # Make sure only fields compatible with adv_train_dataset remain, matching training code.
        utility_train_for_mix = utility_train_for_mix.remove_columns(
            [c for c in utility_train_for_mix.column_names if c not in adv_train_dataset.column_names and c not in {"dataset_id", "Safe_Model"}]
        )
        compute_interleaving_report(
            adv_train_dataset,
            utility_train_for_mix,
            probabilities=[args.prob_adv, args.prob_utility],
            stopping_strategy=args.stopping_strategy,
        )

    # --- Optional machine-readable summary ---
    summary = {
        "data_path": str(data_path),
        "utility": args.utility,
        "probabilities": [args.prob_adv, args.prob_utility],
        "stopping_strategy": args.stopping_strategy,
        "adversarial_training": {
            "raw_csv_rows": int(len(raw_train_df)),
            "matched_behavior_ids": int(merged_train_df["BehaviorID"].nunique()),
            "expanded_rows": int(len(merged_train_df)),
            "target_count_mean": float(per_behavior_train_df["TargetCount"].mean()),
            "target_count_min": int(per_behavior_train_df["TargetCount"].min()),
            "target_count_max": int(per_behavior_train_df["TargetCount"].max()),
        },
        "adversarial_validation": {
            "raw_csv_rows": int(len(raw_val_df)),
            "matched_behavior_ids": int(merged_val_df["BehaviorID"].nunique()),
            "expanded_rows": int(len(merged_val_df)),
        },
        "adversarial_test": {
            "raw_csv_rows": int(len(raw_test_df)),
            "matched_behavior_ids": int(merged_test_df["BehaviorID"].nunique()),
            "expanded_rows": int(len(merged_test_df)),
        },
    }

    if utility_train_ds is not None:
        summary["utility"] = {
            "train_rows_after_preprocessing": int(len(utility_train_ds)),
            "eval_rows_after_preprocessing": int(len(utility_eval_ds)),
        }

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_summary_json(save_path, summary)
        print_section("Saved summary")
        print(f"Wrote: {save_path}")

    print_section("Done")
    print("This script mirrors the data preprocessing used by CAT and shows both raw and expanded counts.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse CAT data files and the final training mixture.

This script is organized in three layers:

1. Raw source files
   - adversarial behavior CSVs
   - target JSON files
   - safe-response CSVs

2. Utility data from Hugging Face
   - ultrachat
   - ultrachat_200k

3. Final training view
   - uses load_adversarial_training_data()
   - this is the closest view to what the trainer actually sees

Typical usage:
    python analyse_dataset.py --data-path ../Reachable-Continuous-AdvTrain/data --utility ultrachat_200k

Optional final training-view inspection:
    python analyse_dataset.py --data-path ../Reachable-Continuous-AdvTrain/data --utility ultrachat_200k \
        --model-name llama-2 --tokenizer-name meta-llama/Llama-2-7b-chat-hf

Notes:
- The utility dataset is loaded from Hugging Face exactly like the training code.
- The default interleaving probabilities are the ones you provided:
    adv = 0.125
    utility = 0.875
- The training-view section uses CAT's own load_adversarial_training_data() helper.
- If tokenizer loading fails, the script still runs the raw-file and utility analyses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None

try:
    import data as cat_data
except Exception:  # pragma: no cover
    cat_data = None


DEFAULT_DATA_PATH = "../Reachable-Continuous-AdvTrain/data/"
DEFAULT_UTILITY = "ultrachat_200k"
DEFAULT_PROB_ADV = 0.125
DEFAULT_PROB_UTILITY = 0.875
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


def paths_for(data_path: Path) -> Dict[str, Path]:
    return {
        "adv_train_csv": data_path / "behavior_datasets" / "extra_behavior_datasets" / "adv_training_behaviors.csv",
        "adv_train_targets": data_path / "optimizer_targets" / "extra_targets" / "adv_training_targets.json",
        "adv_val_csv": data_path / "behavior_datasets" / "harmbench_behaviors_text_val.csv",
        "adv_val_targets": data_path / "optimizer_targets" / "extra_targets" / "adv_training_val_targets.json",
        "adv_test_csv": data_path / "behavior_datasets" / "harmbench_behaviors_text_all.csv",
        "adv_test_targets": data_path / "optimizer_targets" / "harmbench_targets_text.json",
        "safe_csv": data_path / "safe_responses" / "adv_training_behaviors_safe_responses.csv",
        "safe_long_csv": data_path / "safe_responses" / "adv_training_behaviors_safe_responses_longer.csv",
    }


def build_behavior_target_tables(behavior_csv: Path, target_json: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw_behavior_df, merged_df, per_behavior_stats_df)."""
    raw_df = safe_read_csv(behavior_csv)
    if "BehaviorID" not in raw_df.columns or "Behavior" not in raw_df.columns:
        raise ValueError(f"{behavior_csv} does not look like a CAT/HarmBench behavior CSV")

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
        row: Dict[str, Any] = {
            "BehaviorID": behavior_id,
            "Behavior": group["Behavior"].iloc[0],
            "TargetCount": len(group),
        }
        for col in ["Category", "Tags", "ContextString"]:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        stats_rows.append(row)

    per_behavior_df = pd.DataFrame(stats_rows).sort_values(["TargetCount", "BehaviorID"], ascending=[False, True])
    return raw_df, merged_df, per_behavior_df


def print_behavior_summary(name: str, raw_df: pd.DataFrame, merged_df: pd.DataFrame, per_behavior_df: pd.DataFrame) -> None:
    print_section(name)
    print(f"Raw CSV rows: {len(raw_df)}")
    print(f"Unique BehaviorIDs in CSV: {raw_df['BehaviorID'].nunique()}")
    print(f"Matched BehaviorIDs after merge: {merged_df['BehaviorID'].nunique()}")
    print(f"Expanded rows (one per target): {len(merged_df)}")

    print(f"Average targets per matched behavior: {per_behavior_df['TargetCount'].mean():.3f}")
    print(f"Min targets per behavior: {int(per_behavior_df['TargetCount'].min())}")
    print(f"Max targets per behavior: {int(per_behavior_df['TargetCount'].max())}")

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
    else:
        print("\nNo Category column present in the raw CSV.")

    print("\nTarget-count distribution per behavior:")
    print(per_behavior_df["TargetCount"].describe().to_string())


def print_examples(df: pd.DataFrame, title: str, columns: Sequence[str], n: int = 5) -> None:
    print_section(title)
    cols = [c for c in columns if c in df.columns]
    if not cols:
        print("No requested columns are present.")
        return
    print(df[cols].head(n).to_string(index=False))


def print_category_examples(raw_df: pd.DataFrame, merged_df: pd.DataFrame, per_behavior_df: pd.DataFrame, n_categories: int = 5, n_examples: int = 2) -> None:
    if "Category" not in raw_df.columns:
        print_section("Example behaviors by category")
        print("No Category column present.")
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
    s = series.fillna("").astype(str)
    char_lengths = s.map(len)
    word_lengths = s.map(lambda x: len(x.split()))
    print_section(f"{label} length statistics")
    print("Characters:")
    print(char_lengths.describe().to_string())
    print("\nWords:")
    print(word_lengths.describe().to_string())


def inspect_safe_response_file(path: Path, n: int = 5) -> pd.DataFrame:
    print_section("Safe-response file preview")
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()

    df = safe_read_csv(path)
    print(f"Rows: {len(df)}")
    cols = [c for c in ["BehaviorID", "Behavior", "Target", "Safe_Response"] if c in df.columns]
    if cols:
        print(df[cols].head(n).to_string(index=False))
    else:
        print(df.head(n).to_string(index=False))
    return df


def cache_dir_has_contents(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def load_utility_dataset(data_path: Path, utility_name: str) -> Tuple[Any, Any]:
    """Load utility data exactly like CAT training code does."""
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

        print("train_ds.column_names:")
        print(train_ds.column_names)
        print("train_ds[0]:")
        print(train_ds[0])
        print("train_ds.features:")
        print(train_ds.features)

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

        # Same filter as the CAT code.
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


def load_tokenizer(model_path: str, fallback_name: Optional[str]) -> Any:
    if not model_path:
        if fallback_name is None or not fallback_name:
            raise ValueError("Provide --model-path for tokenizer loading")
        model_path = fallback_name

    if AutoTokenizer is None:
        raise RuntimeError("transformers is not available in this environment")

    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def analyse_final_training_view(
    data_path: Path,
    utility_data: str,
    probabilities: Sequence[float],
    model_path: str,
    model_name: str,
    tokenizer_name: Optional[str],
    stopping_strategy: str,
    diverse_safe_answers: bool,
    restricted_trainingset_size: Optional[int],
    or_replace_n: int,
    max_examples: int,
) -> None:
    if cat_data is None:
        print_section("Final training view")
        print("Could not import CAT's data.py, so load_adversarial_training_data() is unavailable.")
        return

    try:
        tokenizer = load_tokenizer(model_path, model_name)
    except Exception as e:
        print_section("Final training view")
        print(f"Skipping load_adversarial_training_data() because tokenizer loading failed: {e}")
        return

    print_section("Final training view (CAT preprocessing)")
    print("Calling load_adversarial_training_data() now...")

    train_data, eval_data = cat_data.load_adversarial_training_data(  # type: ignore[attr-defined]
        data_path=str(data_path) + "/",
        utility_data=utility_data,
        probabilities=list(probabilities),
        model_name=model_name,
        tokenizer=tokenizer,
        stopping_strategy=stopping_strategy,
        diverse_safe_answers=diverse_safe_answers,
        restricted_trainingset_size=restricted_trainingset_size,
        or_replace_n=or_replace_n,
    )

    print(f"Final train rows: {len(train_data)}")
    print(f"Final eval rows: {len(eval_data)}")
    print(f"Train columns: {train_data.column_names}")
    print(f"Eval columns: {eval_data.column_names}")

    # dataset_id counts in the final train view
    counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    for i in range(len(train_data)):
        ex = train_data[i]
        did = ex.get("dataset_id", None)
        try:
            if did is not None:
                counts[int(did)] += 1
        except Exception:
            pass
    print("\nDataset_id counts in final train data:")
    print(counts)

    print("\nFirst few training examples:")
    for i in range(min(max_examples, len(train_data))):
        ex = train_data[i]
        print("\n" + "-" * 88)
        print(f"Example {i}")
        print(f"dataset_id: {ex.get('dataset_id')}")
        print(f"User: {ex.get('User')}")
        print(f"Model: {ex.get('Model')}")
        if "Safe_Model" in ex:
            print(f"Safe_Model: {ex.get('Safe_Model')}")

    print("\nFirst few eval examples:")
    for i in range(min(max_examples, len(eval_data))):
        ex = eval_data[i]
        print("\n" + "-" * 88)
        print(f"Example {i}")
        print(f"dataset_id: {ex.get('dataset_id')}")
        print(f"User: {ex.get('User')}")
        print(f"Model: {ex.get('Model')}")
        if "Safe_Model" in ex:
            print(f"Safe_Model: {ex.get('Safe_Model')}")

    # Interleaving report based on the actual returned datasets.
    print("\nFinal train set includes the CAT-built mix of adversarial, safe, and utility examples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse CAT data and final training mixture.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Path to the CAT data directory")
    parser.add_argument("--utility", type=str, default=DEFAULT_UTILITY, choices=["None", "none", "ultrachat", "ultrachat_200k"], help="Utility dataset to inspect")
    parser.add_argument("--prob-adv", type=float, default=DEFAULT_PROB_ADV, help="Adversarial interleaving probability")
    parser.add_argument("--prob-utility", type=float, default=DEFAULT_PROB_UTILITY, help="Utility interleaving probability")
    parser.add_argument("--stopping-strategy", type=str, default=DEFAULT_STOPPING_STRATEGY, help="Hugging Face interleave_datasets stopping strategy")
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="CAT chat-template name, used by model_utils.get_chat_template()",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Hugging Face model ID or local path used to load the tokenizer/model",
    )
    parser.add_argument("--tokenizer-name", type=str, default="", help="Tokenizer name/path to load for the final training view. Defaults to --model-name if omitted.")
    parser.add_argument("--diverse-safe-answers", action="store_true", help="Use the longer safe-response file in the final training view")
    parser.add_argument("--restricted-trainingset-size", type=int, default=None, help="Optional cap on the final adversarial training set")
    parser.add_argument("--or-replace-n", type=int, default=1000, help="Number of utility examples replaced by OR-Bench data in the final training view")
    parser.add_argument("--max-behavior-examples", type=int, default=5, help="How many raw behavior examples to print")
    parser.add_argument("--max-category-examples", type=int, default=2, help="How many examples per category to print")
    parser.add_argument("--max-utility-examples", type=int, default=3, help="How many utility examples to print per split")
    parser.add_argument("--max-final-examples", type=int, default=5, help="How many final training-view examples to print")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save a machine-readable summary")
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    p = paths_for(data_path)

    print_section("Resolved inputs")
    print(f"Data path: {data_path}")
    print(f"Utility dataset: {args.utility}")
    print(f"Probabilities: [adv={args.prob_adv}, utility={args.prob_utility}]")
    print(f"Stopping strategy: {args.stopping_strategy}")
    print(f"Model name: {args.model_name or '(not set)'}")
    print(f"Tokenizer name: {args.tokenizer_name or '(default to model name or skipped)'}")
    print(f"Diverse safe answers: {args.diverse_safe_answers}")
    print(f"Restricted training set size: {args.restricted_trainingset_size}")
    for name, path in p.items():
        print(f"{name}: {path}")

    # ------------------------------------------------------------
    # Raw adversarial source analysis
    # ------------------------------------------------------------
    raw_train_df, merged_train_df, per_behavior_train_df = build_behavior_target_tables(p["adv_train_csv"], p["adv_train_targets"])
    print_behavior_summary("Adversarial training set", raw_train_df, merged_train_df, per_behavior_train_df)
    text_length_stats("Adversarial training behaviors", raw_train_df["Behavior"])
    print_examples(raw_train_df, "Raw adversarial training preview", ["BehaviorID", "Category", "Behavior", "Tags", "ContextString"], n=args.max_behavior_examples)
    print_category_examples(raw_train_df, merged_train_df, per_behavior_train_df, n_categories=5, n_examples=args.max_category_examples)

    safe_df = inspect_safe_response_file(p["safe_csv"], n=args.max_behavior_examples)
    if p["safe_long_csv"].exists():
        print_section("Longer safe-response file preview")
        long_safe_df = safe_read_csv(p["safe_long_csv"])
        print(f"Rows: {len(long_safe_df)}")
        cols = [c for c in ["BehaviorID", "Behavior", "Target", "Safe_Response"] if c in long_safe_df.columns]
        if cols:
            print(long_safe_df[cols].head(args.max_behavior_examples).to_string(index=False))
        else:
            print(long_safe_df.head(args.max_behavior_examples).to_string(index=False))

    # Validation/test raw source analysis
    raw_val_df, merged_val_df, per_behavior_val_df = build_behavior_target_tables(p["adv_val_csv"], p["adv_val_targets"])
    print_behavior_summary("Adversarial validation set", raw_val_df, merged_val_df, per_behavior_val_df)
    raw_test_df, merged_test_df, per_behavior_test_df = build_behavior_target_tables(p["adv_test_csv"], p["adv_test_targets"])
    print_behavior_summary("Adversarial test set", raw_test_df, merged_test_df, per_behavior_test_df)

    # ------------------------------------------------------------
    # Utility data from Hugging Face (same preprocessing as CAT)
    # ------------------------------------------------------------
    utility_train_ds = None
    utility_eval_ds = None
    if args.utility not in {"None", "none"}:
        utility_train_ds, utility_eval_ds = load_utility_dataset(data_path, args.utility)
        print_section(f"Utility dataset: {args.utility}")
        print(f"Train rows after preprocessing/filtering: {len(utility_train_ds)}")
        print(f"Eval rows after preprocessing/filtering: {len(utility_eval_ds)}")

        # Show prompt lengths for the first user message, which is the important part for CAT.
        if len(utility_train_ds) > 0:
            first_user_messages = [utility_train_ds[i]["User"][0] for i in range(min(len(utility_train_ds), 5000))]
            text_length_stats(f"{args.utility} train first-user-message", pd.Series(first_user_messages))

        preview_utility_examples(utility_train_ds, "train", n=args.max_utility_examples)
        preview_utility_examples(utility_eval_ds, "eval", n=args.max_utility_examples)

    # ------------------------------------------------------------
    # Final training view using CAT's own helper
    # ------------------------------------------------------------
    if args.model_name:
        analyse_final_training_view(
            data_path=data_path,
            utility_data=args.utility,
            probabilities=[args.prob_adv, args.prob_utility],
            model_path=args.model_path,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name or None,
            stopping_strategy=args.stopping_strategy,
            diverse_safe_answers=args.diverse_safe_answers,
            restricted_trainingset_size=args.restricted_trainingset_size,
            or_replace_n=args.or_replace_n,
            max_examples=args.max_final_examples,
        )
    else:
        print_section("Final training view")
        print("Skipping load_adversarial_training_data() because --model-name was not provided.")
        print("Provide --model-name (and optionally --tokenizer-name) to inspect the exact training mixture.")

    # ------------------------------------------------------------
    # Optional summary JSON
    # ------------------------------------------------------------
    summary: Dict[str, Any] = {
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print_section("Saved summary")
        print(f"Wrote: {save_path}")

    print_section("Done")
    print("This script separates raw-file analysis from the final CAT training-view analysis.")


if __name__ == "__main__":
    main()

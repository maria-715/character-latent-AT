#!/usr/bin/env python3

"""
Compact adversarial embedding-space analysis.

This script reads existing attack trace JSONL files only.
It does NOT rerun training.
It does NOT recompute attacks.

It produces compact aggregate summaries for:

1. Chosen perturbation = perturbation at best-loss inner step.
2. Final perturbation = perturbation at last inner step.
3. Best inner-step distribution.
4. Distances to original embedding.
5. Distances to nearest embedding.
6. Distances to top-k nearest neighbours.
7. Compact token-type and position summaries.
8. Optional aggregation across multiple epsilon/seed/model runs.

Expected trace files inside each attack_traces directory:
- step_summaries.jsonl
- token_records.jsonl

Optional:
- final_token_records.jsonl
- final_summaries.jsonl
- attack_metadata.jsonl

Single-run usage:

python analyse_attack_traces.py ^
  --trace_dir experiments\\CAT_adv_eps005_42_relativelr\\attack_traces ^
  --out_dir experiments\\CAT_adv_eps005_42_relativelr\\compact_space_analysis ^
  --epsilon 0.05 ^
  --seed 42 ^
  --run_id CAT_adv_eps005_42_relativelr

Multi-run usage:

python analyse_attack_traces.py ^
  --manifest manifest.csv ^
  --out_dir experiments\\all_space_analysis

python Continuous-AdvTrain/analyse_attack_traces.py ^
  --manifest Continuous-AdvTrain/to_run_global.csv ^
  --out_dir experiments\\all_space_analysis_global

Manifest format:

run_id,epsilon,seed,trace_dir
CAT_adv_eps005_42_relativelr,0.05,42,experiments/CAT_adv_eps005_42_relativelr/attack_traces
CAT_adv_eps005_43_relativelr,0.05,43,experiments/CAT_adv_eps005_43_relativelr/attack_traces
CAT_adv_eps010_42_relativelr,0.10,42,experiments/CAT_adv_eps010_42_relativelr/attack_traces
"""

import argparse
import ast
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Basic IO
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        print(f"[WARN] Skipping empty CSV: {path}")
        return
    df.to_csv(path, index=False)
    print(f"[OK] Saved {path}")


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] Saved {path}")


def resolve_trace_file(trace_dir: Path, filename: str, required: bool = True) -> Path | None:
    """
    Robustly find a trace file.

    First checks trace_dir / filename.
    Then searches recursively under trace_dir.

    This avoids confusing Windows path issues and nested folders.
    """

    direct = trace_dir / filename
    if direct.exists():
        return direct

    matches = list(trace_dir.rglob(filename))
    if matches:
        print(f"[INFO] Found {filename} at {matches[0]}")
        return matches[0]

    if required:
        jsonl_files = sorted(trace_dir.rglob("*.jsonl"))
        print(f"[ERROR] Could not find required file {filename} under {trace_dir}")
        print("[ERROR] JSONL files found under trace_dir:")
        for p in jsonl_files[:100]:
            print(f"  - {p}")
        raise FileNotFoundError(f"Could not find {filename} under {trace_dir}")

    return None


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    if path is None or not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse {path}, line {line_idx}: {e}")

    return rows


def read_jsonl_df(path: Path | None) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ============================================================
# Type helpers
# ============================================================

def parse_list(x):
    """
    Handles lists stored either as actual Python lists or stringified lists.
    """

    if isinstance(x, list):
        return x

    if x is None:
        return []

    if isinstance(x, float) and np.isnan(x):
        return []

    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return []

    return []


def make_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_bool(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df


def mean_safe(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def median_safe(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.median())


def q_safe(x, q):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(q))


def min_safe(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.min())


def max_safe(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.max())


def frac_false(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float((~x.astype(bool)).mean())


# ============================================================
# Loading traces
# ============================================================

def keep_last_run_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    step_summaries.jsonl (and token_records.jsonl) are written in append mode,
    so a job that crashed and was resubmitted into the same output directory
    leaves earlier, aborted attempts' rows sitting before the rows of the run
    that actually finished. Every restart trains from scratch (load_checkpoint
    is always false in this codebase), so trainer_global_step resets to 0 at
    the start of each attempt. A drop in trainer_global_step therefore marks
    the boundary of a new attempt. Keep only the rows from the last one.
    """

    if df.empty or "trainer_global_step" not in df.columns:
        return df

    gstep = df["trainer_global_step"].to_numpy()
    restart_starts = np.nonzero(np.r_[False, gstep[1:] < gstep[:-1]])[0]

    if len(restart_starts) == 0:
        return df

    last_start = restart_starts[-1]
    print(
        f"[INFO] Detected {len(restart_starts)} restart(s) in step_summaries.jsonl; "
        f"dropping {last_start} row(s) from earlier aborted attempt(s), "
        f"keeping the final {len(df) - last_start} row(s)."
    )
    return df.iloc[last_start:].reset_index(drop=True)


def load_step_summaries(trace_dir: Path) -> pd.DataFrame:
    path = resolve_trace_file(trace_dir, "step_summaries.jsonl", required=True)
    df = read_jsonl_df(path)

    numeric_cols = [
        "trainer_global_step",
        "attack_call_idx",
        "step",
        "loss",
        "affirmative_responses",
        "num_perturbed_tokens",
        "mean_l2_to_original",
        "median_l2_to_original",
        "p90_l2_to_original",
        "p99_l2_to_original",
        "min_l2_to_original",
        "max_l2_to_original",
        "mean_dist_to_nearest",
        "median_dist_to_nearest",
        "p90_dist_to_nearest",
        "p99_dist_to_nearest",
        "min_dist_to_nearest",
        "max_dist_to_nearest",
        "mean_original_rank",
        "median_original_rank",
        "max_original_rank",
        "frac_original_is_nearest",
        "frac_original_in_topk",
        "frac_nearest_changed",
        "mean_eps_fraction",
        "max_eps_fraction",
    ]

    df = make_numeric(df, numeric_cols)

    if df.empty:
        raise ValueError(f"{path} exists but contains no rows")

    required = {"attack_id", "step", "loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"step_summaries.jsonl is missing columns: {missing}")

    df = keep_last_run_segment(df)

    return df


def load_selected_token_records(trace_dir: Path, best_steps: pd.DataFrame) -> pd.DataFrame:
    """
    Load only token rows from:
    - chosen step = best_loss_step
    - final step = final_step

    This avoids materialising unnecessary analysis outputs.
    """

    token_path = resolve_trace_file(trace_dir, "token_records.jsonl", required=True)

    best_step_map = dict(zip(best_steps["attack_id"], best_steps["best_loss_step"]))
    final_step_map = dict(zip(best_steps["attack_id"], best_steps["final_step"]))

    selected = []

    with token_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse token_records line {line_idx}: {e}")
                continue

            attack_id = row.get("attack_id")
            step = row.get("step")

            if attack_id not in best_step_map:
                continue

            is_chosen = step == best_step_map[attack_id]
            is_final = step == final_step_map[attack_id]

            if is_chosen:
                row["chosen_or_final"] = "chosen"
                selected.append(row)

            if is_final:
                # If chosen == final, this deliberately duplicates the row:
                # once as chosen and once as final.
                final_row = dict(row)
                final_row["chosen_or_final"] = "final"
                selected.append(final_row)

    df = pd.DataFrame(selected)

    if df.empty:
        raise ValueError(
            "No selected token records found. "
            "Check that token_records.jsonl has attack_id and step fields matching step_summaries.jsonl."
        )

    numeric_cols = [
        "trainer_global_step",
        "attack_call_idx",
        "step",
        "loss",
        "affirmative_responses",
        "batch_index",
        "token_position",
        "original_token_id",
        "l2_to_original",
        "nearest_token_id",
        "dist_to_nearest",
        "original_rank",
        "original_dist_from_vocab_matrix",
    ]

    df = make_numeric(df, numeric_cols)
    df = make_bool(df, ["nearest_is_original", "original_in_topk"])

    return df


# ============================================================
# Best-step logic
# ============================================================

def build_best_steps(df_steps: pd.DataFrame) -> pd.DataFrame:
    """
    One row per attack_id:
    - best_loss_step: step with minimum attack loss
    - final_step: maximum inner step
    - first/final/best loss
    """

    df_steps = df_steps.sort_values(["attack_id", "step"]).copy()

    # idx_best = df_steps.groupby("attack_id")["loss"].idxmin()
    # Keep only rows with a valid loss for best-step selection.
    valid_loss = df_steps["loss"].notna()

    bad = df_steps.groupby("attack_id")["loss"].apply(lambda s: s.notna().sum() == 0)
    if bad.any():
        print("[WARN] attack_ids with no valid loss values:", bad[bad].index.tolist())

    idx_best = (
        df_steps.loc[valid_loss]
        .groupby("attack_id")["loss"]
        .idxmin()
    )
    
    best = df_steps.loc[idx_best].copy()

    best = best.rename(
        columns={
            "step": "best_loss_step",
            "loss": "best_loss",
        }
    )

    keep = [
        "attack_id",
        "trainer_global_step",
        "attack_call_idx",
        "best_loss_step",
        "best_loss",
    ]
    keep = [c for c in keep if c in best.columns]
    best = best[keep]

    first_loss = (
        df_steps.groupby("attack_id")
        .first(numeric_only=False)["loss"]
        .rename("first_loss")
        .reset_index()
    )

    final_loss = (
        df_steps.groupby("attack_id")
        .last(numeric_only=False)["loss"]
        .rename("final_loss")
        .reset_index()
    )

    final_step = (
        df_steps.groupby("attack_id")["step"]
        .max()
        .rename("final_step")
        .reset_index()
    )

    num_inner_steps = (
        df_steps.groupby("attack_id")["step"]
        .nunique()
        .rename("num_inner_steps")
        .reset_index()
    )

    out = best.merge(first_loss, on="attack_id", how="left")
    out = out.merge(final_loss, on="attack_id", how="left")
    out = out.merge(final_step, on="attack_id", how="left")
    out = out.merge(num_inner_steps, on="attack_id", how="left")

    out["loss_drop_first_to_best"] = out["first_loss"] - out["best_loss"]
    out["loss_drop_first_to_final"] = out["first_loss"] - out["final_loss"]

    return out


# ============================================================
# Token aggregation
# ============================================================

def aggregate_topk_distances(sub: pd.DataFrame, label: str, row: dict, top_k: int):
    """
    Adds aggregate top-k distance columns to row.

    Example:
    chosen_top1_distance_mean
    chosen_top2_distance_mean
    ...
    """

    topk_by_rank = defaultdict(list)

    for _, tok_row in sub.iterrows():
        distances = parse_list(tok_row.get("topk_distances", []))

        for rank, dist in enumerate(distances, start=1):
            if rank > top_k:
                break
            try:
                topk_by_rank[rank].append(float(dist))
            except Exception:
                continue

    for rank in range(1, top_k + 1):
        values = topk_by_rank.get(rank, [])

        row[f"{label}_top{rank}_distance_mean"] = mean_safe(values)
        row[f"{label}_top{rank}_distance_median"] = median_safe(values)
        row[f"{label}_top{rank}_distance_p90"] = q_safe(values, 0.90)
        row[f"{label}_top{rank}_distance_min"] = min_safe(values)
        row[f"{label}_top{rank}_distance_max"] = max_safe(values)


def aggregate_one_label(sub: pd.DataFrame, label: str, row: dict, top_k: int):
    """
    Aggregate token rows for chosen or final perturbation for one attack.
    """

    row[f"{label}_num_tokens"] = int(len(sub))

    row[f"{label}_mean_l2_to_original"] = mean_safe(sub["l2_to_original"])
    row[f"{label}_median_l2_to_original"] = median_safe(sub["l2_to_original"])
    row[f"{label}_p90_l2_to_original"] = q_safe(sub["l2_to_original"], 0.90)
    row[f"{label}_p99_l2_to_original"] = q_safe(sub["l2_to_original"], 0.99)
    row[f"{label}_min_l2_to_original"] = min_safe(sub["l2_to_original"])
    row[f"{label}_max_l2_to_original"] = max_safe(sub["l2_to_original"])

    row[f"{label}_mean_dist_to_nearest"] = mean_safe(sub["dist_to_nearest"])
    row[f"{label}_median_dist_to_nearest"] = median_safe(sub["dist_to_nearest"])
    row[f"{label}_p90_dist_to_nearest"] = q_safe(sub["dist_to_nearest"], 0.90)
    row[f"{label}_p99_dist_to_nearest"] = q_safe(sub["dist_to_nearest"], 0.99)
    row[f"{label}_min_dist_to_nearest"] = min_safe(sub["dist_to_nearest"])
    row[f"{label}_max_dist_to_nearest"] = max_safe(sub["dist_to_nearest"])

    if "nearest_is_original" in sub.columns:
        row[f"{label}_frac_nearest_changed"] = frac_false(sub["nearest_is_original"])
        row[f"{label}_frac_original_is_nearest"] = mean_safe(sub["nearest_is_original"].astype(float))

    if "original_in_topk" in sub.columns:
        row[f"{label}_frac_original_not_topk"] = frac_false(sub["original_in_topk"])
        row[f"{label}_frac_original_in_topk"] = mean_safe(sub["original_in_topk"].astype(float))

    if "original_rank" in sub.columns:
        row[f"{label}_mean_original_rank"] = mean_safe(sub["original_rank"])
        row[f"{label}_median_original_rank"] = median_safe(sub["original_rank"])
        row[f"{label}_p90_original_rank"] = q_safe(sub["original_rank"], 0.90)
        row[f"{label}_max_original_rank"] = max_safe(sub["original_rank"])

    aggregate_topk_distances(sub, label, row, top_k)


def build_attack_level_chosen_vs_final(
    df_selected_tokens: pd.DataFrame,
    best_steps: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Produces one compact row per attack call.
    """

    best_map = {
        row["attack_id"]: row.to_dict()
        for _, row in best_steps.iterrows()
    }

    rows = []

    for attack_id, df_attack in df_selected_tokens.groupby("attack_id"):
        base = best_map.get(attack_id, {})

        row = {
            "attack_id": attack_id,
            "trainer_global_step": base.get("trainer_global_step"),
            "attack_call_idx": base.get("attack_call_idx"),
            "num_inner_steps": base.get("num_inner_steps"),
            "best_loss_step": base.get("best_loss_step"),
            "final_step": base.get("final_step"),
            "first_loss": base.get("first_loss"),
            "best_loss": base.get("best_loss"),
            "final_loss": base.get("final_loss"),
            "loss_drop_first_to_best": base.get("loss_drop_first_to_best"),
            "loss_drop_first_to_final": base.get("loss_drop_first_to_final"),
        }

        chosen = df_attack[df_attack["chosen_or_final"] == "chosen"].copy()
        final = df_attack[df_attack["chosen_or_final"] == "final"].copy()

        if not chosen.empty:
            aggregate_one_label(chosen, "chosen", row, top_k)

        if not final.empty:
            aggregate_one_label(final, "final", row, top_k)

        if (
            "chosen_mean_l2_to_original" in row
            and "final_mean_l2_to_original" in row
        ):
            row["delta_final_minus_chosen_mean_l2_to_original"] = (
                row["final_mean_l2_to_original"] - row["chosen_mean_l2_to_original"]
            )

        if (
            "chosen_mean_dist_to_nearest" in row
            and "final_mean_dist_to_nearest" in row
        ):
            row["delta_final_minus_chosen_mean_dist_to_nearest"] = (
                row["final_mean_dist_to_nearest"] - row["chosen_mean_dist_to_nearest"]
            )

        if (
            "chosen_mean_original_rank" in row
            and "final_mean_original_rank" in row
        ):
            row["delta_final_minus_chosen_mean_original_rank"] = (
                row["final_mean_original_rank"] - row["chosen_mean_original_rank"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Compact summaries
# ============================================================

def build_best_step_distribution(df_attack: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_attack["best_loss_step"]
        .value_counts()
        .sort_index()
        .rename_axis("best_loss_step")
        .reset_index(name="count")
    )

    total = out["count"].sum()
    out["fraction"] = out["count"] / total if total else np.nan

    return out


def build_run_summary(df_attack: pd.DataFrame, run_id: str, epsilon: float, seed: int | None) -> pd.DataFrame:
    """
    One row summary for one run.
    """

    row = {
        "run_id": run_id,
        "epsilon": epsilon,
        "seed": seed,
        "num_attacks": df_attack["attack_id"].nunique(),
        "mean_best_loss_step": mean_safe(df_attack["best_loss_step"]),
        "median_best_loss_step": median_safe(df_attack["best_loss_step"]),
        "frac_best_step_0": float((df_attack["best_loss_step"] == 0).mean()),
        "frac_best_step_last": float((df_attack["best_loss_step"] == df_attack["final_step"]).mean()),
        "mean_first_loss": mean_safe(df_attack["first_loss"]),
        "mean_best_loss": mean_safe(df_attack["best_loss"]),
        "mean_final_loss": mean_safe(df_attack["final_loss"]),
        "mean_loss_drop_first_to_best": mean_safe(df_attack["loss_drop_first_to_best"]),
        "mean_loss_drop_first_to_final": mean_safe(df_attack["loss_drop_first_to_final"]),
    }

    metric_cols = [
        "chosen_mean_l2_to_original",
        "chosen_median_l2_to_original",
        "chosen_p90_l2_to_original",
        "chosen_max_l2_to_original",
        "chosen_mean_dist_to_nearest",
        "chosen_median_dist_to_nearest",
        "chosen_p90_dist_to_nearest",
        "chosen_min_dist_to_nearest",
        "chosen_frac_nearest_changed",
        "chosen_frac_original_in_topk",
        "chosen_frac_original_not_topk",
        "chosen_mean_original_rank",
        "chosen_median_original_rank",
        "chosen_p90_original_rank",
        "chosen_max_original_rank",

        "final_mean_l2_to_original",
        "final_median_l2_to_original",
        "final_p90_l2_to_original",
        "final_max_l2_to_original",
        "final_mean_dist_to_nearest",
        "final_median_dist_to_nearest",
        "final_p90_dist_to_nearest",
        "final_min_dist_to_nearest",
        "final_frac_nearest_changed",
        "final_frac_original_in_topk",
        "final_frac_original_not_topk",
        "final_mean_original_rank",
        "final_median_original_rank",
        "final_p90_original_rank",
        "final_max_original_rank",

        "delta_final_minus_chosen_mean_l2_to_original",
        "delta_final_minus_chosen_mean_dist_to_nearest",
        "delta_final_minus_chosen_mean_original_rank",
    ]

    for col in metric_cols:
        if col in df_attack.columns:
            row[col] = mean_safe(df_attack[col])

    # Include top-k mean distance columns.
    for label in ["chosen", "final"]:
        for rank in range(1, 6):
            for stat in ["mean", "median", "p90", "min", "max"]:
                col = f"{label}_top{rank}_distance_{stat}"
                if col in df_attack.columns:
                    row[col] = mean_safe(df_attack[col])

    return pd.DataFrame([row])


def build_topk_distance_by_rank(
    df_attack: pd.DataFrame,
    run_id: str,
    epsilon: float,
    seed: int | None,
    top_k: int,
) -> pd.DataFrame:
    rows = []

    for label in ["chosen", "final"]:
        for rank in range(1, top_k + 1):
            row = {
                "run_id": run_id,
                "epsilon": epsilon,
                "seed": seed,
                "chosen_or_final": label,
                "topk_rank": rank,
            }

            for stat in ["mean", "median", "p90", "min", "max"]:
                col = f"{label}_top{rank}_distance_{stat}"
                row[f"attack_avg_{stat}_distance"] = mean_safe(df_attack[col]) if col in df_attack.columns else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def build_token_type_summary(
    df_selected_tokens: pd.DataFrame,
    run_id: str,
    epsilon: float,
    seed: int | None,
    top_n: int,
) -> pd.DataFrame:
    """
    Compact token-level summary.

    Uses chosen rows only, because chosen is what enters your conceptual analysis.
    """

    df = df_selected_tokens[df_selected_tokens["chosen_or_final"] == "chosen"].copy()

    if df.empty or "original_token" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(["original_token_id", "original_token"]).agg(
        count=("original_token", "size"),
        attacks_seen=("attack_id", "nunique"),
        mean_l2_to_original=("l2_to_original", "mean"),
        median_l2_to_original=("l2_to_original", "median"),
        p90_l2_to_original=("l2_to_original", lambda x: x.quantile(0.90)),
        max_l2_to_original=("l2_to_original", "max"),
        mean_dist_to_nearest=("dist_to_nearest", "mean"),
        median_dist_to_nearest=("dist_to_nearest", "median"),
        p90_dist_to_nearest=("dist_to_nearest", lambda x: x.quantile(0.90)),
        min_dist_to_nearest=("dist_to_nearest", "min"),
        frac_nearest_changed=("nearest_is_original", lambda x: float((~x.astype(bool)).mean())),
        frac_original_not_topk=("original_in_topk", lambda x: float((~x.astype(bool)).mean())),
        mean_original_rank=("original_rank", "mean"),
        max_original_rank=("original_rank", "max"),
    ).reset_index()

    grouped["run_id"] = run_id
    grouped["epsilon"] = epsilon
    grouped["seed"] = seed

    grouped = grouped.sort_values(
        [
            "frac_nearest_changed",
            "frac_original_not_topk",
            "mean_original_rank",
            "p90_l2_to_original",
            "count",
        ],
        ascending=[False, False, False, False, False],
    )

    return grouped.head(top_n)


def build_position_summary(
    df_selected_tokens: pd.DataFrame,
    run_id: str,
    epsilon: float,
    seed: int | None,
    top_n: int,
) -> pd.DataFrame:
    df = df_selected_tokens[df_selected_tokens["chosen_or_final"] == "chosen"].copy()

    if df.empty or "token_position" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("token_position").agg(
        count=("token_position", "size"),
        attacks_seen=("attack_id", "nunique"),
        mean_l2_to_original=("l2_to_original", "mean"),
        median_l2_to_original=("l2_to_original", "median"),
        p90_l2_to_original=("l2_to_original", lambda x: x.quantile(0.90)),
        max_l2_to_original=("l2_to_original", "max"),
        mean_dist_to_nearest=("dist_to_nearest", "mean"),
        median_dist_to_nearest=("dist_to_nearest", "median"),
        min_dist_to_nearest=("dist_to_nearest", "min"),
        frac_nearest_changed=("nearest_is_original", lambda x: float((~x.astype(bool)).mean())),
        frac_original_not_topk=("original_in_topk", lambda x: float((~x.astype(bool)).mean())),
        mean_original_rank=("original_rank", "mean"),
        max_original_rank=("original_rank", "max"),
    ).reset_index()

    grouped["run_id"] = run_id
    grouped["epsilon"] = epsilon
    grouped["seed"] = seed

    grouped = grouped.sort_values(
        [
            "frac_nearest_changed",
            "frac_original_not_topk",
            "mean_original_rank",
            "p90_l2_to_original",
            "count",
        ],
        ascending=[False, False, False, False, False],
    )

    return grouped.head(top_n)


# ============================================================
# Plots
# ============================================================

def plot_best_step_distribution(best_dist: pd.DataFrame, plots_dir: Path):
    if best_dist.empty:
        return

    plt.figure()
    plt.bar(best_dist["best_loss_step"], best_dist["fraction"])
    plt.xlabel("Chosen inner step, i.e. best-loss step")
    plt.ylabel("Fraction of attacks")
    plt.title("Distribution of chosen inner step")
    save_plot(plots_dir / "best_step_distribution.png")


def plot_chosen_vs_final_attack_level(df_attack: pd.DataFrame, plots_dir: Path):
    if df_attack.empty:
        return

    if {"chosen_mean_l2_to_original", "final_mean_l2_to_original"}.issubset(df_attack.columns):
        plt.figure()
        plt.scatter(
            df_attack["chosen_mean_l2_to_original"],
            df_attack["final_mean_l2_to_original"],
            alpha=0.4,
            s=12,
        )
        plt.xlabel("Chosen mean L2 to original")
        plt.ylabel("Final mean L2 to original")
        plt.title("Chosen vs final movement from original")
        save_plot(plots_dir / "chosen_vs_final_l2_to_original.png")

    if {"chosen_mean_dist_to_nearest", "final_mean_dist_to_nearest"}.issubset(df_attack.columns):
        plt.figure()
        plt.scatter(
            df_attack["chosen_mean_dist_to_nearest"],
            df_attack["final_mean_dist_to_nearest"],
            alpha=0.4,
            s=12,
        )
        plt.xlabel("Chosen mean distance to nearest")
        plt.ylabel("Final mean distance to nearest")
        plt.title("Chosen vs final distance to nearest embedding")
        save_plot(plots_dir / "chosen_vs_final_dist_to_nearest.png")

    if {"chosen_mean_l2_to_original", "chosen_mean_dist_to_nearest"}.issubset(df_attack.columns):
        plt.figure()
        plt.scatter(
            df_attack["chosen_mean_l2_to_original"],
            df_attack["chosen_mean_dist_to_nearest"],
            alpha=0.4,
            s=12,
        )
        plt.xlabel("Chosen mean L2 to original")
        plt.ylabel("Chosen mean distance to nearest")
        plt.title("Chosen movement vs distance to nearest embedding")
        save_plot(plots_dir / "chosen_l2_vs_dist_to_nearest.png")


def plot_topk_distance(topk: pd.DataFrame, plots_dir: Path):
    if topk.empty:
        return

    plt.figure()

    for label, sub in topk.groupby("chosen_or_final"):
        sub = sub.sort_values("topk_rank")
        plt.plot(
            sub["topk_rank"],
            sub["attack_avg_mean_distance"],
            marker="o",
            label=label,
        )

    plt.xlabel("Nearest-neighbour rank")
    plt.ylabel("Mean distance")
    plt.title("Top-k nearest-neighbour distances")
    plt.legend()
    save_plot(plots_dir / "topk_distance_by_rank.png")


def plot_multi_run_epsilon_summary(all_run_summary: pd.DataFrame, plots_dir: Path):
    if all_run_summary.empty or "epsilon" not in all_run_summary.columns:
        return

    metrics = [
        ("mean_best_loss_step", "Mean chosen inner step", "mean_best_step_by_epsilon.png"),
        ("chosen_mean_l2_to_original", "Chosen mean L2 to original", "chosen_l2_to_original_by_epsilon.png"),
        ("chosen_mean_dist_to_nearest", "Chosen mean distance to nearest", "chosen_dist_to_nearest_by_epsilon.png"),
        ("final_mean_l2_to_original", "Final mean L2 to original", "final_l2_to_original_by_epsilon.png"),
        ("final_mean_dist_to_nearest", "Final mean distance to nearest", "final_dist_to_nearest_by_epsilon.png"),
        ("chosen_frac_nearest_changed", "Chosen fraction nearest changed", "chosen_frac_nearest_changed_by_epsilon.png"),
        ("final_frac_nearest_changed", "Final fraction nearest changed", "final_frac_nearest_changed_by_epsilon.png"),
    ]

    for metric, ylabel, filename in metrics:
        if metric not in all_run_summary.columns:
            continue

        plt.figure()

        for seed, sub in all_run_summary.groupby("seed"):
            sub = sub.sort_values("epsilon")
            plt.plot(sub["epsilon"], sub[metric], marker="o", label=f"seed={seed}")

        avg = all_run_summary.groupby("epsilon")[metric].mean().reset_index()
        avg = avg.sort_values("epsilon")
        plt.plot(avg["epsilon"], avg[metric], marker="x", linewidth=3, label="mean across seeds")

        plt.xlabel("epsilon")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        save_plot(plots_dir / filename)


# ============================================================
# Report
# ============================================================

def write_single_run_report(
    out_dir: Path,
    run_id: str,
    epsilon: float,
    seed: int | None,
    run_summary: pd.DataFrame,
    best_dist: pd.DataFrame,
    topk: pd.DataFrame,
):
    path = out_dir / "report.txt"

    lines = []

    lines.append("# Compact attack-space analysis report\n")
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Epsilon: {epsilon}")
    lines.append(f"Seed: {seed}\n")

    lines.append("## What this report measures\n")
    lines.append("Chosen perturbation = perturbation at the best-loss inner step.")
    lines.append("Final perturbation = perturbation at the last inner step.")
    lines.append("All token-level values are aggregated using mean, median, p90, min, and max where relevant.\n")

    if not run_summary.empty:
        row = run_summary.iloc[0]

        lines.append("## Core distances\n")

        lines.append("### Chosen perturbation\n")
        lines.append(f"- Mean L2 to original embedding: {row.get('chosen_mean_l2_to_original', np.nan):.8f}")
        lines.append(f"- Median L2 to original embedding: {row.get('chosen_median_l2_to_original', np.nan):.8f}")
        lines.append(f"- P90 L2 to original embedding: {row.get('chosen_p90_l2_to_original', np.nan):.8f}")
        lines.append(f"- Max L2 to original embedding: {row.get('chosen_max_l2_to_original', np.nan):.8f}")
        lines.append(f"- Mean distance to nearest embedding: {row.get('chosen_mean_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Median distance to nearest embedding: {row.get('chosen_median_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- P90 distance to nearest embedding: {row.get('chosen_p90_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Min distance to nearest embedding: {row.get('chosen_min_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Fraction nearest changed: {row.get('chosen_frac_nearest_changed', np.nan):.6f}")
        lines.append(f"- Fraction original still in top-k: {row.get('chosen_frac_original_in_topk', np.nan):.6f}")
        lines.append(f"- Mean original-token rank: {row.get('chosen_mean_original_rank', np.nan):.6f}\n")

        lines.append("### Final perturbation\n")
        lines.append(f"- Mean L2 to original embedding: {row.get('final_mean_l2_to_original', np.nan):.8f}")
        lines.append(f"- Median L2 to original embedding: {row.get('final_median_l2_to_original', np.nan):.8f}")
        lines.append(f"- P90 L2 to original embedding: {row.get('final_p90_l2_to_original', np.nan):.8f}")
        lines.append(f"- Max L2 to original embedding: {row.get('final_max_l2_to_original', np.nan):.8f}")
        lines.append(f"- Mean distance to nearest embedding: {row.get('final_mean_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Median distance to nearest embedding: {row.get('final_median_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- P90 distance to nearest embedding: {row.get('final_p90_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Min distance to nearest embedding: {row.get('final_min_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Fraction nearest changed: {row.get('final_frac_nearest_changed', np.nan):.6f}")
        lines.append(f"- Fraction original still in top-k: {row.get('final_frac_original_in_topk', np.nan):.6f}")
        lines.append(f"- Mean original-token rank: {row.get('final_mean_original_rank', np.nan):.6f}\n")

        lines.append("### Chosen versus final\n")
        lines.append(f"- Mean final-minus-chosen L2 to original: {row.get('delta_final_minus_chosen_mean_l2_to_original', np.nan):.8f}")
        lines.append(f"- Mean final-minus-chosen distance to nearest: {row.get('delta_final_minus_chosen_mean_dist_to_nearest', np.nan):.8f}")
        lines.append(f"- Mean final-minus-chosen original rank: {row.get('delta_final_minus_chosen_mean_original_rank', np.nan):.8f}\n")

    lines.append("## Best inner-step distribution\n")
    if not best_dist.empty:
        lines.append(best_dist.to_string(index=False))
    lines.append("")

    lines.append("## Top-k neighbour distance by rank\n")
    if not topk.empty:
        lines.append(topk.to_string(index=False))
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- If best_loss_step is usually the final step, the attack is still improving at the end of the 10 inner steps.")
    lines.append("- If chosen and final distances are nearly identical, the chosen perturbation is usually the final perturbation.")
    lines.append("- If chosen_frac_nearest_changed is close to 0 and original rank is close to 1, perturbations remain in the original token's local nearest-neighbour cell.")
    lines.append("- If L2 to original increases but distance to nearest stays stable, the perturbation moves inside a local cell without changing nearest token.")
    lines.append("- If top-1 to top-5 distances are very close, the perturbed point lies in a dense/ambiguous part of embedding space.")
    lines.append("- If top-1 is much smaller than top-2, the perturbed point has a clear nearest vocabulary embedding.")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved {path}")


def write_multi_run_report(
    out_dir: Path,
    all_run_summary: pd.DataFrame,
    epsilon_summary: pd.DataFrame,
    all_best_dist: pd.DataFrame,
):
    path = out_dir / "report.txt"

    lines = []
    lines.append("# Multi-run compact attack-space analysis report\n")

    lines.append("## Runs loaded\n")
    lines.append(f"- Number of runs: {all_run_summary['run_id'].nunique() if not all_run_summary.empty else 0}")
    if not all_run_summary.empty:
        lines.append(f"- Epsilons: {sorted(all_run_summary['epsilon'].dropna().unique())}")
        lines.append(f"- Seeds: {sorted(all_run_summary['seed'].dropna().unique())}")
    lines.append("")

    lines.append("## Epsilon-level summary\n")
    if not epsilon_summary.empty:
        lines.append(epsilon_summary.to_string(index=False))
    lines.append("")

    lines.append("## Best-step distribution across runs\n")
    if not all_best_dist.empty:
        pivot = all_best_dist.groupby(["epsilon", "best_loss_step"]).agg(
            mean_fraction=("fraction", "mean"),
            total_count=("count", "sum"),
        ).reset_index()
        lines.append(pivot.to_string(index=False))
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- Compare chosen_mean_l2_to_original across epsilon to see whether larger epsilon actually moves points further from original embeddings.")
    lines.append("- Compare chosen_mean_dist_to_nearest across epsilon to see whether perturbations move closer to neighbouring vocabulary embeddings.")
    lines.append("- Compare chosen_frac_nearest_changed and final_frac_nearest_changed to detect actual nearest-neighbour boundary crossings.")
    lines.append("- Compare mean_best_loss_step across epsilon to see whether stronger epsilon changes the attack optimisation dynamics.")
    lines.append("- Check seed variation before making claims about geometry.")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved {path}")


# ============================================================
# Single-run processing
# ============================================================

def process_one_run(
    trace_dir: Path,
    out_dir: Path,
    run_id: str,
    epsilon: float,
    seed: int | None,
    top_k: int,
    top_n: int,
    make_plots: bool = True,
) -> dict[str, pd.DataFrame]:

    print(f"[INFO] Processing run_id={run_id}, epsilon={epsilon}, seed={seed}")
    print(f"[INFO] Trace dir: {trace_dir}")

    ensure_dir(out_dir)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    df_steps = load_step_summaries(trace_dir)
    best_steps = build_best_steps(df_steps)

    df_selected_tokens = load_selected_token_records(trace_dir, best_steps)

    df_attack = build_attack_level_chosen_vs_final(
        df_selected_tokens=df_selected_tokens,
        best_steps=best_steps,
        top_k=top_k,
    )

    df_attack["run_id"] = run_id
    df_attack["epsilon"] = epsilon
    df_attack["seed"] = seed

    run_summary = build_run_summary(df_attack, run_id, epsilon, seed)
    best_dist = build_best_step_distribution(df_attack)
    best_dist["run_id"] = run_id
    best_dist["epsilon"] = epsilon
    best_dist["seed"] = seed

    topk = build_topk_distance_by_rank(df_attack, run_id, epsilon, seed, top_k)

    token_summary = build_token_type_summary(df_selected_tokens, run_id, epsilon, seed, top_n)
    position_summary = build_position_summary(df_selected_tokens, run_id, epsilon, seed, top_n)

    # Save compact outputs.
    save_csv(df_attack, tables_dir / "attack_level_chosen_vs_final.csv")
    save_csv(run_summary, tables_dir / "run_summary.csv")
    save_csv(best_dist, tables_dir / "best_step_distribution.csv")
    save_csv(topk, tables_dir / "topk_distance_by_rank.csv")
    save_csv(token_summary, tables_dir / "token_type_summary_top.csv")
    save_csv(position_summary, tables_dir / "position_summary_top.csv")

    # Save selected token records only if requested? Here we do not save them by default,
    # because you asked to avoid too much data.
    # The attack-level summary contains the compact information.

    if make_plots:
        plot_best_step_distribution(best_dist, plots_dir)
        plot_chosen_vs_final_attack_level(df_attack, plots_dir)
        plot_topk_distance(topk, plots_dir)

    write_single_run_report(
        out_dir=out_dir,
        run_id=run_id,
        epsilon=epsilon,
        seed=seed,
        run_summary=run_summary,
        best_dist=best_dist,
        topk=topk,
    )

    return {
        "attack_level": df_attack,
        "run_summary": run_summary,
        "best_step_distribution": best_dist,
        "topk_distance_by_rank": topk,
        "token_type_summary_top": token_summary,
        "position_summary_top": position_summary,
    }


# ============================================================
# Multi-run aggregation
# ============================================================

def run_outputs_exist(run_out_dir: Path) -> bool:
    """
    Detect whether a run has already been processed.

    We treat a run as complete when its core per-run outputs exist.
    """

    required = [
        run_out_dir / "tables" / "attack_level_chosen_vs_final.csv",
        run_out_dir / "tables" / "run_summary.csv",
        run_out_dir / "tables" / "best_step_distribution.csv",
        run_out_dir / "report.txt",
    ]
    return all(p.exists() for p in required)


def aggregate_many_runs(manifest_path: Path, out_dir: Path, top_k: int, top_n: int):
    ensure_dir(out_dir)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    manifest = pd.read_csv(manifest_path)

    required_cols = {"run_id", "epsilon", "trace_dir"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if "seed" not in manifest.columns:
        manifest["seed"] = np.nan

    all_attack = []
    all_run_summary = []
    all_best_dist = []
    all_topk = []
    all_token_summary = []
    all_position_summary = []

    for _, row in manifest.iterrows():
        run_id = str(row["run_id"])
        epsilon = float(row["epsilon"])
        seed = None if pd.isna(row["seed"]) else int(row["seed"])
        trace_dir = Path(str(row["trace_dir"]))

        run_out_dir = out_dir / "per_run" / run_id

        if run_outputs_exist(run_out_dir):
            print(f"[INFO] Skipping {run_id}: existing outputs found in {run_out_dir}")
            continue

        result = process_one_run(
            trace_dir=trace_dir,
            out_dir=run_out_dir,
            run_id=run_id,
            epsilon=epsilon,
            seed=seed,
            top_k=top_k,
            top_n=top_n,
            make_plots=False,
        )

        all_attack.append(result["attack_level"])
        all_run_summary.append(result["run_summary"])
        all_best_dist.append(result["best_step_distribution"])
        all_topk.append(result["topk_distance_by_rank"])

        if not result["token_type_summary_top"].empty:
            all_token_summary.append(result["token_type_summary_top"])

        if not result["position_summary_top"].empty:
            all_position_summary.append(result["position_summary_top"])

    df_all_attack = pd.concat(all_attack, ignore_index=True) if all_attack else pd.DataFrame()
    df_all_run_summary = pd.concat(all_run_summary, ignore_index=True) if all_run_summary else pd.DataFrame()
    df_all_best_dist = pd.concat(all_best_dist, ignore_index=True) if all_best_dist else pd.DataFrame()
    df_all_topk = pd.concat(all_topk, ignore_index=True) if all_topk else pd.DataFrame()
    df_all_token_summary = pd.concat(all_token_summary, ignore_index=True) if all_token_summary else pd.DataFrame()
    df_all_position_summary = pd.concat(all_position_summary, ignore_index=True) if all_position_summary else pd.DataFrame()

    # Epsilon-level aggregation across seeds/runs.
    if not df_all_run_summary.empty:
        epsilon_summary = df_all_run_summary.groupby("epsilon").agg(
            num_runs=("run_id", "nunique"),
            seeds_seen=("seed", "nunique"),
            total_attacks=("num_attacks", "sum"),

            mean_best_loss_step=("mean_best_loss_step", "mean"),
            std_best_loss_step=("mean_best_loss_step", "std"),
            mean_frac_best_step_last=("frac_best_step_last", "mean"),

            mean_chosen_l2_to_original=("chosen_mean_l2_to_original", "mean"),
            std_chosen_l2_to_original=("chosen_mean_l2_to_original", "std"),
            mean_chosen_dist_to_nearest=("chosen_mean_dist_to_nearest", "mean"),
            std_chosen_dist_to_nearest=("chosen_mean_dist_to_nearest", "std"),
            mean_chosen_frac_nearest_changed=("chosen_frac_nearest_changed", "mean"),
            mean_chosen_frac_original_in_topk=("chosen_frac_original_in_topk", "mean"),
            mean_chosen_original_rank=("chosen_mean_original_rank", "mean"),

            mean_final_l2_to_original=("final_mean_l2_to_original", "mean"),
            std_final_l2_to_original=("final_mean_l2_to_original", "std"),
            mean_final_dist_to_nearest=("final_mean_dist_to_nearest", "mean"),
            std_final_dist_to_nearest=("final_mean_dist_to_nearest", "std"),
            mean_final_frac_nearest_changed=("final_frac_nearest_changed", "mean"),
            mean_final_frac_original_in_topk=("final_frac_original_in_topk", "mean"),
            mean_final_original_rank=("final_mean_original_rank", "mean"),

            mean_delta_final_minus_chosen_l2=(
                "delta_final_minus_chosen_mean_l2_to_original",
                "mean",
            ),
            mean_delta_final_minus_chosen_dist_nearest=(
                "delta_final_minus_chosen_mean_dist_to_nearest",
                "mean",
            ),
        ).reset_index()
    else:
        epsilon_summary = pd.DataFrame()

    save_csv(manifest, tables_dir / "manifest_used.csv")
    save_csv(df_all_attack, tables_dir / "all_attack_level_chosen_vs_final.csv")
    save_csv(df_all_run_summary, tables_dir / "all_run_summary.csv")
    save_csv(epsilon_summary, tables_dir / "epsilon_summary.csv")
    save_csv(df_all_best_dist, tables_dir / "all_best_step_distribution.csv")
    save_csv(df_all_topk, tables_dir / "all_topk_distance_by_rank.csv")
    save_csv(df_all_token_summary, tables_dir / "all_token_type_summary_top.csv")
    save_csv(df_all_position_summary, tables_dir / "all_position_summary_top.csv")

    plot_multi_run_epsilon_summary(df_all_run_summary, plots_dir)

    write_multi_run_report(
        out_dir=out_dir,
        all_run_summary=df_all_run_summary,
        epsilon_summary=epsilon_summary,
        all_best_dist=df_all_best_dist,
    )

    print(f"[DONE] Multi-run compact analysis saved to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace_dir", type=str, help="Single attack_traces directory")
    mode.add_argument("--manifest", type=str, help="CSV with run_id, epsilon, seed, trace_dir")

    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--epsilon", type=float, default=None, help="Required for single-run mode")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_n", type=int, default=50)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.trace_dir is not None:
        if args.epsilon is None:
            raise ValueError("--epsilon is required in single-run mode")

        trace_dir = Path(args.trace_dir)
        run_id = args.run_id or trace_dir.parent.name

        process_one_run(
            trace_dir=trace_dir,
            out_dir=out_dir,
            run_id=run_id,
            epsilon=float(args.epsilon),
            seed=args.seed,
            top_k=args.top_k,
            top_n=args.top_n,
            make_plots=True,
        )

        print(f"[DONE] Single-run compact analysis saved to {out_dir}")

    else:
        aggregate_many_runs(
            manifest_path=Path(args.manifest),
            out_dir=out_dir,
            top_k=args.top_k,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
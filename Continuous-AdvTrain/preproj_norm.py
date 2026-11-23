#!/usr/bin/env python3
"""
Best-loss-step embedding-space analysis.

This script answers a specific question:

  For each attack, which inner step achieved the best loss, and how far from the
  original embedding was the perturbation at that chosen step?

It reads existing trace files only. It does not rerun training or recompute attacks.

Expected files inside --trace_dir:
  - step_summaries.jsonl
  - token_records.jsonl

Optional files are ignored if present.

Outputs:
  - attack_level_best_step.csv
      One row per attack, with best-loss-step metadata and attack-level geometry.

  - best_step_summary.csv
      One row per best-loss step, aggregating attacks that chose that step.

  - best_step_distribution.csv
      Counts and fractions of best-loss steps.

  - report.txt
      Compact human-readable summary.

The key implementation detail is that the script selects token rows only at the
best-loss step and, by default, excludes unperturbed token rows so they do not
pull the distances toward zero.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import json
import pandas as pd
import numpy as np
from pathlib import Path

LOGFILE = "Continuous-AdvTrain/slurm-96008.out"
TRACE_DIR = Path("experiments/CAT_adv_eps005_42_relativelr_REFAIT/attack_traces")  # adjust
OUT_CSV = "experiments/all_space_analysis_100innerloops/best_loss_step_summary_eps004.csv"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        print(f"[WARN] Skipping empty CSV: {path}")
        return
    df.to_csv(path, index=False)
    print(f"[OK] Saved {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse {path} line {line_idx}: {e}")
    return rows


def read_jsonl_df(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def resolve_trace_file(trace_dir: Path, filename: str, required: bool = True) -> Path | None:
    direct = trace_dir / filename
    if direct.exists():
        return direct

    matches = list(trace_dir.rglob(filename))
    if matches:
        print(f"[INFO] Found {filename} at {matches[0]}")
        return matches[0]

    if required:
        raise FileNotFoundError(f"Could not find {filename} under {trace_dir}")
    return None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_list(x: Any) -> list[Any]:
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


def mean_safe(x: Iterable[Any]) -> float:
    s = pd.Series(list(x)).dropna()
    return float(s.mean()) if len(s) else np.nan


def median_safe(x: Iterable[Any]) -> float:
    s = pd.Series(list(x)).dropna()
    return float(s.median()) if len(s) else np.nan


def q_safe(x: Iterable[Any], q: float) -> float:
    s = pd.Series(list(x)).dropna()
    return float(s.quantile(q)) if len(s) else np.nan


def min_safe(x: Iterable[Any]) -> float:
    s = pd.Series(list(x)).dropna()
    return float(s.min()) if len(s) else np.nan


def max_safe(x: Iterable[Any]) -> float:
    s = pd.Series(list(x)).dropna()
    return float(s.max()) if len(s) else np.nan


def frac_true(x: Iterable[Any]) -> float:
    s = pd.Series(list(x)).dropna()
    return float(s.astype(bool).mean()) if len(s) else np.nan


def frac_false(x: Iterable[Any]) -> float:
    s = pd.Series(list(x)).dropna()
    return float((~s.astype(bool)).mean()) if len(s) else np.nan


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Loading trace files
# ---------------------------------------------------------------------------


def load_step_summaries(trace_dir: Path) -> pd.DataFrame:
    path = resolve_trace_file(trace_dir, "step_summaries.jsonl", required=True)
    df = read_jsonl_df(path)
    if df.empty:
        raise ValueError(f"{path} exists but contains no rows")

    required = {"attack_id", "step", "loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"step_summaries.jsonl is missing columns: {sorted(missing)}")

    df = to_numeric(
        df,
        [
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
        ],
    )
    return df


def build_best_steps(df_steps: pd.DataFrame) -> pd.DataFrame:
    """
    One row per attack_id.

    best_loss_step is the step with the minimum loss.
    final_step is the maximum step observed for that attack.
    """

    df_steps = df_steps.sort_values(["attack_id", "step"]).copy()
    valid = df_steps["loss"].notna()
    if not valid.any():
        raise ValueError("No valid loss values found in step_summaries.jsonl")

    idx_best = df_steps.loc[valid].groupby("attack_id")["loss"].idxmin()
    best = df_steps.loc[idx_best].copy()

    best = best.rename(columns={"step": "best_loss_step", "loss": "best_loss"})
    keep = [c for c in ["attack_id", "trainer_global_step", "attack_call_idx", "best_loss_step", "best_loss"] if c in best.columns]
    best = best[keep]

    first_loss = df_steps.groupby("attack_id").first(numeric_only=False)["loss"].rename("first_loss").reset_index()
    final_loss = df_steps.groupby("attack_id").last(numeric_only=False)["loss"].rename("final_loss").reset_index()
    final_step = df_steps.groupby("attack_id")["step"].max().rename("final_step").reset_index()
    num_inner_steps = df_steps.groupby("attack_id")["step"].nunique().rename("num_inner_steps").reset_index()

    out = best.merge(first_loss, on="attack_id", how="left")
    out = out.merge(final_loss, on="attack_id", how="left")
    out = out.merge(final_step, on="attack_id", how="left")
    out = out.merge(num_inner_steps, on="attack_id", how="left")

    out["loss_drop_first_to_best"] = out["first_loss"] - out["best_loss"]
    out["loss_drop_first_to_final"] = out["first_loss"] - out["final_loss"]
    return out


def is_perturbed_row(row: dict[str, Any], tol: float = 1e-12) -> bool:
    for key in ("is_perturbed", "perturbed", "token_perturbed", "is_token_perturbed"):
        if key in row and row[key] is not None:
            try:
                return bool(row[key])
            except Exception:
                pass

    try:
        return float(row.get("l2_to_original", 0.0)) > tol
    except Exception:
        return False


def load_best_step_token_rows(
    trace_dir: Path,
    best_steps: pd.DataFrame,
    perturbed_only: bool = True,
) -> pd.DataFrame:
    """
    Load token rows only for each attack's chosen best-loss step.

    This is the critical filter that keeps the analysis focused on the selected
    perturbation rather than all steps or all tokens.
    """

    token_path = resolve_trace_file(trace_dir, "token_records.jsonl", required=True)
    best_map = dict(zip(best_steps["attack_id"], best_steps["best_loss_step"]))

    selected: list[dict[str, Any]] = []
    with token_path.open("r", encoding="utf-8", errors="ignore") as f:
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
            if attack_id not in best_map:
                continue
            if step != best_map[attack_id]:
                continue
            if perturbed_only and not is_perturbed_row(row):
                continue

            selected.append(row)

    df = pd.DataFrame(selected)
    if df.empty:
        raise ValueError(
            "No selected token rows were found. Check that token_records.jsonl has "
            "attack_id and step fields matching step_summaries.jsonl."
        )

    df = to_numeric(
        df,
        [
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
        ],
    )
    for col in ["nearest_is_original", "original_in_topk"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_attack_level_summary(
    df_selected_tokens: pd.DataFrame,
    best_steps: pd.DataFrame,
    epsilon: float,
    top_k: int,
) -> pd.DataFrame:
    """
    One row per attack.

    Important: metrics are computed from the selected best-loss-step token rows
    for that attack.
    """

    best_map = best_steps.set_index("attack_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []

    for attack_id, sub in df_selected_tokens.groupby("attack_id"):
        base = best_map.get(attack_id, {})
        row: dict[str, Any] = {
            "attack_id": attack_id,
            "trainer_global_step": base.get("trainer_global_step"),
            "attack_call_idx": base.get("attack_call_idx"),
            "best_loss_step": base.get("best_loss_step"),
            "final_step": base.get("final_step"),
            "num_inner_steps": base.get("num_inner_steps"),
            "first_loss": base.get("first_loss"),
            "best_loss": base.get("best_loss"),
            "final_loss": base.get("final_loss"),
            "loss_drop_first_to_best": base.get("loss_drop_first_to_best"),
            "loss_drop_first_to_final": base.get("loss_drop_first_to_final"),
            "selected_token_count": int(len(sub)),
        }

        # Token-level distances at the chosen step.
        row["mean_l2_to_original"] = mean_safe(sub["l2_to_original"])
        row["median_l2_to_original"] = median_safe(sub["l2_to_original"])
        row["p90_l2_to_original"] = q_safe(sub["l2_to_original"], 0.90)
        row["p99_l2_to_original"] = q_safe(sub["l2_to_original"], 0.99)
        row["min_l2_to_original"] = min_safe(sub["l2_to_original"])
        row["max_l2_to_original"] = max_safe(sub["l2_to_original"])

        row["mean_dist_to_nearest"] = mean_safe(sub["dist_to_nearest"])
        row["median_dist_to_nearest"] = median_safe(sub["dist_to_nearest"])
        row["p90_dist_to_nearest"] = q_safe(sub["dist_to_nearest"], 0.90)
        row["p99_dist_to_nearest"] = q_safe(sub["dist_to_nearest"], 0.99)
        row["min_dist_to_nearest"] = min_safe(sub["dist_to_nearest"])
        row["max_dist_to_nearest"] = max_safe(sub["dist_to_nearest"])

        if "nearest_is_original" in sub.columns:
            row["frac_nearest_changed"] = frac_false(sub["nearest_is_original"])
            row["frac_original_is_nearest"] = frac_true(sub["nearest_is_original"])

        if "original_in_topk" in sub.columns:
            row["frac_original_in_topk"] = frac_true(sub["original_in_topk"])
            row["frac_original_not_topk"] = frac_false(sub["original_in_topk"])

        if "original_rank" in sub.columns:
            row["mean_original_rank"] = mean_safe(sub["original_rank"])
            row["median_original_rank"] = median_safe(sub["original_rank"])
            row["p90_original_rank"] = q_safe(sub["original_rank"], 0.90)
            row["max_original_rank"] = max_safe(sub["original_rank"])

        # Boundary-style quantities.
        row["mean_l2_over_epsilon"] = row["mean_l2_to_original"] / epsilon if epsilon else np.nan
        row["median_l2_over_epsilon"] = row["median_l2_to_original"] / epsilon if epsilon else np.nan
        row["mean_slack_to_boundary"] = epsilon - row["mean_l2_to_original"] if epsilon else np.nan
        row["max_l2_over_epsilon"] = row["max_l2_to_original"] / epsilon if epsilon else np.nan

        # Top-k nearest-neighbour distances if present.
        if "topk_distances" in sub.columns:
            topk_by_rank: dict[int, list[float]] = defaultdict(list)
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
                row[f"top{rank}_distance_mean"] = mean_safe(values)
                row[f"top{rank}_distance_median"] = median_safe(values)
                row[f"top{rank}_distance_p90"] = q_safe(values, 0.90)
                row[f"top{rank}_distance_min"] = min_safe(values)
                row[f"top{rank}_distance_max"] = max_safe(values)

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Attach run-level epsilon/seed later in main, but keep the attack-level table self-contained.
    out = out.sort_values(["best_loss_step", "attack_id"]).reset_index(drop=True)
    return out


def build_best_step_summary(df_attack: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    Aggregate attacks by the step that won the best-loss competition.

    This is the table that directly answers:
      "For attacks whose best-loss step was k, how far from original were they?"
    """

    if df_attack.empty:
        return pd.DataFrame()

    out = (
        df_attack.groupby("best_loss_step", as_index=False)
        .agg(
            attacks=("attack_id", "nunique"),
            mean_selected_tokens=("selected_token_count", "mean"),
            median_selected_tokens=("selected_token_count", "median"),
            mean_l2_to_original=("mean_l2_to_original", "mean"),
            median_l2_to_original=("mean_l2_to_original", "median"),
            p90_l2_to_original=("mean_l2_to_original", lambda s: s.quantile(0.90)),
            min_l2_to_original=("mean_l2_to_original", "min"),
            max_l2_to_original=("mean_l2_to_original", "max"),
            mean_l2_over_epsilon=("mean_l2_over_epsilon", "mean"),
            median_l2_over_epsilon=("mean_l2_over_epsilon", "median"),
            mean_slack_to_boundary=("mean_slack_to_boundary", "mean"),
            mean_dist_to_nearest=("mean_dist_to_nearest", "mean"),
            median_dist_to_nearest=("mean_dist_to_nearest", "median"),
            mean_original_rank=("mean_original_rank", "mean"),
            median_original_rank=("mean_original_rank", "median"),
            frac_nearest_changed=("frac_nearest_changed", "mean"),
            frac_original_in_topk=("frac_original_in_topk", "mean"),
        )
        .sort_values("best_loss_step")
        .reset_index(drop=True)
    )

    out["epsilon"] = epsilon
    return out


def build_best_step_distribution(df_attack: pd.DataFrame, epsilon: float, run_id: str, seed: int | None) -> pd.DataFrame:
    out = (
        df_attack["best_loss_step"]
        .value_counts()
        .sort_index()
        .rename_axis("best_loss_step")
        .reset_index(name="count")
    )
    total = float(out["count"].sum()) if len(out) else 0.0
    out["fraction"] = out["count"] / total if total else np.nan
    out["run_id"] = run_id
    out["epsilon"] = epsilon
    out["seed"] = seed
    return out


def build_report(
    out_dir: Path,
    run_id: str,
    epsilon: float,
    seed: int | None,
    df_attack: pd.DataFrame,
    step_summary: pd.DataFrame,
    best_dist: pd.DataFrame,
) -> None:
    path = out_dir / "report.txt"
    lines: list[str] = []

    lines.append("# Best-loss-step embedding-space report\n")
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Epsilon: {epsilon}")
    lines.append(f"Seed: {seed}\n")

    lines.append("## What this report measures\n")
    lines.append("The script selects the token rows at each attack's best-loss inner step.")
    lines.append("It then excludes unperturbed rows by default and summarizes distances from the original embedding.")
    lines.append("The main quantity of interest is mean L2 to original at the chosen best-loss step.\n")

    if not df_attack.empty:
        lines.append("## Overall attack-level summary\n")
        lines.append(f"- Attacks: {df_attack['attack_id'].nunique()}")
        lines.append(f"- Mean selected-token count: {df_attack['selected_token_count'].mean():.4f}")
        lines.append(f"- Mean L2 to original: {df_attack['mean_l2_to_original'].mean():.8f}")
        lines.append(f"- Median L2 to original: {df_attack['mean_l2_to_original'].median():.8f}")
        lines.append(f"- P90 L2 to original: {df_attack['mean_l2_to_original'].quantile(0.90):.8f}")
        lines.append(f"- Mean L2 / epsilon: {(df_attack['mean_l2_to_original'] / epsilon).mean():.8f}")
        lines.append(f"- Mean slack to boundary: {(epsilon - df_attack['mean_l2_to_original']).mean():.8f}\n")

    lines.append("## Best-loss-step distribution\n")
    if not best_dist.empty:
        lines.append(best_dist.to_string(index=False))
    lines.append("")

    lines.append("## Best-loss-step distance summary\n")
    if not step_summary.empty:
        lines.append(step_summary.to_string(index=False))
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- mean_l2_to_original is the average distance of the chosen perturbation from the original embedding.")
    lines.append("- mean_l2_over_epsilon tells you how close the chosen perturbation is to the epsilon radius.")
    lines.append("- mean_slack_to_boundary = epsilon - mean_l2_to_original; smaller values mean you are closer to the boundary.")
    lines.append("- If best_loss_step is often the final step, the attack keeps improving until the end of the inner loop.")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved {path}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_one_run(
    trace_dir: Path,
    out_dir: Path,
    run_id: str,
    epsilon: float,
    seed: int | None,
    perturbed_only: bool,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    print(f"[INFO] Processing run_id={run_id}, epsilon={epsilon}, seed={seed}")
    print(f"[INFO] Trace dir: {trace_dir}")

    ensure_dir(out_dir)
    tables_dir = out_dir / "tables"
    ensure_dir(tables_dir)

    df_steps = load_step_summaries(trace_dir)
    best_steps = build_best_steps(df_steps)
    df_selected_tokens = load_best_step_token_rows(trace_dir, best_steps, perturbed_only=perturbed_only)

    df_attack = build_attack_level_summary(
        df_selected_tokens=df_selected_tokens,
        best_steps=best_steps,
        epsilon=epsilon,
        top_k=top_k,
    )
    df_attack["run_id"] = run_id
    df_attack["epsilon"] = epsilon
    df_attack["seed"] = seed

    step_summary = build_best_step_summary(df_attack, epsilon)
    step_summary["run_id"] = run_id
    step_summary["seed"] = seed

    best_dist = build_best_step_distribution(df_attack, epsilon, run_id, seed)

    save_csv(best_steps, tables_dir / "best_steps_per_attack.csv")
    save_csv(df_selected_tokens, tables_dir / "selected_best_step_token_rows.csv")
    save_csv(df_attack, tables_dir / "attack_level_best_step.csv")
    save_csv(step_summary, tables_dir / "best_step_summary.csv")
    save_csv(best_dist, tables_dir / "best_step_distribution.csv")

    build_report(out_dir, run_id, epsilon, seed, df_attack, step_summary, best_dist)

    return {
        "best_steps": best_steps,
        "selected_tokens": df_selected_tokens,
        "attack_level": df_attack,
        "step_summary": step_summary,
        "best_dist": best_dist,
    }


def aggregate_many_runs(manifest_path: Path, out_dir: Path, perturbed_only: bool, top_k: int) -> None:
    ensure_dir(out_dir)
    tables_dir = out_dir / "tables"
    ensure_dir(tables_dir)

    manifest = pd.read_csv(manifest_path)
    required_cols = {"run_id", "epsilon", "trace_dir"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if "seed" not in manifest.columns:
        manifest["seed"] = np.nan

    all_attack: list[pd.DataFrame] = []
    all_step_summary: list[pd.DataFrame] = []
    all_best_dist: list[pd.DataFrame] = []

    for _, row in manifest.iterrows():
        run_id = str(row["run_id"])
        epsilon = float(row["epsilon"])
        seed = None if pd.isna(row["seed"]) else int(row["seed"])
        trace_dir = Path(str(row["trace_dir"]))
        run_out = out_dir / "per_run" / run_id

        result = process_one_run(
            trace_dir=trace_dir,
            out_dir=run_out,
            run_id=run_id,
            epsilon=epsilon,
            seed=seed,
            perturbed_only=perturbed_only,
            top_k=top_k,
        )

        all_attack.append(result["attack_level"])
        all_step_summary.append(result["step_summary"])
        all_best_dist.append(result["best_dist"])

    df_all_attack = pd.concat(all_attack, ignore_index=True) if all_attack else pd.DataFrame()
    df_all_step_summary = pd.concat(all_step_summary, ignore_index=True) if all_step_summary else pd.DataFrame()
    df_all_best_dist = pd.concat(all_best_dist, ignore_index=True) if all_best_dist else pd.DataFrame()

    save_csv(manifest, tables_dir / "manifest_used.csv")
    save_csv(df_all_attack, tables_dir / "all_attack_level_best_step.csv")
    save_csv(df_all_step_summary, tables_dir / "all_best_step_summary.csv")
    save_csv(df_all_best_dist, tables_dir / "all_best_step_distribution.csv")

    if not df_all_step_summary.empty:
        epsilon_summary = (
            df_all_step_summary.groupby("epsilon", as_index=False)
            .agg(
                runs=("run_id", "nunique"),
                total_attacks=("attacks", "sum"),
                mean_l2_to_original=("mean_l2_to_original", "mean"),
                median_l2_to_original=("mean_l2_to_original", "median"),
                mean_l2_over_epsilon=("mean_l2_over_epsilon", "mean"),
                mean_slack_to_boundary=("mean_slack_to_boundary", "mean"),
                mean_best_step=("best_loss_step", "mean"),
            )
            .sort_values("epsilon")
            .reset_index(drop=True)
        )
    else:
        epsilon_summary = pd.DataFrame()

    save_csv(epsilon_summary, tables_dir / "epsilon_summary.csv")
    print(f"[DONE] Multi-run analysis saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-loss-step embedding-space analysis")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace_dir", type=str, help="Single attack_traces directory")
    mode.add_argument("--manifest", type=str, help="CSV with run_id, epsilon, trace_dir[, seed]")

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=None, help="Required in single-run mode")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--include_unperturbed",
        action="store_true",
        help="Keep token rows even if they appear unperturbed. Default is to exclude them.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Reserved for top-k distance summaries")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    perturbed_only = not args.include_unperturbed

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
            perturbed_only=perturbed_only,
            top_k=args.top_k,
        )
        print(f"[DONE] Single-run analysis saved to {out_dir}")

    else:
        aggregate_many_runs(
            manifest_path=Path(args.manifest),
            out_dir=out_dir,
            perturbed_only=perturbed_only,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Seed-level aggregation of Mean L2 to original embedding.

Reads only the per-run tables/run_summary.csv files that
analyse_attack_traces.py has already produced under
experiments/all_space_analysis_global/per_run/<run_id>/tables/run_summary.csv.
It does NOT rerun any analysis or touch trace files.

Runs are grouped into "families" (same configuration, different seed) by
removing the seed value as a token from run_id, using the seed column from
the manifest. For example:

CAT_adv_eps005_42_relativelr_REFAIT  (seed=42)  -> CAT_adv_eps005_relativelr_REFAIT
CAT_adv_eps005_123_relativelr_REFAIT (seed=123) -> CAT_adv_eps005_relativelr_REFAIT
CAT_llama3b_005                      (seed=42)  -> CAT_llama3b_005   (42 not in name)
CAT_llama3b_005_123                  (seed=123) -> CAT_llama3b_005
CAT_llama3b_123_otherparams          (seed=123) -> CAT_llama3b_otherparams

Within each family, mean and standard deviation (ddof=1, i.e. divide by n-1)
of chosen_mean_l2_to_original and final_mean_l2_to_original are computed
across seeds.

Usage:

python Continuous-AdvTrain/aggregate_seed_family_summary.py ^
  --manifest Continuous-AdvTrain/to_run_global.csv ^
  --per_run_dir experiments/all_space_analysis_global/per_run ^
  --out_dir experiments/all_space_analysis_global
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def family_of(run_id: str, seed) -> str:
    """
    Removes the seed value from run_id, so that runs differing only by seed
    collapse into the same family. Handles two conventions seen in our
    manifests:
    - seed as its own underscore-delimited token, e.g. "..._123_..." -> "..."
    - seed fused onto the word "seed", e.g. "..._seed123_..." -> "..._seed_..."
    If the seed does not appear in run_id at all (e.g. the default seed run
    has no seed suffix in its name), run_id is returned unchanged.
    """

    if pd.isna(seed):
        return run_id

    seed_str = str(int(seed))
    parts = run_id.split("_")

    new_parts = []
    changed = False
    for p in parts:
        if p == seed_str:
            changed = True
            continue
        if p == f"seed{seed_str}":
            new_parts.append("seed")
            changed = True
            continue
        new_parts.append(p)

    if not changed:
        return run_id

    return "_".join(new_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--per_run_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    per_run_dir = Path(args.per_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, m in manifest.iterrows():
        run_id = str(m["run_id"])
        seed = m["seed"]
        epsilon = float(m["epsilon"])

        summary_path = per_run_dir / run_id / "tables" / "run_summary.csv"
        if not summary_path.exists():
            print(f"[WARN] Skipping {run_id}: no run_summary.csv found at {summary_path}")
            continue

        summary = pd.read_csv(summary_path)
        if summary.empty:
            print(f"[WARN] Skipping {run_id}: run_summary.csv is empty")
            continue

        r = summary.iloc[0]

        rows.append({
            "family": family_of(run_id, seed),
            "run_id": run_id,
            "epsilon": epsilon,
            "seed": None if pd.isna(seed) else int(seed),
            "chosen_mean_l2_to_original": r.get("chosen_mean_l2_to_original", np.nan),
            "final_mean_l2_to_original": r.get("final_mean_l2_to_original", np.nan),
            "chosen_top1_distance_mean": r.get("chosen_top1_distance_mean", np.nan),
            "chosen_top2_distance_mean": r.get("chosen_top2_distance_mean", np.nan),
            "final_top1_distance_mean": r.get("final_top1_distance_mean", np.nan),
            "final_top2_distance_mean": r.get("final_top2_distance_mean", np.nan),
            "chosen_frac_nearest_changed": r.get("chosen_frac_nearest_changed", np.nan),
            "final_frac_nearest_changed": r.get("final_frac_nearest_changed", np.nan),
            "chosen_mean_original_rank": r.get("chosen_mean_original_rank", np.nan),
            "final_mean_original_rank": r.get("final_mean_original_rank", np.nan),
            # Best inner-step statistics.
            # best_loss_step is zero-indexed (0..9), so add 1 when
            # expressing it as iteration number (1..10).
            "mean_chosen_iteration": (
                r.get("mean_best_loss_step", np.nan) + 1
            ),
            "median_chosen_iteration": (
                r.get("median_best_loss_step", np.nan) + 1
            ),
            "frac_final_iteration": r.get("frac_best_step_last", np.nan),
        })

    df = pd.DataFrame(rows)

    per_run_path = out_dir / "tables" / "family_seed_l2_per_run.csv"
    df.to_csv(per_run_path, index=False)
    print(f"[OK] Saved {per_run_path}")

    summary = df.groupby(["family", "epsilon"]).agg(
        n_seeds=("seed", "nunique"),
        seeds=("seed", lambda s: sorted(s.dropna().unique().tolist())),
        mean_chosen_l2_to_original=("chosen_mean_l2_to_original", "mean"),
        std_chosen_l2_to_original=("chosen_mean_l2_to_original", lambda x: x.std(ddof=1)),
        mean_final_l2_to_original=("final_mean_l2_to_original", "mean"),
        std_final_l2_to_original=("final_mean_l2_to_original", lambda x: x.std(ddof=1)),
        mean_chosen_top1_distance=("chosen_top1_distance_mean", "mean"),
        std_chosen_top1_distance=("chosen_top1_distance_mean", lambda x: x.std(ddof=1)),
        mean_chosen_top2_distance=("chosen_top2_distance_mean", "mean"),
        std_chosen_top2_distance=("chosen_top2_distance_mean", lambda x: x.std(ddof=1)),
        mean_final_top1_distance=("final_top1_distance_mean", "mean"),
        std_final_top1_distance=("final_top1_distance_mean", lambda x: x.std(ddof=1)),
        mean_final_top2_distance=("final_top2_distance_mean", "mean"),
        std_final_top2_distance=("final_top2_distance_mean", lambda x: x.std(ddof=1)),
        mean_chosen_frac_nearest_changed=("chosen_frac_nearest_changed", "mean"),
        std_chosen_frac_nearest_changed=("chosen_frac_nearest_changed", lambda x: x.std(ddof=1)),
        mean_final_frac_nearest_changed=("final_frac_nearest_changed", "mean"),
        std_final_frac_nearest_changed=("final_frac_nearest_changed", lambda x: x.std(ddof=1)),
        mean_chosen_original_rank=("chosen_mean_original_rank", "mean"),
        std_chosen_original_rank=("chosen_mean_original_rank", lambda x: x.std(ddof=1)),
        mean_final_original_rank=("final_mean_original_rank", "mean"),
        std_final_original_rank=("final_mean_original_rank", lambda x: x.std(ddof=1)),

        mean_chosen_iteration=(
            "mean_chosen_iteration",
            "mean",
        ),
        std_chosen_iteration=(
            "mean_chosen_iteration",
            lambda x: x.std(ddof=1),
        ),

        mean_median_chosen_iteration=(
            "median_chosen_iteration",
            "mean",
        ),
        std_median_chosen_iteration=(
            "median_chosen_iteration",
            lambda x: x.std(ddof=1),
        ),

        mean_frac_final_iteration=(
            "frac_final_iteration",
            "mean",
        ),
        std_frac_final_iteration=(
            "frac_final_iteration",
            lambda x: x.std(ddof=1),
        ),
    ).reset_index()

    summary = summary.sort_values(["family", "epsilon"])

    summary_path = out_dir / "tables" / "family_seed_l2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved {summary_path}")

    report_path = out_dir / "family_seed_l2_summary.txt"
    lines = []
    lines.append("# Mean L2 to original embedding, aggregated over seeds\n")
    lines.append(
        "Standard deviation uses ddof=1 (divide by n-1), i.e. the sample "
        "standard deviation, since each family only has a handful of seeds "
        "and is being treated as a sample from a larger population of possible seeds.\n"
    )

    for _, row in summary.iterrows():
        lines.append(f"## {row['family']} (epsilon={row['epsilon']})")
        lines.append(f"- Seeds: {row['seeds']} (n={row['n_seeds']})")
        lines.append(
            f"- Chosen mean L2 to original: {row['mean_chosen_l2_to_original']:.8f} "
            f"+/- {row['std_chosen_l2_to_original']:.8f}"
        )
        lines.append(
            f"- Final mean L2 to original:  {row['mean_final_l2_to_original']:.8f} "
            f"+/- {row['std_final_l2_to_original']:.8f}"
        )
        lines.append(
            f"- Chosen attack_avg_mean_distance (topk_rank=1): {row['mean_chosen_top1_distance']:.8f} "
            f"+/- {row['std_chosen_top1_distance']:.8f}"
        )
        lines.append(
            f"- Chosen attack_avg_mean_distance (topk_rank=2): {row['mean_chosen_top2_distance']:.8f} "
            f"+/- {row['std_chosen_top2_distance']:.8f}"
        )
        lines.append(
            f"- Final attack_avg_mean_distance (topk_rank=1):  {row['mean_final_top1_distance']:.8f} "
            f"+/- {row['std_final_top1_distance']:.8f}"
        )
        lines.append(
            f"- Final attack_avg_mean_distance (topk_rank=2):  {row['mean_final_top2_distance']:.8f} "
            f"+/- {row['std_final_top2_distance']:.8f}"
        )
        lines.append(
            f"- Chosen fraction nearest changed: {row['mean_chosen_frac_nearest_changed']:.8f} "
            f"+/- {row['std_chosen_frac_nearest_changed']:.8f}"
        )
        lines.append(
            f"- Final fraction nearest changed:  {row['mean_final_frac_nearest_changed']:.8f} "
            f"+/- {row['std_final_frac_nearest_changed']:.8f}"
        )
        lines.append(
            f"- Chosen mean original-token rank: {row['mean_chosen_original_rank']:.8f} "
            f"+/- {row['std_chosen_original_rank']:.8f}"
        )
        lines.append(
            f"- Final mean original-token rank:  {row['mean_final_original_rank']:.8f} "
            f"+/- {row['std_final_original_rank']:.8f}"
        )

        lines.append(
            f"- Mean chosen iteration (1-10): "
            f"{row['mean_chosen_iteration']:.4f} "
            f"+/- {row['std_chosen_iteration']:.4f}"
        )
        lines.append(
            f"- Median chosen iteration (1-10), averaged across seeds: "
            f"{row['mean_median_chosen_iteration']:.4f} "
            f"+/- {row['std_median_chosen_iteration']:.4f}"
        )
        lines.append(
            f"- Fraction selecting final (10th) iteration: "
            f"{row['mean_frac_final_iteration']:.6f} "
            f"+/- {row['std_frac_final_iteration']:.6f}\n"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved {report_path}")


if __name__ == "__main__":
    main()

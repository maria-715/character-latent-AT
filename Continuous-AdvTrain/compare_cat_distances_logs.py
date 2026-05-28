"""
Compare Continuous Adversarial Training log files using custom run labels.

What it does:
- Reads one or more log files, or directories containing log files.
- Extracts per-block iteration L2 metrics and block losses.
- Uses your provided label as the run name everywhere in the output.
- Writes:
  * CSV summary for each block
  * CSV summary aggregated by run
- Prints compact comparison tables.

Two run-level summaries are reported:
- all_iters_* : averages over all attack iterations inside each block, then across blocks
- best_*      : averages over the selected best iteration inside each block, then across blocks

Important:
- Attack loss is logged once per block (from the "Total loss ..." line), not per iteration.
  So there is no separate "best attack loss". It is shared by the whole block.

Usage:
  python compare_cat_logs.py logs/*.out --labels eps0.0 eps0.1 eps0.2
  python compare_cat_logs.py /path/to/log_dir --labels base low mid high
  python compare_cat_logs.py run1/slurm.out run2/slurm.out --labels model_a model_b
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ITER_RE = re.compile(
    r"\[Iter\s+(?P<iter>\d+)\]\s+L2\(orig\):\s+mean=(?P<orig_mean>[-+0-9.eE]+),\s+max=(?P<orig_max>[-+0-9.eE]+)\s+\|\s+L2\(nearest\):\s+mean=(?P<near_mean>[-+0-9.eE]+),\s+min=(?P<near_min>[-+0-9.eE]+)"
)
BEST_RE = re.compile(
    r"\[BEST\s+@\s+Iter\s+(?P<iter>\d+)\]\s+L2\(orig\):\s+mean=(?P<orig_mean>[-+0-9.eE]+),\s+max=(?P<orig_max>[-+0-9.eE]+)\s+\|\s+L2\(nearest\):\s+mean=(?P<near_mean>[-+0-9.eE]+),\s+min=(?P<near_min>[-+0-9.eE]+)"
)
LOSS_RE = re.compile(
    r"Total loss\s+(?P<total>[-+0-9.eE]+);\s+Away loss\s+(?P<away>[-+0-9.eE]+);\s+Toward loss\s+(?P<toward>[-+0-9.eE]+);\s+DPO loss\s+(?P<dpo>[-+0-9.eE]+);\s+Utility loss\s+(?P<utility>[-+0-9.eE]+);\s+All attack losses\s+\[(?P<attack>[^\]]*)\];\s+Affirmative responses\s+\[(?P<aff>[^\]]*)\]"
)


def flatten_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(Path(x) for x in glob.glob(str(p / "*")) if Path(x).is_file()))
        elif any(ch in item for ch in "*?["):
            paths.extend(sorted(Path(x) for x in glob.glob(item) if Path(x).is_file()))
        else:
            paths.append(p)

    seen = set()
    uniq: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            uniq.append(p)
    return uniq


def try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_attack_losses(text: str) -> List[float]:
    vals: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = try_float(part)
        if v is not None and math.isfinite(v):
            vals.append(v)
    return vals


def parse_log(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = path.read_text(errors="replace")

    blocks: List[Dict[str, Any]] = []
    current_iters: List[Dict[str, Any]] = []
    current_best: Optional[Dict[str, Any]] = None
    current_loss: Optional[Dict[str, Any]] = None
    block_idx = -1

    for line in text.splitlines():
        m_iter = ITER_RE.search(line)
        if m_iter:
            current_iters.append(
                {
                    "iter": int(m_iter.group("iter")),
                    "orig_mean": float(m_iter.group("orig_mean")),
                    "orig_max": float(m_iter.group("orig_max")),
                    "near_mean": float(m_iter.group("near_mean")),
                    "near_min": float(m_iter.group("near_min")),
                }
            )
            continue

        m_best = BEST_RE.search(line)
        if m_best:
            current_best = {
                "iter": int(m_best.group("iter")),
                "orig_mean": float(m_best.group("orig_mean")),
                "orig_max": float(m_best.group("orig_max")),
                "near_mean": float(m_best.group("near_mean")),
                "near_min": float(m_best.group("near_min")),
            }
            continue

        m_loss = LOSS_RE.search(line)
        if m_loss:
            block_idx += 1
            attack_losses = parse_attack_losses(m_loss.group("attack"))
            current_loss = {
                "total_loss": float(m_loss.group("total")),
                "away_loss": float(m_loss.group("away")),
                "toward_loss": float(m_loss.group("toward")),
                "dpo_loss": float(m_loss.group("dpo")),
                "utility_loss": float(m_loss.group("utility")),
                "attack_loss_mean": stats.fmean(attack_losses) if attack_losses else None,
                "attack_loss_min": min(attack_losses) if attack_losses else None,
                "attack_loss_max": max(attack_losses) if attack_losses else None,
                "attack_loss_n": len(attack_losses),
            }

            if current_iters:
                iter_orig = [x["orig_mean"] for x in current_iters]
                iter_near = [x["near_mean"] for x in current_iters]

                best_gap = None
                if current_best is not None:
                    best_gap = current_best["orig_mean"] - current_best["near_mean"]

                row = {
                    "file": path.name,
                    "path": str(path),
                    "block_idx": block_idx,
                    "n_iters": len(current_iters),
                    "iter_first": current_iters[0]["iter"],
                    "iter_last": current_iters[-1]["iter"],
                    "iters_orig_l2_mean": stats.fmean(iter_orig),
                    "iters_orig_l2_last": current_iters[-1]["orig_mean"],
                    "iters_near_l2_mean": stats.fmean(iter_near),
                    "iters_near_l2_last": current_iters[-1]["near_mean"],
                    "iters_gap_mean": stats.fmean([a - b for a, b in zip(iter_orig, iter_near)]),
                    "iters_gap_last": current_iters[-1]["orig_mean"] - current_iters[-1]["near_mean"],
                    "best_iter": current_best["iter"] if current_best else None,
                    "best_orig_l2": current_best["orig_mean"] if current_best else None,
                    "best_near_l2": current_best["near_mean"] if current_best else None,
                    "best_gap_l2": best_gap,
                    **current_loss,
                }
                blocks.append(row)

            current_iters = []
            current_best = None
            current_loss = None

    run_summary: Dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "n_blocks": len(blocks),
    }

    if blocks:
        # Summary over all attack iterations (really: block averages of iteration-level values)
        all_map = {
            "iters_orig_l2_mean": "all_iters_orig_l2",
            "iters_near_l2_mean": "all_iters_near_l2",
            "iters_gap_mean": "all_iters_gap_l2",
            "total_loss": "all_total_loss",
            "away_loss": "all_away_loss",
            "toward_loss": "all_toward_loss",
            "utility_loss": "all_utility_loss",
            "attack_loss_mean": "all_attack_loss",
        }
        for row_key, out_prefix in all_map.items():
            vals = [r[row_key] for r in blocks if r.get(row_key) is not None]
            run_summary[f"{out_prefix}_mean"] = stats.fmean(vals) if vals else None
            run_summary[f"{out_prefix}_last"] = vals[-1] if vals else None

        # Summary over the selected best iteration in each block
        best_map = {
            "best_orig_l2": "best_orig_l2",
            "best_near_l2": "best_near_l2",
            "best_gap_l2": "best_gap_l2",
        }
        for row_key, out_prefix in best_map.items():
            vals = [r[row_key] for r in blocks if r.get(row_key) is not None]
            run_summary[f"{out_prefix}_mean"] = stats.fmean(vals) if vals else None
            run_summary[f"{out_prefix}_last"] = vals[-1] if vals else None

        run_summary["all_iters_gap_vs_orig"] = (
            run_summary["all_iters_orig_l2_mean"] - run_summary["all_iters_near_l2_mean"]
            if run_summary.get("all_iters_orig_l2_mean") is not None and run_summary.get("all_iters_near_l2_mean") is not None
            else None
        )
        run_summary["best_gap_vs_orig"] = (
            run_summary["best_orig_l2_mean"] - run_summary["best_near_l2_mean"]
            if run_summary.get("best_orig_l2_mean") is not None and run_summary.get("best_near_l2_mean") is not None
            else None
        )

    return blocks, run_summary


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Log files and/or directories (globs allowed).")
    ap.add_argument("--outdir", default="cat_log_compare", help="Output directory.")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels, one per input path after glob expansion.",
    )
    args = ap.parse_args()

    paths = flatten_inputs(args.inputs)
    if not paths:
        raise SystemExit("No log files found.")

    if args.labels is not None and len(args.labels) not in (0, len(paths)):
        raise SystemExit(
            f"--labels must be omitted or provide exactly {len(paths)} labels after glob expansion; got {len(args.labels)}."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    label_map: Dict[str, str] = {}
    if args.labels:
        label_map = {str(path): label for path, label in zip(paths, args.labels)}

    block_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    for path in paths:
        rows, summary = parse_log(path)
        run_name = label_map.get(str(path), path.stem)
        summary["run_name"] = run_name
        for row in rows:
            row["run_name"] = run_name
        block_rows.extend(rows)
        run_rows.append(summary)

    run_rows.sort(key=lambda r: r.get("run_name", r.get("file", "")))
    block_rows.sort(key=lambda r: (r.get("run_name", r.get("file", "")), r["block_idx"]))

    block_fields = [
        "run_name", "file", "path", "block_idx", "n_iters", "iter_first", "iter_last",
        "iters_orig_l2_mean", "iters_orig_l2_last", "iters_near_l2_mean", "iters_near_l2_last",
        "iters_gap_mean", "iters_gap_last", "best_iter", "best_orig_l2", "best_near_l2",
        "best_gap_l2", "total_loss", "away_loss", "toward_loss", "dpo_loss",
        "utility_loss", "attack_loss_mean", "attack_loss_min", "attack_loss_max", "attack_loss_n",
    ]
    run_fields = sorted({k for row in run_rows for k in row.keys()})
    write_csv(outdir / "block_level_summary.csv", block_rows, block_fields)
    write_csv(outdir / "run_level_summary.csv", run_rows, run_fields)

    print("\nRun-level summary: all attack iterations inside each block")
    print("=" * 140)
    header_all = [
        "run_name", "n_blocks",
        "all_iters_orig_l2_mean", "all_iters_near_l2_mean", "all_iters_gap_l2_mean",
        "all_attack_loss_mean", "all_total_loss_mean", "all_utility_loss_mean",
    ]
    print(" | ".join(f"{h:>24}" for h in header_all))
    print("-" * 140)
    for r in run_rows:
        print(
            " | ".join(
                f"{fmt(r.get(h), 6):>24}" if isinstance(r.get(h), float) or r.get(h) is None else f"{str(r.get(h)):>24}"
                for h in header_all
            )
        )

    print("\nRun-level summary: best selected iteration in each block")
    print("=" * 140)
    header_best = [
        "run_name", "n_blocks",
        "best_orig_l2_mean", "best_near_l2_mean", "best_gap_l2_mean",
    ]
    print(" | ".join(f"{h:>24}" for h in header_best))
    print("-" * 140)
    for r in run_rows:
        print(
            " | ".join(
                f"{fmt(r.get(h), 6):>24}" if isinstance(r.get(h), float) or r.get(h) is None else f"{str(r.get(h)):>24}"
                for h in header_best
            )
        )

    print(f"\nWrote:\n  {outdir / 'block_level_summary.csv'}\n  {outdir / 'run_level_summary.csv'}")


if __name__ == "__main__":
    main()

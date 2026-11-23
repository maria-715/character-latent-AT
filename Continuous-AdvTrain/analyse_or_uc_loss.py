"""
Analyse the UC (UltraChat) vs OR (over-refusal) loss diagnostic across a fixed set of
CAT_llama3b_likeZephyr training runs.

For each run this pulls every "Total loss ...; UC loss ...; OR loss ...;" line out of its
slurm-<job>.out log (one line per compute_loss call / micro-batch, in order), and:

  - saves the raw parsed series to a per-run CSV (Continuous-AdvTrain/or_loss_analysis/<name>.csv)
  - computes summary stats per run (OR coverage, trend direction/slope for toward_loss,
    uc_loss and or_loss, early-vs-late window means) into a single summary CSV
  - plots grouped comparisons: same-epsilon overlays across seeds (the direct test for
    whether a seed's or_loss stalls/climbs while toward_loss dives -- the "fork" the
    collapse hypothesis predicts), and same-seed overlays across the epsilon sweep

Usage:
  python analyse_or_uc_loss.py
  python analyse_or_uc_loss.py --logs-dir . --out-dir or_loss_analysis --smooth 25

Run yourself from anywhere; --logs-dir defaults to this script's own directory, where the
slurm-*.out files already live.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Fixed run table, as given: (name, seed, slurm_job_id)
# ============================================================================
RUN_TABLE = [
    ("CAT_llama3b_likeZephyr_42_utility_100_NEW", 42, 114748),
    ("CAT_llama3b_likeZephyr_123_utility_100_NEW", 123, 114751),
    ("CAT_llama3b_likeZephyr_456_utility_100_NEW", 456, 114752),
    ("CAT_llama3b_likeZephyr_eps0075_42_relativelr_utility_100_NEW", 42, 114753),
    ("CAT_llama3b_likeZephyr_eps0150_42_relativelr_utility_100_NEW", 42, 114989),
    ("CAT_llama3b_likeZephyr_eps0300_42_relativelr_utility_100_NEW", 42, 114990),
    ("CAT_llama3b_likeZephyr_eps0600_42_relativelr_utility_100_NEW", 42, 114991),
    ("CAT_llama3b_likeZephyr_eps0600_123_relativelr_utility_100_NEW", 123, 114992),
    ("CAT_llama3b_likeZephyr_eps0600_456_relativelr_utility_100_NEW", 456, 114993),
    ("CAT_llama3b_likeZephyr_42_utility_500_NEW", 42, 114994),
    ("CAT_llama3b_likeZephyr_123_utility_500_NEW", 123, 114995),
    ("CAT_llama3b_likeZephyr_456_utility_500_NEW", 456, 114996),
    ("CAT_llama3b_likeZephyr_eps0075_42_relativelr_utility_500_NEW", 42, 114997),
    ("CAT_llama3b_likeZephyr_eps0150_42_relativelr_utility_500_NEW", 42, 114998),
    ("CAT_llama3b_likeZephyr_eps0300_42_relativelr_utility_500_NEW", 42, 114999),
    ("CAT_llama3b_likeZephyr_eps0600_42_relativelr_utility_500_NEW", 42, 115000),
    ("CAT_llama3b_likeZephyr_eps0600_123_relativelr_utility_500_NEW", 123, 115001),
    ("CAT_llama3b_likeZephyr_eps0600_456_relativelr_utility_500_NEW", 456, 115002),
]

# One line per compute_loss call, e.g.:
#   Total loss 1.156...; Away loss -2.096...; Toward loss 2.331...; DPO loss 0;
#   Utility loss 1.039...; UC loss 1.039...; OR loss N/A; All attack losses [...]; ...
LOSS_LINE_RE = re.compile(
    r"Total loss\s+(?P<total>[-+0-9.eE]+);\s+Away loss\s+(?P<away>[-+0-9.eE]+);\s+"
    r"Toward loss\s+(?P<toward>[-+0-9.eE]+);\s+DPO loss\s+(?P<dpo>[-+0-9.eE]+);\s+"
    r"Utility loss\s+(?P<utility>[-+0-9.eE]+);\s+"
    r"UC loss\s+(?P<uc>N/A|[-+0-9.eE]+);\s+OR loss\s+(?P<or>N/A|[-+0-9.eE]+);"
)


def parse_run_name(name: str) -> Dict[str, object]:
    """Pull epsilon (None = baseline/no eps override) and the utility_100/500 tag out of the run name."""
    m_eps = re.search(r"eps(\d{4})", name)
    eps = int(m_eps.group(1)) / 1000.0 if m_eps else None

    m_util = re.search(r"utility_(\d+)", name)
    utility_variant = m_util.group(1) if m_util else "unknown"

    return {"eps": eps, "eps_label": f"eps{eps:.3f}" if eps is not None else "baseline", "utility_variant": utility_variant}


def parse_log(path: Path) -> pd.DataFrame:
    """Parse every matching loss line into a sequential (iteration-indexed) DataFrame."""
    rows = []
    for i, line in enumerate(path.read_text(errors="replace").splitlines()):
        m = LOSS_LINE_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "iteration": len(rows),
                "away_loss": float(m.group("away")),
                "toward_loss": float(m.group("toward")),
                "utility_loss": float(m.group("utility")),
                "uc_loss": None if m.group("uc") == "N/A" else float(m.group("uc")),
                "or_loss": None if m.group("or") == "N/A" else float(m.group("or")),
            }
        )
    return pd.DataFrame(rows)


def slope(df: pd.DataFrame, col: str) -> Optional[float]:
    """Least-squares slope of `col` vs iteration, dropping missing values. None if <2 points."""
    sub = df[["iteration", col]].dropna()
    if len(sub) < 2:
        return None
    x = sub["iteration"].to_numpy()
    y = sub[col].to_numpy()
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return None
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def window_means(df: pd.DataFrame, col: str, frac: float = 0.1):
    """Mean of `col` over the first `frac` and last `frac` of rows with a non-missing value."""
    sub = df[["iteration", col]].dropna()
    if sub.empty:
        return None, None
    n = max(1, int(len(sub) * frac))
    first = float(sub[col].iloc[:n].mean())
    last = float(sub[col].iloc[-n:].mean())
    return first, last


def summarize_run(name: str, meta: Dict[str, object], df: pd.DataFrame) -> Dict[str, object]:
    n_total = len(df)
    n_uc = int(df["uc_loss"].notna().sum())
    n_or = int(df["or_loss"].notna().sum())

    toward_first, toward_last = window_means(df, "toward_loss")
    uc_first, uc_last = window_means(df, "uc_loss")
    or_first, or_last = window_means(df, "or_loss")

    toward_slope = slope(df, "toward_loss")
    uc_slope = slope(df, "uc_loss")
    or_slope = slope(df, "or_loss")

    # "Fork" heuristic: toward_loss dropping (negative slope) while or_loss is flat/rising
    # (slope >= 0, or missing) is the signature the collapse hypothesis predicts.
    forked = None
    if toward_slope is not None and or_slope is not None:
        forked = bool(toward_slope < 0 and or_slope >= 0)

    return {
        "name": name,
        "seed": meta["seed"],
        "utility_variant": meta["utility_variant"],
        "eps": meta["eps"],
        "eps_label": meta["eps_label"],
        "n_logged_steps": n_total,
        "uc_coverage": n_uc / n_total if n_total else None,
        "or_coverage": n_or / n_total if n_total else None,
        "toward_loss_first": toward_first,
        "toward_loss_last": toward_last,
        "toward_loss_delta": (toward_last - toward_first) if (toward_first is not None and toward_last is not None) else None,
        "toward_loss_slope": toward_slope,
        "uc_loss_first": uc_first,
        "uc_loss_last": uc_last,
        "uc_loss_delta": (uc_last - uc_first) if (uc_first is not None and uc_last is not None) else None,
        "uc_loss_slope": uc_slope,
        "or_loss_first": or_first,
        "or_loss_last": or_last,
        "or_loss_delta": (or_last - or_first) if (or_first is not None and or_last is not None) else None,
        "or_loss_slope": or_slope,
        "possible_collapse_fork": forked,
    }


def rolling_mean_skip_none(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over only the non-missing entries (index-aligned), for smoothing sparse or_loss."""
    return series.dropna().rolling(window=window, min_periods=1).mean()


def bin_by_progress(df: pd.DataFrame, col: str, n_bins: int) -> pd.Series:
    """Mean of `col` within each of n_bins equal-width bins spanning this run's full
    iteration range, ignoring missing values. This puts runs of slightly different length
    (different seeds don't log exactly the same number of steps) onto a common 0-100%
    "training progress" x-axis so they can be averaged together bin-by-bin."""
    max_iter = df["iteration"].max() if not df.empty else None
    if max_iter is None or pd.isna(max_iter) or max_iter <= 0:
        return pd.Series([float("nan")] * n_bins, index=range(n_bins))

    bin_edges = np.linspace(0, max_iter, n_bins + 1)
    sub = df[["iteration", col]].dropna()
    result = pd.Series([float("nan")] * n_bins, index=range(n_bins))
    if sub.empty:
        return result

    bin_idx = np.clip(np.digitize(sub["iteration"], bin_edges[1:-1]), 0, n_bins - 1)
    means = sub.groupby(bin_idx)[col].mean()
    result.update(means)
    return result


def plot_group(runs: List[tuple], title: str, out_path: Path, smooth: int) -> None:
    """One figure, 2 stacked panels (toward_loss, then uc_loss+or_loss), one line per run in `runs`."""
    fig, (ax_toward, ax_utility) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, df) in enumerate(runs):
        color = colors[i % len(colors)]
        ax_toward.plot(
            df["iteration"], df["toward_loss"].rolling(window=smooth, min_periods=1).mean(),
            label=label, color=color, linewidth=1.2,
        )
        uc_smooth = rolling_mean_skip_none(df.set_index("iteration")["uc_loss"], smooth)
        or_smooth = rolling_mean_skip_none(df.set_index("iteration")["or_loss"], smooth)
        ax_utility.plot(uc_smooth.index, uc_smooth.values, label=f"{label} (UC)", color=color, linewidth=1.2, linestyle="-")
        ax_utility.plot(or_smooth.index, or_smooth.values, label=f"{label} (OR)", color=color, linewidth=1.6, linestyle="--")

    ax_toward.set_ylabel("toward_loss")
    ax_toward.grid(alpha=0.3)
    ax_toward.legend(fontsize=8)
    ax_utility.set_ylabel("uc_loss (solid) / or_loss (dashed)")
    ax_utility.set_xlabel("logged step (compute_loss call order)")
    ax_utility.grid(alpha=0.3)
    ax_utility.legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_group_mean_std(runs: List[tuple], title: str, out_path: Path, n_bins: int = 50) -> None:
    """One figure, 2 stacked panels, mean +/- std across the given runs (same eps/regime,
    different seeds), binned by % of training progress so runs of slightly different
    logged-step counts are still directly comparable point-for-point."""
    binned = {
        col: [bin_by_progress(df, col, n_bins) for _, df in runs]
        for col in ("toward_loss", "uc_loss", "or_loss")
    }
    x = (np.arange(n_bins) + 0.5) / n_bins * 100  # bin centers, as % of training progress

    fig, (ax_toward, ax_utility) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    def plot_band(ax, col: str, color: str, label: str, linestyle: str = "-") -> None:
        mat = pd.concat(binned[col], axis=1)
        mean = mat.mean(axis=1, skipna=True)
        std = mat.std(axis=1, ddof=0, skipna=True)
        ax.plot(x, mean, color=color, linewidth=1.6, linestyle=linestyle, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    plot_band(ax_toward, "toward_loss", "tab:blue", "toward_loss")
    plot_band(ax_utility, "uc_loss", "tab:blue", "uc_loss", linestyle="-")
    plot_band(ax_utility, "or_loss", "tab:orange", "or_loss", linestyle="--")

    n_seeds = len(runs)
    ax_toward.set_ylabel("toward_loss")
    ax_toward.grid(alpha=0.3)
    ax_toward.legend(fontsize=8)
    ax_utility.set_ylabel("uc_loss (solid) / or_loss (dashed)")
    ax_utility.set_xlabel(f"training progress (%)  --  mean ± std across {n_seeds} seeds")
    ax_utility.grid(alpha=0.3)
    ax_utility.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", type=Path, default=Path(__file__).resolve().parent, help="Directory containing slurm-<job>.out files")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "or_loss_analysis", help="Output directory for CSVs and plots")
    parser.add_argument("--smooth", type=int, default=25, help="Rolling-average window for plots (default: 25)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    plots_dir = args.out_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    parsed_runs: Dict[str, Dict[str, object]] = {}
    summaries = []

    for name, seed, slurm_id in RUN_TABLE:
        log_path = args.logs_dir / f"slurm-{slurm_id}.out"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found, skipping {name}")
            continue

        meta = parse_run_name(name)
        meta["seed"] = seed
        meta["slurm_id"] = slurm_id

        df = parse_log(log_path)
        if df.empty:
            print(f"WARNING: no 'UC loss / OR loss' lines found in {log_path} (name={name}); skipping")
            continue

        df.to_csv(raw_dir / f"{name}.csv", index=False)
        parsed_runs[name] = {"meta": meta, "df": df}
        summaries.append(summarize_run(name, meta, df))
        print(f"Parsed {name}: {len(df)} logged steps "
              f"(UC coverage {summaries[-1]['uc_coverage']:.1%}, OR coverage {summaries[-1]['or_coverage']:.1%})")

    if not summaries:
        raise SystemExit("No runs parsed successfully -- nothing to analyse.")

    summary_df = pd.DataFrame(summaries).sort_values(["utility_variant", "eps", "seed"], na_position="first")
    summary_csv = args.out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table to {summary_csv}\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary_df[[
        "name", "utility_variant", "eps_label", "seed", "n_logged_steps",
        "or_coverage", "toward_loss_delta", "or_loss_delta", "possible_collapse_fork",
    ]].to_string(index=False))

    # ---- grouped comparison plots ----
    # (a) same epsilon/regime, across seeds -- mean +/- std, the direct fork/bimodality test.
    # Baseline is included here (compared only against its own other seeds, never against
    # the eps sweep) since it has its own training regime and isn't meant to sit on the same
    # epsilon axis as the eps0.075-0.600 runs -- see (b) below, where it's excluded.
    for utility_variant in sorted({m["meta"]["utility_variant"] for m in parsed_runs.values()}):
        for eps_label in sorted({m["meta"]["eps_label"] for m in parsed_runs.values()}):
            group = [
                (f"seed{m['meta']['seed']}", m["df"])
                for name, m in parsed_runs.items()
                if m["meta"]["utility_variant"] == utility_variant and m["meta"]["eps_label"] == eps_label
            ]
            if len(group) < 2:
                continue  # mean/std across seeds only makes sense with >=2 seeds
            regime_note = " (baseline: different training regime, not part of the eps sweep)" if eps_label == "baseline" else ""
            plot_group_mean_std(
                group,
                title=f"utility_{utility_variant}, {eps_label} -- across seeds{regime_note}",
                out_path=plots_dir / f"by_seed_utility{utility_variant}_{eps_label}.png",
            )

    # (b) same seed, across the epsilon sweep only -- baseline deliberately excluded, since
    # it's a different training regime and isn't comparable to the eps0.075-0.600 points.
    for utility_variant in sorted({m["meta"]["utility_variant"] for m in parsed_runs.values()}):
        for seed in sorted({m["meta"]["seed"] for m in parsed_runs.values()}):
            group = [
                (m["meta"]["eps_label"], m["df"])
                for name, m in parsed_runs.items()
                if m["meta"]["utility_variant"] == utility_variant
                and m["meta"]["seed"] == seed
                and m["meta"]["eps"] is not None  # excludes baseline
            ]
            group.sort(key=lambda t: t[0])
            if len(group) < 2:
                continue
            plot_group(
                group,
                title=f"utility_{utility_variant}, seed {seed} -- across epsilon (baseline excluded)",
                out_path=plots_dir / f"by_eps_utility{utility_variant}_seed{seed}.png",
                smooth=args.smooth,
            )

    print(f"\nDone. Raw per-run CSVs: {raw_dir}\nGrouped plots: {plots_dir}\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()

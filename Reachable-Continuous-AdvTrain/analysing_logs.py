"""
plot_reachable_cat_logs_meaningful.py

Reads raw log lines like:
{'global_step': 779, 'loss': -2.1341, 'away_loss': -5.08, 'toward_loss': 0.5,
 'utility_loss': 0.156, 'attack_loss': 80.88, 'index': 2, 'harm_loss': 80.85,
 'reach_loss': 0.00033, 'nn_dist': 5.88e-07, 'nn_entropy': 0.0,
 'affirmative': 0, 'adv_norm': 0.0364, 'epoch': 4.99}

Produces plots that are directly interpretable for your method.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_log_file(log_path: str | Path) -> pd.DataFrame:
    rows: List[Dict] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "{'global_step':" not in line:
                continue
            start = line.find("{")
            end = line.rfind("}")
            if start < 0 or end < 0 or end <= start:
                continue

            payload = line[start : end + 1]
            try:
                rec = ast.literal_eval(payload)
            except Exception:
                continue

            if isinstance(rec, dict) and "global_step" in rec:
                rows.append(rec)

    if not rows:
        raise RuntimeError("No parseable log dictionaries with 'global_step' found.")

    df = pd.DataFrame(rows)

    # Convert obvious numeric columns
    for col in df.columns:
        if col in {"global_step", "epoch"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col != "index":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "index" in df.columns:
        df["is_attack_row"] = df["index"].notna()
    else:
        df["is_attack_row"] = False

    return df


def aggregate_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by global_step.
    - outer losses: mean over all rows at that step
    - attack stats: mean over rows that contain chosen-attack stats
    """
    df = df.copy()

    outer_cols = [c for c in ["loss", "away_loss", "toward_loss", "utility_loss", "dpo_loss"] if c in df.columns]
    attack_cols = [
        c for c in [
            "attack_loss",
            "harm_loss",
            "reach_loss",
            "nn_dist",
            "nn_entropy",
            "affirmative",
            "adv_norm",
            "index",
        ] if c in df.columns
    ]

    outer = df.groupby("global_step", as_index=True)[outer_cols].mean(numeric_only=True)
    outer = outer.add_prefix("outer_")
    outer["n_rows"] = df.groupby("global_step").size()

    if df["is_attack_row"].any():
        att = df[df["is_attack_row"]].copy()
        attack = att.groupby("global_step", as_index=True)[attack_cols].mean(numeric_only=True)
        attack = attack.add_prefix("chosen_")
        attack["attack_rows"] = att.groupby("global_step").size()
        summary = outer.join(attack, how="left")
    else:
        summary = outer.copy()

    summary = summary.reset_index().sort_values("global_step")
    return summary


def add_derived_metrics(summary: pd.DataFrame, lambda_weight: float) -> pd.DataFrame:
    df = summary.copy()

    # Weighted reachability contribution is what actually enters the attack loss
    if "chosen_reach_loss" in df.columns:
        df["chosen_reach_contribution"] = lambda_weight * df["chosen_reach_loss"]
        df["chosen_reach_share"] = df["chosen_reach_contribution"] / (df["chosen_harm_loss"].abs() + 1e-8)

    if "chosen_harm_loss" in df.columns and "chosen_reach_contribution" in df.columns:
        df["chosen_attack_total_reconstructed"] = df["chosen_harm_loss"] + df["chosen_reach_contribution"]

    if "chosen_nn_dist" in df.columns:
        df["chosen_nn_dist_log10"] = np.log10(df["chosen_nn_dist"] + 1e-12)

    if "chosen_adv_norm" in df.columns:
        df["chosen_adv_norm_log10"] = np.log10(df["chosen_adv_norm"] + 1e-12)

    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_lines(
    df: pd.DataFrame,
    cols: List[str],
    title: str,
    outfile: Path,
    window: int = 20,
    ylabel: str | None = None,
) -> None:
    plt.figure(figsize=(11, 5.5))

    plotted = False
    for col in cols:
        if col in df.columns:
            y = df[col].astype(float)
            plt.plot(df["global_step"], smooth(y, window), label=col)
            plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("global_step")
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    outfile: Path,
    alpha: float = 0.45,
) -> None:
    if x not in df.columns or y not in df.columns:
        return

    d = df[[x, y]].dropna()
    if d.empty:
        return

    plt.figure(figsize=(7, 5.5))
    plt.scatter(d[x], d[y], s=12, alpha=alpha)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_hist(df: pd.DataFrame, col: str, title: str, outfile: Path, bins: int = 20) -> None:
    if col not in df.columns:
        return

    d = df[col].dropna()
    if d.empty:
        return

    plt.figure(figsize=(8, 5.5))
    plt.hist(d, bins=bins)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_dual_axis(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    title: str,
    outfile: Path,
    window: int = 20,
    left_label: str | None = None,
    right_label: str | None = None,
) -> None:
    if left_col not in df.columns or right_col not in df.columns:
        return

    d = df[[left_col, right_col, "global_step"]].dropna()
    if d.empty:
        return

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    ax1.plot(d["global_step"], smooth(d[left_col], window), color="C0", label=left_col)
    ax2.plot(d["global_step"], smooth(d[right_col], window), color="C1", label=right_col)

    ax1.set_title(title)
    ax1.set_xlabel("global_step")
    ax1.set_ylabel(left_label or left_col, color="C0")
    ax2.set_ylabel(right_label or right_col, color="C1")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str, help="Path to the training log")
    parser.add_argument("--outdir", type=str, default="meaningful_plots")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--lambda_weight", type=float, default=50.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = parse_log_file(args.log_path)
    summary = aggregate_by_step(raw)
    summary = add_derived_metrics(summary, lambda_weight=args.lambda_weight)

    summary.to_csv(outdir / "step_summary.csv", index=False)

    # ----------------------------
    # Core training plots
    # ----------------------------
    plot_lines(
        summary,
        ["outer_loss", "outer_utility_loss", "outer_away_loss", "outer_toward_loss"],
        "Outer training losses",
        outdir / "outer_losses.png",
        window=args.window,
    )

    # ----------------------------
    # Chosen attack decomposition
    # ----------------------------
    plot_lines(
        summary,
        ["chosen_harm_loss", "chosen_reach_contribution", "chosen_attack_loss"],
        "Chosen attack: harm vs weighted reachability contribution",
        outdir / "chosen_attack_decomposition.png",
        window=args.window,
    )

    plot_lines(
        summary,
        ["chosen_reach_share"],
        "Chosen attack: reachability share of attack objective",
        outdir / "chosen_reach_share.png",
        window=args.window,
        ylabel="weighted reach / |harm|",
    )

    # Optional: show whether the reconstructed attack objective matches stored attack_loss
    if "chosen_attack_total_reconstructed" in summary.columns:
        plot_lines(
            summary,
            ["chosen_attack_loss", "chosen_attack_total_reconstructed"],
            "Chosen attack loss vs reconstructed objective",
            outdir / "attack_loss_check.png",
            window=args.window,
        )

    # ----------------------------
    # Geometry / space analysis
    # ----------------------------
    plot_lines(
        summary,
        ["chosen_nn_dist_log10"],
        "Chosen nearest-neighbor distance (log10 scale)",
        outdir / "chosen_nn_dist_log10.png",
        window=args.window,
        ylabel="log10(nn_dist + 1e-12)",
    )

    plot_lines(
        summary,
        ["chosen_adv_norm"],
        "Chosen perturbation norm",
        outdir / "chosen_adv_norm.png",
        window=args.window,
        ylabel="L2 norm",
    )

    plot_lines(
        summary,
        ["chosen_index"],
        "Chosen inner-loop index",
        outdir / "chosen_index_over_time.png",
        window=args.window,
        ylabel="best inner-loop step",
    )

    plot_hist(
        summary,
        "chosen_index",
        "Distribution of chosen inner-loop index",
        outdir / "chosen_index_hist.png",
        bins=20,
    )

    # ----------------------------
    # Behaviour
    # ----------------------------
    plot_lines(
        summary,
        ["chosen_affirmative"],
        "Chosen attack affirmative signal",
        outdir / "chosen_affirmative.png",
        window=args.window,
        ylabel="affirmative (0/1)",
    )

    # ----------------------------
    # Tradeoff scatter plots
    # ----------------------------
    plot_scatter(
        summary,
        "chosen_reach_share",
        "outer_utility_loss",
        "Utility vs reachability share",
        outdir / "utility_vs_reach_share.png",
    )

    plot_scatter(
        summary,
        "chosen_reach_contribution",
        "outer_utility_loss",
        "Utility vs weighted reachability contribution",
        outdir / "utility_vs_reach_contribution.png",
    )

    plot_scatter(
        summary,
        "chosen_nn_dist_log10",
        "outer_utility_loss",
        "Utility vs log nearest-neighbor distance",
        outdir / "utility_vs_nn_dist_log10.png",
    )

    plot_scatter(
        summary,
        "chosen_adv_norm",
        "outer_utility_loss",
        "Utility vs perturbation norm",
        outdir / "utility_vs_advnorm.png",
    )

    plot_scatter(
        summary,
        "chosen_attack_loss",
        "outer_utility_loss",
        "Utility vs chosen attack loss",
        outdir / "utility_vs_attack_loss.png",
    )

    # ----------------------------
    # Joint views that help interpret the method
    # ----------------------------
    plot_dual_axis(
        summary,
        "chosen_harm_loss",
        "chosen_reach_contribution",
        "Harm loss and weighted reach contribution",
        outdir / "harm_and_reach_dual_axis.png",
        window=args.window,
        left_label="harm_loss",
        right_label="lambda * reach_loss",
    )

    plot_dual_axis(
        summary,
        "chosen_nn_dist_log10",
        "chosen_adv_norm",
        "Log nearest-neighbor distance and perturbation norm",
        outdir / "nn_dist_and_adv_norm_dual_axis.png",
        window=args.window,
        left_label="log10(nn_dist)",
        right_label="adv_norm",
    )

    print(f"Saved CSV: {outdir / 'step_summary.csv'}")
    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
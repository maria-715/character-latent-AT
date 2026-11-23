"""
Plot training loss curves from Continuous-AdvTrain log files.

Two log formats are supported, and this script picks whichever is present:

1. slurm-*.out files: during training, the HF Trainer logging call in
   adversarial_training.py prints one dict per logging step to stdout, e.g.:
     {'global_step': 12, 'loss': 1.23, 'away_loss': ..., 'epoch': 0.5}
   This is the preferred format since it carries global_step/epoch directly.

2. experiments/<run>/run_experiments.log: the hydra-managed logger, which
   survives even if the slurm .out files are gone. It only has the text line
     Total loss 1.23; Away loss ...; Toward loss ...; DPO loss ...; Utility loss ...; All attack losses [...]; ...
   (no global_step/epoch), so the x-axis falls back to "iteration" (the
   row count of logged records) when this format is used.

Usage:
  python plot_loss_curve.py slurm-98496.out
  python plot_loss_curve.py slurm-98496.out --metrics loss away_loss toward_loss utility_loss
  python plot_loss_curve.py run1.out run2.out --labels eps0.05 eps0.1 --smooth 20
  python plot_loss_curve.py logs_dir/ --out loss_curve.png
  python plot_loss_curve.py experiments/CAT_adv_eps005_42_relativelr/run_experiments.log
"""

from __future__ import annotations

import argparse
import ast
import glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

METRIC_DICT_RE = re.compile(r"^\{'global_step':.*\}\s*$")
LOSS_LINE_RE = re.compile(
    r"Total loss\s+(?P<total>[-+0-9.eE]+);\s+Away loss\s+(?P<away>[-+0-9.eE]+);\s+"
    r"Toward loss\s+(?P<toward>[-+0-9.eE]+);\s+DPO loss\s+(?P<dpo>[-+0-9.eE]+);\s+"
    r"Utility loss\s+(?P<utility>[-+0-9.eE]+);\s+All attack losses\s+\[(?P<attack>[^\]]*)\]"
)

DEFAULT_METRICS = ["loss"]
ALL_METRICS = ["loss", "away_loss", "toward_loss", "utility_loss", "dpo_loss", "attack_loss"]


def flatten_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for pattern in ("*.out", "*.log"):
                paths.extend(sorted(Path(x) for x in glob.glob(str(p / pattern)) if Path(x).is_file()))
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


def parse_attack_losses(text: str) -> List[float]:
    vals: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return vals


def parse_log(path: Path) -> List[Dict[str, Any]]:
    dict_rows: List[Dict[str, Any]] = []
    text_rows: List[Dict[str, Any]] = []

    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if METRIC_DICT_RE.match(stripped):
            try:
                record = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                continue
            record["iteration"] = len(dict_rows)
            dict_rows.append(record)
            continue

        m = LOSS_LINE_RE.search(line)
        if m:
            attack_losses = parse_attack_losses(m.group("attack"))
            text_rows.append(
                {
                    "iteration": len(text_rows),
                    "global_step": None,
                    "epoch": None,
                    "loss": float(m.group("total")),
                    "away_loss": float(m.group("away")),
                    "toward_loss": float(m.group("toward")),
                    "dpo_loss": float(m.group("dpo")),
                    "utility_loss": float(m.group("utility")),
                    "attack_loss": (sum(attack_losses) / len(attack_losses)) if attack_losses else None,
                }
            )

    # Prefer the dict-style records (they carry real global_step/epoch); fall
    # back to the "Total loss ..." text line, which is what survives in
    # run_experiments.log once the slurm .out stdout capture is gone.
    return dict_rows if dict_rows else text_rows


def rolling_mean(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return values
    out: List[Optional[float]] = []
    buf: List[float] = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from Continuous-AdvTrain training logs.")
    parser.add_argument("inputs", nargs="+", help="Log file(s), directories, or glob patterns (e.g. slurm-*.out)")
    parser.add_argument("--labels", nargs="*", default=None, help="Run labels, one per input file (in order)")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        choices=ALL_METRICS,
        help=f"Which loss component(s) to plot, one subplot each. Choices: {ALL_METRICS}",
    )
    parser.add_argument(
        "--x",
        choices=["iteration", "global_step", "epoch"],
        default="global_step",
        help="What to use for the x-axis (default: global_step)",
    )
    parser.add_argument("--smooth", type=int, default=1, help="Rolling-average window size (default: 1, no smoothing)")
    parser.add_argument(
        "--name",
        default=None,
        help="Run name: used as the legend label (when --labels isn't given) and appended to the output filename",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output image path (default: loss_curve.png, or loss_curve_<name>.png)"
    )
    args = parser.parse_args()

    paths = flatten_inputs(args.inputs)
    if not paths:
        raise SystemExit("No log files found for the given inputs.")

    if args.labels:
        labels = args.labels
    elif args.name:
        labels = [args.name] * len(paths)
    else:
        labels = [p.stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit(f"Got {len(paths)} log files but {len(labels)} labels; these must match.")

    if args.out is not None:
        out_path = args.out
    elif args.name:
        safe_name = re.sub(r"[^\w\-.]+", "_", args.name).strip("_")
        out_path = Path(f"Continuous-AdvTrain/loss_curves/loss_curve_{safe_name}.png")
    else:
        out_path = Path("Continuous-AdvTrain/loss_curves/loss_curve.png")

    parsed: List[tuple] = []
    for path, label in zip(paths, labels):
        rows = parse_log(path)
        if not rows:
            print(f"Warning: no loss records found in {path}")
            continue
        parsed.append((label, rows))

    if not parsed:
        raise SystemExit("No loss records parsed from any input file.")

    x_key = args.x
    if x_key != "iteration" and any(r.get(x_key) is None for _, rows in parsed for r in rows):
        print(
            f"Note: '{x_key}' isn't available for one or more logs (falling back to the "
            "text-based 'Total loss' line, which has no global_step/epoch); using "
            "'iteration' for the x-axis instead."
        )
        x_key = "iteration"

    n_metrics = len(args.metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(9, 3.5 * n_metrics), sharex=True, squeeze=False)
    axes = [row[0] for row in axes]

    for label, rows in parsed:
        x_vals = [r[x_key] for r in rows]
        for ax, metric in zip(axes, args.metrics):
            y_vals = [r.get(metric) for r in rows]
            y_smoothed = rolling_mean(y_vals, args.smooth)
            ax.plot(x_vals, y_smoothed, label=label, linewidth=1.2)

    for ax, metric in zip(axes, args.metrics):
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel(x_key)
    fig.suptitle(f"Training loss — {args.name}" if args.name else "Training loss")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

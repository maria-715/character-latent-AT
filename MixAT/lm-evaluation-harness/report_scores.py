#!/usr/bin/env python3
"""Print mmlu / arc-easy / arc-challenge accuracy for a list (or range) of slurm job ids.

Usage:
    python report_scores.py 101687 101688-101691 101906
    python report_scores.py 101687-101691 --root /path/to/search

Looks for a file named slurm-<id>.out (searched recursively under --root,
default: the repo root) and pulls the accuracy scores out of the results
dict that lm-evaluation-harness dumps to stdout.
"""

import argparse
import re
from pathlib import Path

METRIC_RE = re.compile(r"'(mmlu|arc_easy|arc_challenge)':\s*\{'acc,none':\s*([0-9.]+)")

TASK_LABELS = [("mmlu", "mmlu"), ("arc_easy", "arc-e"), ("arc_challenge", "arc-c")]


def expand_ids(args):
    ids = []
    for arg in args:
        if "-" in arg:
            start, end = arg.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(arg))
    return ids


def find_slurm_file(root, slurm_id):
    matches = sorted(root.rglob(f"slurm-{slurm_id}.out"))
    return matches[0] if matches else None


def parse_scores(path):
    text = path.read_text(errors="ignore")
    scores = {}
    for task, value in METRIC_RE.findall(text):
        if task not in scores:
            scores[task] = float(value)
    return scores


def format_score(value):
    return f"{value * 100:.2f}".replace(".", ",") + "%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="+", help="slurm ids and/or ranges, e.g. 101687 101688-101691")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]),
                         help="directory to search for slurm-<id>.out files (default: repo root)")
    args = parser.parse_args()

    root = Path(args.root)
    slurm_ids = expand_ids(args.ids)

    lines = []
    for slurm_id in slurm_ids:
        lines.append(str(slurm_id))
        path = find_slurm_file(root, slurm_id)
        if path is None:
            lines.append("file not found")
        else:
            scores = parse_scores(path)
            for task, label in TASK_LABELS:
                if task in scores:
                    lines.append(format_score(scores[task]))
                else:
                    lines.append("n/a")
        lines.append("")

    print("\n".join(lines).rstrip())


if __name__ == "__main__":
    main()

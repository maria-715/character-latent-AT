#!/usr/bin/env python3
"""Print safe compliance rate / unsafe refusal rate / overall accuracy for a list (or range) of slurm job ids.

Usage:
    python report_scores.py 100713 100707-100710 82786
    python report_scores.py 100713-100720 --root /path/to/search

Looks for a file named slurm-<id>.out (searched recursively under --root,
default: the repo root) and pulls the rates out of the "XSTest Results"
block that xstest_eval.sh dumps to stdout.
"""

import argparse
import re
from pathlib import Path

METRICS = [
    ("safe", re.compile(r"Safe compliance rate:\s*([0-9.]+)%")),
    ("unsafe", re.compile(r"Unsafe refusal rate:\s*([0-9.]+)%")),
    ("overall", re.compile(r"Overall accuracy:\s*([0-9.]+)%")),
]


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
    for name, pattern in METRICS:
        match = pattern.search(text)
        if match:
            scores[name] = float(match.group(1))
    return scores


def format_score(value):
    return f"{value:.2f}".replace(".", ",") + "%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="+", help="slurm ids and/or ranges, e.g. 100713 100707-100710")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
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
            for name, _ in METRICS:
                if name in scores:
                    lines.append(format_score(scores[name]))
                else:
                    lines.append("n/a")
        lines.append("")

    print("\n".join(lines).rstrip())


if __name__ == "__main__":
    main()

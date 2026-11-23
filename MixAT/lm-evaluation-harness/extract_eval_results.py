#!/usr/bin/env python3
"""Extract mmlu / arc_easy / arc_challenge accuracy from slurm-*.out log files.

Usage:
    python extract_eval_results.py 101687-101691 101906-101910 102086-102090
    python extract_eval_results.py 75878

Each argument is either a single job id or an inclusive range "start-end".
For every job id, looks for slurm-<id>.out in the same directory as this
script (or --dir) and prints one row per file with the accuracy of mmlu,
arc_easy and arc_challenge (decimal comma instead of decimal point).
Missing files or missing tasks are reported as empty fields.
"""

import argparse
import re
from pathlib import Path

TASKS = ["mmlu", "arc_easy", "arc_challenge"]


def parse_ranges(args):
    job_ids = []
    for arg in args:
        if "-" in arg:
            start, end = arg.split("-", 1)
            job_ids.extend(range(int(start), int(end) + 1))
        else:
            job_ids.append(int(arg))
    return job_ids


def extract_accuracies(text):
    values = {}
    for task in TASKS:
        m = re.search(r"'%s': \{[^}]*?'acc,none': ([0-9.eE+-]+)" % re.escape(task), text)
        values[task] = m.group(1) if m else ""
    return values


def format_number(value):
    return value.replace(".", ",") if value else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranges", nargs="+", help="job id or start-end range, e.g. 101687-101691")
    parser.add_argument("--dir", default=str(Path(__file__).parent), help="directory containing slurm-*.out files")
    args = parser.parse_args()

    job_ids = parse_ranges(args.ranges)
    directory = Path(args.dir)

    for job_id in job_ids:
        path = directory / f"slurm-{job_id}.out"
        print(job_id)
        if not path.exists():
            print("not found")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        values = extract_accuracies(text)
        for task in TASKS:
            print(format_number(values[task]))


if __name__ == "__main__":
    main()

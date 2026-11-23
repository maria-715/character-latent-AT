#!/usr/bin/env python3
import json
from pathlib import Path
import statistics as stats
import csv
import matplotlib.pyplot as plt

INPUT_DIRS = [
    "2026-06-29/19-33-10/0",
    "2026-06-29/19-33-10/1",
    "2026-06-30/02-24-18/1",
    "2026-06-30/02-24-18/3",
]

BASE_IN = Path(__file__).resolve().parent / "outputs"
BASE_OUT = Path("p_harmful_reports")
BASE_OUT.mkdir(parents=True, exist_ok=True)

rows = []

for rel_dir in INPUT_DIRS:
    run_dir = BASE_IN / rel_dir
    file = run_dir / "run.json"
    if not file.exists():
        print(f"Skipping missing file: {file}")
        continue

    with file.open() as f:
        data = json.load(f)

    scores = []
    for run in data.get("runs", []):
        for step in run.get("steps", []):
            sr = step.get("scores", {}).get("strong_reject", {})
            vals = sr.get("p_harmful", [])
            for v in vals:
                if v is not None:
                    scores.append(float(v))

    if not scores:
        print(f"No p_harmful scores found in {file}")
        continue

    scores_sorted = sorted(scores)

    def pct(p):
        k = int(round((len(scores_sorted) - 1) * p))
        return scores_sorted[k]

    safe_name = rel_dir.replace("/", "__")
    out_txt = BASE_OUT / f"{safe_name}_stats.txt"
    out_png = BASE_OUT / f"{safe_name}_p_harmful_distribution.png"

    stats_text = (
        f"file:    {file}\n"
        f"count:   {len(scores)}\n"
        f"min:     {min(scores):.6f}\n"
        f"p25:     {pct(0.25):.6f}\n"
        f"median:  {pct(0.50):.6f}\n"
        f"mean:    {stats.mean(scores):.6f}\n"
        f"p75:     {pct(0.75):.6f}\n"
        f"p90:     {pct(0.90):.6f}\n"
        f"p95:     {pct(0.95):.6f}\n"
        f"max:     {max(scores):.6f}\n"
    )
    out_txt.write_text(stats_text)

    plt.figure(figsize=(8, 4.5))
    plt.hist(scores, bins=50)
    plt.xlabel("p_harmful")
    plt.ylabel("count")
    plt.title(f"Distribution of p_harmful\n{rel_dir}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    rows.append([
        rel_dir,
        len(scores),
        min(scores),
        pct(0.25),
        pct(0.50),
        stats.mean(scores),
        pct(0.75),
        pct(0.90),
        pct(0.95),
        max(scores),
    ])

    print(f"Saved: {out_txt}")
    print(f"Saved: {out_png}")

csv_path = BASE_OUT / "summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run_dir", "count", "min", "p25", "median", "mean", "p75", "p90", "p95", "max"])
    writer.writerows(rows)

print(f"Saved: {csv_path}")
print(f"All outputs are in: {BASE_OUT}")
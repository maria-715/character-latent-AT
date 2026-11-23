"""
Show what makes the unpushed commits so large. Read-only: changes nothing.

Usage (from the Code folder, the root of the git repo):
    python repo_management/git_size_report.py
    python repo_management/git_size_report.py --base origin/main --top 40
"""
import argparse
import subprocess
import sys
from collections import Counter
from pathlib import PurePosixPath

GITHUB_FILE_LIMIT = 100 * 1024 * 1024   # GitHub rejects files above this
GITHUB_WARN = 50 * 1024 * 1024          # GitHub warns above this


def git(*args, input=None):
    r = subprocess.run(["git", *args], input=input, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def mb(n):
    return f"{n / 1024 / 1024:9.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main", help="What is already on GitHub (default origin/main)")
    ap.add_argument("--top", type=int, default=30, help="How many of the largest files to list")
    args = ap.parse_args()

    # Files added or changed in the unpushed commits (as they are in HEAD)
    changed = [l for l in git("diff", "--name-only", "--diff-filter=AM", args.base, "HEAD").splitlines() if l]
    if not changed:
        sys.exit(f"No added/changed files between {args.base} and HEAD.")

    # Sizes of those files in HEAD
    tree = git("ls-tree", "-r", "-l", "HEAD")
    size_of = {}
    for line in tree.splitlines():
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) == 4 and parts[1] == "blob" and parts[3] != "-":
            size_of[path] = int(parts[3])

    files = [(size_of.get(p, 0), p) for p in changed]
    total = sum(s for s, _ in files)

    by_top = Counter()
    by_dir2 = Counter()
    by_ext = Counter()
    n_ext = Counter()
    for s, p in files:
        pp = PurePosixPath(p)
        top = pp.parts[0] + "/" if len(pp.parts) > 1 else "(root files)"
        by_top[top] += s
        d2 = "/".join(pp.parts[:2]) + "/" if len(pp.parts) > 2 else top
        by_dir2[d2] += s
        ext = pp.suffix.lower() or "(none)"
        by_ext[ext] += s
        n_ext[ext] += 1

    print(f"Unpushed commits: {len(files)} added/changed files, {mb(total).strip()} in total\n")

    over = [(s, p) for s, p in files if s > GITHUB_FILE_LIMIT]
    warn = [(s, p) for s, p in files if GITHUB_WARN < s <= GITHUB_FILE_LIMIT]
    if over:
        print(f"!! {len(over)} files are OVER GitHub's 100 MB limit (push will be rejected):")
        for s, p in sorted(over, reverse=True):
            print(f"   {mb(s)}  {p}")
        print()
    if warn:
        print(f"{len(warn)} files are between 50 and 100 MB (GitHub warns):")
        for s, p in sorted(warn, reverse=True):
            print(f"   {mb(s)}  {p}")
        print()

    print("By top-level folder:")
    for d, s in by_top.most_common():
        print(f"   {mb(s)}  {d}")

    print("\nBy sub-folder (largest 20):")
    for d, s in by_dir2.most_common(20):
        print(f"   {mb(s)}  {d}")

    print("\nBy file type:")
    for e, s in by_ext.most_common(15):
        print(f"   {mb(s)}  {e:12} ({n_ext[e]} files)")

    print(f"\nLargest {args.top} files:")
    for s, p in sorted(files, reverse=True)[: args.top]:
        print(f"   {mb(s)}  {p}")


if __name__ == "__main__":
    main()

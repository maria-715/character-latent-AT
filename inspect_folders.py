"""
Inventory of the subfolders in a directory: what each one actually contains.
Read-only: nothing is modified or uploaded. No extra packages needed.

Usage (from inside the experiments folder):
    python ../inspect_folders.py .
    python ../inspect_folders.py . --csv inventory.csv

For each subfolder it reports:
  kind      : full model / LoRA adapter / checkpoints / weights (other) / no weights
  size      : total GB (wandb logs counted separately)
  weights   : number of weight files (.safetensors, .bin, .pt, ...)
  ckpts     : number of checkpoint-* subfolders
  types     : the most common file extensions
  modified  : most recent file modification date
  notes     : wandb logs present, unreadable files, etc.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".onnx", ".gguf", ".msgpack"}
# Files that are not weights even if they end in .bin / .pt
NOT_WEIGHTS = {"training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"}
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def inspect(folder: Path) -> dict:
    ext_sizes = Counter()
    n_files = n_weights = unreadable = 0
    size = wandb_size = 0
    latest = 0.0
    names = set()
    ckpt_dirs = 0

    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        in_wandb = "wandb" in Path(dirpath).relative_to(folder).parts
        if Path(dirpath) == folder:
            ckpt_dirs = sum(1 for d in dirnames if d.lower().startswith("checkpoint"))
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                if os.path.islink(fp):
                    continue
                st = os.stat(fp)
            except OSError:
                unreadable += 1
                continue
            if in_wandb:
                wandb_size += st.st_size
                continue
            n_files += 1
            size += st.st_size
            latest = max(latest, st.st_mtime)
            ext = Path(name).suffix.lower() or "(none)"
            ext_sizes[ext] += st.st_size
            names.add(name.lower())
            if ext in WEIGHT_EXTS and name.lower() not in NOT_WEIGHTS:
                n_weights += 1

    if "adapter_config.json" in names and n_weights:
        kind = "LoRA adapter"
    elif "config.json" in names and n_weights:
        kind = "full model"
    elif ckpt_dirs and n_weights:
        kind = "checkpoints"
    elif n_weights:
        kind = "weights (other)"
    else:
        kind = "no weights"

    notes = []
    if ckpt_dirs:
        notes.append(f"{ckpt_dirs} checkpoint dirs")
    if wandb_size:
        notes.append(f"wandb logs {wandb_size / 1e9:.2f} GB")
    if unreadable:
        notes.append(f"{unreadable} unreadable files")

    top = ", ".join(f"{e} {s / 1e9:.2f}GB" if s >= 1e8 else e for e, s in ext_sizes.most_common(4))
    return {
        "folder": folder.name,
        "kind": kind,
        "size_gb": round(size / 1e9, 2),
        "files": n_files,
        "weights": n_weights,
        "types": top,
        "modified": time.strftime("%Y-%m-%d", time.localtime(latest)) if latest else "",
        "notes": "; ".join(notes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help='Folder to inspect ("." = current folder)')
    ap.add_argument("--csv", help="Also save the table to this CSV file (opens in Excel)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dirs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                  key=lambda p: p.name.lower())
    if not dirs:
        sys.exit(f"No subfolders in {root}")

    print(f"Inspecting {len(dirs)} folders in {root} ...\n")
    rows = []
    for i, d in enumerate(dirs, 1):
        print(f"\r  {i}/{len(dirs)}", end="", flush=True)
        rows.append(inspect(d))
    print("\r" + " " * 20 + "\r", end="")

    w = min(max(len(r["folder"]) for r in rows), 70)
    header = f"{'folder':{w}}  {'kind':15} {'GB':>7} {'files':>6} {'wts':>4}  {'modified':10}  types / notes"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = r["folder"] if len(r["folder"]) <= w else r["folder"][: w - 1] + "…"
        extra = r["types"] + (f"  [{r['notes']}]" if r["notes"] else "")
        print(f"{name:{w}}  {r['kind']:15} {r['size_gb']:7.2f} {r['files']:6} {r['weights']:4}  {r['modified']:10}  {extra}")

    # Summary by kind
    print("\nSummary:")
    by_kind = Counter()
    gb_kind = Counter()
    for r in rows:
        by_kind[r["kind"]] += 1
        gb_kind[r["kind"]] += r["size_gb"]
    for k, n in by_kind.most_common():
        print(f"  {k:15} {n:4} folders  {gb_kind[k]:8.1f} GB")

    # Folders with weights that don't start with the most common prefix
    prefixes = Counter(r["folder"].split("_")[0] for r in rows if r["kind"] != "no weights")
    if prefixes:
        main_prefix = prefixes.most_common(1)[0][0]
        odd = [r["folder"] for r in rows if r["kind"] != "no weights" and not r["folder"].startswith(main_prefix + "_")]
        if odd:
            print(f"\nFolders WITH weights that don't start with '{main_prefix}_' (check these):")
            for o in odd:
                print(f"  - {o}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nSaved table to {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()

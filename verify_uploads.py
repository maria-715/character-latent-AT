"""
Check that every uploaded folder really is on the Hugging Face Hub, byte for byte.
Read-only: nothing is deleted, changed or uploaded.

Usage (from the experiments folder, in hf-upload-env):
    python ../verify_uploads.py . --exclude "*.jsonl"            # full check (checksums)
    python ../verify_uploads.py . --exclude "*.jsonl" --quick    # sizes only, much faster

Use the same --exclude you used for the upload.

For each folder it reports:
  OK        every uploaded file is on the Hub with identical content
  PROBLEM   a file is missing on the Hub or its content differs
and it lists the files that exist ONLY on your computer (excluded files, logs, wandb ...),
which would be lost if you deleted the folder.

Results are also saved to verify_report.json.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

LOG_FILE = Path("upload_log.json")
REPORT_FILE = Path("verify_report.json")
# Files not expected on the Hub (.log files ARE expected, via upload_missing_logs.py)
IGNORE = [
    "*.tmp", "__pycache__/*", "*/__pycache__/*", ".git/*",
    "*.ipynb_checkpoints*", "Thumbs.db", "desktop.ini",
    "wandb/*", "*/wandb/*",
]
SKIP_DIRS = {"wandb", "__pycache__", ".git", ".ipynb_checkpoints"}


def local_files(p: Path, exclude):
    """Returns (uploaded, local_only) where each is a list of (abs_path, rel_path, size)."""
    patterns = IGNORE + list(exclude)
    uploaded, local_only = [], []
    for dirpath, dirnames, filenames in os.walk(p):
        in_skipped = any(part in SKIP_DIRS for part in Path(dirpath).relative_to(p).parts)
        for name in filenames:
            fp = os.path.join(dirpath, name)
            rel = Path(fp).relative_to(p).as_posix()
            try:
                if os.path.islink(fp):
                    continue
                size = os.stat(fp).st_size
            except OSError:
                local_only.append((fp, rel, None))  # unreadable (e.g. wandb shortcut)
                continue
            if in_skipped or any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in patterns):
                local_only.append((fp, rel, size))
            else:
                uploaded.append((fp, rel, size))
    return uploaded, local_only


def sha256_of(fp: str) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(fp: str, size: int) -> str:
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_files(api: HfApi, repo_id: str, prefix: str) -> dict:
    """{rel_path: (size, sha256_or_None, blob_id_or_None)} for files under prefix."""
    out = {}
    path = None if prefix in (".", "") else prefix
    for item in api.list_repo_tree(repo_id, path_in_repo=path, recursive=True, expand=True):
        if not hasattr(item, "size"):  # folders
            continue
        rel = item.path if path is None else item.path[len(path) + 1:]
        lfs = getattr(item, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha is None and isinstance(lfs, dict):
            sha = lfs.get("sha256")
        out[rel] = (item.size, sha, getattr(item, "blob_id", None))
    return out


def fmt_gb(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e8 else f"{b / 1e6:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help='The folder you uploaded from ("." = current folder)')
    ap.add_argument("--exclude", nargs="*", default=[], metavar="PATTERN",
                    help='The same --exclude you used for the upload, e.g. "*.jsonl"')
    ap.add_argument("--quick", action="store_true", help="Compare sizes only (no checksums)")
    args = ap.parse_args()

    if not LOG_FILE.exists():
        sys.exit(f"{LOG_FILE} not found; run this from the folder where you ran the upload.")
    log = json.loads(LOG_FILE.read_text())
    root = Path(args.root).resolve()

    # Folders with weights that are NOT in the upload log
    logged = set(log)
    not_logged = [p.name for p in sorted(root.iterdir()) if p.is_dir() and str(p) not in logged
                  and any(f.suffix == ".safetensors" for f in p.glob("*.safetensors"))]

    api = HfApi()
    report = {}
    n_ok = n_bad = 0
    local_only_total = 0
    print(f"Checking {len(log)} uploaded folders ({'sizes only' if args.quick else 'full checksums'}) ...\n")

    for i, (folder, entry) in enumerate(sorted(log.items()), 1):
        p = Path(folder)
        repo_id, prefix = entry["repo_id"], entry.get("path", ".")
        name = p.name
        problems = []
        if not p.exists():
            print(f"[{i}/{len(log)}] {name}: local folder no longer exists, skipped")
            continue
        try:
            remote = remote_files(api, repo_id, prefix)
        except Exception as e:
            problems.append(f"could not list repo {repo_id}: {type(e).__name__}: {e}")
            remote = {}

        uploaded, local_only = local_files(p, args.exclude)
        for fp, rel, size in uploaded:
            if rel not in remote:
                problems.append(f"MISSING on Hub: {rel}")
                continue
            r_size, r_sha, r_blob = remote[rel]
            if r_size != size:
                problems.append(f"SIZE differs: {rel} (local {size}, Hub {r_size})")
                continue
            if args.quick:
                continue
            if r_sha:
                if sha256_of(fp) != r_sha:
                    problems.append(f"CONTENT differs: {rel}")
            elif r_blob:
                if git_blob_sha1(fp, size) != r_blob:
                    problems.append(f"CONTENT differs: {rel}")

        lo_bytes = sum(s for _, _, s in local_only if s)
        local_only_total += lo_bytes
        status = "PROBLEM" if problems else "OK"
        n_bad += bool(problems)
        n_ok += not problems
        lo_note = f"  | only on PC: {len(local_only)} files, {fmt_gb(lo_bytes)}" if local_only else ""
        print(f"[{i}/{len(log)}] {status:7} {name}  ({len(uploaded)} files checked){lo_note}")
        for msg in problems:
            print(f"           - {msg}")
        report[name] = {
            "repo_id": repo_id, "status": status, "problems": problems,
            "files_checked": len(uploaded),
            "only_on_pc": [rel for _, rel, _ in local_only],
            "only_on_pc_bytes": lo_bytes,
        }

    REPORT_FILE.write_text(json.dumps(report, indent=2))

    # Which kinds of files exist only locally, across all folders
    kinds = {}
    for r in report.values():
        for rel in r["only_on_pc"]:
            key = "wandb/" if rel.startswith("wandb/") or "/wandb/" in rel else (Path(rel).suffix.lower() or "(no ext)")
            kinds[key] = kinds.get(key, 0) + 1

    print("\n" + "=" * 60)
    print(f"OK: {n_ok}    PROBLEM: {n_bad}")
    if not_logged:
        print(f"\n{len(not_logged)} folders with weights are NOT in the upload log (never uploaded):")
        for n in not_logged:
            print(f"  - {n}")
    if kinds:
        print(f"\nFiles that exist ONLY on your PC ({fmt_gb(local_only_total)} in total), by type:")
        for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {k:12} {n:5} files")
        print("These would be lost if you delete the folders.")
    print(f"\nFull report: {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()
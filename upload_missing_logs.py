"""
Add the .log files that upload_models.py skipped to the repos that already exist.
Only .log files are uploaded; nothing else is touched. wandb/ folders are left out.

Usage (from the experiments folder, in hf-upload-env):
    python ../upload_missing_logs.py . --dry-run
    python ../upload_missing_logs.py .

Safe to rerun: folders already done are recorded in logs_uploaded.json and skipped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

LOG_FILE = Path("upload_log.json")
DONE_FILE = Path("logs_uploaded.json")
SKIP_DIRS = {"wandb", "__pycache__", ".git", ".ipynb_checkpoints"}


def find_logs(p: Path):
    out, unreadable = [], 0
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.lower().endswith(".log"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                if os.path.islink(fp):
                    continue
                size = os.stat(fp).st_size
            except OSError:
                unreadable += 1
                continue
            out.append((fp, Path(fp).relative_to(p).as_posix(), size))
    return out, unreadable


def with_retries(fn, attempts=5, base_wait=30):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if i == attempts:
                raise
            print(f"    ! {type(e).__name__}: {e}\n    retrying in {base_wait * i}s ({i}/{attempts})")
            time.sleep(base_wait * i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help='The folder you uploaded from ("." = current folder)')
    ap.add_argument("--dry-run", action="store_true", help="Only list the .log files that would be uploaded")
    args = ap.parse_args()

    if not LOG_FILE.exists():
        sys.exit(f"{LOG_FILE} not found; run this from the folder where you ran the upload.")
    log = json.loads(LOG_FILE.read_text())
    done = json.loads(DONE_FILE.read_text()) if DONE_FILE.exists() else {}

    api = HfApi()
    todo, total_files, total_bytes = [], 0, 0
    for folder, entry in sorted(log.items()):
        p = Path(folder)
        if not p.exists():
            continue
        logs, unreadable = find_logs(p)
        if not logs:
            continue
        size = sum(s for _, _, s in logs)
        status = "done" if folder in done else "todo"
        note = f" ({unreadable} unreadable skipped)" if unreadable else ""
        print(f"[{status}] {p.name:70s} {len(logs):3} .log files, {size / 1e6:8.1f} MB{note}")
        if status == "todo":
            todo.append((folder, entry, logs))
            total_files += len(logs)
            total_bytes += size

    print(f"\n{len(todo)} folders to update, {total_files} .log files, {total_bytes / 1e9:.2f} GB")
    if args.dry_run or not todo:
        return

    failed = 0
    for n, (folder, entry, logs) in enumerate(todo, 1):
        repo_id, prefix = entry["repo_id"], entry.get("path", ".")
        prefix = "" if prefix in (".", "") else prefix.rstrip("/") + "/"
        print(f"\n({n}/{len(todo)}) {Path(folder).name} -> {repo_id}")
        ops = [CommitOperationAdd(path_in_repo=prefix + rel, path_or_fileobj=fp) for fp, rel, _ in logs]
        try:
            for i in range(0, len(ops), 50):
                chunk = ops[i:i + 50]
                with_retries(lambda: api.create_commit(repo_id=repo_id, operations=chunk,
                                                       commit_message="Add training logs"))
        except Exception as e:
            failed += 1
            print(f"    ✗ FAILED: {type(e).__name__}: {e} (rerun to retry)")
            continue
        done[folder] = {"repo_id": repo_id, "files": [rel for _, rel, _ in logs],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        DONE_FILE.write_text(json.dumps(done, indent=2))
        print(f"    ✓ {len(logs)} .log files added")
        time.sleep(1)

    print(f"\nFinished. {len(todo) - failed} folders updated" + (f", {failed} failed (rerun to retry)" if failed else ""))


if __name__ == "__main__":
    main()

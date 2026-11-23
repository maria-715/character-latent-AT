"""
Bulk-upload local model folders to the Hugging Face Hub.

Each selected subfolder of ROOT becomes one repo:  <namespace>/<prefix><folder-name>
(or, with --single-repo NAME, a subfolder inside one repo).

Folder selection:
  By default, only folders that contain weight files (.safetensors, .bin, .pt, ...) are uploaded.
  --include-prefix CAT_         only upload folders whose names start with this
  --skip-prefix all analyse     skip folders whose names start with these (default)
Both are case-insensitive.

File selection:
  --exclude "*.jsonl"           leave these files out of every upload

Usage (from inside the folder that holds the models, use "." as the path):
    hf auth login
    python upload_models.py . --dry-run
    python upload_models.py . --exclude "*.jsonl" --dry-run
    python upload_models.py . --exclude "*.jsonl"
    python upload_models.py . --single-repo cat-checkpoints

Already-uploaded folders are recorded in upload_log.json and skipped on rerun.
Folders that fail are recorded in upload_failed.json and retried on the next run.
"""
import argparse
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

LOG_FILE = Path("upload_log.json")
FAILED_FILE = Path("upload_failed.json")
IGNORE = [
    "*.tmp", "__pycache__/*", "*/__pycache__/*", ".git/*",
    "*.ipynb_checkpoints*", "Thumbs.db", "desktop.ini",
    "wandb/*", "*/wandb/*",          # Weights & Biases run logs
]
SKIP_DIRS = {"wandb", "__pycache__", ".git", ".ipynb_checkpoints"}
DEFAULT_SKIP_PREFIXES = ["all", "analyse"]
WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".onnx", ".gguf", ".msgpack"}
NOT_WEIGHTS = {"training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"}


def has_weights(p: Path) -> bool:
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if Path(name).suffix.lower() in WEIGHT_EXTS and name.lower() not in NOT_WEIGHTS:
                return True
    return False


def repo_safe(name: str) -> str:
    """HF repo names: letters, digits, '-', '_', '.'; max 96 chars."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return name[:96] or "model"


def folder_size_gb(p: Path, exclude=()) -> tuple[float, int]:
    """Total size in GB of what would be uploaded; returns (size, unreadable_file_count)."""
    total, unreadable = 0, 0
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            try:
                fp = os.path.join(dirpath, name)
                rel = Path(fp).relative_to(p).as_posix()
                if any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                    continue
                if os.path.islink(fp):
                    continue
                total += os.stat(fp).st_size
            except OSError:
                unreadable += 1
    return total / 1e9, unreadable


def files_to_upload(p: Path, exclude=()) -> tuple[list[tuple[str, str]], int]:
    """(local_path, path_in_folder) pairs to upload, never touching skipped dirs.
    Unreadable files (e.g. broken wandb shortcuts on Windows) are skipped, not fatal."""
    patterns = list(IGNORE) + list(exclude)
    out, unreadable = [], 0
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            fp = os.path.join(dirpath, name)
            rel = Path(fp).relative_to(p).as_posix()
            if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in patterns):
                continue
            try:
                if os.path.islink(fp) or not os.path.isfile(fp):
                    continue
                os.stat(fp)
            except OSError:
                unreadable += 1
                continue
            out.append((fp, rel))
    return out, unreadable


def upload_files(api, repo_id: str, files, path_in_repo: str, message: str, batch: int = 50) -> None:
    """Upload a list of files in commits of at most `batch` files."""
    prefix = "" if path_in_repo in (".", "") else path_in_repo.rstrip("/") + "/"
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        ops = [CommitOperationAdd(path_in_repo=prefix + rel, path_or_fileobj=fp) for fp, rel in chunk]
        part = f" (part {i // batch + 1})" if len(files) > batch else ""
        with_retries(lambda: api.create_commit(repo_id=repo_id, operations=ops,
                                               commit_message=message + part))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def with_retries(fn, attempts=5, base_wait=30):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # network errors, rate limits (HTTP 429), timeouts
            if i == attempts:
                raise
            wait = base_wait * i
            print(f"    ! {type(e).__name__}: {e}\n    retrying in {wait}s ({i}/{attempts})")
            time.sleep(wait)


def model_card(folder: Path) -> str:
    """Minimal README for a LoRA adapter, linking it to its base model."""
    base = None
    cfg = folder / "adapter_config.json"
    try:
        base = json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (OSError, ValueError):
        pass
    meta = ["---", "library_name: peft", "tags:", "- lora", "- adversarial-training"]
    if base and "/" in base and not Path(base).is_absolute() and ":" not in base:
        meta.insert(1, f"base_model: {base}")
    meta.append("---")
    body = [
        f"# {folder.name}",
        "",
        "LoRA adapter from Master's thesis experiments on adversarial training for LLM robustness.",
    ]
    if base:
        body += ["", f"Base model: `{base}`"]
    return "\n".join(meta + [""] + body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help='Folder containing one subfolder per model ("." = current folder)')
    ap.add_argument("--namespace", help="User or org to upload to (default: your account)")
    ap.add_argument("--prefix", default="", help="Prefix added to every repo name")
    ap.add_argument("--public", action="store_true", help="Create public repos (default: private)")
    ap.add_argument("--single-repo", metavar="NAME", help="Put every model as a subfolder of one repo")
    ap.add_argument("--include-prefix", nargs="*", default=[],
                    help="Only upload folders whose names start with these (e.g. CAT_)")
    ap.add_argument("--skip-prefix", nargs="*", default=DEFAULT_SKIP_PREFIXES,
                    help="Skip folders whose names start with these (default: all analyse)")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="PATTERN",
                    help='Leave matching files out of every upload, e.g. --exclude "*.jsonl"')
    ap.add_argument("--include-no-weights", action="store_true",
                    help="Also upload folders that contain no weight files (skipped by default)")
    ap.add_argument("--collection", metavar="SLUG",
                    help='Add each uploaded model to this collection, e.g. "maria/cat-models-66f1a2b3c4d5e6"')
    ap.add_argument("--no-card", action="store_true",
                    help="Don't generate a README.md for folders that lack one")
    ap.add_argument("--limit", type=int, help="Upload at most this many models (e.g. --limit 1 to test)")
    ap.add_argument("--dry-run", action="store_true", help="List what would be uploaded, with sizes")
    ap.add_argument("--pause", type=float, default=2.0, help="Seconds to wait between models")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    include = tuple(s.lower() for s in args.include_prefix)
    skip = tuple(s.lower() for s in args.skip_prefix)

    def selected(p: Path) -> bool:
        n = p.name.lower()
        if include and not n.startswith(include):
            return False
        if skip and n.startswith(skip):
            return False
        return args.include_no_weights or has_weights(p)

    all_dirs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name.lower())
    folders = [p for p in all_dirs if selected(p)]
    skipped = [p for p in all_dirs if not selected(p)]

    print(f"Root: {root}")
    if skipped:
        print(f"\nSkipping {len(skipped)} folders:")
        for p in skipped:
            print(f"  - {p.name}")
    if not folders:
        sys.exit("\nNo model folders selected.")

    api = HfApi()
    namespace = args.namespace or api.whoami()["name"]
    private = not args.public
    log = load_json(LOG_FILE)
    failed = load_json(FAILED_FILE)

    print()
    total = 0.0
    todo = []
    for f in folders:
        size, unreadable = folder_size_gb(f, args.exclude)
        total += size
        status = "done" if str(f) in log else "todo"
        note = f"  ({unreadable} unreadable files will be skipped)" if unreadable else ""
        print(f"[{status}] {f.name:70s} {size:8.2f} GB{note}")
        if status == "todo":
            todo.append((f, size))
    print(f"\n{len(folders)} model folders, {total:.1f} GB total, {len(todo)} left to upload "
          f"({'private' if private else 'public'} -> {namespace})")
    if args.collection:
        print(f"Collection: https://huggingface.co/collections/{args.collection}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: uploading only {len(todo)} this run")
    if args.dry_run:
        return

    def add_to_collection(repo_id: str) -> bool:
        if not args.collection:
            return False
        try:
            with_retries(lambda: api.add_collection_item(
                args.collection, item_id=repo_id, item_type="model", exists_ok=True), attempts=3, base_wait=10)
            return True
        except Exception as e:
            print(f"    ! could not add to collection ({type(e).__name__}: {e}); rerun to retry")
            return False

    # Models uploaded in an earlier run but not yet in the collection
    if args.collection:
        pending = [k for k, v in log.items() if not v.get("in_collection")]
        if pending:
            print(f"\nAdding {len(pending)} previously uploaded models to the collection ...")
            for k in pending:
                if add_to_collection(log[k]["repo_id"]):
                    log[k]["in_collection"] = True
            save_json(LOG_FILE, log)

    if args.single_repo:
        single_id = f"{namespace}/{repo_safe(args.single_repo)}"
        with_retries(lambda: api.create_repo(single_id, private=private, exist_ok=True))

    for n, (f, size) in enumerate(todo, 1):
        if args.single_repo:
            repo_id, path_in_repo = single_id, repo_safe(f.name)
        else:
            repo_id, path_in_repo = f"{namespace}/{repo_safe(args.prefix + f.name)}", "."
        print(f"\n({n}/{len(todo)}) {f.name}  ->  {repo_id}/{'' if path_in_repo == '.' else path_in_repo}  [{size:.2f} GB]")

        try:
            if not args.single_repo:
                with_retries(lambda: api.create_repo(repo_id, private=private, exist_ok=True))
            files, unreadable = files_to_upload(f, args.exclude)
            if unreadable:
                print(f"    (skipping {unreadable} unreadable files)")
            if not files:
                raise RuntimeError("no uploadable files found")
            upload_files(api, repo_id, files, path_in_repo, f"Upload {f.name}")
            if not args.single_repo and not args.no_card and not (f / "README.md").exists():
                card = model_card(f).encode("utf-8")
                with_retries(lambda: api.upload_file(
                    path_or_fileobj=card, path_in_repo="README.md",
                    repo_id=repo_id, commit_message="Add model card"))
        except Exception as e:
            print(f"    ✗ FAILED: {type(e).__name__}: {e}\n    (moving on; rerun the script to retry)")
            failed[str(f)] = {"repo_id": repo_id, "error": f"{type(e).__name__}: {e}",
                              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_json(FAILED_FILE, failed)
            continue

        log[str(f)] = {"repo_id": repo_id, "path": path_in_repo,
                       "size_gb": round(size, 3), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "in_collection": add_to_collection(repo_id)}
        save_json(LOG_FILE, log)
        if failed.pop(str(f), None) is not None:
            save_json(FAILED_FILE, failed)
        print("    ✓ done")
        time.sleep(args.pause)

    print(f"\nFinished. Uploaded log: {LOG_FILE.resolve()}")
    if failed:
        print(f"{len(failed)} folders failed; see {FAILED_FILE.resolve()} and rerun to retry.")


if __name__ == "__main__":
    main()
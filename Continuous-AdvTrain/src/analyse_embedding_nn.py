#!/usr/bin/env python3
"""Nearest-neighbour distance statistics for token embedding spaces.

For each model, computes the L2 distance from every vocabulary embedding to its
nearest other vocab embedding, then reports summary statistics (mean, std,
percentiles, min, max). This gives a baseline sense of how tightly packed a
model's token embedding space is -- useful context when interpreting
adversarial-attack epsilon budgets defined in that same embedding space
(compare against src/embedding_attack.py, which tracks the same L2 quantity
for perturbed tokens during an attack).

Run examples:
    python src/analyse_embedding_nn.py --out-dir embedding_nn_analysis
    python src/analyse_embedding_nn.py --model-paths HuggingFaceH4/zephyr-7b-beta --out-dir embedding_nn_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from model_utils import get_embed_weights, load_model_and_tokenizer

DEFAULT_MODEL_PATHS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


@torch.no_grad()
def compute_nearest_neighbours(embed_weights: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For every row, find the L2 distance and id of its nearest other row.

    Chunked over rows so we never materialize the full [vocab_size, vocab_size]
    distance matrix at once (mirrors the chunking approach in
    embedding_attack.py::compute_nearest_neighbour_info_chunked).
    """

    weights = embed_weights.float()
    vocab_size = weights.shape[0]
    norms = (weights ** 2).sum(dim=1)

    nearest_dist = torch.empty(vocab_size, dtype=torch.float32, device=weights.device)
    nearest_id = torch.empty(vocab_size, dtype=torch.long, device=weights.device)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = weights[start:end]

        dist_sq = norms[start:end].unsqueeze(1) + norms.unsqueeze(0) - 2 * chunk @ weights.T
        dist_sq = torch.clamp(dist_sq, min=0.0)

        row_idx = torch.arange(end - start, device=weights.device)
        col_idx = torch.arange(start, end, device=weights.device)
        dist_sq[row_idx, col_idx] = float("inf")  # exclude self

        chunk_min = dist_sq.min(dim=1)
        nearest_dist[start:end] = torch.sqrt(chunk_min.values)
        nearest_id[start:end] = chunk_min.indices

    return nearest_dist.cpu(), nearest_id.cpu()


def safe_convert_ids_to_tokens(tokenizer, ids: list[int]) -> list[str]:
    vocab_len = len(tokenizer)
    in_range = [token_id for token_id in ids if token_id < vocab_len]
    converted = dict(zip(in_range, tokenizer.convert_ids_to_tokens(in_range)))
    return [converted.get(token_id, "<no_token_text>") for token_id in ids]


def summarize(distances: pd.Series, embed_norms: pd.Series) -> dict:
    out = {
        "vocab_size": int(len(distances)),
        "nn_distance_mean": float(distances.mean()),
        "nn_distance_std": float(distances.std()),
        "nn_distance_min": float(distances.min()),
        "nn_distance_max": float(distances.max()),
        "embed_norm_mean": float(embed_norms.mean()),
        "embed_norm_std": float(embed_norms.std()),
    }
    for q in QUANTILES:
        out[f"nn_distance_p{int(q * 100)}"] = float(distances.quantile(q))
    return out


def analyse_model(model_path: str, chunk_size: int, dtype: str, out_dir: Path) -> dict:
    print(f"[INFO] Loading {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, dtype=dtype)
    embed_weights = get_embed_weights(model)
    vocab_size, embed_dim = embed_weights.shape
    print(f"[INFO] embedding matrix: vocab_size={vocab_size}, dim={embed_dim}")

    nearest_dist, nearest_id = compute_nearest_neighbours(embed_weights, chunk_size)
    with torch.no_grad():
        embed_norms = torch.norm(embed_weights.float(), p=2, dim=1).detach().cpu()

    token_ids = list(range(vocab_size))
    token_strs = safe_convert_ids_to_tokens(tokenizer, token_ids)
    nearest_id_list = nearest_id.tolist()
    nearest_token_strs = safe_convert_ids_to_tokens(tokenizer, nearest_id_list)

    df = pd.DataFrame(
        {
            "token_id": token_ids,
            "token": token_strs,
            "embed_norm": embed_norms.numpy(),
            "nn_distance": nearest_dist.numpy(),
            "nn_token_id": nearest_id_list,
            "nn_token": nearest_token_strs,
        }
    )

    model_slug = model_path.split("/")[-1]
    model_out_dir = out_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(model_out_dir / "per_token_nn_distances.csv", index=False)
    print(f"[OK] Saved {model_out_dir / 'per_token_nn_distances.csv'}")

    summary = summarize(df["nn_distance"], df["embed_norm"])
    summary["model_path"] = model_path

    with (model_out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved {model_out_dir / 'summary.json'}")

    del model
    torch.cuda.empty_cache()

    return summary


def build_report(summaries: list[dict], out_dir: Path) -> None:
    lines = ["# Embedding nearest-neighbour distance report\n"]
    lines.append(
        "For each model, this reports the L2 distance from every vocab embedding "
        "to its nearest other vocab embedding. Use the percentiles here as a "
        "reference scale when interpreting an attack epsilon defined in the same "
        "embedding space (e.g. src/embedding_attack.py, preproj_norm.py).\n"
    )

    summary_df = pd.DataFrame(summaries).set_index("model_path")
    lines.append(summary_df.to_string())

    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    summary_df.to_csv(out_dir / "combined_summary.csv")
    print(f"[OK] Saved {out_dir / 'report.txt'}")
    print(f"[OK] Saved {out_dir / 'combined_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding-space nearest-neighbour distance statistics")
    parser.add_argument("--model-paths", nargs="+", default=DEFAULT_MODEL_PATHS)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048, help="Rows processed per distance-matrix chunk")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_path in args.model_paths:
        summary = analyse_model(model_path, args.chunk_size, args.dtype, out_dir)
        summaries.append(summary)

    build_report(summaries, out_dir)
    print("[DONE]")


if __name__ == "__main__":
    main()

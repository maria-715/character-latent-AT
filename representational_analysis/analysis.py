"""
analysis.py

Phase-1 analysis primitives only: load extracted representations into an
xarray.DataArray, and compute per-layer, per-prompt cosine similarity / L2
distance between any two checkpoints (not just "vs. base" -- eps-vs-eps,
seed-vs-seed at fixed eps, untrimmed-vs-trimmed, or vs.-base are all the same
function call with different checkpoint ids).

No aggregation across prompts happens anywhere in this module -- that's left
to report/plotting code, once there's real data to decide how (mean/median,
std vs. IQR, which layers to summarize, etc). Deliberately minimal for the
first pass; drift-from-base convenience wrappers, multi-checkpoint trajectory
comparisons, seed-divergence aggregation, and PCA/centroid analysis are phase
2, built on top of these two primitives once there's something to look at.

Memory note: load_representations() loads the full (checkpoint, prompt, layer,
hidden_dim) array for every checkpoint matched into RAM as float16. For the
full 44-checkpoint manifest across all three prompt sets this is on the order
of 10+ GB per pooling mode -- use the `checkpoint_ids=` and/or `dataset=`
filters to scope down if that's a problem.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DEFAULT_REPR_DIR = HERE / "representations"


def _load_checkpoint_metadata(repr_dir):
    metas = {}
    for meta_path in sorted(repr_dir.glob("*/metadata.json")):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        metas[meta["id"]] = meta
    return metas


def load_representations(repr_dir=DEFAULT_REPR_DIR, checkpoint_ids=None, dataset=None, pooling="last_token"):
    """
    Returns an xarray.DataArray with dims (checkpoint, prompt, layer, hidden_dim).
    Checkpoint coordinates carry every checkpoints.yaml field (trajectory, eps,
    relative_lr, seed, or_variant, hyperparam_config, base_model) for filtering
    via .sel()/.where()/.groupby().

    checkpoint_ids: optional list to restrict which extracted checkpoints to load.
    dataset: optional prompt-set name (see prompt_sets.yaml) to restrict to a
             single prompt dataset instead of all extracted prompts.
    pooling: "last_token" or "mean_pooled".
    """
    repr_dir = Path(repr_dir)
    if pooling not in ("last_token", "mean_pooled"):
        raise ValueError(f"pooling must be 'last_token' or 'mean_pooled', got {pooling!r}")

    metas = _load_checkpoint_metadata(repr_dir)
    if checkpoint_ids is not None:
        missing = [cid for cid in checkpoint_ids if cid not in metas]
        if missing:
            raise ValueError(f"checkpoint_ids not found under {repr_dir}: {missing}")
        metas = {cid: metas[cid] for cid in checkpoint_ids}

    keep_prompt_ids = None
    if dataset is not None:
        registry = pd.read_parquet(repr_dir / "prompts_registry.parquet")
        keep_prompt_ids = set(registry.loc[registry["dataset"] == dataset, "prompt_id"])
        if not keep_prompt_ids:
            raise ValueError(f"dataset {dataset!r} not found in {repr_dir / 'prompts_registry.parquet'}")

    coord_fields = ["trajectory", "eps", "relative_lr", "seed", "or_variant", "hyperparam_config", "base_model"]
    coords_by_field = {f: [] for f in coord_fields}
    per_checkpoint_arrays = []
    ckpt_ids_used = []
    common_prompt_ids = None

    for ckpt_id, meta in metas.items():
        ckpt_dir = repr_dir / ckpt_id
        with np.load(ckpt_dir / "hidden_states.npz") as npz:
            arr = npz[pooling]  # (n_layers, n_prompts, hidden_dim)
        prompt_index = pd.read_parquet(ckpt_dir / "prompt_index.parquet")
        prompt_ids = prompt_index["prompt_id"].tolist()

        if keep_prompt_ids is not None:
            mask = [pid in keep_prompt_ids for pid in prompt_ids]
            arr = arr[:, mask, :]
            prompt_ids = [pid for pid, m in zip(prompt_ids, mask) if m]

        if common_prompt_ids is None:
            common_prompt_ids = prompt_ids
        elif prompt_ids != common_prompt_ids:
            raise ValueError(
                f"Checkpoint {ckpt_id!r} has a different prompt set/ordering than earlier checkpoints "
                "in this load -- extraction must have used a different prompt_sets.yaml across runs. "
                "Re-run extract_representations.py --force with a consistent prompt_sets.yaml."
            )

        per_checkpoint_arrays.append(arr)
        ckpt_ids_used.append(ckpt_id)
        for f in coord_fields:
            coords_by_field[f].append(meta.get(f))

    if not per_checkpoint_arrays:
        raise ValueError("No checkpoints matched -- check checkpoint_ids/dataset and that extraction has run.")

    stacked = np.stack(per_checkpoint_arrays, axis=0)  # (checkpoint, layer, prompt, hidden_dim)
    stacked = np.moveaxis(stacked, 1, 2)  # -> (checkpoint, prompt, layer, hidden_dim)

    return xr.DataArray(
        stacked,
        dims=("checkpoint", "prompt", "layer", "hidden_dim"),
        coords={
            "checkpoint": ckpt_ids_used,
            "prompt": common_prompt_ids,
            "layer": np.arange(stacked.shape[2]),
            **{f: ("checkpoint", coords_by_field[f]) for f in coord_fields},
        },
        name=f"hidden_states_{pooling}",
    )


def cosine_similarity_per_layer(da, ckpt_id_a, ckpt_id_b):
    """(layer, prompt) cosine similarity between two checkpoints' representations,
    for every prompt and every layer present in `da`. Generic two-checkpoint
    comparison -- pass any two ids (eps-vs-eps, seed-vs-seed, untrimmed-vs-trimmed,
    vs.-base, ...)."""
    a = da.sel(checkpoint=ckpt_id_a).values.astype(np.float32)  # (prompt, layer, hidden_dim)
    b = da.sel(checkpoint=ckpt_id_b).values.astype(np.float32)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(denom > 0, dot / denom, np.nan)
    return sim.T  # (layer, prompt)


def l2_distance_per_layer(da, ckpt_id_a, ckpt_id_b):
    """(layer, prompt) L2 distance between two checkpoints' representations,
    for every prompt and every layer present in `da`. Same generic-pair semantics
    as cosine_similarity_per_layer."""
    a = da.sel(checkpoint=ckpt_id_a).values.astype(np.float32)
    b = da.sel(checkpoint=ckpt_id_b).values.astype(np.float32)
    dist = np.linalg.norm(a - b, axis=-1)  # (prompt, layer)
    return dist.T  # (layer, prompt)


if __name__ == "__main__":
    da = load_representations()
    print(da)
    ids = list(da.coords["checkpoint"].values)
    if len(ids) >= 2:
        sim = cosine_similarity_per_layer(da, ids[0], ids[1])
        dist = l2_distance_per_layer(da, ids[0], ids[1])
        print(f"cosine_similarity_per_layer({ids[0]}, {ids[1]}): shape={sim.shape} mean={np.nanmean(sim):.4f}")
        print(f"l2_distance_per_layer({ids[0]}, {ids[1]}): shape={dist.shape} mean={np.nanmean(dist):.4f}")

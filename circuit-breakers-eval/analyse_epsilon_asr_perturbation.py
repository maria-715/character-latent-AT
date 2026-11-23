#!/usr/bin/env python3

"""
Perturbation size vs. ASR, per CAT-trained model.

For each slurm log of a Soft-Prompt (softopt) attack run, extracts:
- eps_multiplier: the CAT training epsilon as it appears in configs/dir names
  (e.g. 0.05, 0.075, 0.4, 0.6). This is NOT a raw embedding-space distance --
  it's a multiplier: embedding_attack.py:123 sets
  self.eps = eps_multiplier * embedding_norm, where embedding_norm is the
  mean L2 norm of that specific model's own token embeddings
  (embedding_attack.py:119). Kept here for logistics/bookkeeping (it's what
  the run directory names and hydra configs use).
- eps_raw: self.eps itself, i.e. the actual per-token L2 ball radius CAT
  trained against, in raw embedding-space units. Read directly from the
  "using eps: X" value logged in experiments/<run>/run_experiments.log --
  not recomputed. This is what's comparable to the attack's perturbation
  size, and it is NOT model-independent: the same eps_multiplier corresponds
  to a different eps_raw for different base models (e.g. 0.075 is raw
  0.0133 for Zephyr-7B-beta but raw 0.0442 for Llama-3-8B-Instruct, because
  their embedding_norm differs).
- embedding_norm: the model's mean per-token embedding L2 norm, also from
  run_experiments.log. Kept for transparency, since it's what turns
  eps_multiplier into eps_raw.
- base_model: which model was attacked, since eps_multiplier is only
  comparable within the same base_model.
- eps_note: set (and eps_multiplier/embedding_norm/eps_raw left as NaN) when
  no epsilon could be resolved -- most commonly because the run has no CAT
  adapter (a raw/baseline model eval). The attack still ran and still has a
  perturbation size in that case; it's only the epsilon that's inapplicable.
- asr: the "Evaluation score" printed at the end of the slurm log.
- perturbation_l2_mean / perturbation_l2_std: mean and sample std dev (ddof=1)
  of the "Avg token distance" logged for every saved solution -- the
  PER-TOKEN L2 distance moved from initialization, averaged over the soft
  prompt's tokens. This is the metric comparable to eps_raw, which is also a
  per-token L2 ball radius (embedding_attack.py's project_l2 clips each
  token's perturbation independently, norm computed with dim=-1). The
  alternative "L2 distance" field softopt.py also logs is a single norm over
  all soft-prompt tokens combined, not per-token, and is not used here.

Reads only files that already exist (slurm-*.out, experiments/*/.hydra/*.yaml,
experiments/*/run_experiments.log). Does not rerun attacks or training.
Outputs are written to circuit-breakers-eval/epsilon_asr_perturbation/.

Every slurm log in SLURM_LOGS is accounted for in the output, whether or not
it was usable: run_status.csv lists every one with status ok/skipped and,
for skipped ones, why (missing file, no distance instrumentation, no
resolvable epsilon, etc.) -- not just a console print that disappears.

Usage:

python analyse_epsilon_asr_perturbation.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Only slurm logs that contain the per-solution distance instrumentation
# ("Saved solution ... L2 distance ..."). Most softopt slurm logs predate this
# instrumentation and have no perturbation-size data at all.
# slurm-104884.out is an incomplete duplicate of slurm-105126.out (same
# adapter/output_dir, run cut short) and is excluded.
SLURM_LOGS = [f"slurm-{i}.out" for i in [116787,116503,116789,116128,116129,116792]]

SOLUTION_PATTERN = re.compile(
    r"(?:Saved solution|Final solution) at iter \d+ \| "
    r"L2 distance: [0-9.eE+-]+ \| "
    r"Avg token distance: ([0-9.eE+-]+)"
)
# softopt.py restarts its inner iteration counter at 0 on every call to
# run() (one call per behavior), so a fresh "Iter: 0 Loss: ..." line marks
# the start of a new behavior's trajectory. Used to group the up-to-
# multiple_gen solutions logged per behavior, so we can tell the first
# threshold-crossing (closest thing to a margin) apart from later ones
# (which tend to have drifted further, since optim_embeds is unconstrained).
ITER_ZERO_PATTERN = re.compile(r"^Iter: 0 Loss: ")
ARG_PATTERN = re.compile(r"^([a-zA-Z_]+): (.*)$")
SCORE_PATTERN = re.compile(r"Evaluation score: ([0-9.eE+-]+)")
EMBEDDING_EPS_PATTERN = re.compile(
    r"L2 norm of embedding weights equals ([0-9.eE+-]+)\. "
    r"eps multiplier is: ([0-9.eE+-]+)\. "
    r"using eps: ([0-9.eE+-]+)\."
)


def parse_slurm_args(lines: list[str]) -> dict:
    args = {}
    for line in lines:
        m = ARG_PATTERN.match(line.strip())
        if m:
            args.setdefault(m.group(1), m.group(2))
    return args


def read_eps_multiplier(experiment_dir: Path) -> float | None:
    """eps as it appears in configs/dir names, from the CAT run's saved hydra config."""
    overrides_path = experiment_dir / ".hydra" / "overrides.yaml"
    if overrides_path.exists():
        m = re.search(r"adversarial\.eps=([0-9.]+)", overrides_path.read_text())
        if m:
            return float(m.group(1))

    config_path = experiment_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        m = re.search(r"adversarial:\n(?:.*\n)*?\s+eps:\s*([0-9.]+)", config_path.read_text())
        if m:
            return float(m.group(1))

    return None


def read_eps_raw(experiment_dir: Path) -> tuple[float, float, float] | None:
    """
    Reads the actual embedding_norm/eps_multiplier/eps_raw triple logged by
    embedding_attack.py's __init__ (self.eps = eps_multiplier * embedding_norm),
    from experiments/<run>/run_experiments.log. Returns
    (embedding_norm, eps_multiplier, eps_raw) or None if not found.
    """
    log_path = experiment_dir / "run_experiments.log"
    if not log_path.exists():
        return None

    m = EMBEDDING_EPS_PATTERN.search(log_path.read_text(errors="replace"))
    if not m:
        return None

    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def skipped(slurm_name: str, reason: str, **known) -> dict:
    """A row for a slurm log that could not be used, recording why."""
    print(f"[WARN] {slurm_name}: skipped -- {reason}")
    row = {"slurm_log": slurm_name, "status": "skipped", "skip_reason": reason}
    row.update(known)
    return row


def process_run(slurm_path: Path, circuit_breakers_dir: Path) -> dict:
    """
    Always returns a row -- either a full data row (status="ok") or a
    skipped row recording why (status="skipped", skip_reason=...), so that
    every entry in SLURM_LOGS is accounted for in the saved output, not just
    the ones that happened to work.

    A run is only "skipped" when the attack data itself (ASR, perturbation
    distances) is unavailable. Runs with no CAT adapter (raw/baseline model
    evals) still ran the same Soft-Prompt attack and still have a
    perturbation norm -- there's just no CAT epsilon to compare it against.
    Those are "ok" with eps_multiplier/embedding_norm/eps_raw left as NaN and
    eps_note explaining why.
    """
    slurm_name = slurm_path.name

    if not slurm_path.exists():
        return skipped(slurm_name, "slurm log file does not exist")

    lines = slurm_path.read_text(errors="replace").splitlines()
    args = parse_slurm_args(lines)

    output_dir = args.get("output_dir")
    adapter = args.get("adapter")
    if output_dir is None or adapter is None:
        return skipped(slurm_name, "missing output_dir/adapter args in slurm log (not a softopt eval run?)")

    run_id = Path(output_dir).name

    eps_multiplier = embedding_norm = eps_raw = None
    if adapter == "None":
        # --adapter was not passed: a raw/baseline model eval with no CAT
        # training and therefore no epsilon to compare against. The attack
        # itself still ran, so ASR/perturbation are still collected below.
        eps_note = "no adapter (raw/baseline model, not CAT-trained)"
    else:
        # adapter is "../experiments/<run_id>/final_model", relative to circuit-breakers-eval/
        experiment_dir = (circuit_breakers_dir / adapter).resolve().parent
        eps_multiplier = read_eps_multiplier(experiment_dir)
        if eps_multiplier is None:
            eps_note = f"could not resolve eps_multiplier from {experiment_dir}/.hydra/"
        else:
            eps_raw_info = read_eps_raw(experiment_dir)
            if eps_raw_info is None:
                eps_note = f"no run_experiments.log (or no matching eps line) under {experiment_dir}"
            else:
                embedding_norm, eps_multiplier_from_log, eps_raw = eps_raw_info
                eps_note = None
                if abs(eps_multiplier_from_log - eps_multiplier) > 1e-6:
                    print(f"[WARN] {run_id}: eps_multiplier mismatch between hydra config ({eps_multiplier}) "
                          f"and run_experiments.log ({eps_multiplier_from_log})")

    score_matches = [SCORE_PATTERN.search(l) for l in lines]
    score_matches = [m for m in score_matches if m]
    if not score_matches:
        return skipped(
            slurm_name, "no 'Evaluation score' line found (run incomplete?)",
            run_id=run_id, eps_multiplier=eps_multiplier, embedding_norm=embedding_norm,
            eps_raw=eps_raw, eps_note=eps_note,
        )
    asr = float(score_matches[-1].group(1))

    token_distances = [float(m.group(1)) for m in (SOLUTION_PATTERN.search(l) for l in lines) if m]
    if not token_distances:
        return skipped(
            slurm_name, "no 'Saved/Final solution ... Avg token distance' lines found "
                        "(predates the distance instrumentation, or not a softopt run)",
            run_id=run_id, eps_multiplier=eps_multiplier, embedding_norm=embedding_norm,
            eps_raw=eps_raw, eps_note=eps_note, asr=asr,
        )

    # Group per-solution distances by behavior so we can compare the first
    # saved solution (smallest perturbation sufficient to cross the loss
    # threshold) against the last one (how far the trajectory had drifted by
    # the time it stopped), rather than only the all-snapshots average above.
    behavior_groups: list[list[float]] = []
    for line in lines:
        if ITER_ZERO_PATTERN.match(line):
            behavior_groups.append([])
        m = SOLUTION_PATTERN.search(line)
        if m:
            if not behavior_groups:
                behavior_groups.append([])
            behavior_groups[-1].append(float(m.group(1)))
    behavior_groups = [g for g in behavior_groups if g]

    first_distances = [g[0] for g in behavior_groups]
    last_distances = [g[-1] for g in behavior_groups]

    return {
        "slurm_log": slurm_name,
        "status": "ok",
        "skip_reason": None,
        "run_id": run_id,
        "base_model": args.get("model_name_or_path"),
        "eps_multiplier": eps_multiplier,
        "embedding_norm": embedding_norm,
        "eps_raw": eps_raw,
        "eps_note": eps_note,
        "asr": asr,
        "perturbation_l2_mean": float(np.mean(token_distances)),
        "perturbation_l2_std": float(np.std(token_distances, ddof=1)),
        "perturbation_l2_first_mean": float(np.mean(first_distances)),
        "perturbation_l2_last_mean": float(np.mean(last_distances)),
        "num_solutions": len(token_distances),
    }


def main():
    circuit_breakers_dir = Path(__file__).resolve().parent
    out_dir = circuit_breakers_dir / "epsilon_asr_perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [process_run(circuit_breakers_dir / name, circuit_breakers_dir) for name in SLURM_LOGS]
    all_df = pd.DataFrame(rows)

    status_path = out_dir / "run_status.csv"
    all_df.to_csv(status_path, index=False)
    print(f"\n[OK] Saved {status_path}  ({len(all_df)} slurm logs considered)")

    n_ok = (all_df["status"] == "ok").sum()
    n_skipped = (all_df["status"] == "skipped").sum()
    print(f"    ok: {n_ok}, skipped: {n_skipped}")
    if n_skipped:
        print("    skip reasons:")
        for reason, count in all_df.loc[all_df["status"] == "skipped", "skip_reason"].value_counts().items():
            print(f"      [{count}] {reason}")

    df = all_df[all_df["status"] == "ok"].drop(columns=["status", "skip_reason"]).sort_values(
        "eps_raw", na_position="first"
    ).reset_index(drop=True)

    csv_path = out_dir / "epsilon_asr_perturbation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved {csv_path}\n")
    print(df.to_string(index=False))

    if df.empty:
        print("\n[WARN] No usable runs, skipping correlations and plot.")
        return

    n_baseline = df["eps_raw"].isna().sum()
    if n_baseline:
        print(f"\n{n_baseline} row(s) have no eps (baseline/no-adapter models) -- "
              f"included in the perturbation data, excluded from eps correlations below.")

    df_eps = df.dropna(subset=["eps_raw"])
    r_perturbation_asr = df["perturbation_l2_mean"].corr(df["asr"])
    r_eps_raw_asr = df_eps["eps_raw"].corr(df_eps["asr"])
    r_eps_multiplier_asr = df_eps["eps_multiplier"].corr(df_eps["asr"])
    print(f"\nPearson r(perturbation_l2_mean, asr) = {r_perturbation_asr:.3f}  (n={len(df)}, includes baseline rows)")
    print(f"Pearson r(eps_raw, asr)               = {r_eps_raw_asr:.3f}  (n={len(df_eps)})")
    print(f"Pearson r(eps_multiplier, asr)        = {r_eps_multiplier_asr:.3f}  (n={len(df_eps)}, mixes base models, see docstring)")

    # One plot: perturbation L2 distance per model (mean +/- std dev), in eps_raw order.
    # Baseline (no-eps) rows sort first and are labeled "baseline" instead of a raw eps value.
    def xtick_label(row) -> str:
        if pd.isna(row.eps_raw):
            return f"baseline (no eps)\n{row.run_id}"
        return f"eps_raw={row.eps_raw:.4f} ({row.eps_multiplier})\n{row.run_id}"

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(df)), df["perturbation_l2_mean"], yerr=df["perturbation_l2_std"], capsize=3)
    plt.xticks(
        range(len(df)),
        [xtick_label(row) for row in df.itertuples()],
        rotation=90, fontsize=6,
    )
    plt.ylabel("L2 distance/token (mean +/- std)")
    plt.xlabel("Model (raw eps used in training, eps_multiplier in config / run id)")
    plt.title("Soft-Prompt per-token perturbation L2 distance per CAT-trained model")
    plt.tight_layout()
    plt.savefig(out_dir / "perturbation_l2_per_model.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_dir / 'perturbation_l2_per_model.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Post-hoc CAT bridging diagnostic: D_train (from training traces) vs D_posthoc (final checkpoint).

Standalone. Does not modify CAT training code, checkpoints, traces or existing analysis outputs.
Everything it writes goes under --out_root (default: experiments/posthoc_bridging_D_mixat/).

Definition (corrected CAT D):
    D_call = mean over perturbed tokens of ||delta_K||_2, where delta_K is the tensor returned by
             EmbeddingSpaceAttack.attack() and used in the CAT outer loss.
    D_run  = mean of D_call over attack calls.
    Radius = mean +/- sample SD (ddof=1) over the 3 training seeds.

Modes
    list       (CPU)  resolve + validate the 12 Llama-3 MixAT-recipe runs, print their configs, and
                      check that the trace-based chosen-D reproduces the paper's Table 3 rows.
                      Writes nothing.
    plan       (CPU)  build the fixed diagnostic plan (which away examples, grouped into which attack
                      calls) once, shared by all checkpoints.  Needs the Llama-3 tokenizer + data.
    attack     (GPU)  one checkpoint (--run_id): reproduce the training-time attack condition and
                      record delta_K for every planned call.  Resumable; skips completed runs.
    summarize  (CPU)  training-time A/B from traces + post-hoc C -> tables.

Training-time condition reproduced in `attack` mode (all verified against the pinned library source,
trl 0.8.3 / peft 0.9.0 / transformers 4.41.2 / accelerate 0.27.2):
  * base model loaded exactly as in adversarial_training_loop (model_utils.load_model_and_tokenizer,
    4-bit NF4, dtype from the run config);
  * EmbeddingSpaceAttack constructed from the (still bf16) embedding matrix *before*
    prepare_model_for_kbit_training, which is what SFTTrainer(peft_config=...) did in training; the
    kbit prep then casts non-quantized params to fp32 in place, so the attack's embed_weights and
    delta are fp32 exactly as in training.  eps / lr / embedding norm are asserted equal to the values
    logged in the run's run_experiments.log;
  * trained adapter loaded read-only (PeftModel.from_pretrained, is_trainable=False);
  * forward wrapped by accelerate's own prepare_model with mixed_precision="fp16" (training used
    fp16=True -> native AMP autocast + convert_outputs_to_fp32);
  * model.train(): Trainer.training_step calls model.train() before compute_loss, so LoRA dropout
    (p=0.1) was active during every training attack;
  * examples tokenized with trl's own SFTTrainer._prepare_non_packed_dataloader, collated with the
    CAT collator, split with AdversarialULTrainer.split_inputs, attacked with attack.attack().
Dropout reproducibility: torch RNG (CPU+CUDA) is re-seeded to (dropout_seed + call_index) immediately
before every attack call, so call i sees the same dropout stream on every checkpoint and a resumed run
is identical to an uninterrupted one (up to GPU kernel non-determinism).
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CAT_DIR = REPO / "Continuous-AdvTrain"
SRC = CAT_DIR / "src"

RUN_ID_RE = re.compile(r"^CAT_llama3b_(0075|0150|0300|0600)_relativelr(_(123|456))?_otherparams$")
EXPECTED_S = [0.075, 0.15, 0.3, 0.6]
EXPECTED_SEEDS = [42, 123, 456]
# Paper Table 3, Llama 3 (MixAT) rows: D (old "chosen" definition), mean, SD.
PAPER_TABLE3 = {0.075: (0.03539, 0.00045), 0.15: (0.05884, 0.00232), 0.3: (0.09078, 0.00334), 0.6: (0.13500, 0.00616)}
# Config keys allowed to differ between the 12 runs.
VARYING_KEYS = {("adversarial", "eps"), ("training", "seed"), ("path", "logging_path")}

EPS_LOG_RE = re.compile(
    r"L2 norm of embedding weights equals ([0-9.eE+-]+)\. eps multiplier is: ([0-9.eE+-]+)\. "
    r"using eps: ([0-9.eE+-]+)\. the learning rate is: ([0-9.eE+-]+) \(relative_lr=(\w+)\)"
)


# ============================================================ utilities

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


# ============================================================ run resolution (list mode + shared)

def resolve_runs(manifest_path, experiments_dir):
    from omegaconf import OmegaConf

    man = pd.read_csv(manifest_path)
    man = man[man.run_id.map(lambda r: bool(RUN_ID_RE.match(r)))].copy()
    if len(man) != 12:
        raise RuntimeError(f"Expected 12 MixAT-recipe runs in {manifest_path}, found {len(man)}")

    runs, ref_flat = [], None
    for _, m in man.iterrows():
        run_dir = experiments_dir / m.run_id
        if not (run_dir / ".hydra" / "config.yaml").exists():
            raise FileNotFoundError(f"{run_dir}/.hydra/config.yaml not found -- pass --experiments_dir pointing at "
                                    f"the directory that contains the CAT run folders")
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        flat = flatten(cfg)
        if ref_flat is None:
            ref_flat = flat
        diff = {k for k in set(flat) | set(ref_flat) if flat.get(k) != ref_flat.get(k)} - VARYING_KEYS
        if diff:
            raise RuntimeError(f"{m.run_id}: config differs from the other runs in {sorted(diff)}")
        if not math.isclose(float(cfg["adversarial"]["eps"]), float(m.epsilon)):
            raise RuntimeError(f"{m.run_id}: manifest eps {m.epsilon} != config eps {cfg['adversarial']['eps']}")
        if int(cfg["training"]["seed"]) != int(m.seed):
            raise RuntimeError(f"{m.run_id}: manifest seed {m.seed} != config seed {cfg['training']['seed']}")

        log_hits = EPS_LOG_RE.findall((run_dir / "run_experiments.log").read_text(errors="replace"))
        if not log_hits or len(set(log_hits)) != 1:
            raise RuntimeError(f"{m.run_id}: expected one consistent eps line in run_experiments.log, got {set(log_hits)}")
        emb_norm, eps_mult, eps_raw, lr, rel = log_hits[0]
        if not math.isclose(float(eps_mult), float(m.epsilon)):
            raise RuntimeError(f"{m.run_id}: logged eps multiplier {eps_mult} != {m.epsilon}")

        adapter_dir = run_dir / "final_model"
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        if adapter_cfg["base_model_name_or_path"] != cfg["path"]["model_path"]:
            raise RuntimeError(f"{m.run_id}: adapter base model {adapter_cfg['base_model_name_or_path']} != config")
        if not math.isclose(adapter_cfg["lora_dropout"], cfg["peft"]["lora_dropout"]):
            raise RuntimeError(f"{m.run_id}: adapter lora_dropout differs from config")

        runs.append(dict(
            run_id=m.run_id, s=float(m.epsilon), seed=int(m.seed), run_dir=run_dir,
            adapter_dir=adapter_dir, trace_dir=experiments_dir / m.run_id / "attack_traces", cfg=cfg,
            log_embedding_norm=float(emb_norm), log_eps=float(eps_raw), log_lr=float(lr), log_relative_lr=rel,
            lora_dropout=adapter_cfg["lora_dropout"], lora_targets=sorted(adapter_cfg["target_modules"]),
        ))

    got = sorted((r["s"], r["seed"]) for r in runs)
    want = sorted((s, sd) for s in EXPECTED_S for sd in EXPECTED_SEEDS)
    if got != want:
        raise RuntimeError(f"Resolved (s, seed) pairs {got} != expected {want}")
    return sorted(runs, key=lambda r: (r["s"], r["seed"]))


def check_output_root(out_root, runs):
    out_root = out_root.resolve()
    for r in runs:
        rd = r["run_dir"].resolve()
        if out_root == rd or rd in out_root.parents or (out_root / r["run_id"]).resolve() == rd:
            raise RuntimeError(f"--out_root {out_root} would write into training run dir {rd}; refusing.")
    return out_root


# ============================================================ training-time D from traces (A, B)

def trace_final_D(trace_dir):
    """Per-call final-iterate D (step K-1 record == tensor returned by attack()) in call order."""
    rows = []
    for r in read_jsonl(trace_dir / "step_summaries.jsonl"):
        rows.append((r["trainer_global_step"], r["attack_id"], r["step"], r["loss"], r["mean_l2_to_original"]))
    df = pd.DataFrame(rows, columns=["g", "attack_id", "step", "loss", "D"])
    # Same restart handling as analyse_attack_traces.keep_last_run_segment.
    g = df["g"].to_numpy()
    starts = np.nonzero(np.r_[False, g[1:] < g[:-1]])[0]
    n_restarts = len(starts)
    if n_restarts:
        df = df.iloc[starts[-1]:].reset_index(drop=True)
    order = df.drop_duplicates("attack_id")["attack_id"].tolist()
    k_last = int(df["step"].max())
    fin = df[df.step == k_last].set_index("attack_id").loc[order].reset_index()
    chosen = df.loc[df.groupby("attack_id")["loss"].idxmin()].set_index("attack_id").loc[order, "D"].to_numpy()
    return fin, chosen, n_restarts


def training_AB(trace_dir):
    fin, chosen, n_restarts = trace_final_D(trace_dir)
    n = len(fin)
    late = fin.iloc[int(math.floor(0.9 * n)):]
    return dict(
        n_train_calls=n, n_restarts=n_restarts,
        D_train_whole=float(fin.D.mean()),
        D_train_last10pct=float(late.D.mean()),
        n_train_calls_last10pct=len(late),
        loss_final_train_last10pct=float(late.loss.mean()),  # L(delta_{K-1}), same as posthoc losses[-1]
        D_chosen_whole=float(np.mean(chosen)),
    )


# ============================================================ mode: list

def mode_list(args):
    runs = resolve_runs(args.manifest, args.experiments_dir)
    print(f"Resolved {len(runs)} runs from {args.manifest}\n")
    c0 = runs[0]["cfg"]
    print("Shared config (identical across all 12, verified):")
    print(f"  base model       : {c0['path']['model_path']} (model_name={c0['path']['model_name']})")
    print(f"  attack           : iters K={c0['adversarial']['iters']}, opt={c0['adversarial']['opt_config']}, "
          f"init_type={c0['adversarial']['init_type']}, relative_lr={c0['adversarial']['relative_lr']}, "
          f"attack_type={c0['adversarial']['attack_type']}")
    print(f"  peft             : {c0['peft']}")
    print(f"  bnb              : {c0['bnb']}")
    print(f"  mixed precision  : fp16={c0['training']['fp16']} bf16={c0['training']['bf16']}; "
          f"dtype={c0['trainer_hparams']['dtype']}; trainer_type={c0['trainer_hparams']['trainer_type']}")
    print(f"  data             : {c0['dataset']['dataset_name']} ({c0['dataset']['data_path']}), "
          f"probabilities={c0['dataset']['probabilities']}, max_seq_length={c0['sfttrainer']['max_seq_length']}, "
          f"per_device_bs={c0['training']['per_device_train_batch_size']}\n")

    rows = []
    for r in runs:
        ab = training_AB(r["trace_dir"])
        rows.append(dict(
            run_id=r["run_id"], s=r["s"], seed=r["seed"], eps_logged=r["log_eps"], inner_lr_logged=r["log_lr"],
            emb_norm_logged=r["log_embedding_norm"], lora_dropout=r["lora_dropout"],
            adapter=str(r["adapter_dir"]),
            adapter_ok=(r["adapter_dir"] / "adapter_model.safetensors").exists(),
            traces_ok=all((r["trace_dir"] / f).exists() for f in
                          ("step_summaries.jsonl", "final_token_records.jsonl", "attack_metadata.jsonl")),
            n_train_calls=ab["n_train_calls"], restarts=ab["n_restarts"], D_chosen=ab["D_chosen_whole"],
        ))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))

    print("\nPaper Table 3 check (old 'chosen' D recomputed from these 12 traces):")
    for s, g in df.groupby("s"):
        m, sd = g.D_chosen.mean(), g.D_chosen.std(ddof=1)
        pm, psd = PAPER_TABLE3[s]
        print(f"  s={s:<6} recomputed {m:.5f} +/- {sd:.5f}   paper {pm:.5f} +/- {psd:.5f}   "
              f"{'MATCH' if round(m, 5) == pm and round(sd, 5) == psd else 'MISMATCH'}")


# ============================================================ mode: plan

def load_cat_modules():
    sys.path.insert(0, str(SRC))
    import adversarial_training  # noqa: F401  (imports data, model_utils, embedding_attack)
    import data
    import embedding_attack
    import model_utils
    return adversarial_training, data, embedding_attack, model_utils


def resolve_data_path(cfg, override):
    if override:
        p = Path(override)
    else:
        # Training ran from Continuous-AdvTrain/ with a path relative to it (see train_CAT.sh).
        p = (CAT_DIR / cfg["dataset"]["data_path"]).resolve()
    return str(p) + os.sep  # data.py concatenates strings onto data_path


def load_away_dataset(data_mod, cfg, tokenizer, data_path):
    """Away (dataset_id 0) rows exactly as data.load_adversarial_training_data builds them.

    utility_data=None skips the UltraChat interleave, which does not touch away rows; the current
    data.py still appends OR rows for llama-3, which are filtered out here.
    """
    train_data, _ = data_mod.load_adversarial_training_data(
        data_path=data_path,
        utility_data=None,
        probabilities=cfg["dataset"]["probabilities"],
        model_name=cfg["path"]["model_name"],
        tokenizer=tokenizer,
        stopping_strategy=cfg["dataset"]["stopping_strategy"],
        diverse_safe_answers=cfg["dataset"]["diverse_safe_answers"],
        restricted_trainingset_size=None,
    )
    return train_data.filter(lambda ex: ex["dataset_id"] == 0)


def plan_rows_fingerprint(plan):
    return hashlib.sha256(json.dumps(
        [plan["call_sizes"], plan["call_row_indices"], plan["rows"]], sort_keys=True).encode()).hexdigest()


def mode_plan(args):
    runs = resolve_runs(args.manifest, args.experiments_dir)
    out_root = check_output_root(args.out_root, runs)
    plan_path = out_root / "diagnostic_plan.json"
    if plan_path.exists():
        print(f"[skip] plan already exists: {plan_path}")
        return

    from transformers import AutoTokenizer

    _, data_mod, _, _ = load_cat_modules()
    cfg = runs[0]["cfg"]
    # The plan only records which away rows (behaviour, target) go into which call; the tokenizer is
    # needed solely because load_adversarial_training_data tokenizes Safe_Model. Tokenization for the
    # attack itself happens in attack mode with the training tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"]["model_path"], use_fast=False)
    away = load_away_dataset(data_mod, cfg, tokenizer, resolve_data_path(cfg, args.data_path))

    # Empirical away-sub-batch-size distribution of the training attack calls of these 12 runs.
    sizes_all = []
    for r in runs:
        sizes_all += [rec["config"]["batch_size"] for rec in read_jsonl(r["trace_dir"] / "attack_metadata.jsonl")]
    values, counts = np.unique(np.array(sizes_all), return_counts=True)
    probs = counts / counts.sum()

    rng = np.random.default_rng(args.plan_seed)
    call_sizes = rng.choice(values, size=args.n_calls, p=probs).astype(int).tolist()
    total = int(sum(call_sizes))
    if total > len(away):
        raise RuntimeError(f"Plan needs {total} examples but only {len(away)} away rows exist")
    perm = rng.permutation(len(away))[:total].tolist()
    call_rows, i = [], 0
    for sz in call_sizes:
        call_rows.append(perm[i:i + sz])
        i += sz

    plan = dict(
        version=1, plan_seed=args.plan_seed, n_calls=args.n_calls, n_away_rows=len(away),
        call_size_distribution={int(v): float(p) for v, p in zip(values, probs)},
        n_training_calls_used_for_distribution=len(sizes_all),
        call_sizes=call_sizes, call_row_indices=call_rows,
        rows={str(j): {"User": away[j]["User"][0], "Model": away[j]["Model"][0]} for j in perm},
    )
    plan["fingerprint"] = plan_rows_fingerprint(plan)
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(plan_path, plan)
    print(f"[ok] wrote {plan_path}: {args.n_calls} calls, {total} examples, "
          f"{len({plan['rows'][str(j)]['User'] for j in perm})} distinct behaviours, fingerprint {plan['fingerprint'][:16]}")


def load_plan(out_root):
    plan = json.loads((out_root / "diagnostic_plan.json").read_text(encoding="utf-8"))
    if plan_rows_fingerprint(plan) != plan["fingerprint"]:
        raise RuntimeError("diagnostic_plan.json fingerprint mismatch (file edited?)")
    return plan


# ============================================================ mode: attack

def training_perturbed_sequences(trace_dir):
    """Set of perturbed-token-id sequences (one per away example) seen at the final inner step in training."""
    seqs = {}
    for r in read_jsonl(trace_dir / "final_token_records.jsonl"):
        seqs.setdefault((r["attack_id"], r["batch_index"]), []).append((r["token_position"], r["original_token_id"]))
    return {tuple(t for _, t in sorted(v)) for v in seqs.values()}


def lora_fingerprint(model, torch):
    s1 = s2 = 0.0
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                s1 += float(p.double().sum())
                s2 += float(p.double().pow(2).sum())
                n += p.numel()
    return {"n_params": n, "sum": s1, "sumsq": s2}


def mode_attack(args):
    import torch
    from accelerate import Accelerator
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    from trl import SFTTrainer

    runs = {r["run_id"]: r for r in resolve_runs(args.manifest, args.experiments_dir)}
    if args.run_id not in runs:
        raise RuntimeError(f"--run_id must be one of {sorted(runs)}")
    run = runs[args.run_id]
    out_root = check_output_root(args.out_root, list(runs.values()))
    out_dir = out_root / run["run_id"]
    summary_path = out_dir / "posthoc_run_summary.json"
    calls_path = out_dir / "posthoc_calls.jsonl"
    if summary_path.exists():
        print(f"[skip] {summary_path} exists")
        return
    plan = load_plan(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    adversarial_training, data_mod, embedding_attack, model_utils = load_cat_modules()
    cfg = run["cfg"]
    adapter_file = run["adapter_dir"] / "adapter_model.safetensors"
    adapter_sha_before = sha256_file(adapter_file)

    # ---- model + tokenizer exactly as adversarial_training_loop (and the SFTTrainer peft path)
    bnb = BitsAndBytesConfig(**cfg["bnb"])
    model, tokenizer = model_utils.load_model_and_tokenizer(
        cfg["path"]["model_path"], bnb_config=bnb,
        padding_side=cfg["trainer_hparams"]["padding_side"], dtype=cfg["trainer_hparams"]["dtype"],
    )
    model_name = cfg["path"]["model_name"]
    formatting_func, collator, response_key = data_mod.get_prompt_formatting_func_and_collator(model_name, tokenizer)

    embed_weights = model_utils.get_embed_weights(model)
    attack_cfg = {k: v for k, v in cfg["adversarial"].items() if k not in ("attack_type", "tracking")}
    attack_cfg["opt_config"] = dict(attack_cfg["opt_config"])  # constructor rescales lr in place
    attack = embedding_attack.EmbeddingSpaceAttack(
        embed_weights, response_key=response_key, tokenizer=tokenizer,
        tracking_config={"enabled": False},  # tracking is observational only; D computed below
        **attack_cfg,
    )
    for name, got, want in (("embedding_norm", attack.embedding_norm, run["log_embedding_norm"]),
                            ("eps", attack.eps, run["log_eps"]),
                            ("lr", attack.opt_config["lr"], run["log_lr"])):
        if not math.isclose(got, want, rel_tol=1e-9, abs_tol=0.0):
            raise RuntimeError(f"{name}: reconstructed {got!r} != training log {want!r}")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if embed_weights.dtype != torch.float32:
        raise RuntimeError(f"embed_weights dtype {embed_weights.dtype}, expected fp32 after kbit prep (as in training)")
    model = PeftModel.from_pretrained(model, str(run["adapter_dir"]), is_trainable=False)
    accelerator = Accelerator(mixed_precision="fp16" if cfg["training"]["fp16"] else "no")
    model = accelerator.prepare_model(model, evaluation_mode=True)

    # ---- weight safety: nothing trainable, no grads, fingerprint to compare at the end
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Some model parameter requires grad after loading")
    fp_before = lora_fingerprint(model, torch)

    # ---- train mode, as in Trainer.training_step
    model.train()
    lora_dropouts = [m for n, m in model.named_modules() if "lora_dropout" in n and isinstance(m, torch.nn.Dropout)]
    if not lora_dropouts or not all(m.training and math.isclose(m.p, run["lora_dropout"]) for m in lora_dropouts):
        raise RuntimeError("LoRA dropout modules missing, not in train mode, or wrong p")

    # ---- fixed diagnostic examples, tokenized with trl's own SFTTrainer code path
    away = load_away_dataset(data_mod, cfg, tokenizer, resolve_data_path(cfg, args.data_path))
    if len(away) != plan["n_away_rows"]:
        raise RuntimeError(f"away rows {len(away)} != plan {plan['n_away_rows']}")
    flat = [j for call in plan["call_row_indices"] for j in call]
    for j in flat:
        if away[j]["User"][0] != plan["rows"][str(j)]["User"] or away[j]["Model"][0] != plan["rows"][str(j)]["Model"]:
            raise RuntimeError(f"away row {j} differs from the plan")
    shim = types.SimpleNamespace(dataset_num_proc=None, dataset_batch_size=1000)
    tokenized = SFTTrainer._prepare_non_packed_dataloader(
        shim, tokenizer, away.select(flat), None, cfg["sfttrainer"]["max_seq_length"],
        formatting_func, True, cfg["training"]["remove_unused_columns"],
    )
    pos = {j: i for i, j in enumerate(flat)}

    train_seqs = training_perturbed_sequences(run["trace_dir"])
    device = next(p for p in model.parameters() if p.device.type == "cuda").device

    # Resume guard: a partially recorded run may only be continued with the same plan and dropout seed.
    meta_path = out_dir / "posthoc_meta.json"
    meta = dict(plan_fingerprint=plan["fingerprint"], dropout_seed=args.dropout_seed, model_mode="train")
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{meta_path} was written with a different plan/dropout seed; refusing to mix.")
    else:
        atomic_write_json(meta_path, meta)
    done = 0
    if calls_path.exists():
        good = []
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break  # truncated final line from an interrupted write
            if rec.get("call") != len(good):
                break
            good.append(line)
        tmp = Path(str(calls_path) + ".tmp")
        tmp.write_text("".join(g + "\n" for g in good), encoding="utf-8")
        os.replace(tmp, calls_path)
        done = len(good)
        print(f"[resume] {done} calls already recorded in {calls_path}")

    with open(calls_path, "a", encoding="utf-8") as fout:
        for ci in range(done, plan["n_calls"]):
            examples = [tokenized[pos[j]] for j in plan["call_row_indices"][ci]]
            batch = collator(examples)
            batch = {k: v.to(device) for k, v in batch.items()}
            _, away_inputs, _, _ = adversarial_training.AdversarialULTrainer.split_inputs(None, batch)

            torch.manual_seed(args.dropout_seed + ci)  # CPU + all CUDA devices
            input_embeds, delta_K, delta_mask, losses, affirm = attack.attack(
                model, away_inputs["input_ids"], away_inputs["labels"], away_inputs["attention_mask"],
            )
            # attack() re-enables LoRA grads on exit (for the outer step in training); undo it here.
            for p in model.parameters():
                p.requires_grad_(False)
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("A model parameter received a gradient")

            with torch.no_grad():
                m = delta_mask.squeeze(-1).bool()
                adv_embeds = attack.get_adv_embeddings(input_embeds, delta_K, delta_mask)
                l2 = torch.norm(adv_embeds[m] - input_embeds[m], p=2, dim=-1).float()  # as in the traces
                dnorm = torch.norm(delta_K[m], p=2, dim=-1).float()
                seqs = [tuple(away_inputs["input_ids"][b][m[b]].tolist()) for b in range(m.shape[0])]
            matched = [s in train_seqs for s in seqs]

            rec = dict(
                call=ci, row_indices=plan["call_row_indices"][ci], batch_size=int(m.shape[0]),
                n_tokens=int(m.sum()), D_call=float(l2.mean()), delta_norm_mean=float(dnorm.mean()),
                max_abs_l2_minus_delta_norm=float((l2 - dnorm).abs().max()),
                frac_at_eps=float((dnorm >= attack.eps * (1 - 1e-4)).float().mean()),
                losses=[float(x) for x in losses], affirmative=[int(x) for x in affirm],
                mask_matches_training=matched,
            )
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"[call {ci}] B={rec['batch_size']} tok={rec['n_tokens']} D={rec['D_call']:.6f} "
                  f"loss9={rec['losses'][-1]:.4f} mask_ok={all(matched)}", flush=True)

    fp_after = lora_fingerprint(model, torch)
    del model
    adapter_sha_after = sha256_file(adapter_file)
    if fp_before != fp_after:
        raise RuntimeError(f"LoRA weights changed: {fp_before} -> {fp_after}")
    if adapter_sha_before != adapter_sha_after:
        raise RuntimeError("Adapter file on disk changed")

    recs = list(read_jsonl(calls_path))
    if [r["call"] for r in recs] != list(range(plan["n_calls"])):
        raise RuntimeError("posthoc_calls.jsonl is incomplete or out of order")
    n_mismatch = int(sum(not x for r in recs for x in r["mask_matches_training"]))
    if n_mismatch and not args.allow_mask_mismatch:
        raise RuntimeError(f"{n_mismatch} diagnostic examples have a perturbed-token sequence never seen in "
                           f"this run's training traces (tokenization/mask mismatch). Summary NOT written.")
    d = np.array([r["D_call"] for r in recs])
    import peft, transformers, trl, accelerate
    atomic_write_json(summary_path, dict(
        run_id=run["run_id"], s=run["s"], seed=run["seed"], eps=attack.eps, inner_lr=attack.opt_config["lr"],
        embedding_norm=attack.embedding_norm, iters=attack.iters, model_mode="train",
        lora_dropout=run["lora_dropout"], dropout_seed=args.dropout_seed, plan_fingerprint=plan["fingerprint"],
        n_calls=len(d), D_posthoc=float(d.mean()), D_call_sd=float(d.std(ddof=1)),
        D_posthoc_se=float(d.std(ddof=1) / math.sqrt(len(d))),
        loss_final_posthoc=float(np.mean([r["losses"][-1] for r in recs])),
        frac_tokens_at_eps=float(np.average([r["frac_at_eps"] for r in recs], weights=[r["n_tokens"] for r in recs])),
        n_mask_mismatch=n_mismatch,
        lora_fingerprint=fp_after, adapter_sha256=adapter_sha_after,
        gpu=torch.cuda.get_device_name(0),
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                      trl=trl.__version__, accelerate=accelerate.__version__),
    ))
    print(f"[ok] {summary_path}: D_posthoc={d.mean():.6f} (SE {d.std(ddof=1)/math.sqrt(len(d)):.6f})")


# ============================================================ mode: summarize

def mode_summarize(args):
    runs = resolve_runs(args.manifest, args.experiments_dir)
    out_root = check_output_root(args.out_root, runs)
    rows = []
    for r in runs:
        ab = training_AB(r["trace_dir"])
        sp = out_root / r["run_id"] / "posthoc_run_summary.json"
        ph = json.loads(sp.read_text()) if sp.exists() else {}
        row = dict(epsilon=r["log_eps"], s=r["s"], seed=r["seed"],
                   D_train_whole=ab["D_train_whole"], D_train_last10pct=ab["D_train_last10pct"],
                   D_posthoc=ph.get("D_posthoc", np.nan), D_posthoc_se_examples=ph.get("D_posthoc_se", np.nan),
                   n_posthoc_calls=ph.get("n_calls", 0), n_train_calls=ab["n_train_calls"],
                   loss9_train_last10pct=ab["loss_final_train_last10pct"],
                   loss9_posthoc=ph.get("loss_final_posthoc", np.nan),
                   n_mask_mismatch=ph.get("n_mask_mismatch", np.nan))
        row["posthoc_vs_whole_pct"] = 100 * (row["D_posthoc"] / row["D_train_whole"] - 1)
        row["posthoc_vs_last10pct_pct"] = 100 * (row["D_posthoc"] / row["D_train_last10pct"] - 1)
        row["last10_vs_whole_pct"] = 100 * (row["D_train_last10pct"] / row["D_train_whole"] - 1)
        rows.append(row)
    per_run = pd.DataFrame(rows)

    def sd(x):
        return x.std(ddof=1)

    per_rad = per_run.groupby(["s", "epsilon"]).agg(
        n_seeds=("seed", "size"),
        D_train_whole_mean=("D_train_whole", "mean"), D_train_whole_sd=("D_train_whole", sd),
        D_train_last10pct_mean=("D_train_last10pct", "mean"), D_train_last10pct_sd=("D_train_last10pct", sd),
        D_posthoc_mean=("D_posthoc", "mean"), D_posthoc_sd=("D_posthoc", sd),
        D_posthoc_se_examples_mean=("D_posthoc_se_examples", "mean"),
        posthoc_vs_whole_pct_mean=("posthoc_vs_whole_pct", "mean"),
        posthoc_vs_last10pct_pct_mean=("posthoc_vs_last10pct_pct", "mean"),
    ).reset_index()

    out_root.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out_root / "summary_per_run.csv", index=False)
    per_rad.to_csv(out_root / "summary_per_radius.csv", index=False)

    pd.set_option("display.width", 250)
    lines = ["# Post-hoc CAT bridging diagnostic (Llama 3, MixAT recipe)", "",
             "## Per run", per_run.round(6).to_string(index=False), "",
             "## Per radius (mean, sample SD over 3 training seeds; SE over diagnostic calls reported separately)",
             per_rad.round(6).to_string(index=False), ""]
    complete = per_run.D_posthoc.notna().all()
    if complete:
        pr = per_rad.sort_values("s")
        lines.append("## Ordering of radii")
        for col in ("D_train_whole_mean", "D_train_last10pct_mean", "D_posthoc_mean"):
            lines.append(f"  {col:<24} strictly increasing with eps: {bool(np.all(np.diff(pr[col]) > 0))}")
        seeds_ok = all(np.all(np.diff(g.sort_values('s').D_posthoc) > 0) for _, g in per_run.groupby("seed"))
        lines.append(f"  D_posthoc increasing with eps within every seed: {seeds_ok}")
        lines.append("")
        lines.append("## Scale of the discrepancy")
        for _, r in pr.iterrows():
            d_whole = r.D_posthoc_mean - r.D_train_whole_mean
            d_late = r.D_posthoc_mean - r.D_train_last10pct_mean
            lines.append(
                f"  s={r.s:<6} posthoc-whole {d_whole:+.5f} ({100*d_whole/r.D_train_whole_mean:+.2f}%, "
                f"{d_whole/r.D_train_whole_sd:+.2f} x seed SD)   posthoc-last10 {d_late:+.5f} "
                f"({100*d_late/r.D_train_last10pct_mean:+.2f}%, {d_late/r.D_train_last10pct_sd:+.2f} x seed SD)")
        gaps = np.diff(pr.D_train_whole_mean.to_numpy())
        lines.append(f"  adjacent-radius gaps in D_train_whole: {np.round(gaps, 5).tolist()}")
    else:
        lines.append(f"[incomplete] post-hoc results missing for: "
                     f"{per_run.loc[per_run.D_posthoc.isna(), ['s', 'seed']].values.tolist()}")
    text = "\n".join(lines)
    (out_root / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


# ============================================================ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["list", "plan", "attack", "summarize"])
    p.add_argument("--run_id", help="attack mode: one of the 12 run ids")
    p.add_argument("--manifest", type=Path, default=CAT_DIR / "to_run_global.csv")
    p.add_argument("--experiments_dir", type=Path, default=REPO / "experiments")
    p.add_argument("--out_root", type=Path, default=None,
                   help="default: <experiments_dir>/posthoc_bridging_D_mixat")
    p.add_argument("--data_path", default=None, help="override dataset.data_path (default: as in training)")
    p.add_argument("--n_calls", type=int, default=100)
    p.add_argument("--plan_seed", type=int, default=0)
    p.add_argument("--dropout_seed", type=int, default=12345)
    p.add_argument("--allow_mask_mismatch", action="store_true")
    args = p.parse_args()
    if args.mode == "attack" and not args.run_id:
        p.error("--run_id is required for --mode attack")
    if args.out_root is None:
        args.out_root = args.experiments_dir / "posthoc_bridging_D_mixat"
    {"list": mode_list, "plan": mode_plan, "attack": mode_attack, "summarize": mode_summarize}[args.mode](args)


if __name__ == "__main__":
    main()

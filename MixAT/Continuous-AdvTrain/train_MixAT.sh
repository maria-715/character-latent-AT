#!/bin/bash

# ===================================
#SBATCH --job-name=train_MixAT
#SBATCH --partition=msc

# The MixAT paper reports 3xA100 40G (or 1xH200) for Llama-3-8B MixAT training.
# Traced: the PAP attack model (mistralai/Mixtral-8x7B-Instruct-v0.1) is routed
# through harmbench/baselines/pap/language_models.py's TogetherAPI class (a remote
# Together.ai API call), not loaded locally via vLLM despite PAP_config_mixat.yaml's
# use_vllm: True -- that field is dead for any model_name_or_path matching 'Mixtral-'.
# So the 3xA100 figure is not something the code requires for this config; it's an
# inference from the paper's own reported hardware, not verified empirically here.
# A single A100 (matching CAT's footprint for the same base model) may well suffice --
# adjust based on actual usage on your cluster.
#SBATCH --gres=gpu:a100:3
#SBATCH --nodelist=oat[11-18]
#SBATCH --nodes=1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

# ---- --help must be side-effect-free: no env exports, no mkdir, no conda, no network ----
print_usage() {
  cat <<'EOF'
Usage: sbatch train_MixAT.sh --logging_path PATH --seed N

Thin Slurm wrapper around MixAT's own vendored Hydra entrypoint
(MixAT/Continuous-AdvTrain/src/run_experiments.py --config-name=Llama3_MixAT
path=path_llama_ul), reproducing the official Llama-3-8B MixAT recipe as
faithfully as possible. It does not reimplement or modify MixAT's training
logic -- it only sets up the environment and forwards Hydra overrides.

Required:
  --logging_path PATH   Output directory for this run. Also becomes
                         hydra.run.dir and path.logging_path, so the final
                         adapter lands at PATH/final_model, and
                         experiments_db/ (bookkeeping) lives under it too --
                         concurrent runs with different --logging_path values
                         no longer share any output location.
  --seed N               One experiment seed, ported from CAT's
                         run_experiments.py: training.seed now drives
                         seed_everything(cfg.training.seed) at the very top
                         of main() (before model/LoRA/Trainer construction),
                         seeding Python random, NumPy, torch CPU/CUDA, and
                         transformers.set_seed together. The same value is
                         also forwarded to random_seed_for_random_library
                         (the per-step PAP-vs-continuous coin flip) and
                         dataset.seed_for_interleave_dataset (data
                         interleaving order), so --seed 42/123/456 give
                         genuinely different, internally consistent complete
                         runs rather than several independently-varying
                         knobs. Note: the schedule-construction step
                         (np.random.permutation building which steps are
                         even eligible for that coin flip) is not seeded by
                         anything in this codebase, so exact bit-for-bit
                         reproduction of one specific seed's schedule is not
                         guaranteed -- only that different seeds are
                         genuinely independent runs.

No epsilon/LR/relative-LR flags are exposed yet -- this script only
reproduces the baseline recipe from Llama3_MixAT.yaml unmodified.

Example:
  sbatch train_MixAT.sh --logging_path ./experiments/original_mixat_llama3_seed42 --seed 42
EOF
}

for arg in "$@"; do
  if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
    print_usage
    exit 0
  fi
done

# ---- everything below this line has real side effects ----

# Refuse to run from anywhere other than MixAT's own vendored Continuous-AdvTrain,
# regardless of where `sbatch` was invoked from, so this can never silently pick up
# the top-level (CAT) Continuous-AdvTrain/ instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$SCRIPT_DIR" in
  */MixAT/Continuous-AdvTrain) : ;;
  *)
    echo "ERROR: expected to run from a .../MixAT/Continuous-AdvTrain directory, got: $SCRIPT_DIR"
    exit 1
    ;;
esac
if [ ! -f "config/Llama3_MixAT.yaml" ] || [ ! -f "src/run_experiments.py" ]; then
  echo "ERROR: config/Llama3_MixAT.yaml or src/run_experiments.py not found under $SCRIPT_DIR"
  echo "This does not look like MixAT's vendored Continuous-AdvTrain checkout."
  exit 1
fi

# ======================================================
# Conda settings
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

# Force conda to use node-local scratch for cache
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
mkdir -p "$XDG_CACHE_HOME"

# Because I'm going to use pip
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR"

# Create conda env
CONDA_BIN=/scratch-ssd/oatml/miniconda3/bin/conda
ENV_NAME="env_mixat"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

NEED_PIP_INSTALL=0
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env $ENV_NAME"
  # MixAT/README.md: "We used Python 3.10.14 in our experiments" -- NOT CAT's 3.11.
  $RUN_LOCK "$CONDA_BIN" create -y -n "$ENV_NAME" \
    python=3.10 pip
  NEED_PIP_INSTALL=1
else
  echo "Conda env $ENV_NAME already exists"
fi

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"

# pip install requirements -- MixAT/README.md's documented install path, not the
# separate (narrower, PAP/GCG-dependency-free) MixAT/Continuous-AdvTrain/requirements.txt.
if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r ../requirements/requirements_mixat.txt
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" python -m spacy download en_core_web_sm
  # Not pinned in requirements_mixat.txt but required by LlamaTokenizer; CAT's own
  # env needed the same manual addition (train_CAT.sh installs it separately too).
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install sentencepiece
fi

# ======================================================

LOGGING_PATH=""
SEED=42

while [[ $# -gt 0 ]]; do
  case $1 in
    --logging_path)
      LOGGING_PATH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$LOGGING_PATH" ]; then
  echo "ERROR: You must provide --logging_path ..."
  echo "Example: sbatch train_MixAT.sh --logging_path ./experiments/original_mixat_llama3_seed42 --seed 42"
  exit 1
fi
mkdir -p "$LOGGING_PATH"

# HF token in .env file (repo root: MixAT/Continuous-AdvTrain -> ../../.env)
ENV_FILE="../../.env"
if [ -f "$ENV_FILE" ]; then
  export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
else
  echo ".env file not found at $ENV_FILE"
  exit 1
fi
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

# PAP generation (attack_type: mixed_perturbed_llama) needs a filled-in attack_model
# token in config/harmbench/PAP_config_mixat.yaml (mistralai/Mixtral-8x7B-Instruct-v0.1
# via vLLM/Together per README). This script does not read or write that file -- it is
# a secret and a one-time manual setup step per MixAT's own README, not something a
# wrapper should touch silently.
if grep -q "token: #<your_token>" config/harmbench/PAP_config_mixat.yaml 2>/dev/null; then
  echo "WARNING: config/harmbench/PAP_config_mixat.yaml still has an unfilled attack_model.token placeholder."
  echo "Llama3_MixAT.yaml's PAP mixing will fail without it. See MixAT/README.md."
fi

# Concurrency: hydra.run.dir/path.logging_path being set to the caller's unique
# --logging_path (same convention as train_CAT.sh) is what makes checkpoints unique
# per run. run_experiments.py now also derives path.experiments_path from that same
# root (ported from CAT's run_experiments.py), so the experiments_db bookkeeping
# JSON/lockfile live under <logging_path>/experiments_db/ instead of a single shared
# ./experiments/experiments_db/ -- no separate experiment_id uniqueness hack needed.
#
# Seeding: training.seed is a real field now (ported from CAT's run_experiments.py,
# consumed by the new seed_everything() call at the very top of main() -- seeds
# Python random, NumPy, torch CPU/CUDA, and transformers.set_seed, all before the
# model/LoRA/Trainer are constructed). random_seed_for_random_library=$SEED and
# dataset.seed_for_interleave_dataset=$SEED are set to the same value so the whole
# run -- PAP/continuous step draws and dataset interleaving order included -- moves
# together with one seed, rather than only the parts MixAT already happened to wire up.
CMD=(
    srun
    python src/run_experiments.py
    --config-name=Llama3_MixAT
    path=path_llama_ul
    "hydra.run.dir=$LOGGING_PATH"
    "path.logging_path=$LOGGING_PATH"
    "training.seed=$SEED"
    "random_seed_for_random_library=$SEED"
    "dataset.seed_for_interleave_dataset=$SEED"
)

echo "================================================================"
echo "Resolved training entrypoint : $(realpath src/run_experiments.py)"
echo "Resolved config              : $(realpath config/Llama3_MixAT.yaml) (path=path_llama_ul)"
echo "Logging / checkpoint path    : $(realpath "$LOGGING_PATH")"
echo "Seed                         : $SEED"
echo "Resolved command:"
printf '  %q' "${CMD[@]}"
echo
echo "================================================================"

"${CMD[@]}"

#!/bin/bash

# ===================================
#SBATCH --job-name=train_LAT
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat[11-18]
#SBATCH --nodes=1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

# Non-interactive launcher for the ORIGINAL LAT jailbreak-robustness training experiment
# (notebooks/jailbreaks.ipynb), ported to train_lat.py. Defaults to the notebook's
# Llama-3-8B-Instruct branch; pass --use_llama2 true for its Llama-2-7b-chat branch instead.
#
# Shaped after Continuous-AdvTrain/train_CAT.sh (conda bootstrap, CLI pattern, .env/HF
# handling, output-dir handling) but does NOT copy CAT's algorithm, losses, dataset
# pipeline, or attack. See the accompanying plan (Phase 3) for the full source/infra
# mapping and for what deliberately differs from CAT's conventions and why.
#
# Unlike train_CAT.sh, argument parsing happens BEFORE any conda/network side effects,
# so `bash train_LAT.sh --help` (or a plain syntax check) is fast and side-effect-free.

# ======================================================
# Defaults -- every ML hyperparameter default below reproduces the ORIGINAL jailbreaks.ipynb
# Llama-3 branch exactly (see train_lat.py for per-parameter sourcing). Pass --use_llama2 to
# switch to the notebook's Llama-2-7b-chat branch instead: MODEL_PATH/EPSILON/INNER_LR/OUTER_LR/
# ADD_COMPLETIONS_PGD are left unset ("") here and resolved to the correct branch's values
# below, right after argument parsing -- but each is still independently overridable by
# passing its own flag explicitly, same as train_lat.py's own resolution logic.
LOGGING_PATH=""
SEED=42                                   # new infra; NOT a jailbreaks.ipynb value (notebook never seeds)
USE_LLAMA2=false
MODEL_PATH=""
EPSILON=""
INNER_LR=""
OUTER_LR=""
PGD_ITERS=16
PGD_LAYERS="embedding,8,16,24,30"
NUM_STEPS=100
MODEL_ITERS_PER_STEP=4
MAX_BATCH_PER_ACC=2
ADD_COMPLETIONS_PGD=""
DIAGNOSTICS=false                         # new infra; opt-in, observational-only PGD inner-loop diagnostics
WANDB_MODE_ARG="offline"                  # non-interactive by default; see W&B section below
WANDB_PROJECT="jailbreaks_test"
WANDB_RUN_NAME=""

print_help() {
  cat <<'EOF'
Usage: sbatch train_LAT.sh --logging_path <dir> [options]

Required:
  --logging_path PATH             Output/run directory. Final adapter is saved to PATH/final_model

Original-LAT hyperparameters (defaults reproduce jailbreaks.ipynb's Llama-3 branch exactly):
  --use_llama2 true|false         default: false. Switches to the notebook's Llama-2-7b-chat
                                   branch: MODEL_PATH/EPSILON/INNER_LR/OUTER_LR/ADD_COMPLETIONS_PGD
                                   below then default to that branch's values unless you also
                                   pass the flag explicitly. pgd_layers/pgd_iterations_per_step/
                                   model_iterations_per_step/max_batch_per_acc/num_steps are
                                   shared by both branches in the notebook and unaffected.
  --model_path STR                default: meta-llama/Meta-Llama-3-8B-Instruct (branch-dependent)
  --epsilon FLOAT                 default: 6.0
  --inner_learning_rate FLOAT     default: 1e-3 (branch-dependent; llama2 branch: 5e-2)
  --outer_learning_rate FLOAT     default: 8e-5 (branch-dependent; llama2 branch: 2e-5)
  --pgd_iterations_per_step INT   default: 16
  --pgd_layers STR                default: embedding,8,16,24,30  (comma-separated)
  --num_steps INT                 default: 100
  --model_iterations_per_step INT default: 4
  --max_batch_per_acc INT         default: 2
  --add_completions_pgd true|false default: true (branch-dependent; llama2 branch: false)

New infrastructure (not in jailbreaks.ipynb):
  --diagnostics true|false        default: false. Opt-in, observational-only PGD inner-loop
                                   diagnostics (perturbation-norm distributions around the
                                   optimizer.step()/clip_attack() boundary). Does not change
                                   training behavior. Written to <logging_path>/diagnostics.csv.
  --seed INT                      default: 42
  --wandb_mode online|offline|disabled  default: offline
  --wandb_project STR             default: jailbreaks_test
  --wandb_run_name STR            default: none (wandb auto-names the run)

  --help                          print this message and exit (no side effects)

Example:
  sbatch train_LAT.sh --logging_path ./experiments/jailbreaks_llama3_run1 --seed 42
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --logging_path) LOGGING_PATH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --use_llama2) USE_LLAMA2="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --epsilon) EPSILON="$2"; shift 2 ;;
    --inner_learning_rate) INNER_LR="$2"; shift 2 ;;
    --outer_learning_rate) OUTER_LR="$2"; shift 2 ;;
    --pgd_iterations_per_step) PGD_ITERS="$2"; shift 2 ;;
    --pgd_layers) PGD_LAYERS="$2"; shift 2 ;;
    --num_steps) NUM_STEPS="$2"; shift 2 ;;
    --model_iterations_per_step) MODEL_ITERS_PER_STEP="$2"; shift 2 ;;
    --max_batch_per_acc) MAX_BATCH_PER_ACC="$2"; shift 2 ;;
    --add_completions_pgd) ADD_COMPLETIONS_PGD="$2"; shift 2 ;;
    --diagnostics) DIAGNOSTICS="$2"; shift 2 ;;
    --wandb_mode) WANDB_MODE_ARG="$2"; shift 2 ;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_run_name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$LOGGING_PATH" ]; then
  echo "ERROR: You must provide --logging_path ..."
  echo "Example: sbatch train_LAT.sh --logging_path ./experiments/jailbreaks_llama3_run1 --seed 42"
  exit 1
fi
mkdir -p "$LOGGING_PATH"

# ======================================================
# Resolve branch-dependent defaults (cell-3 of jailbreaks.ipynb): only fills in flags left
# unset above, so explicitly passing e.g. --outer_learning_rate still overrides this.
if [ "$USE_LLAMA2" = "true" ]; then
  : "${MODEL_PATH:=meta-llama/Llama-2-7b-chat-hf}"
  : "${EPSILON:=6.0}"
  : "${INNER_LR:=5e-2}"
  : "${OUTER_LR:=2e-5}"
  : "${ADD_COMPLETIONS_PGD:=false}"
else
  : "${MODEL_PATH:=meta-llama/Meta-Llama-3-8B-Instruct}"
  : "${EPSILON:=6.0}"
  : "${INNER_LR:=1e-3}"
  : "${OUTER_LR:=8e-5}"
  : "${ADD_COMPLETIONS_PGD:=true}"
fi

# ======================================================
# Conda settings (mirrors Continuous-AdvTrain/train_CAT.sh; separate env, LAT's own deps)
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
mkdir -p "$XDG_CACHE_HOME"

export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR"

CONDA_BIN=/scratch-ssd/oatml/miniconda3/bin/conda
ENV_NAME="env_LAT"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

# $RUN_LOCK "$CONDA_BIN" env remove -y -n "$ENV_NAME"

NEED_PIP_INSTALL=0
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env $ENV_NAME"
  $RUN_LOCK "$CONDA_BIN" create -y -n "$ENV_NAME" \
    python=3.11 pip
  NEED_PIP_INSTALL=1
else
  echo "Conda env $ENV_NAME already exists"
fi

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"

if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
  # deepspeed and transformer-lens are commented out directly in requirements.txt (both
  # optional in latent_at, both unused by this training path, both incompatible at import
  # time with the rest of this pinned environment) -- see requirements.txt for why.
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r requirements.txt
  # einops is imported unconditionally by latent_at/laa/attacks.py (GDAdversary's module)
  # but is missing from requirements.txt upstream. Installed explicitly here, rather than
  # editing the library or requirements.txt, so latent_at/ stays untouched beyond the
  # breakpoint() fix and the dependency gap is visible in the bootstrap step that needs it.
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install einops
  # jaxtyping: not a latent_at dependency, but the external `tasks` repo (pinned above) needs
  # it -- tasks/__init__.py unconditionally imports tasks/ioi/IOITask.py, which imports
  # jaxtyping, and latent_at/lat_methods.py's top-level `from tasks...import MMLUTask`
  # triggers that whole chain. Not declared in tasks' own install story
  # (install_tasks_from_github.sh just clones it, no requirements file is installed for it);
  # added here for the same reason as einops. More such gaps in `tasks`' own dependencies may
  # still surface -- its full dependency list hasn't been enumerable from this environment.
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install jaxtyping

fi

/scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip uninstall -y deepspeed

# ======================================================
# Pin the external `tasks` dependency (magikarp01/tasks) that latent_at/lat_methods.py
# imports unconditionally at module level. install_tasks_from_github.sh only clones HEAD
# with no pin; its own comment names commit 6ed4f42 as the known-good revision -- pin to
# that explicitly so the job is reproducible regardless of upstream changes.
TASKS_REPO_URL="https://github.com/magikarp01/tasks.git"
TASKS_COMMIT="6ed4f42"

if [ ! -d "tasks" ]; then
  echo "Cloning $TASKS_REPO_URL"
  git clone "$TASKS_REPO_URL" tasks
fi
( cd tasks && git checkout "$TASKS_COMMIT" )


# ======================================================
# HF auth: LAT's OWN convention (python-dotenv + HUGGINGFACE_API_KEY, passed explicitly
# as token=), confirmed from latent_at/lat_datasets.py and notebooks/jailbreaks.ipynb.
# This is NOT the same as Continuous-AdvTrain's HF_TOKEN convention (an implicitly-read
# env var) -- do not assume the two are interchangeable without checking .env.
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
  HF_KEY_VALUE=$(grep -E '^HUGGINGFACE_API_KEY=' "$ENV_FILE" | cut -d '=' -f2-)
  if [ -z "$HF_KEY_VALUE" ]; then
    echo "WARNING: HUGGINGFACE_API_KEY not found in $ENV_FILE (LAT's own convention)." >&2
    echo "         Falling back to HF_TOKEN (Continuous-AdvTrain's convention) if present." >&2
    HF_KEY_VALUE=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
  fi
  export HUGGINGFACE_API_KEY="$HF_KEY_VALUE"

  WANDB_KEY_VALUE=$(grep -E '^WANDB_API_KEY=' "$ENV_FILE" | cut -d '=' -f2-)
  if [ -n "$WANDB_KEY_VALUE" ]; then
    export WANDB_API_KEY="$WANDB_KEY_VALUE"
  fi
else
  echo "ERROR: .env file not found at $ENV_FILE" >&2
  exit 1
fi
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

# ======================================================
# W&B: ProjectedGradLAT.train() (latent_at/lat_methods.py, LATBaseClass.train) calls
# wandb.init()/wandb.finish() unconditionally -- this is ORIGINAL LAT behavior and is left
# untouched. Made non-interactive purely via environment: WANDB_MODE=offline needs no
# login/network and will not block waiting for an API key. Switch to --wandb_mode online
# if you want live syncing (requires WANDB_API_KEY in .env or prior `wandb login`).
export WANDB_MODE="$WANDB_MODE_ARG"
export WANDB_DIR="$LOGGING_PATH/wandb"
mkdir -p "$WANDB_DIR"

# ======================================================
CMD=(
    srun
    python train_lat.py
    --logging_path "$LOGGING_PATH"
    --seed "$SEED"
    --model_path "$MODEL_PATH"
    --epsilon "$EPSILON"
    --inner_learning_rate "$INNER_LR"
    --outer_learning_rate "$OUTER_LR"
    --pgd_iterations_per_step "$PGD_ITERS"
    --pgd_layers "$PGD_LAYERS"
    --num_steps "$NUM_STEPS"
    --model_iterations_per_step "$MODEL_ITERS_PER_STEP"
    --max_batch_per_acc "$MAX_BATCH_PER_ACC"
    --wandb_project "$WANDB_PROJECT"
)

if [ "$USE_LLAMA2" = "true" ]; then
    CMD+=(--use_llama2)
else
    CMD+=(--no-use_llama2)
fi

if [ "$ADD_COMPLETIONS_PGD" = "true" ]; then
    CMD+=(--add_completions_pgd)
else
    CMD+=(--no-add_completions_pgd)
fi

if [ "$DIAGNOSTICS" = "true" ]; then
    CMD+=(--diagnostics)
else
    CMD+=(--no-diagnostics)
fi

if [ -n "$WANDB_RUN_NAME" ]; then
    CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

"${CMD[@]}"

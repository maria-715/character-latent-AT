#!/bin/bash -l

# ===================================
#SBATCH --job-name=train_CAT
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH -A plggdvps01-gpu-gh200

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

# ======================================================
# Helios env settings
# GPU nodes are aarch64 (Grace CPU) - use the ML-bundle module instead of
# a hand-built conda env (Anaconda is discouraged on this cluster, and
# /scratch-ssd paths from the OATML cluster don't exist here).
set -e

module purge
module load ML-bundle/24.06a

# Keep pip cache off $HOME (10GB quota) and on fast scratch instead
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export XDG_CACHE_HOME=$SCRATCH/.cache
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

export TMPDIR=$SCRATCH/tmp
mkdir -p "$TMPDIR"

ENV_NAME="env_CAT"
ENV_PATH=$SCRATCH/envs/$ENV_NAME

NEED_PIP_INSTALL=0
if [ ! -d "$ENV_PATH" ]; then
  echo "Creating venv $ENV_NAME at $ENV_PATH"
  # --system-site-packages inherits the ML-bundle module's preinstalled
  # torch (built for aarch64+CUDA) instead of pip-installing the x86-only
  # torch==2.5.1+cu124 pinned in requirements.txt, which doesn't exist for GH200.
  python3 -m venv --system-site-packages "$ENV_PATH"
  NEED_PIP_INSTALL=1
else
  echo "venv $ENV_NAME already exists"
fi

source "$ENV_PATH/bin/activate"

# pip install requirements
if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
  pip install -r requirements_helios.txt
  pip install bitsandbytes
  pip install sentencepiece
fi

# ======================================================

# Output folder
LOGGING_PATH=""
EPS=0.05
ITERS=10
SEED=42
RELATIVE_LR=false
CONFIG_NAME="adv_train_ul"
PATH_ARG="example_path"
ADVERSARIAL_OPT_CONFIG_LR=0.1
MAX_STEPS=-1
OR_REPLACE_N=1000

while [[ $# -gt 0 ]]; do
  case $1 in
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --eps)
      EPS="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --logging_path)
      LOGGING_PATH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --relative_lr)
      RELATIVE_LR="$2"
      shift 2
      ;;
    --config_name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --path)
      PATH_ARG="$2"
      shift 2
      ;;
    --adversarial_opt_config_lr)
      ADVERSARIAL_OPT_CONFIG_LR="$2"
      shift 2
      ;;
    --or_replace_n)
      OR_REPLACE_N="$2"
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
  echo "Example: sbatch script.sh --logging_path ./experiments_eps01 --eps 0.1 --seed 42"
  exit 1
fi
mkdir -p "$LOGGING_PATH"

# HF token in .env file
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
  export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
else
  echo ".env file not found"
  exit 1
fi
export HF_HOME=$SCRATCH/.cache/huggingface
mkdir -p "$HF_HOME"

CMD=(
    srun
    python src/run_experiments.py
    --config-name="$CONFIG_NAME"
    "path=$PATH_ARG"
    "adversarial.eps=$EPS"
    "adversarial.iters=$ITERS"
    "adversarial.relative_lr=$RELATIVE_LR"
    "hydra.run.dir=$LOGGING_PATH"
    "path.logging_path=$LOGGING_PATH"
    "training.seed=$SEED"
    "dataset.or_replace_n=$OR_REPLACE_N"
)

if [ "$RELATIVE_LR" = "true" ]; then
    CMD+=(
        "adversarial.opt_config.lr=$ADVERSARIAL_OPT_CONFIG_LR"
    )
fi

if [ "$MAX_STEPS" -ne -1 ]; then
    CMD+=(
        "training.max_steps=$MAX_STEPS"
    )
fi

"${CMD[@]}"

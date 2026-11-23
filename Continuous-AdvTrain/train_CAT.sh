#!/bin/bash

# ===================================
#SBATCH --job-name=train_CAT
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat[11-18]
#SBATCH --nodes=1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

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
ENV_NAME="env_CAT"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

# $CONDA_BIN remove --name $ENV_NAME --all -y

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

# pip install requirements
if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r requirements.txt
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install bitsandbytes
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install sentencepiece
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
OR_WEIGHT=0.0595

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
    --or_weight)
      OR_WEIGHT="$2"
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
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

# srun python src/run_experiments.py --config-name=adv_train_ul_utility path=example_path \
#      adversarial.eps=$EPS \
#      adversarial.iters=$ITERS \
#      adversarial.relative_lr=$RELATIVE_LR \
#      adversarial.opt_config.lr=0.1 \
#      hydra.run.dir=$LOGGING_PATH \
#      path.logging_path=$LOGGING_PATH \
#      training.seed=$SEED

# srun python src/run_experiments.py --config-name=adv_train_ul_utility path=example_path \
#      adversarial.eps=$EPS \
#      adversarial.iters=$ITERS \
#      adversarial.relative_lr=$RELATIVE_LR \
#      hydra.run.dir=$LOGGING_PATH \
#      path.logging_path=$LOGGING_PATH \
#      training.seed=$SEED

#srun python src/run_experiments.py --config-name=adv_train_ul path=example_path

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
    "trainer_hparams.or_weight=$OR_WEIGHT"
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

#!/bin/bash

# ===================================
#SBATCH --job-name=run_embedding_attack_newpeft
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
ENV_NAME="env_run_embedding_attack_newpeft"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

# $CONDA_BIN remove --name $ENV_NAME --all -y

ENV_DIR="$CONDA_ENVS_PATH/$ENV_NAME"
SETUP_MARKER="$ENV_DIR/.newpeft_setup_complete"

# `conda create` makes the env show up in `conda env list` long before pip
# finishes installing into it. If two jobs are submitted around the same time
# and the env doesn't exist yet, a second job checking "does it exist" too
# soon would think setup is done and run against a half-installed env. Gate
# on a completion marker instead, and hold the lock for the whole build (not
# per pip command) so a concurrent job waits for the first one to finish
# rather than racing it.
if [ ! -f "$SETUP_MARKER" ]; then
  "$RUN_LOCK" env TMPDIR="$TMPDIR" bash -c "
    set -e
    if [ ! -f '$SETUP_MARKER' ]; then
      if ! '$CONDA_BIN' env list | awk '{print \$1}' | grep -qx '$ENV_NAME'; then
        echo 'Creating conda env $ENV_NAME'
        '$CONDA_BIN' create -y -n '$ENV_NAME' python=3.11 pip
      fi
      source /scratch-ssd/oatml/miniconda3/bin/activate '$ENV_NAME'
      pip install -r requirements_newpeft.txt
      touch '$SETUP_MARKER'
    fi
  "
fi

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"

# ======================================================

echo "Running command: $0 $*"

# Output folder
LOGGING_PATH=""
MODEL=""
ADAPTER=""
SEED=42


while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;

    --adapter)
      ADAPTER="$2"
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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done


if [ -z "$MODEL" ]; then
  echo "ERROR: --model required"
  exit 1
fi
if [ -z "$LOGGING_PATH" ]; then
  echo "ERROR: --logging_path required"
  exit 1
fi
mkdir -p "$LOGGING_PATH"


CMD=(
  python evaluation/evaluate.py
  -m "$MODEL"
  --benchmark data/harmbench_test_std.json
  --output_dir "$LOGGING_PATH"
  --seed "$SEED"
)

if [ -n "$ADAPTER" ]; then
  CMD+=( --adapter "$ADAPTER" )
fi

echo "Running command: ${CMD[*]}"
srun "${CMD[@]}"

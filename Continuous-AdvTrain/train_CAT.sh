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
EPS=0.5
SEED=42

while [[ $# -gt 0 ]]; do
  case $1 in
    --eps)
      EPS="$2"
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

if [ -z "$LOGGING_PATH" ]; then
  echo "ERROR: You must provide --logging_path ..."
  echo "Example: sbatch script.sh --logging_path ./experiments_eps01 --eps 0.1 --seed 42"
  exit 1
fi
mkdir -p "$LOGGING_PATH"


srun python src/run_experiments.py --config-name=adv_train_ul path=example_path \
     adversarial.eps=$EPS \
     hydra.run.dir=$LOGGING_PATH \
     path.logging_path=$LOGGING_PATH \
     training.seed=$SEED

#srun python src/run_experiments.py --config-name=adv_train_ul path=example_path

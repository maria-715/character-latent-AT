#!/bin/bash

# ===================================
#SBATCH --job-name=AdversariaLLM
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat[11-18]
#SBATCH --nodes=1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

set -euo pipefail

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
ENV_NAME="env_adversarialllm"
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
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -e .
fi

# ======================================================

MODEL=""
ADAPTER_PATH=""
ATTACK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --adapter_path)   ADAPTER_PATH="$2"; shift 2 ;;
    --attack)  ATTACK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$ATTACK" ]]; then
  echo "Usage: $0 --model <hf-or-config-name> --attack <name> [--dataset <name>]"
  exit 1
fi

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

python run_attacks.py -m \
  model="$MODEL" \
  models.${MODEL}.adapter_path="$ADAPTER_PATH" \
  dataset=adv_behaviors \
  dataset.idx="range(0,300)" \
  attack="$ATTACK" \
  hydra.launcher.timeout_min=240

python run_judges.py \
  judge=strong_reject
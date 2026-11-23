#!/bin/bash

# ===================================
#SBATCH --job-name=AdversariaLLM
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat[12-17]
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

export PIXI_HOME=/scratch-ssd/$USER/.pixi
mkdir -p "$PIXI_HOME"

# Create conda env
CONDA_BIN=/scratch-ssd/oatml/miniconda3/bin/conda
ENV_NAME="env_adversariallm"
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

if ! command -v pixi >/dev/null 2>&1; then
  echo "Installing pixi into $ENV_NAME"
  $RUN_LOCK "$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge -y pixi
fi

# pip install requirements
if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
    # /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
    # /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install --no-build-isolation -r requirements.txt
    # /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -e .
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pixi install --locked
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

ARGS=(
  "model=${MODEL}"
  "attack=${ATTACK}"
  "dataset=jbb_behaviors"
  "hydra.launcher.timeout_min=240"
  "hydra.launcher.partition=msc"
)

if [[ -n "$ADAPTER_PATH" ]]; then
    ARGS+=("models.${MODEL}.adapter_path=${ADAPTER_PATH}")
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

pixi run python run_attacks.py "${ARGS[@]}"

# pixi run python run_judges.py \
#   judge=strong_reject

# pixi run python run_judges.py \
#   classifier=strong_reject \
#   run_paths='["outputs/2026-07-24/03-53-07/0/run.json"]'

pixi run python run_judges.py \
  classifier=strong_reject
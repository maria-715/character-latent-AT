#!/bin/bash -l

# ===================================
#SBATCH --job-name=run_eval_harmless
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

ENV_NAME="env_harmless"
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
  # vllm has no aarch64 wheel and must build from source. pip's default build
  # isolation creates a fresh sandbox without the ML-bundle's CUDA torch, so
  # vllm's setup.py can't detect CUDA and fails with "Unknown runtime
  # environment". --no-build-isolation lets the build see our system torch,
  # and VLLM_TARGET_DEVICE forces the device detection explicitly.
  export VLLM_TARGET_DEVICE=cuda
  pip install --no-build-isolation -r ../requirements/requirements_utility_helios.txt
fi

# ======================================================

echo "Running command: $0 $*"

if [ $# -ne 4 ]; then
    echo "Usage: $0  <model> <peft>(None if no) <model_name> <seed>"
    exit 1
fi

# HF token in .env file
ENV_FILE="../../.env"
if [ -f "$ENV_FILE" ]; then
  export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
else
  echo ".env file not found"
  exit 1
fi
export HF_HOME=$SCRATCH/.cache/huggingface
mkdir -p "$HF_HOME"

echo "Running command: python evaluation.py $1 $2 $3 None True False $4"
python evaluation.py $1 $2 $3 None True False $4


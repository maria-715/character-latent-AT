#!/bin/bash

# ===================================
#SBATCH --job-name=run_utility_test
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
ENV_NAME="env_utility_test"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

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
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r ../requirements/requirements_utility.txt
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -e .
fi

# ======================================================

echo "Running command: $0 $*"

# I added the seed
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model> <seed>"
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
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"
echo "Running command: lm_eval --model hf --model_args pretrained=$1,parallelize=True,dtype=bfloat16,load_in_4bit=True --tasks arc_easy,arc_challenge,mmlu --device cuda:0 --batch_size auto:12 --output_path ../utility_eval/results/{$1}_{$2} --log_samples --cache_requests true --seed $2"
lm_eval --model hf \
    --model_args pretrained=$1,parallelize=True,dtype=bfloat16,load_in_4bit=True \
    --tasks arc_easy,arc_challenge,mmlu \
    --device cuda:0 \
    --batch_size auto:12 \
    --output_path ../utility_eval/results/{$1}_{$2} \
    --log_samples \
    --cache_requests true \
    --seed $2
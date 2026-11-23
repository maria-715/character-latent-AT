#!/bin/bash

# ===================================
#SBATCH --job-name=run_mt_bench
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat11

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
ENV_NAME="env_mt_bench"
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
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -e "../fastchat/[model_worker,llm_judge]"
fi

# ======================================================

if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_path> model_id peft_path load_in_4bit"
    exit 1
fi

if [ "$4" == "True" ]; then
  python gen_model_answer.py --model-path $1 --model-id $2 --peft $3 --load_in_4bit --dtype bfloat16
else
  python gen_model_answer.py --model-path $1 --model-id $2 --peft $3 --dtype bfloat16
fi

python gen_judgment.py --model-list $2  --parallel 8  --judge-model "qwen-32b"  # Added Qwen-32B
python show_result.py --model-list $2 
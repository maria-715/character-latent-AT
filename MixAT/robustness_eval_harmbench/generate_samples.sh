#!/bin/bash

# ===================================
#SBATCH --job-name=run_generate_samples
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:2 # 2 for AutoDAN
#SBATCH --nodelist=oat14

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
ENV_NAME="env_harmbench"
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

if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r ../requirements/requirements_harmbench.txt
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -e "../harmbench/requirement/FastChat[model_worker,webui]"
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install flash-attn --no-build-isolation
fi

# ======================================================

if [ $# -ne 2 ]; then
    echo "Usage: $0 <attack> <model>"
    exit 1
fi


base_save_dir="../robustness_eval_harmbench/results"
method_name=$1
experiment_name=$2
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_test.csv"
save_dir="${base_save_dir}/${method_name}/${experiment_name}/test_cases"
start_idx=0
end_idx=40
behavior_ids_subset="0,1,2,3,5,6,7,8,10,12,13,14,16,18,19,21,22,23,24,25,28,29,30,31,33,34,35,37,39,40,41,42,43,44,45,46,47,49,50,51"
run_id=""
overwrite="False"
verbose="True"


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
./scripts/generate_test_cases.sh $method_name $experiment_name $behaviors_path $save_dir $start_idx $end_idx $behavior_ids_subset
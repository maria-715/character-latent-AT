#!/bin/bash

# ===================================
#SBATCH --job-name=run_generate_samples_newpeft
#SBATCH --partition=msc

# -CHANGED HERE- (to 4)
#SBATCH --gres=gpu:a100:1 # 2 for AutoDAN
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
ENV_NAME="env_harmbench_newpeft"
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

if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
  # fschat (installed below) requires unpinned torch/transformers, which would
  # otherwise silently upgrade them past what the LoRA adapters and flash-attn
  # were built against. Pin just those to known-working versions on every
  # install in this block.
  CONSTRAINTS_FILE="$TMPDIR/pinned_versions.txt"
  cat > "$CONSTRAINTS_FILE" <<EOF
torch==2.3.0
numpy==1.26.4
transformers<4.46.0
EOF

  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" -r ../requirements/requirements_harmbench_newpeft.txt
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" -e "../harmbench/requirement/FastChat[model_worker,webui]"

  # flash-attn compiles CUDA kernels from source and can take a very long time.
  # env_harmbench pins the same python/torch versions, so reuse its already-built
  # copy instead of recompiling when it's available.
  OLD_ENV_SITE_PACKAGES="$CONDA_ENVS_PATH/env_harmbench/lib/python3.11/site-packages"
  NEW_ENV_SITE_PACKAGES="$CONDA_ENVS_PATH/$ENV_NAME/lib/python3.11/site-packages"
  if compgen -G "$OLD_ENV_SITE_PACKAGES/flash_attn*" > /dev/null; then
    echo "Reusing prebuilt flash-attn from env_harmbench"
    cp -r "$OLD_ENV_SITE_PACKAGES"/flash_attn* "$NEW_ENV_SITE_PACKAGES"/
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" einops
  else
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" flash-attn --no-build-isolation
  fi
fi

# ======================================================

# I added the seed
if [ $# -ne 3 ]; then
    echo "Usage: $0 <attack> <model> <seed>"
    exit 1
fi


base_save_dir="../robustness_eval_harmbench/results"
method_name=$1
experiment_name=$2
seed=$3

# -CHANGED HERE-
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
./scripts/generate_test_cases_newpeft.sh $method_name $experiment_name $behaviors_path $save_dir $start_idx $end_idx $behavior_ids_subset $seed

# behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_all.csv"
# save_dir="${base_save_dir}/${method_name}/${experiment_name}/test_cases"

# # HF token in .env file
# ENV_FILE="../../.env"
# if [ -f "$ENV_FILE" ]; then
#   export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
# else
#   echo ".env file not found"
#   exit 1
# fi
# export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
# mkdir -p "$HF_HOME"

# # One behavior per process is maximally parallel.
# # Increase to 2, 4, 5, 10, etc. if the attack has expensive startup overhead.
# CHUNK_SIZE="${CHUNK_SIZE:-1}"

# # Use the GPUs allocated by Slurm.
# NUM_GPUS=$(python - <<'PY'
# import torch
# print(torch.cuda.device_count())
# PY
# )

# if [ "$NUM_GPUS" -lt 1 ]; then
#   echo "No GPUs visible. Exiting."
#   exit 1
# fi

# # Count rows in the behavior CSV.
# NUM_BEHAVIORS=$(python - <<PY
# import csv
# with open("$behaviors_path", "r", encoding="utf-8") as f:
#     print(sum(1 for _ in csv.DictReader(f)))
# PY
# )

# echo "NUM_GPUS=$NUM_GPUS"
# echo "NUM_BEHAVIORS=$NUM_BEHAVIORS"
# echo "CHUNK_SIZE=$CHUNK_SIZE"
# echo "save_dir=$save_dir"

# mkdir -p "$save_dir"

# job_idx=0
# running_jobs=0

# for start_idx in $(seq 0 "$CHUNK_SIZE" $((NUM_BEHAVIORS - 1))); do
#   end_idx=$((start_idx + CHUNK_SIZE))
#   if [ "$end_idx" -gt "$NUM_BEHAVIORS" ]; then
#     end_idx="$NUM_BEHAVIORS"
#   fi

#   gpu_id=$((job_idx % NUM_GPUS))

#   echo "Launching behaviors [$start_idx, $end_idx) on GPU $gpu_id"

#   CUDA_VISIBLE_DEVICES="$gpu_id" python -u generate_test_cases.py \
#     --method_name "$method_name" \
#     --experiment_name "$experiment_name" \
#     --behaviors_path "$behaviors_path" \
#     --save_dir "$save_dir" \
#     --behavior_start_idx "$start_idx" \
#     --behavior_end_idx "$end_idx" \
#     --verbose &

#   job_idx=$((job_idx + 1))
#   running_jobs=$((running_jobs + 1))

#   # Keep at most one running process per GPU.
#   if [ "$running_jobs" -ge "$NUM_GPUS" ]; then
#     wait -n
#     running_jobs=$((running_jobs - 1))
#   fi
# done

# wait

# echo "All generation jobs finished."
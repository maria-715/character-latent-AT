#!/bin/bash

# ===================================
#SBATCH --job-name=run_generate_samples
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat13

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

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"
# ======================================================

source /opt/rh/devtoolset-10/enable

model_name=$1
behaviors_path=$2
test_cases_path=$3
save_path=$4
max_new_tokens=$5
incremental_update=$6
lora_exp="$7"
multiple_exp="$8"
begin_num="${9}"
end_num="${10}"
step_num="${11}"
final_num="${12}"
list_num="${13}"
temp_repeats="${14}"

echo "model_name=$model_name"
echo "behaviors_path=$behaviors_path"
echo "test_cases_path=$test_cases_path"
echo "max_new_tokens=$max_new_tokens"
echo "save_path=$save_path"
echo "incremental_update=$incremental_update"
echo "lora_exp=$lora_exp"
echo "multiple_exp=$multiple_exp"
echo "begin_num=$begin_num"
echo "end_num=$end_num"
echo "step_num=$step_num"
echo "final_num=$final_num"
echo "list_num=$list_num"
echo "temp_repeats=$temp_repeats"

# Construct the base command
cmd="python -u generate_completions.py \
    --model_name $model_name \
    --behaviors_path $behaviors_path \
    --test_cases_path $test_cases_path \
    --save_path $save_path \
    --max_new_tokens $max_new_tokens \
    --generate_with_vllm"

# Check if incremental_update is provided and equals "True"
if [[ ! -z "$incremental_update" && "$incremental_update" == "True" ]]; then
    cmd="$cmd --incremental_update"
fi

if [[ ! -z "$lora_exp" && "$lora_exp" == "True" ]]; then
    cmd="$cmd --lora_scaling_experiments"
fi

if [[ ! -z "$multiple_exp" && "$multiple_exp" == "True" ]]; then
    cmd="$cmd --multiple_checkpoints"
fi

if [[ ! -z "$begin_num" ]]; then
    cmd="$cmd --begin $begin_num"
fi

if [[ ! -z "$end_num" ]]; then
    cmd="$cmd --end $end_num"
fi

if [[ ! -z "$step_num" ]]; then
    cmd="$cmd --step $step_num"
fi

if [[ ! -z "$final_num" ]]; then
    cmd="$cmd --final $final_num"
fi

if [[ ! -z "$temp_repeats" ]]; then
    cmd="$cmd --temperature_repeats $temp_repeats"
fi

if [[ ! -z "$list_num" ]]; then
    cmd="$cmd --list_num \"$list_num\""
fi

echo "Executing command: $cmd"

# Execute the command
eval $cmd

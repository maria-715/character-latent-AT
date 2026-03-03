#!/bin/bash

# ===================================
#SBATCH --job-name=run_generate_samples
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
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

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"
# ======================================================

lora_exp="$3"
multiple_exp="$4"
begin_num="${5}"
end_num="${6}"
step_num="${7}"
final_num="${8}"
list_num="${9}"
echo "lora_exp=$lora_exp"
echo "multiple_exp=$multiple_exp"
echo "begin_num=$begin_num"
echo "end_num=$end_num"
echo "step_num=$step_num"
echo "final_num=$final_num"
echo "list_num=$list_num"

cmd="python -u merge_test_cases.py --method_name $1 --save_dir $2"

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

if [[ ! -z "$list_num" ]]; then
    cmd="$cmd --list_num $list_num"
fi

# Execute the command
eval $cmd

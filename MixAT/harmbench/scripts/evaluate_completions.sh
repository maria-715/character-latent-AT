#!/bin/bash

# ===================================
#SBATCH --job-name=run_generate_samples
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
ENV_NAME="env_harmbench"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"
# ======================================================

source /opt/rh/devtoolset-10/enable

# I added the seed and changed the numbers
cls_path="$1"
behaviors_path=$2
completions_path=$3
save_path=$4
seed=$5
lora_exp="$6"
multiple_exp="$7"
begin_num="${8}"
end_num="${9}"
step_num="${10}"
final_num="${11}"
list_num="${12}"
temp_repeats="${13}"
window_book="${14}"
overlap_book="${15}"
window_lyrics="${16}"
overlap_lyrics="${17}"
threshold="${18}"

echo "cls_path=$cls_path"
echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"
echo "seed=$seed"
echo "lora_exp=$lora_exp"
echo "multiple_exp=$multiple_exp"
echo "begin_num=$begin_num"
echo "end_num=$end_num"
echo "step_num=$step_num"
echo "final_num=$final_num"
echo "list_num=$list_num"
echo "temp_repeats=$temp_repeats"
echo "window_book=$window_book"
echo "overlap_book=$overlap_book"
echo "window_lyrics=$window_lyrics"
echo "overlap_lyrics=$overlap_lyrics"
echo "threshold=$threshold"

cmd="python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --seed $seed \
    --save_path $save_path "

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

if [[ ! -z "$window_book" ]]; then
    cmd="$cmd --window_size_book $window_book"
fi

if [[ ! -z "$window_lyrics" ]]; then
    cmd="$cmd --window_size_lyrics $window_lyrics"
fi

if [[ ! -z "$overlap_book" ]]; then
    cmd="$cmd --overlap_size_book $overlap_book"
fi

if [[ ! -z "$overlap_lyrics" ]]; then
    cmd="$cmd --overlap_size_lyrics $overlap_lyrics"
fi

if [[ ! -z "$threshold" ]]; then
    cmd="$cmd --threshold $threshold"
fi

cmd="$cmd --include_advbench_metric"
# Execute the command
eval $cmd

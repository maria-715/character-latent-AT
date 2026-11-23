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

partition=your_partition
base_save_dir=./results
base_log_dir=./slurm_logs
behaviors_path=data/behavior_datasets/harmbench_behaviors_text_val.csv
methods=("GCG" "EnsembleGCG" "PAIR" "AutoDAN")
experiment_names=("llama2_7b" "llama2_7b" "llama2_7b" "llama2_7b")
model_names=("llama2_7b" "llama2_7b" "llama2_7b" "llama2_7b")
gpu_counts=(1 1 1 1)
max_new_tokens=512
incremental_update="False"

# Loop through each method
for method_name in "${methods[@]}"; do
    # Loop over each index
    for ((index=0; index<${#model_names[@]}; index++)); do
        experiment_name="${experiment_names[$index]}"
        model_name="${model_names[$index]}"
        gpu_count=${gpu_counts[$index]}
        save_path="$base_save_dir/$method_name/$experiment_name/completions/${model_name}.json"
        test_cases_path="$base_save_dir/$method_name/$experiment_name/test_cases/test_cases.json"
        job_name="generate_completions_${method_name}_${experiment_name}_${model_name}"
        log_path="$base_log_dir/generate_completions/${method_name}/${experiment_name}_${model_name}.log"

        sbatch --partition=$partition --job-name=$job_name --nodes=1 --gpus-per-node=$gpu_count --time=5:00:00 --output=$log_path \
            scripts/generate_completions.sh $model_name $behaviors_path $test_cases_path $save_path $max_new_tokens $incremental_update
    done
done
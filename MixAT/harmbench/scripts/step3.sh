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

partition=your_partition
base_save_dir=./results
base_log_dir=./slurm_logs
behaviors_path=./data/behavior_datasets/harmbench_behaviors_text_val.csv
methods=("GCG" "EnsembleGCG" "PAIR" "AutoDAN")
experiment_names=("llama2_7b" "llama2_7b" "llama2_7b" "llama2_7b")
model_names=("llama2_7b" "llama2_7b" "llama2_7b" "llama2_7b")
gpu_count=1  # Number of GPUs to use for the behavior classifier
cls_path="cais/HarmBench-Llama-2-13b-cls"

for method_name in "${methods[@]}"; do
    # Loop over each index
    for ((index=0; index<${#model_names[@]}; index++)); do
        experiment_name="${experiment_names[$index]}"
        # experiment_name=mixtral_attacker_llm
        model_name=${model_names[$index]}
        completions_path="$base_save_dir/$method_name/$experiment_name/completions/${model_name}.json"
        save_path="$base_save_dir/$method_name/$experiment_name/results/${model_name}.json"

        job_name="evaluate_completions_${method_name}_${experiment_name}"
        output_log="$base_log_dir/evaluate_completions/${method_name}/${experiment_name}_${model_name}.log"

        sbatch --partition=$partition --job-name=$job_name --nodes=1 --gpus-per-node=$gpu_count --time=5:00:00 --output=$output_log \
            scripts/evaluate_completions.sh $cls_path $behaviors_path $completions_path $save_path
    done
done
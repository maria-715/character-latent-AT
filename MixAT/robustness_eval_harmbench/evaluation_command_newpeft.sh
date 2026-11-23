#!/bin/bash

# ===================================
#SBATCH --job-name=run_evaluate_commands
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
ENV_NAME="env_harmbench_newpeft"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"
# ======================================================

base_save_dir="../robustness_eval_harmbench/results"
method_name=$1
experiment=$2
seed=$3
save_dir="$base_save_dir/$method_name/$experiment/test_cases"

experiment_dir="$base_save_dir/$method_name/$experiment"
model_name=$2
test_cases_path="$experiment_dir/test_cases/test_cases.json"
behaviors_path="./data/behavior_datasets/harmbench_behaviors_text_test.csv"
max_new_tokens=512
incremental_update="False"
save_path_completions="$experiment_dir/completions/${model_name}.json"

cls_path="cais/HarmBench-Llama-2-13b-cls"
completions_path="$base_save_dir/$method_name/$experiment/completions/$model_name.json"
save_path="$base_save_dir/$method_name/$experiment/results/$model_name.json"

# merge_test_cases and evaluate_completions never load the target model's LoRA
# adapter (only the AutoDAN test-case merge and the HarmBench classifier), so
# they run fine in the original env_harmbench. Only completions generation
# loads the adapter and needs the newpeft env/script.
./scripts/merge_test_cases.sh $method_name $save_dir
./scripts/generate_completions_newpeft.sh $model_name $behaviors_path $test_cases_path $save_path_completions $max_new_tokens $incremental_update
./scripts/evaluate_completions.sh $cls_path $behaviors_path $completions_path $save_path $seed

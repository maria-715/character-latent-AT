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
behaviors_path=./data/behavior_datasets/harmbench_behaviors_text_val.csv
methods=("GCG" "EnsembleGCG" "PAIR" "AutoDAN")
experiments=("llama2_7b")


for method_name in "${methods[@]}"; do
    for experiment_name in "${experiments[@]}"; do
        save_dir="$base_save_dir/$method_name/$experiment_name/test_cases"
        job_name="merge_test_cases_${method_name}_${experiment_name}"
        log_path="$base_log_dir/merge_test_cases/${method_name}_${experiment_name}.log"

        # Call sbatch with arguments
        sbatch --partition=$partition --job-name=$job_name --nodes=1 --gpus-per-node=0 --time=0:10:00 --output=$log_path \
            scripts/merge_test_cases.sh $method_name $save_dir
    done
done
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

partition=your_partition
base_save_dir=./results
base_log_dir=./slurm_logs
behaviors_path=./data/behavior_datasets/harmbench_behaviors_text_val.csv
methods=("GCG")
experiments=("llama2_7b")
start=0
end=80
# Note: some methods jointly optimize over all behaviors and require step=$end (e.g., EnsembleGCG)
step=20  # Reduce to split test cases into more jobs
behavior_ids_subset=""
run_id=""  # Empty string means no run ID
overwrite="False"
verbose="False"

for method_name in "${methods[@]}"; do
    for experiment_name in "${experiments[@]}"; do
        echo "Generating $method_name, $experiment_name"
        for start_idx in $(seq "$start" "$step" "$end"); do
            end_idx=$(($start_idx + $step))
            if [ "$end_idx" -gt "$end" ]; then
                end_idx="$end"
            fi
            save_dir="$base_save_dir/$method_name/$experiment_name/test_cases"
            job_name="generate_test_cases_${method_name}_${experiment_name}_${start_idx}_${end_idx}"
            log_path="$base_log_dir/generate_test_cases/${method_name}/${experiment_name}/start_idx_${start_idx}.log"
            sbatch --partition=$partition --job-name=$job_name --nodes=1 --gpus-per-node=1 --time=24:00:00 --output=$log_path \
                scripts/generate_test_cases.sh $method_name $experiment_name $behaviors_path $save_dir $start_idx $end_idx "$behavior_ids_subset" "$run_id" $overwrite $verbose
        done
    done
done
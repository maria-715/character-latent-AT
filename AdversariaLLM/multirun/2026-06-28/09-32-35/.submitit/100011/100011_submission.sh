#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/auto/users/ms25mds/MSc_Project/AdversariaLLM/multirun/2026-06-28/09-32-35/.submitit/%j/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_attacks
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/auto/users/ms25mds/MSc_Project/AdversariaLLM/multirun/2026-06-28/09-32-35/.submitit/%j/%j_0_log.out
#SBATCH --partition=msc
#SBATCH --signal=USR2@60
#SBATCH --time=240
#SBATCH --wckey=submitit

# setup

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /auto/users/ms25mds/MSc_Project/AdversariaLLM/multirun/2026-06-28/09-32-35/.submitit/%j/%j_%t_log.out --error /auto/users/ms25mds/MSc_Project/AdversariaLLM/multirun/2026-06-28/09-32-35/.submitit/%j/%j_%t_log.err /auto/users/ms25mds/MSc_Project/AdversariaLLM/.pixi/envs/default/bin/python -u -m submitit.core._submit /auto/users/ms25mds/MSc_Project/AdversariaLLM/multirun/2026-06-28/09-32-35/.submitit/%j

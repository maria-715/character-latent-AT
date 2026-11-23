#!/bin/bash

#SBATCH --job-name=run_i-gcg
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:4
#SBATCH --nodelist=oat11
#SBATCH --nodes=1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ======================================================

# Conda settings

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
mkdir -p "$XDG_CACHE_HOME"

export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR"

CONDA_BIN=/scratch-ssd/oatml/miniconda3/bin/conda
ENV_NAME="env_i-gcg"
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

# pip install requirements
if [ "$NEED_PIP_INSTALL" -eq 1 ]; then
    $RUN_LOCK env TMPDIR="$TMPDIR" pip install -r requirements.txt || exit 1
    $RUN_LOCK env TMPDIR="$TMPDIR" pip install peft==0.11.1
    $RUN_LOCK env TMPDIR="$TMPDIR" pip install fschat==0.2.23 --no-deps
fi

# ======================================================

MODEL=""
ADAPTER=""
OUTPUT_DIR=""
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --adapter)
            ADAPTER="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model required"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output_dir required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 1) Generate suffix initialisation

CMD1=(
python attack_llm_core_best_update_our_target.py
--behaviors_config behaviors_ours_config.json
--model_path "$MODEL"
--seed "$SEED"
)

if [ -n "$ADAPTER" ]; then
CMD1+=( --adapter_path "$ADAPTER" )
fi

echo "${CMD1[@]}"
srun "${CMD1[@]}"

# 2) Generate new config

srun python generate_our_config.py

# 3) Run I-GCG attack

CMD3=(
python run_multiple_attack_our_target.py
--behaviors_config behaviors_ours_config_init.json
--model_path "$MODEL"
--output_path "$OUTPUT_PATH"
--seed "$SEED"
)

if [ -n "$ADAPTER" ]; then
CMD3+=( --adapter_path "$ADAPTER" )
fi

echo "${CMD3[@]}"
srun "${CMD3[@]}"
#!/bin/bash

# ===================================
#SBATCH --job-name=XSTest_eval
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
ENV_NAME="env_XSTest"
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
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r requirements.txt
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install bitsandbytes
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install sentencepiece
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install --upgrade transformers
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip uninstall -y peft
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install --no-cache-dir git+https://github.com/huggingface/peft
fi

# ======================================================

# Defaults
MODEL_PATH="HuggingFaceH4/zephyr-7b-beta"
ADAPTER=""
OUTPUT="results/xstest_outputs_raw_mistral.jsonl"
SEED=0
STRICT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --adapter)
            ADAPTER="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ADAPTER" ]; then
    ADAPTER_ARG=""
else
    ADAPTER_ARG="--adapter $ADAPTER"
fi

if [ "$STRICT" -eq 1 ]; then
    STRICT_ARG="--strict"
else
    STRICT_ARG=""
fi

CSV_OUTPUT="${OUTPUT%.jsonl}.csv"
CLASSIFIED_OUTPUT="${OUTPUT%.jsonl}_classified.csv"

# HF token in .env file
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
  export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
else
  echo ".env file not found"
  exit 1
fi
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

echo "=== GENERATION ==="
python generate_xstest_outputs.py \
    --model_path "$MODEL_PATH" \
    $ADAPTER_ARG \
    --output "$OUTPUT" \
    --seed "$SEED"

echo "=== JSONL → CSV ==="
python jsonl_to_csv.py \
    --input "$OUTPUT" \
    --output "$CSV_OUTPUT"

echo "=== CLASSIFICATION ==="
python evaluation/classify_completions_strmatch.py \
    --input "$CSV_OUTPUT" \
    --output "$CLASSIFIED_OUTPUT" \
    $STRICT_ARG

echo "=== METRICS ==="
python compute_metrics.py \
    --input "$CLASSIFIED_OUTPUT"
#!/bin/bash
# ===================================
#SBATCH --job-name=generate_answers_to_or-bench
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat[11-18]
#SBATCH --nodes=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch generate_answers_to_or-bench.sh \
    --model-id MODEL \
    --input INPUT_CSV \
    --output OUTPUT_CSV \
    [--batch-size N] \
    [--precision bf16|fp16|fp32] \
    [--env-file PATH]

Required:
  --model-id     Hugging Face model id
  --input        Input CSV path
  --output       Output CSV path

Optional:
  --batch-size   Default: 64
  --precision    Default: bf16
  --env-file     Default: ../.env
  --limit        Only process the first N (unfinished) prompts -- use to test
EOF
}

MODEL_ID=""
INPUT_CSV=""
OUTPUT_CSV=""
BATCH_SIZE=64
PRECISION="bf16"
ENV_FILE="../.env"
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --input)
      INPUT_CSV="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_CSV="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-64}"
      shift 2
      ;;
    --precision)
      PRECISION="${2:-bf16}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:-../.env}"
      shift 2
      ;;
    --limit)
      LIMIT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_ID" || -z "$INPUT_CSV" || -z "$OUTPUT_CSV" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

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

CONDA_BIN=/scratch-ssd/oatml/miniconda3/bin/conda
ENV_NAME="env_prompting_llms"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

NEED_PIP_INSTALL=0
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env $ENV_NAME"
  $RUN_LOCK "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.11 pip
  NEED_PIP_INSTALL=1
else
  echo "Conda env $ENV_NAME already exists"
fi

source /scratch-ssd/oatml/miniconda3/bin/activate "$ENV_NAME"

if [[ "$NEED_PIP_INSTALL" -eq 1 ]]; then
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -r requirements.txt
fi

# ======================================================

if [[ -f "$ENV_FILE" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)"
else
  echo ".env file not found: $ENV_FILE"
  exit 1
fi

export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

echo "Running:"
echo "  model_id   = $MODEL_ID"
echo "  input_csv  = $INPUT_CSV"
echo "  output_csv = $OUTPUT_CSV"
echo "  batch_size = $BATCH_SIZE"
echo "  precision  = $PRECISION"

EXTRA_ARGS=()
if [[ -n "$LIMIT" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

srun --output=slurm-%j-$(basename "$OUTPUT_CSV" .csv).out \
     --error=slurm-%j-$(basename "$OUTPUT_CSV" .csv).err \
  python generate_answers_to_or-bench.py \
    --model_id "$MODEL_ID" \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_CSV" \
    --batch_size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    "${EXTRA_ARGS[@]}"


# Examples:

# sbatch generate_answers_to_or-bench.sh \
#   --model-id HuggingFaceH4/zephyr-7b-beta \
#   --input datasets/or-bench_train_and_eval/or-bench-from80kto1k_train.csv \
#   --output utility_zephyr_7b_beta_from80kto1k_train_SHORTER.csv

# sbatch generate_answers_to_or-bench.sh \
#   --model-id meta-llama/Meta-Llama-3-8B-Instruct \
#   --input datasets/or-bench_train_and_eval/or-bench-from80kto1k_train.csv \
#   --output utility_llama3_8b_instruct_from80kto1k_train_SHORTER.csv

# sbatch generate_answers_to_or-bench.sh \
#   --model-id HuggingFaceH4/zephyr-7b-beta \
#   --input datasets/or-bench_train_and_eval/or-bench-from80kto1k_eval.csv \
#   --output utility_zephyr_7b_beta_from80kto1k_eval_SHORTER.csv

# sbatch generate_answers_to_or-bench.sh \
#   --model-id meta-llama/Meta-Llama-3-8B-Instruct \
#   --input datasets/or-bench_train_and_eval/or-bench-from80kto1k_eval.csv \
#   --output utility_llama3_8b_instruct_from80kto1k_eval_SHORTER.csv
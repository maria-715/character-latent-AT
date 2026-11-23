#!/bin/bash

# ===================================
#SBATCH --job-name=posthoc_D_CAT
#SBATCH --partition=msc

#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat[11-18]
#SBATCH --nodes=1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===================================

# Post-hoc CAT bridging diagnostic (Llama 3, MixAT recipe, 12 final checkpoints).
# Submit from Continuous-AdvTrain/ (same directory as train_CAT.sh):
#
#   sbatch posthoc_bridging/run_posthoc_bridging_D.sh
#   sbatch posthoc_bridging/run_posthoc_bridging_D.sh --experiments_dir ../experiments
#   sbatch posthoc_bridging/run_posthoc_bridging_D.sh --run_id CAT_llama3b_0075_relativelr_otherparams
#
# Read-only w.r.t. checkpoints and traces. Outputs: <experiments_dir>/posthoc_bridging_D_mixat/.
# Each checkpoint runs in its own python process. Finished checkpoints are skipped and an interrupted
# checkpoint resumes from its last recorded call, so the job can simply be resubmitted.

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

# Same env as train_CAT.sh (same pinned trl/peft/transformers/accelerate as the training runs)
CONDA_BIN=/scratch-ssd/oatml/miniconda3/bin/conda
ENV_NAME="env_CAT"
RUN_LOCK=/scratch-ssd/oatml/run_locked.sh

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
fi

# ======================================================

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

# ======================================================

echo "Running command: $0 $*"

if [ ! -f src/run_experiments.py ]; then
  echo "ERROR: submit this from Continuous-AdvTrain/ (sbatch posthoc_bridging/run_posthoc_bridging_D.sh)"
  exit 1
fi

SCRIPT=posthoc_bridging/posthoc_bridging_D.py
EXPERIMENTS_DIR="../experiments"
RUN_IDS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --experiments_dir)
      EXPERIMENTS_DIR="$2"
      shift 2
      ;;
    --run_id)
      RUN_IDS+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ ${#RUN_IDS[@]} -eq 0 ]; then
  RUN_IDS=(
    CAT_llama3b_0075_relativelr_otherparams
    CAT_llama3b_0075_relativelr_123_otherparams
    CAT_llama3b_0075_relativelr_456_otherparams
    CAT_llama3b_0150_relativelr_otherparams
    CAT_llama3b_0150_relativelr_123_otherparams
    CAT_llama3b_0150_relativelr_456_otherparams
    CAT_llama3b_0300_relativelr_otherparams
    CAT_llama3b_0300_relativelr_123_otherparams
    CAT_llama3b_0300_relativelr_456_otherparams
    CAT_llama3b_0600_relativelr_otherparams
    CAT_llama3b_0600_relativelr_123_otherparams
    CAT_llama3b_0600_relativelr_456_otherparams
  )
fi

COMMON=(--experiments_dir "$EXPERIMENTS_DIR")

# 1) resolve + validate the 12 runs (aborts the job on any config/trace inconsistency)
echo "Running command: python $SCRIPT --mode list ${COMMON[*]}"
srun python "$SCRIPT" --mode list "${COMMON[@]}" || exit 1

# 2) fixed diagnostic plan shared by all checkpoints (no-op if it already exists)
echo "Running command: python $SCRIPT --mode plan ${COMMON[*]}"
srun python "$SCRIPT" --mode plan "${COMMON[@]}" || exit 1

# 3) one checkpoint per python process
FAILED=()
for RID in "${RUN_IDS[@]}"; do
  echo "================ $RID ================"
  echo "Running command: python $SCRIPT --mode attack --run_id $RID ${COMMON[*]}"
  if ! srun python "$SCRIPT" --mode attack --run_id "$RID" "${COMMON[@]}"; then
    echo "!!! $RID failed; continuing with the next checkpoint"
    FAILED+=("$RID")
  fi
done

# 4) tables (training-time A/B from traces vs post-hoc C)
echo "Running command: python $SCRIPT --mode summarize ${COMMON[*]}"
srun python "$SCRIPT" --mode summarize "${COMMON[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED checkpoints: ${FAILED[*]}"
  exit 1
fi

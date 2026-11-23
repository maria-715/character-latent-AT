#!/bin/bash
# ===================================
#SBATCH --job-name=extract_representations
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
  sbatch run_extraction.sh [--force] [--checkpoints PATH] [--prompt-sets PATH]

Optional:
  --force          Re-extract even if a checkpoint's output already exists
  --checkpoints    Default: checkpoints.yaml
  --prompt-sets    Default: prompt_sets.yaml
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      EXTRA_ARGS+=(--force)
      shift
      ;;
    --checkpoints)
      EXTRA_ARGS+=(--checkpoints "$2")
      shift 2
      ;;
    --prompt-sets)
      EXTRA_ARGS+=(--prompt-sets "$2")
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
ENV_NAME="env_repr_analysis"
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

if [[ -f "../.env" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "../.env" | cut -d '=' -f2-)"
else
  echo ".env file not found"
  exit 1
fi

export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"

srun python extract_representations.py "${EXTRA_ARGS[@]}"

#!/bin/bash

# ===================================
#SBATCH --job-name=run_evaluate_samples_newpeft
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
  # fschat (installed below) requires unpinned torch/transformers, which would
  # otherwise silently upgrade them past what the LoRA adapters and flash-attn
  # were built against. Pin just those to known-working versions on every
  # install in this block.
  CONSTRAINTS_FILE="$TMPDIR/pinned_versions.txt"
  cat > "$CONSTRAINTS_FILE" <<EOF
torch==2.3.0
numpy==1.26.4
transformers<4.46.0
EOF

  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" -r ../requirements/requirements_harmbench_newpeft.txt
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
  /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" -e "../harmbench/requirement/FastChat[model_worker,webui]"

  # flash-attn compiles CUDA kernels from source and can take a very long time.
  # env_harmbench pins the same python/torch versions, so reuse its already-built
  # copy instead of recompiling when it's available.
  OLD_ENV_SITE_PACKAGES="$CONDA_ENVS_PATH/env_harmbench/lib/python3.11/site-packages"
  NEW_ENV_SITE_PACKAGES="$CONDA_ENVS_PATH/$ENV_NAME/lib/python3.11/site-packages"
  if compgen -G "$OLD_ENV_SITE_PACKAGES/flash_attn*" > /dev/null; then
    echo "Reusing prebuilt flash-attn from env_harmbench"
    cp -r "$OLD_ENV_SITE_PACKAGES"/flash_attn* "$NEW_ENV_SITE_PACKAGES"/
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" einops
  else
    /scratch-ssd/oatml/run_locked.sh env TMPDIR="$TMPDIR" pip install -c "$CONSTRAINTS_FILE" flash-attn --no-build-isolation
  fi
fi
/scratch-ssd/ms25mds/conda_envs/env_harmbench_newpeft/bin/pip install anthropic==0.85.0

# ======================================================

if [ $# -ne 3 ]; then
    echo "Usage: $0 <attack> <model> <seed>"
    exit 1
fi


# HF token in .env file
ENV_FILE="../../.env"
if [ -f "$ENV_FILE" ]; then
  export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | cut -d '=' -f2-)
else
  echo ".env file not found"
  exit 1
fi
export HF_HOME=/scratch-ssd/$USER/.cache/huggingface
mkdir -p "$HF_HOME"
./../robustness_eval_harmbench/evaluation_command_newpeft.sh $1 $2 $3

"""
Non-interactive driver for the ORIGINAL LAT jailbreak-robustness training experiment.

This is a faithful port of notebooks/jailbreaks.ipynb into a plain script, so it can be run
headlessly from train_LAT.sh via Slurm instead of from Jupyter. It does not modify the LAT
algorithm: latent_at/ is imported and used as-is (ProjectedGradLAT, process_generic_chat_dataset,
LatentAdversarialTrainingDataCollator).

The notebook's cell 3 branches on a boolean `use_llama2` between a Llama-2-7b-chat and a
Llama-3-8B-Instruct configuration; --use_llama2 here mirrors that same boolean and switches
between the two branches' *differing* settings only (model path, def_loss_coefs, prompt
formatting, and the branch-specific defaults for --model_path/--epsilon/
--inner_learning_rate/--outer_learning_rate/--add_completions_pgd, each still explicitly
overridable). Everything the notebook does NOT branch on (pgd_layers, pgd_iterations_per_step,
model_iterations_per_step, max_batch_per_acc, num_steps, LoRA config, SYS_PROMPT string,
dataset/column names, MODEL_LAYERS_MODULE) stays a single shared value/default either way,
exactly as in the notebook. Default is --no-use_llama2, i.e. the Llama-3 branch, unchanged
from before this flag existed.

Every hyperparameter below is either:
  - sourced from notebooks/jailbreaks.ipynb (see the per-parameter comments), or
  - a ProjectedGradLAT / LoraConfig constructor default the notebook never overrode, or
  - new infrastructure added for non-interactive execution (seeding, argument parsing,
    the final checkpoint save) -- each such case is called out explicitly below.

Not ported from the notebook (deliberately, not by omission):
  - advbench_data / harmbench_data (HarmBenchTask): constructed in the notebook but never
    used afterwards -- dead code with respect to training.
  - the pre-/post-training single-prompt generation cells: evaluation is out of scope for
    this phase.
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as hf_set_seed

from latent_at import (
    LatentAdversarialTrainingDataCollator,
    ProjectedGradLAT,
    process_generic_chat_dataset,
)

# ----------------------------------------------------------------------------------------
# Fixed values copied verbatim from notebooks/jailbreaks.ipynb, cells 3/7/10.
# Not exposed on the CLI in this phase -- see accompanying plan for rationale.
# ----------------------------------------------------------------------------------------
ADV_LOSS_COEFS = {"toward": 0.5, "away": 0.5}  # cell-3, shared by both branches (identical dict)

# cell-3: def_loss_coefs is the one loss-side setting that DOES differ by branch.
DEF_LOSS_COEFS_LLAMA3 = {"kl": 0.1, "toward": 0.5, "away": 0.5}
DEF_LOSS_COEFS_LLAMA2 = {"sft": 1.5, "toward": 0.5, "away": 0.5}

# cell-3: branch-specific defaults for otherwise-shared CLI flags. Each remains explicitly
# overridable via its own flag; these only supply the default when the flag is omitted.
BRANCH_DEFAULTS_LLAMA3 = dict(
    model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    epsilon=6.0,
    inner_learning_rate=1e-3,
    outer_learning_rate=8e-5,
    add_completions_pgd=True,
)
BRANCH_DEFAULTS_LLAMA2 = dict(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    epsilon=6.0,
    inner_learning_rate=5e-2,
    outer_learning_rate=2e-5,
    add_completions_pgd=False,
)

LORA_R = 64  # cell-10
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]  # cell-10
# lora_alpha / lora_dropout / bias / task_type: not set by the notebook -> peft.LoraConfig's
# own defaults (8, 0.0, "none", None). Third-party library defaults, not a LAT or notebook value.

BATCH_SIZE = 16  # cell-7 (both lat_dataloader and sft_dataloader)
TRUNCATE_LENGTH = 2048  # cell-7
MODEL_LAYERS_MODULE = "base_model.model.model.layers"  # cell-10

SYS_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as "
    "possible, while being safe. Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. Please ensure that your "
    "responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)  # cell-7

LAT_DATASET = "LLM-LAT/harmful-dataset"  # cell-7
LAT_ADV_COLUMN = "rejected"
LAT_DEF_COLUMN = "chosen"

SFT_DATASET = "LLM-LAT/benign-dataset"  # cell-7
SFT_ADV_COLUMN = "refusal"
SFT_DEF_COLUMN = "response"


def seed_everything(seed: int) -> None:
    """
    NOT present in jailbreaks.ipynb -- the notebook never seeds anything.
    New infrastructure, mirroring Continuous-AdvTrain/src/run_experiments.py's
    seed_everything() convention, so Slurm runs are reproducible.
    """
    logging.info(f"Seeding everything with seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def parse_pgd_layers(raw: str) -> list:
    """'embedding,8,16,24,30' -> ['embedding', 8, 16, 24, 30] (cell-10's literal list)."""
    layers = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            layers.append(int(tok))
        else:
            layers.append(tok)
    return layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Non-interactive port of notebooks/jailbreaks.ipynb. Defaults reproduce the "
            "notebook's Llama-3-8B-Instruct branch exactly; pass --use_llama2 for the "
            "notebook's Llama-2-7b-chat branch instead."
        )
    )
    p.add_argument(
        "--logging_path",
        required=True,
        help="Output/run directory. The final LoRA adapter is saved to <logging_path>/final_model.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="New infra -- jailbreaks.ipynb does not seed anything. Default mirrors Continuous-AdvTrain.",
    )
    p.add_argument(
        "--use_llama2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "cell-3's `use_llama2` boolean. Switches model_path/def_loss_coefs/prompt "
            "formatting to the notebook's Llama-2-7b-chat branch, and switches the DEFAULT "
            "(each still independently overridable) of --model_path/--epsilon/"
            "--inner_learning_rate/--outer_learning_rate/--add_completions_pgd to that "
            "branch's values. Default: False (Llama-3 branch, unchanged)."
        ),
    )
    # Defaults for these five are None here and resolved in main() against
    # BRANCH_DEFAULTS_LLAMA{2,3} based on --use_llama2, so omitting a flag picks the correct
    # branch default instead of always silently reproducing the Llama-3 value.
    p.add_argument("--model_path", default=None, help="cell-3; branch-dependent default, see --use_llama2")
    p.add_argument("--epsilon", type=float, default=None, help="cell-3; branch-dependent default, see --use_llama2")
    p.add_argument("--inner_learning_rate", type=float, default=None, help="cell-3; branch-dependent default, see --use_llama2")
    p.add_argument("--outer_learning_rate", type=float, default=None, help="cell-3; branch-dependent default, see --use_llama2")
    p.add_argument("--pgd_iterations_per_step", type=int, default=16, help="cell-10, shared by both branches")
    p.add_argument(
        "--pgd_layers",
        default="embedding,8,16,24,30",
        help="Comma-separated; non-numeric tokens kept as strings. cell-10 default: embedding,8,16,24,30 (shared by both branches)",
    )
    p.add_argument("--num_steps", type=int, default=100, help="cell-10, shared by both branches")
    p.add_argument("--model_iterations_per_step", type=int, default=4, help="cell-10, shared by both branches")
    p.add_argument("--max_batch_per_acc", type=int, default=2, help="cell-10, shared by both branches")
    p.add_argument(
        "--add_completions_pgd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cell-3; branch-dependent default, see --use_llama2",
    )
    p.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "New infra, off by default. Records (does not alter) per-PGD-iteration, "
            "per-attacked-layer perturbation-norm statistics around the "
            "optimizer.step()/clip_attack() boundary, to <logging_path>/diagnostics.csv "
            "and concise console output. See latent_at/diagnostics.py."
        ),
    )
    p.add_argument("--wandb_project", default="jailbreaks_test", help="cell-12 project_name")
    p.add_argument(
        "--wandb_run_name",
        default=None,
        help="Not set by the notebook (ProjectedGradLAT.train()'s name= defaults to None).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Resolve branch-dependent defaults (cell-3): only fills in flags the user left unset.
    branch_defaults = BRANCH_DEFAULTS_LLAMA2 if args.use_llama2 else BRANCH_DEFAULTS_LLAMA3
    for key, default_val in branch_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_val)
    def_loss_coefs = DEF_LOSS_COEFS_LLAMA2 if args.use_llama2 else DEF_LOSS_COEFS_LLAMA3

    os.makedirs(args.logging_path, exist_ok=True)
    seed_everything(args.seed)

    # HF auth: reproduces jailbreaks.ipynb cell-1 exactly (load_dotenv() + HUGGINGFACE_API_KEY,
    # passed explicitly as token=). This is LAT's own convention, distinct from
    # Continuous-AdvTrain's implicit HF_TOKEN env var -- see train_LAT.sh for how the two
    # are reconciled before this script ever runs.
    load_dotenv()
    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
    if hf_access_token is None:
        logging.warning(
            "HUGGINGFACE_API_KEY is not set. Falling back to unauthenticated / ambient HF auth, "
            "which will fail for a gated model such as meta-llama/Meta-Llama-3-8B-Instruct."
        )

    model_dtype = torch.bfloat16  # cell-5
    device = "cuda"  # cell-5

    # ======= Model + tokenizer (cell-5; model_dtype/device/tokenizer setup shared by both branches) ======= #
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        token=hf_access_token,
        torch_dtype=model_dtype,
    ).to(device)

    # NOTE: the notebook's tokenizer load for the llama3 branch does not pass token= explicitly
    # (only the model load does). Passed here explicitly for both, for robustness in a headless
    # context where no ambient `huggingface-cli login` state can be assumed -- a deliberate,
    # documented deviation for auth robustness only, not a training-behavior change.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=hf_access_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ======= Data (cell-7) ======= #
    if args.use_llama2:
        # cell-7 llama2 branch: use the tokenizer's own chat template (Llama-2-chat's
        # [INST]<<SYS>>...<</SYS>>...[/INST]), with SYS_PROMPT passed through as system_prompt=
        # (see apply_chat_formatting in latent_at/lat_datasets.py for exactly how it's applied).
        use_tokenizer_template = True
        custom_prompt_template = None
        custom_completion_template = None
    else:
        # cell-7 llama3 branch
        use_tokenizer_template = False
        custom_prompt_template = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{SYS_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        custom_completion_template = "{completion}"

    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=LAT_DATASET,
        adv_column=LAT_ADV_COLUMN,
        def_column=LAT_DEF_COLUMN,
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=SYS_PROMPT,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
    )
    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=TRUNCATE_LENGTH,
        ),
    )

    # interleaving supervised finetuning with LAT stabilizes training (cell-7 comment)
    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=SFT_DATASET,
        adv_column=SFT_ADV_COLUMN,
        def_column=SFT_DEF_COLUMN,
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=SYS_PROMPT,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
        add_eos_token=True,
    )
    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=TRUNCATE_LENGTH,
        ),
    )

    # ======= LoRA + Trainer (cell-10) ======= #
    peft_config = LoraConfig(r=LORA_R, target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(model, peft_config)

    pgd_layers = parse_pgd_layers(args.pgd_layers)

    pgd_trainer = ProjectedGradLAT(
        model=model,  # model
        dataloader=lat_dataloader,  # dataloader for lat
        sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
        adv_loss_coefs=ADV_LOSS_COEFS,  # adversary's loss coefs
        def_loss_coefs=def_loss_coefs,  # model's loss coefs (branch-dependent: sft vs kl)
        pgd_layers=pgd_layers,  # what layers to attack
        pgd_iterations_per_step=args.pgd_iterations_per_step,  # how many steps of PGD to do
        model_layers=list(range(0, model.config.num_hidden_layers)),  # model layers to train
        epsilon=args.epsilon,  # attack l2 constraint
        inner_learning_rate=args.inner_learning_rate,  # adversary lr
        outer_learning_rate=args.outer_learning_rate,  # model lr
        model_iterations_per_step=args.model_iterations_per_step,  # how many times to train on each step
        num_steps=args.num_steps,  # number of epochs
        max_batch_per_acc=args.max_batch_per_acc,  # max size of a minibatch
        only_train_lora=True,  # train using low rank adapters
        l2_regularization=0,  # coef for l2 weight regularization
        model_layers_module=MODEL_LAYERS_MODULE,  # where the model layers are
        reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step
        add_completions_pgd=args.add_completions_pgd,  # whether to add PGD over completion tokens
        diagnostics=args.diagnostics,  # opt-in, observational-only PGD inner-loop diagnostics
        diagnostics_dir=args.logging_path,
    )

    # ======= Run! (cell-12) ======= #
    pgd_trainer.train(project_name=args.wandb_project, name=args.wandb_run_name)

    # ======= Save Model =======
    # NEW: the notebook's own save_pretrained call (cell-12) is commented out in the source --
    # the notebook never actually persisted a checkpoint. Added here per requirement that a
    # completed Slurm job leave a clearly identifiable final adapter, mirroring
    # Continuous-AdvTrain's `<checkpoint_path>/final_model` naming convention.
    final_dir = os.path.join(args.logging_path, "final_model")
    pgd_trainer.model.save_pretrained(final_dir)
    logging.info(f"Saved final LAT adapter to {final_dir}")


if __name__ == "__main__":
    main()

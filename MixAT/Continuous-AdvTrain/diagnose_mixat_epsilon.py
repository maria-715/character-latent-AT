"""
Read-only diagnostic: is MixAT's continuous-attack epsilon ball binding?

Loads only the embedding weights of a checkpoint and compares the configured
L2 epsilon ball (as computed in embedding_attack.py::EmbeddingSpaceAttack,
`self.eps = eps_multiplier * mean(||embed_row||_2)`) against the deterministic
per-step SignSGD update norm (`lr * sqrt(hidden_size)`) and its K-step ceiling.

No training, no gradients, no fine-tuning, no writes to the checkpoint or to
this repo. Everything runs under torch.no_grad() and the model is only used
to read `model.model.embed_tokens.weight`.

Defaults mirror Continuous-AdvTrain/config/Llama3_MixAT.yaml:
  adversarial.eps = 0.075, adversarial.opt_config.lr = 1.0e-04, adversarial.iters = 10
"""

import argparse
import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--eps-multiplier", type=float, default=0.075)
    parser.add_argument("--inner-lr", type=float, default=1.0e-4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="dtype to load the checkpoint in before casting the embedding matrix to float32 for the norm",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch_dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(args.model_path)
    hidden_size = config.hidden_size

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    with torch.no_grad():
        embed_weights = model.model.embed_tokens.weight.detach().float()
        embedding_norm = torch.norm(embed_weights, p=2, dim=-1).mean().item()

    raw_epsilon = args.eps_multiplier * embedding_norm
    step_norm = args.inner_lr * math.sqrt(hidden_size)
    k_step_norm = args.iters * step_norm

    print(f"model_path                   = {args.model_path}")
    print(f"hidden_size                  = {hidden_size}")
    print(f"mean embedding-row L2 norm   = {embedding_norm:.6f}")
    print(f"eps multiplier (config)      = {args.eps_multiplier}")
    print(f"raw L2 epsilon               = {raw_epsilon:.6f}")
    print(f"SignSGD step L2 norm         = {step_norm:.6f}")
    print(f"K * step L2 norm  (K={args.iters})     = {k_step_norm:.6f}")
    print(f"raw_epsilon / step_norm      = {raw_epsilon / step_norm:.4f}")
    print(f"raw_epsilon / (K*step_norm)  = {raw_epsilon / k_step_norm:.4f}")
    print()
    ratio = raw_epsilon / k_step_norm
    if ratio >= 1.0:
        print(f"raw_epsilon >= K*step_norm ({ratio:.2f}x) -> the eps ball is likely NON-BINDING: "
              f"{args.iters} fixed-magnitude SignSGD steps cannot reach the ball edge.")
    else:
        print(f"raw_epsilon < K*step_norm ({ratio:.2f}x) -> the eps ball is likely the ACTIVE constraint: "
              f"projection can clip perturbations before {args.iters} steps are exhausted.")


if __name__ == "__main__":
    main()

import argparse
import json
from datasets import load_dataset
import torch
import sys
import os
from tqdm import tqdm
import subprocess
import math
import re
import random
import numpy as np

sys.path.append(os.path.abspath("../Continuous-AdvTrain/src"))
from model_utils import (
    load_model_and_tokenizer,
    load_checkpoint,
)


def print_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU] allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


def print_gpu_stats():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
         "--format=csv,nounits,noheader"]
    )
    print("[GPU]", result.decode().strip())


def generate_batch(model, tokenizer, prompts):

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]

    # print("\n=== FORMATTED PROMPT SAMPLE ===\n")
    # print(formatted[0][:min(500, len(formatted[0]))])
    # print("\n==============================\n")

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, prompt_len:]

    # input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    # generated_tokens = [
    #     outputs[i, input_lengths[i]:]
    #     for i in range(outputs.shape[0])
    # ]

    responses = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return [r.strip() for r in responses]


ASSISTANT_PREFIX_RE = re.compile(
    r'^\s*(?:<\|assistant\|>|<\|assistant\|>|assistant\|>|'
    r'\|assistant\|>|\|>\s*|istant\|>)+\s*',
    re.IGNORECASE
)

def normalise_completion(s: str) -> str:
    s = s.strip()
    s = ASSISTANT_PREFIX_RE.sub("", s)
    return s.lower().lstrip()


def main(args):

    # I added the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.eval()

    # print("\nCHAT TEMPLATE:\n", tokenizer.chat_template)

    if args.adapter is not None:
        model = load_checkpoint(model, {"load_checkpoint_path": args.adapter})

    dataset = load_dataset("Paul/XSTest")["train"]

    output_dir = os.path.dirname(args.output)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, "w") as f:

        batch_size = args.batch_size
        for i in tqdm(range(0, len(dataset), batch_size), total=math.ceil(len(dataset) / batch_size), desc="Generating XSTest responses"):
            
            if i % 10 == 0:
                print_gpu_usage()
                print_gpu_stats()

            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            prompts = batch["prompt"]
            types = batch["type"]
            labels = batch["label"]

            responses = generate_batch(model, tokenizer, prompts)

            # if i == 0:
            #     for j in range(min(5, len(prompts))):
            #         print("\n====================")
            #         print("\nPROMPT:\n", prompts[j])
            #         print("\nRESPONSE:\n", responses[j])

            for prompt, type, label, response in zip(prompts, types, labels, responses):
                                
                record = {
                    "prompt": prompt,
                    "completion_raw": response,
                    "completion": normalise_completion(response),
                    "type": type,
                    "label": label,
                }   
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")

    main(args)
"""
generate_utility_dataset_colab.py

Colab/GPU version. Reads a CSV of prompts (single "prompt" column) and
produces a new CSV with "prompt","response" columns, generating each
response with a Hugging Face causal LM on GPU, with batched generation for
speed and 4-bit quantization by default so it comfortably fits a free-tier
Colab T4 (16GB VRAM).

Usage (run from a Colab cell with `!python ...`, or as a normal script):

    !python generate_utility_dataset_colab.py \
        --model_id HuggingFaceH4/zephyr-7b-beta \
        --input harmless_synthetic_1200.csv \
        --output /content/drive/MyDrive/utility_zephyr_7b_beta.csv \
        --hf_token YOUR_HF_TOKEN \
        --batch_size 8

    !python generate_utility_dataset_colab.py \
        --model_id meta-llama/Meta-Llama-3-8B-Instruct \
        --input harmless_synthetic_1200.csv \
        --output /content/drive/MyDrive/utility_llama3_8b_instruct.csv \
        --hf_token YOUR_HF_TOKEN \
        --batch_size 8

Write --output to a Google-Drive-mounted path (see README) so progress
survives a Colab disconnect -- /content/ storage is wiped when the runtime
resets.
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# DEFAULT_SYSTEM_PROMPT = (
#     "You are a helpful, knowledgeable assistant. Answer the user's request "
#     "directly and naturally, scaling your length and level of detail to what "
#     "the request actually calls for -- concise requests get concise answers, "
#     "requests for detailed or step-by-step content get real detail. "
#     "Do not start with stock phrases like 'Certainly!', 'Sure!', or 'I'd be "
#     "happy to help!' -- just answer. Do not add disclaimers, safety caveats, "
#     "or meta-commentary about being an AI unless the request genuinely "
#     "requires it. Do not end with 'let me know if you have any questions' or "
#     "similar filler. Write in clear, natural prose. Make sure your response "
#     "reaches a clear, complete conclusion rather than trailing off. "
#     "When writing a persuasive article or essay, do not open with a generic "
#     "'big picture' framing sentence like 'In today's fast-paced world...', "
#     "'As society becomes increasingly aware...', or similar broad, sweeping "
#     "openers -- start from something specific instead, such as a concrete "
#     "detail, a fact, a scenario, or the topic itself, and vary your opening "
#     "approach across different pieces. Also avoid the formulaic "
#     "'Firstly... Secondly... In conclusion...' essay structure -- organize "
#     "the argument in whatever way fits the specific topic."
# )

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."


def get_terminator_ids(tokenizer):
    """EOS token id(s) to stop generation on. Handles Llama-3's extra
    <|eot_id|> end-of-turn token in addition to the standard eos token."""
    terminators = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in terminators:
            terminators.append(eot_id)
    except Exception:
        pass
    return terminators


def load_done_prompts(output_path):
    if not os.path.exists(output_path):
        return set()
    try:
        done_df = pd.read_csv(output_path)
        return set(done_df["prompt"].astype(str).tolist())
    except Exception:
        return set()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = argparse.ArgumentParser(description="Generate utility-dataset responses on Colab GPU.")
    parser.add_argument("--model_id", required=True, help="HF repo id, e.g. HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--input", required=True, help="Path to input CSV with a 'prompt' column")
    parser.add_argument("--output", required=True, help="Path to output CSV to create/append to")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face access token")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                         help="Generous ceiling so responses reach a natural EOS instead of "
                              "being cut off -- not a target length, just a safety cap")
    parser.add_argument("--temperature", type=float, default=0.0,
                         help="0.0 = greedy decoding (matches eval_harmless's generation setup). "
                              "Set > 0 to re-enable sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Only used when --temperature > 0")
    parser.add_argument("--repetition_penalty", type=float, default=1.15,
                         help="Guards against repetitive loops now that generations can run longer")
    parser.add_argument("--batch_size", type=int, default=8, help="Prompts generated per GPU forward pass")
    parser.add_argument("--precision", choices=["4bit", "fp16", "bf16"], default="4bit",
                         help="4bit fits a free T4 comfortably and is fastest per-VRAM; "
                              "use fp16/bf16 if you have a bigger GPU (A100/L4) and want max quality")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N (unfinished) prompts -- use to test speed first")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no GPU detected by torch. In Colab: Runtime -> Change runtime type -> GPU.")
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    if not args.hf_token:
        print("WARNING: no --hf_token given and HF_TOKEN env var not set. "
              "This will fail for gated models like meta-llama/Meta-Llama-3-8B-Instruct.")
    else:
        login(token=args.hf_token)

    print(f"Loading tokenizer and model: {args.model_id} ({args.precision})")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token or None)
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(token=args.hf_token or None, device_map="auto")
    if args.precision == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.precision == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    else:  # bf16
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f}s")

    terminators = get_terminator_ids(tokenizer)

    # ---- Load input, figure out what's left to do ----
    df = pd.read_csv(args.input)
    if "prompt" not in df.columns:
        print("ERROR: input CSV must have a column named 'prompt'.")
        sys.exit(1)

    done = load_done_prompts(args.output)
    remaining = [p for p in df["prompt"].astype(str).tolist() if p not in done]
    print(f"Total prompts: {len(df)} | already done: {len(done)} | remaining: {len(remaining)}")

    if args.limit is not None:
        remaining = remaining[: args.limit]
        print(f"--limit set: only processing {len(remaining)} prompts this run.")

    if not remaining:
        print("Nothing to do -- output file already has a response for every prompt.")
        return

    write_header = not os.path.exists(args.output)
    out_f = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writerow(["prompt", "response"])
        out_f.flush()

    batch_times = []
    total_done = 0
    for batch in chunks(remaining, args.batch_size):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": args.system_prompt},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            gen_kwargs.update(do_sample=False)  # greedy -- temperature/top_p are meaningless here

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(**enc, **gen_kwargs)
        elapsed = time.time() - start
        batch_times.append(elapsed)

        input_len = enc["input_ids"].shape[-1]  # same for every row: left-padded
        for i, prompt in enumerate(batch):
            response_ids = output_ids[i][input_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            writer.writerow([prompt, response])
        out_f.flush()
        os.fsync(out_f.fileno())

        total_done += len(batch)
        avg = sum(batch_times) / len(batch_times)
        remaining_batches = (len(remaining) - total_done) / args.batch_size
        eta_min = (avg * remaining_batches) / 60
        print(f"[{total_done}/{len(remaining)}] batch took {elapsed:.1f}s "
              f"({elapsed/len(batch):.1f}s/prompt) | ETA {eta_min:.0f} min")

    out_f.close()
    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()

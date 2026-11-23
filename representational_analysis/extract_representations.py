"""
extract_representations.py

For every checkpoint in checkpoints.yaml, and every prompt in every prompt set
listed in prompt_sets.yaml: format the prompt exactly as CAT training does
(model_utils.get_chat_template), run a forward pass with output_hidden_states=True,
and save per-layer, per-prompt hidden states plus the first-generated-token logits.
Nothing is aggregated across prompts here -- that happens only in analysis.py /
report code, on top of the raw saved arrays.

Three pooled representations are saved per layer, per prompt:
  - last_token:      hidden state at the LAST token of (formatted instruction +
                      response_key), i.e. immediately before the model would
                      generate its first response token. This is the true
                      "generation point" / decision representation (matches
                      add_generation_prompt=True convention, e.g. in
                      MixAT/eval_harmless/evaluation.py). For Llama-3 this is the
                      last token of "<|start_header_id|>assistant<|end_header_id|>\\n\\n",
                      NOT the <|eot_id|> that closes the user's turn.
  - last_user_token:  hidden state at the last token of the raw instruction text
                      itself, i.e. immediately before the chat template's closing
                      tokens (e.g. <|eot_id|>) begin. Computed from
                      instruction_start_token/instruction_end_token (see below);
                      these boundaries are derived by tokenizing the chat
                      template's literal prefix/suffix text separately and using
                      their lengths, which can be off by a token at the seam in
                      rare cases where BPE merges differently across the
                      concatenation boundary (tokenizer enc(a+b) != enc(a)+enc(b)
                      is a known issue elsewhere in this codebase, see
                      model_utils.py's Llama2 comment) -- raw input_ids are also
                      saved per prompt so this can be re-derived/verified exactly
                      later without rerunning extraction.
  - mean_pooled:      mean over every token of (formatted instruction + response_key),
                      i.e. including chat-template boilerplate, not just the
                      instruction span. instruction_start_token/instruction_end_token
                      + input_ids are saved specifically so an instruction-only mean
                      can be computed later without rerunning extraction.

Usage:
    python extract_representations.py [--checkpoints checkpoints.yaml] [--prompt-sets prompt_sets.yaml] [--force]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Continuous-AdvTrain" / "src"))
import model_utils  # noqa: E402

# Matches the quantization CAT training and MixAT/eval_harmless/evaluation.py both use --
# extracting representations under the same precision the model is actually trained/
# evaluated with, rather than switching to full bf16 for extraction only.
NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_prompt_set(name, spec, base_dir):
    fmt = spec["format"]
    if fmt == "plain_txt":
        path = (base_dir / spec["source"]).resolve()
        with open(path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif fmt == "csv_column":
        path = (base_dir / spec["source"]).resolve()
        df = pd.read_csv(path)
        texts = df[spec["column"]].dropna().astype(str).tolist()
        sample_size = spec.get("sample_size")
        if sample_size is not None and sample_size < len(texts):
            rng = np.random.RandomState(spec.get("sample_seed", 0))
            idx = np.sort(rng.choice(len(texts), size=sample_size, replace=False))
            texts = [texts[i] for i in idx]
    else:
        raise ValueError(f"Unsupported prompt_sets format: {fmt!r} for prompt set {name!r}")

    return [(f"{name}_{i:05d}", name, text) for i, text in enumerate(texts)]


def load_all_prompts(prompt_sets_path):
    with open(prompt_sets_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = prompt_sets_path.resolve().parent
    prompts = []
    for name, spec in cfg["prompt_sets"].items():
        set_prompts = load_prompt_set(name, spec, base_dir)
        print(f"[INFO] prompt set {name!r}: {len(set_prompts)} prompts")
        prompts.extend(set_prompts)
    return prompts


def load_checkpoints(checkpoints_path):
    with open(checkpoints_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["checkpoints"]


def output_complete(out_dir):
    return (
        (out_dir / "hidden_states.npz").exists()
        and (out_dir / "first_token_logits.npz").exists()
        and (out_dir / "prompt_index.parquet").exists()
        and (out_dir / "metadata.json").exists()
    )


def extract_for_checkpoint(ckpt, prompts, repr_dir, force):
    ckpt_id = ckpt["id"]
    out_dir = repr_dir / ckpt_id

    if output_complete(out_dir) and not force:
        print(f"[OK] {ckpt_id}: already extracted, skipping (--force to redo)")
        return

    adapter_path = None
    if ckpt.get("adapter_path"):
        adapter_path = (HERE / ckpt["adapter_path"]).resolve()
        if not adapter_path.exists():
            print(f"[WARN] {ckpt_id}: adapter path does not exist, skipping: {adapter_path}")
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_utils.get_model_name(ckpt["base_model"])
    first_user_msg, _, _, response_key = model_utils.get_chat_template(model_name)

    # Instruction token boundaries: tokenize the template's literal prefix/suffix
    # (the text surrounding "{instruction}") separately so their lengths can be
    # used as offsets into the full tokenized sequence. See the seam caveat in
    # the module docstring -- exact for every template checked so far (no
    # trailing/leading partial-word issues at these particular boundaries), but
    # not guaranteed in general, hence saving raw input_ids too.
    _prefix_text, _sep, _suffix_text = first_user_msg.partition("{instruction}")
    assert _sep == "{instruction}", f"first_user_msg for {model_name!r} has no '{{instruction}}' placeholder"

    print(f"[INFO] {ckpt_id}: loading base model {ckpt['base_model']}")
    model, tokenizer = model_utils.load_model_and_tokenizer(ckpt["base_model"], bnb_config=NF4_CONFIG, dtype="bf16")

    prefix_ids = tokenizer(_prefix_text, add_special_tokens=False)["input_ids"] if _prefix_text else []
    suffix_ids = tokenizer(_suffix_text, add_special_tokens=False)["input_ids"] if _suffix_text else []

    if adapter_path is not None:
        print(f"[INFO] {ckpt_id}: loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    model.eval()

    n_prompts = len(prompts)
    last_token_arr = None
    last_user_token_arr = None
    mean_pooled_arr = None
    logits_arr = None
    prompt_index_rows = []
    n_layers = hidden_dim = vocab_size = None

    with torch.no_grad():
        for i, (prompt_id, dataset, text) in enumerate(prompts):
            user_turn_text = first_user_msg.format(instruction=text)
            full_text = user_turn_text + response_key  # generation-point prefix, see module docstring

            user_turn_ids = tokenizer(user_turn_text, add_special_tokens=False)["input_ids"]
            instruction_start_token = len(prefix_ids)
            instruction_end_token = len(user_turn_ids) - len(suffix_ids)
            print("\nRecovered instruction:")
            print(
                tokenizer.decode(
                    ids[instruction_start_token:instruction_end_token]
                )
            )

            enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            if i == 0:
                print("=" * 80)
                print("RAW INSTRUCTION:")
                print(text)

                print("\nFULL FORMATTED PROMPT:")
                print(repr(full_text))

                ids = enc["input_ids"][0].cpu().tolist()

                print("\nTOKEN IDS:")
                print(ids)

                print("\nTOKENS:")
                for j, tok_id in enumerate(ids):
                    print(
                        f"{j:3d} "
                        f"{tok_id:8d} "
                        f"{repr(tokenizer.decode([tok_id]))}"
                    )

                print("\nInstruction token span:")
                print(instruction_start_token, instruction_end_token)

                print("\nLast instruction token:")
                print(
                    instruction_end_token - 1,
                    repr(tokenizer.decode([ids[instruction_end_token - 1]]))
                )

                print("\nLast token (generation point):")
                print(
                    len(ids) - 1,
                    repr(tokenizer.decode([ids[-1]]))
                )

                print("=" * 80)

            enc = {k: v.to(model.device) for k, v in enc.items()}

            outputs = model(**enc, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple(len = n_layers+1), each (1, seq, hidden)

            if last_token_arr is None:
                n_layers = len(hidden_states)
                hidden_dim = hidden_states[0].shape[-1]
                vocab_size = outputs.logits.shape[-1]
                last_token_arr = np.zeros((n_layers, n_prompts, hidden_dim), dtype=np.float16)
                last_user_token_arr = np.zeros((n_layers, n_prompts, hidden_dim), dtype=np.float16)
                mean_pooled_arr = np.zeros((n_layers, n_prompts, hidden_dim), dtype=np.float16)
                logits_arr = np.zeros((n_prompts, vocab_size), dtype=np.float16)

            for layer_idx, layer_hidden in enumerate(hidden_states):
                layer_hidden = layer_hidden[0]  # (seq, hidden) -- batch size is always 1 here
                last_token_arr[layer_idx, i] = layer_hidden[-1].to(torch.float16).cpu().numpy()
                last_user_token_arr[layer_idx, i] = (
                    layer_hidden[instruction_end_token - 1].to(torch.float16).cpu().numpy()
                )
                mean_pooled_arr[layer_idx, i] = layer_hidden.mean(dim=0).to(torch.float16).cpu().numpy()

            first_token_logits = outputs.logits[0, -1]
            logits_arr[i] = first_token_logits.to(torch.float16).cpu().numpy()
            first_pred_id = int(first_token_logits.argmax().item())

            prompt_index_rows.append(
                {
                    "prompt_id": prompt_id,
                    "dataset": dataset,
                    "formatted_prompt": full_text,
                    "formatted_token_length": enc["input_ids"].shape[-1],
                    "input_ids": enc["input_ids"][0].cpu().tolist(),
                    "instruction_start_token": instruction_start_token,
                    "instruction_end_token": instruction_end_token,
                    "first_predicted_token_id": first_pred_id,
                    "first_predicted_token_str": tokenizer.decode([first_pred_id]),
                }
            )

            if (i + 1) % 100 == 0 or (i + 1) == n_prompts:
                print(f"[INFO] {ckpt_id}: {i + 1}/{n_prompts} prompts done")

    np.savez_compressed(
        out_dir / "hidden_states.npz",
        last_token=last_token_arr,
        last_user_token=last_user_token_arr,
        mean_pooled=mean_pooled_arr,
    )
    np.savez_compressed(out_dir / "first_token_logits.npz", logits=logits_arr)
    pd.DataFrame(prompt_index_rows).to_parquet(out_dir / "prompt_index.parquet", index=False)

    metadata = dict(ckpt)
    metadata["n_hidden_state_layers"] = n_layers
    metadata["hidden_dim"] = hidden_dim
    metadata["vocab_size"] = vocab_size
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] {ckpt_id}: extraction complete -> {out_dir}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default=str(HERE / "checkpoints.yaml"))
    parser.add_argument("--prompt-sets", default=str(HERE / "prompt_sets.yaml"))
    parser.add_argument("--force", action="store_true", help="Re-extract even if output already exists")
    args = parser.parse_args()

    repr_dir = HERE / "representations"
    repr_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_all_prompts(Path(args.prompt_sets))
    print(f"[INFO] {len(prompts)} total prompts across {len({p[1] for p in prompts})} datasets")

    registry_path = repr_dir / "prompts_registry.parquet"
    if not registry_path.exists() or args.force:
        pd.DataFrame(prompts, columns=["prompt_id", "dataset", "prompt_text"]).to_parquet(registry_path, index=False)
        print(f"[OK] wrote {registry_path}")

    checkpoints = load_checkpoints(Path(args.checkpoints))
    print(f"[INFO] {len(checkpoints)} checkpoints in manifest")

    for ckpt in checkpoints:
        try:
            extract_for_checkpoint(ckpt, prompts, repr_dir, args.force)
        except Exception as e:  # noqa: BLE001 -- one bad checkpoint shouldn't abort the whole batch
            print(f"[WARN] {ckpt['id']}: extraction failed, skipping: {e!r}")


if __name__ == "__main__":
    main()

"""
trim_or_answers.py

Takes the OR-Bench answers we already generated (long, well-written, complete
answers -- generated with a generous token budget so they reach a natural
conclusion) and produces a shorter version of each response by cutting it at
the last full sentence/list-item boundary that still fits within the actual
per-example SFT training budget (max_seq_length minus that row's real
prompt+chat-template token count, computed with the real tokenizer).

Content completeness doesn't matter here -- a response that only covers half
of what the model would have said, but ends cleanly on a full sentence, is a
good training target. A response cut off mid-word/mid-clause is not. This
script never produces the latter: it only ever cuts at sentence or
list-item boundaries, never mid-thought.

No CLI args -- edit MAX_SEQ_LENGTH / JOBS below and run directly.
"""

import csv
import os
import re
import sys

from transformers import AutoTokenizer

# model_utils.get_chat_template lives in the CAT training codebase -- import
# it directly so the prompt+template formatting used here is guaranteed to
# match what training actually does, rather than re-deriving it by hand.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Continuous-AdvTrain", "src"))
import model_utils  # noqa: E402

# Must match sfttrainer.max_seq_length in the training configs
# (llama_likeZephyr_adv_train_ul.yaml / llama_likeLlama2_adv_train_ul.yaml).
MAX_SEQ_LENGTH = 256

HERE = os.path.dirname(__file__)

JOBS = [
    {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_name": "llama-3",
        "input_csv": os.path.join(HERE, "datasets", "or-bench_with_answers", "utility_llama3_8b_instruct_from80kto1k_train.csv"),
        "output_csv": os.path.join(HERE, "utility_llama3_8b_instruct_from80kto1k_train_TRIMMED.csv"),
    },
    {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_name": "llama-3",
        "input_csv": os.path.join(HERE, "datasets", "or-bench_with_answers", "utility_llama3_8b_instruct_from80kto1k_eval.csv"),
        "output_csv": os.path.join(HERE, "utility_llama3_8b_instruct_from80kto1k_eval_TRIMMED.csv"),
    },
    {
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "model_name": "mistral",  # zephyr uses the "mistral" chat-template branch in model_utils.get_chat_template
        "input_csv": os.path.join(HERE, "datasets", "or-bench_with_answers", "utility_zephyr_7b_beta_from80kto1k_train.csv"),
        "output_csv": os.path.join(HERE, "utility_zephyr_7b_beta_from80kto1k_train_TRIMMED.csv"),
    },
    {
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "model_name": "mistral",
        "input_csv": os.path.join(HERE, "datasets", "or-bench_with_answers", "utility_zephyr_7b_beta_from80kto1k_eval.csv"),
        "output_csv": os.path.join(HERE, "utility_zephyr_7b_beta_from80kto1k_eval_TRIMMED.csv"),
    },
]


def load_hf_token():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    env_file = os.path.join(HERE, "..", ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    return ""


# Boundaries we're allowed to cut at: end of a sentence (. ! ? optionally
# followed by a closing quote/paren), or a paragraph/line break -- the latter
# catches list items that don't end in terminal punctuation (e.g. a bolded
# header line). Every candidate is then checked by _is_dangling() to reject
# cuts that land right after a bare list marker ("6.") or a header/label
# that ends in ":" promising content that isn't there -- those aren't
# mid-word breaks, but they still read as an unfinished thought.
_SENTENCE_END_RE = re.compile(r"[.!?][\"'\)\]]*(?=\s|$)")
_LINE_BREAK_RE = re.compile(r"\n+")
_LEADING_MARKER_RE = re.compile(r"^\s*\*{0,2}\s*(\d+[.\)]|[-*•])?\s*\*{0,2}\s*")


def _is_dangling(line_fragment):
    stripped = line_fragment.strip()
    if not stripped:
        return True
    no_bold = stripped.rstrip("*").rstrip()
    if not no_bold or no_bold.endswith(":"):
        return True
    after_marker = _LEADING_MARKER_RE.sub("", stripped, count=1)
    after_marker_letters = re.sub(r"[^a-zA-Z]", "", after_marker)
    return len(after_marker_letters) < 3


def find_boundaries(text):
    raw_positions = {m.end() for m in _SENTENCE_END_RE.finditer(text)}
    raw_positions.update(m.start() for m in _LINE_BREAK_RE.finditer(text))
    raw_positions.add(len(text))

    good = []
    for pos in raw_positions:
        line_start = text.rfind("\n", 0, pos) + 1  # 0 if no preceding newline
        fragment = text[line_start:pos]
        if not _is_dangling(fragment):
            good.append(pos)
    return sorted(set(good))


def trim_response(tokenizer, first_user_msg, response_template, prompt, response):
    """Returns (trimmed_response, was_trimmed, hit_fallback)."""
    boundaries = find_boundaries(response)

    prefix_text = first_user_msg.format(instruction=prompt)

    def total_len(candidate):
        full_text = prefix_text + response_template.format(target=candidate)
        return len(tokenizer(full_text, add_special_tokens=False)["input_ids"])

    # Fast path: whole response already fits, no trimming needed.
    if total_len(response.strip()) <= MAX_SEQ_LENGTH:
        return response.strip(), False, False

    # Walk boundaries from longest to shortest, keep the first (longest) that fits.
    for pos in reversed(boundaries):
        candidate = response[:pos].strip()
        if not candidate:
            continue
        if total_len(candidate) <= MAX_SEQ_LENGTH:
            return candidate, True, False

    # Fallback: not even the first sentence fits (very long prompt, or a
    # response with no punctuation at all) -- hard-truncate at the token
    # level as a last resort. Rare; counted and reported at the end.
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    response_prefix_template_ids = tokenizer(response_template.format(target=""), add_special_tokens=False)["input_ids"]
    budget = MAX_SEQ_LENGTH - len(prefix_ids) - len(response_prefix_template_ids)
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    truncated_ids = response_ids[: max(budget, 0)]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip(), True, True


def process_job(job):
    print(f"=== {job['input_csv']} -> {job['output_csv']} ===")
    tokenizer = AutoTokenizer.from_pretrained(job["model_id"], token=load_hf_token() or None)

    first_user_msg, _, response_template, _ = model_utils.get_chat_template(job["model_name"])

    with open(job["input_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    trimmed_count = 0
    fallback_count = 0
    out_rows = []
    for row in rows:
        new_response, was_trimmed, hit_fallback = trim_response(
            tokenizer, first_user_msg, response_template, row["prompt"], row["response"]
        )
        out_rows.append({"prompt": row["prompt"], "response": new_response})
        trimmed_count += int(was_trimmed)
        fallback_count += int(hit_fallback)

    with open(job["output_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"  rows: {len(rows)} | trimmed: {trimmed_count} | hit hard-truncate fallback: {fallback_count}")


def main():
    for job in JOBS:
        process_job(job)


if __name__ == "__main__":
    main()

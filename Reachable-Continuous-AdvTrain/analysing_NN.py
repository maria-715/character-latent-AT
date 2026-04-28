import argparse
import json
from collections import Counter
import matplotlib.pyplot as plt


# Loading
def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Helpers
def flatten_tokens(nn_tokens):
    return [t["token"] for sample in nn_tokens for t in sample]


# Plots

# Are we collapsing to a few tokens?
def plot_token_frequency(data, top_k=20):
    all_tokens = []

    for entry in data:
        all_tokens.extend(flatten_tokens(entry["nn_tokens"]))

    counts = Counter(all_tokens)
    most_common = counts.most_common(top_k)

    tokens, freqs = zip(*most_common)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(tokens)), freqs)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(f"Top {top_k} NN tokens")
    plt.tight_layout()
    plt.show()


# Are we exploring many tokens or collapsing?
def plot_diversity(data):
    diversity = []

    for entry in data:
        tokens = set(flatten_tokens(entry["nn_tokens"]))
        diversity.append(len(tokens))

    plt.figure()
    plt.plot(diversity)
    plt.title("NN token diversity over time")
    plt.xlabel("step")
    plt.ylabel("unique tokens")
    plt.show()


# Do NN assignments stabilize or keep changing?
def plot_stability(data):
    stability = []

    for i in range(1, len(data)):
        prev = flatten_tokens(data[i - 1]["nn_tokens"])
        curr = flatten_tokens(data[i]["nn_tokens"])

        same = sum(p == c for p, c in zip(prev, curr))
        stability.append(same / len(prev))

    plt.figure()
    plt.plot(stability)
    plt.title("NN stability over time")
    plt.xlabel("step")
    plt.ylabel("fraction unchanged")
    plt.show()


# Debug/printing
def print_examples(data, num_steps=5):
    print("\n=== SAMPLE NN SEQUENCES ===")
    for entry in data[:num_steps]:
        print(f"\nStep {entry['step']}")
        for i, sample in enumerate(entry["nn_text"]):
            print(f"Sample {i}: {sample}")


# Main
def main():
    parser = argparse.ArgumentParser(description="Analyse NN token sequences")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to nn_sequences.jsonl"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of most frequent tokens to plot"
    )

    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plotting"
    )

    parser.add_argument(
        "--print_examples",
        action="store_true",
        help="Print example NN sequences"
    )

    args = parser.parse_args()

    data = load_data(args.input)

    print(f"Loaded {len(data)} steps")

    if args.print_examples:
        print_examples(data)

    if not args.no_plots:
        plot_token_frequency(data, args.top_k)
        plot_diversity(data)
        plot_stability(data)


if __name__ == "__main__":
    main()
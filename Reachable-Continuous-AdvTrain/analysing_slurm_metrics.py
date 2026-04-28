import argparse
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


PATTERN = re.compile(
    r"harm_loss=(?P<harm_loss>[-+]?\d*\.\d+|\d+); "
    r"reach_loss=(?P<reach_loss>[-+]?\d*\.\d+|\d+); "
    r"nn_dist=(?P<nn_dist>[-+]?\d*\.\d+|\d+); "
    r"nn_entropy=(?P<nn_entropy>[-+]?\d*\.\d+|\d+); "
    r"affirmative=(?P<affirmative>\d+); "
    r"adv_norm=(?P<adv_norm>[-+]?\d*\.\d+|\d+); "
    r"nn_change_from_input=(?P<nn_change>[-+]?\d*\.\d+|\d+); "
    r"nn_margin=(?P<nn_margin>[-+]?\d*\.\d+|\d+); "
    r"actual_perturbed_fraction=(?P<pert_frac>[-+]?\d*\.\d+|\d+);"
)


def parse_slurm(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                data = {k: float(v) for k, v in match.groupdict().items()}
                rows.append(data)

    df = pd.DataFrame(rows)
    df["step"] = range(len(df))
    return df


# ===== Plot helpers =====

def save_plot(fig, output_dir, name):
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_core(df, output_dir):
    fig = plt.figure(figsize=(10,5))
    plt.plot(df["step"], df["harm_loss"], label="harm_loss")
    plt.plot(df["step"], df["reach_loss"], label="reach_loss")
    plt.legend()
    plt.title("Harm vs Reach loss")
    plt.xlabel("step")

    save_plot(fig, output_dir, "harm_vs_reach")


def plot_nn_behavior(df, output_dir):
    fig = plt.figure(figsize=(10,5))
    plt.plot(df["step"], df["nn_change"], label="nn_change_from_input")
    plt.plot(df["step"], df["pert_frac"], label="perturbed_fraction")
    plt.legend()
    plt.title("Token change & perturbation fraction")

    save_plot(fig, output_dir, "token_change")


def plot_margin(df, output_dir):
    fig = plt.figure(figsize=(10,5))
    plt.plot(df["step"], df["nn_margin"])
    plt.title("NN margin")

    save_plot(fig, output_dir, "nn_margin")


def plot_distance(df, output_dir):
    fig = plt.figure(figsize=(10,5))
    plt.plot(df["step"], df["nn_dist"])
    plt.title("NN distance")

    save_plot(fig, output_dir, "nn_distance")


def plot_norm(df, output_dir):
    fig = plt.figure(figsize=(10,5))
    plt.plot(df["step"], df["adv_norm"])
    plt.title("Perturbation norm")

    save_plot(fig, output_dir, "adv_norm")


# ===== Main =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to SLURM log")
    parser.add_argument("--output_dir", default="plots_analysing_slurm_metrics", help="Where to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = parse_slurm(args.input)

    print(f"Parsed {len(df)} steps")
    print(f"Saving plots to: {args.output_dir}")

    plot_core(df, args.output_dir)
    plot_nn_behavior(df, args.output_dir)
    plot_margin(df, args.output_dir)
    plot_distance(df, args.output_dir)
    plot_norm(df, args.output_dir)


if __name__ == "__main__":
    main()
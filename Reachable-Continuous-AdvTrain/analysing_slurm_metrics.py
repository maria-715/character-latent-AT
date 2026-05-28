import argparse
import ast
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_scalar(v):
    v = v.strip()
    if v == "None":
        return None
    if v == "nan":
        return float("nan")
    if re.fullmatch(r"[-+]?\d+", v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def parse_slurm(path):
    rows = []
    current_step = None

    # Matches key=value; pairs in the semicolon log line
    pair_re = re.compile(r"([A-Za-z_]+)=([^;]+);")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # 1) Capture step from the dict line
            if line.startswith("{'global_step'") or line.startswith('{"global_step"'):
                try:
                    d = ast.literal_eval(line)
                    if isinstance(d, dict) and "global_step" in d:
                        current_step = int(d["global_step"])
                except Exception:
                    pass
                continue

            # 2) Parse only the semicolon metric line
            if "index=" not in line or "attack_loss=" not in line:
                continue

            # Start from the first metric field, ignore the text outputs before it
            start = line.find("index=")
            metric_part = line[start:]

            pairs = pair_re.findall(metric_part)
            if not pairs:
                continue

            row = {k: parse_scalar(v) for k, v in pairs}

            # Keep only actual attack rows
            if row.get("index") is None:
                continue

            row["global_step"] = current_step if current_step is not None else len(rows)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["global_step", "index"]).reset_index(drop=True)
    return df


def save_plot(fig, output_dir, name):
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_if_present(df, xcol, ycol, title, output_dir, name):
    if xcol not in df.columns or ycol not in df.columns:
        print(f"Skipping {name}: missing {xcol} or {ycol}")
        return

    fig = plt.figure(figsize=(10, 5))
    plt.plot(df[xcol], df[ycol])
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    save_plot(fig, output_dir, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="plots_analysing_slurm_metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = parse_slurm(args.input)
    print(f"Parsed {len(df)} steps")
    print(f"Saving plots to {args.output_dir}")

    if df.empty:
        print("No attack rows found. Check that the log contains lines with index=0, index=1, etc.")
        return

    xcol = "global_step" if "global_step" in df.columns else "step"

    plot_if_present(df, xcol, "harm_loss", "Harm loss", args.output_dir, "harm_loss")
    plot_if_present(df, xcol, "reach_loss", "Reach loss", args.output_dir, "reach_loss")
    plot_if_present(df, xcol, "nn_entropy", "NN entropy", args.output_dir, "nn_entropy")
    plot_if_present(df, xcol, "nn_change_from_input", "NN change from input", args.output_dir, "nn_change_from_input")
    plot_if_present(df, xcol, "nn_change_between_steps", "NN change between steps", args.output_dir, "nn_change_between_steps")
    plot_if_present(df, xcol, "nn_margin", "NN margin", args.output_dir, "nn_margin")
    plot_if_present(df, xcol, "nn_margin_std", "NN margin std", args.output_dir, "nn_margin_std")
    plot_if_present(df, xcol, "nn_unique_tokens", "Unique NN tokens", args.output_dir, "nn_unique_tokens")
    plot_if_present(df, xcol, "actual_perturbed_fraction", "Actual perturbed fraction", args.output_dir, "actual_perturbed_fraction")
    plot_if_present(df, xcol, "reach_to_harm_ratio", "Reach / harm ratio", args.output_dir, "reach_to_harm_ratio")
    plot_if_present(df, xcol, "adv_norm", "Adversarial norm", args.output_dir, "adv_norm")


if __name__ == "__main__":
    main()
import re
import numpy as np
import matplotlib.pyplot as plt

filename = "circuit-breakers-eval/slurm-107090.out"

pattern = re.compile(
    r"Saved solution at iter (\d+) \| "
    r"L2 distance: ([0-9.eE+-]+) \| "
    r"Avg token distance: ([0-9.eE+-]+)"
)

iters = []
l2 = []
avg = []

with open(filename) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            iters.append(int(m.group(1)))
            l2.append(float(m.group(2)))
            avg.append(float(m.group(3)))

print(f"Found {len(iters)} saved solutions")

if len(iters) > 0:

    print()

    print(f"L2 distance")
    print(f"  mean = {np.mean(l2):.4f}")
    print(f"  std  = {np.std(l2):.4f}")
    print(f"  min  = {np.min(l2):.4f}")
    print(f"  max  = {np.max(l2):.4f}")

    print()

    print(f"Average token distance")
    print(f"  mean = {np.mean(avg):.4f}")
    print(f"  std  = {np.std(avg):.4f}")
    print(f"  min  = {np.min(avg):.4f}")
    print(f"  max  = {np.max(avg):.4f}")

    plt.figure(figsize=(7,5))
    plt.plot(iters, l2, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("L2 distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(iters, avg, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Average token distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
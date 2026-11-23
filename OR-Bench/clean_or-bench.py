import csv

input_file = "OR-Bench/datasets/or-bench_train_and_eval/or-bench-from80kto1k_train.csv"
output_file = "OR-Bench/datasets/harmless_format/or-bench-from80kto1k_train-clean.txt"
keep_every = 1 # 10  # Keep 1 out of every 10 prompts

with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    reader = csv.reader(infile)

    # Skip header
    next(reader)

    # Write the output header
    outfile.write("prompt\n")

    # Write every Nth prompt
    for i, row in enumerate(reader):
        if i % keep_every == 0:
            outfile.write(row[0] + "\n")

print(f"Created {output_file}")
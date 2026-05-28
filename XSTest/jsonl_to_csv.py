import json
import pandas as pd
import argparse

def main(input_path, output_path):
    rows = []
    with open(input_path) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    main(args.input, args.output)
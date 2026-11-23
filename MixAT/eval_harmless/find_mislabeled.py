#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

TARGET_ANSWER = "Sorry, I can't do that."
TARGET_LABEL = 1


def analyze_file(json_file: Path):
    """Analyze a single JSON file."""

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON does not contain a list")

    total_entries = 0
    label_1_count = 0
    mistaken_1_count = 0
    flagged = []

    for index, component in enumerate(data):
        if not isinstance(component, dict):
            continue

        total_entries += 1

        label = component.get("label")
        answer = component.get("answer")

        if label == 1:
            label_1_count += 1

            # These entries were labeled 1, but should count as 0
            if answer == TARGET_ANSWER:
                mistaken_1_count += 1

                flagged.append({
                    "index": index,
                    "component": component,
                })

    # Original average:
    # all original label-1 entries / all entries
    before = (
        label_1_count / total_entries
        if total_entries
        else 0
    )

    # Correct the mistaken 1s by treating them as 0.
    # The total number of entries DOES NOT change.
    corrected_label_1 = label_1_count - mistaken_1_count

    after = (
        corrected_label_1 / total_entries
        if total_entries
        else 0
    )

    return {
        "total_entries": total_entries,
        "label_1_count": label_1_count,
        "mistaken_1_count": mistaken_1_count,
        "corrected_label_1": corrected_label_1,
        "before": before,
        "after": after,
        "flagged": flagged,
    }
   


def analyze_folder(folder: Path):
    """Analyze every JSON file in the folder."""

    results = {}

    for json_file in sorted(folder.glob("*.json")):
        try:
            results[json_file.name] = analyze_file(json_file)

        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"Warning: could not process {json_file.name}: {exc}")

    return results


def write_report(results, output_file: Path):
    """Write per-file statistics and flagged components."""

    with output_file.open("w", encoding="utf-8") as f:

        f.write("PER-FILE LABEL REPORT\n")
        f.write("=" * 80 + "\n\n")

        for filename, stats in results.items():

            f.write(f"{filename}\n")
            f.write("-" * 80 + "\n")

            f.write(f"Total entries:                 {stats['total_entries']}\n")
            f.write(f"Original label 1 entries:      {stats['label_1_count']}\n")
            f.write(f"Mistaken label 1s -> 0:        {stats['mistaken_1_count']}\n")
            f.write(f"Corrected label 1 entries:     {stats['corrected_label_1']}\n")

            f.write("\n")

            f.write(
                f"Label-1 average BEFORE:         "
                f"{stats['before']:.6f} "
                f"({stats['before'] * 100:.2f}%)\n"
            )

            f.write(
                f"Label-1 average AFTER:          "
                f"{stats['after']:.6f} "
                f"({stats['after'] * 100:.2f}%)\n"
            )

            difference = stats["before"] - stats["after"]

            f.write(
                f"Difference:                     "
                f"{difference:.6f} "
                f"({difference * 100:.2f} percentage points)\n"
            )

            # Print flagged components for this file
            if stats["flagged"]:

                f.write("\nFLAGGED COMPONENTS\n")
                f.write("-" * 80 + "\n\n")

                for entry in stats["flagged"]:

                    f.write(
                        f"Component index: {entry['index']}\n"
                    )

                    f.write(
                        json.dumps(
                            entry["component"],
                            indent=4,
                            ensure_ascii=False,
                        )
                    )

                    f.write("\n\n")

            else:
                f.write("\nNo mistaken label-1 entries found.\n")

            f.write("\n" + "=" * 80 + "\n\n")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze label-1 rates per JSON file and find "
            "mistaken refusal labels."
        )
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing JSON files",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("flagged_report.txt"),
        help="Output report file",
    )

    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")

    results = analyze_folder(args.folder)

    write_report(
        results,
        args.output,
    )

    # Also print a concise per-file summary to terminal
    print()
    print("PER-FILE RESULTS")
    print("=" * 100)

    for filename, stats in results.items():

        difference = stats["before"] - stats["after"]

        print(f"\n{filename}")
        print(f"  Entries:          {stats['total_entries']}")
        print(f"  Label 1:          {stats['label_1_count']}")
        print(f"  Mistaken 1s:      {stats['mistaken_1_count']}")
        print(
            f"  Before:           "
            f"{stats['before']:.6f} "
            f"({stats['before'] * 100:.2f}%)"
        )
        print(
            f"  After:            "
            f"{stats['after']:.6f} "
            f"({stats['after'] * 100:.2f}%)"
        )
        print(
            f"  Difference:       "
            f"{difference:.6f} "
            f"({difference * 100:.2f} pp)"
        )

    print()
    print("=" * 100)
    print(f"Full report written to: {args.output}")


if __name__ == "__main__":
    main()
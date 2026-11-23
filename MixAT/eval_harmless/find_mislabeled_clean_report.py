#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def keep_filename(filename: str) -> bool:
    """
    Keep a file if:
      1. It does not contain 'OR', OR
      2. It contains 'OR_remaining', OR
      3. It contains 'OR_from80kto1k'.
    """
    return (
        "OR" not in filename
        or "OR_remaining" in filename
        or "OR_from80kto1k" in filename
    )


def clean_report(input_file: Path, output_file: Path):
    lines = input_file.read_text(encoding="utf-8").splitlines(keepends=True)

    sections = []
    current_filename = None
    current_lines = []

    for line in lines:
        stripped = line.strip()

        # A new file section starts whenever a line ends in .json
        if stripped.endswith(".json"):

            # Save previous section
            if current_filename is not None:
                sections.append(
                    (current_filename, "".join(current_lines))
                )

            current_filename = stripped
            current_lines = [line]

        elif current_filename is not None:
            current_lines.append(line)

    # Save final section
    if current_filename is not None:
        sections.append(
            (current_filename, "".join(current_lines))
        )

    if not sections:
        print("No JSON file sections found in the input report.")
        return

    kept_sections = []
    kept_files = []
    removed_files = []

    for filename, section in sections:

        if keep_filename(filename):
            kept_sections.append(section.rstrip())
            kept_files.append(filename)
        else:
            removed_files.append(filename)

    # Write cleaned report
    with output_file.open("w", encoding="utf-8") as f:

        f.write("CLEANED FLAGGED COMPONENTS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "Included files:\n"
            "  - filenames without 'OR'\n"
            "  - filenames containing 'OR_remaining'\n"
            "  - filenames containing 'OR_from80kto1k'\n\n"
        )

        f.write(f"Files found: {len(sections)}\n")
        f.write(f"Files kept: {len(kept_files)}\n")
        f.write(f"Files removed: {len(removed_files)}\n\n")

        f.write("=" * 80 + "\n\n")

        for section in kept_sections:
            f.write(section)
            f.write("\n\n")

    print("Done.")
    print(f"Input report:  {input_file}")
    print(f"Output report: {output_file}")
    print(f"Files found:   {len(sections)}")
    print(f"Files kept:    {len(kept_files)}")
    print(f"Files removed: {len(removed_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove unwanted file sections from the JSON analysis report."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Original report .txt file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cleaned_report.txt"),
        help="Cleaned output file (default: cleaned_report.txt)",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    clean_report(args.input, args.output)


if __name__ == "__main__":
    main()
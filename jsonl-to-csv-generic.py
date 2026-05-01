#!/usr/bin/env python3
"""
Generic JSONL to CSV converter.
Reads the first row to determine column names, then converts all records to CSV.
Handles nested structures by flattening with dot notation (e.g., metadata.temperature).
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten nested dictionaries.
    
    Example:
        {"a": 1, "b": {"c": 2}} -> {"a": 1, "b.c": 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to JSON string for CSV
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def get_column_names(record: dict) -> list:
    """Extract and sort column names from a record (flattens nested keys)."""
    flattened = flatten_dict(record)
    return sorted(flattened.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL file to CSV format. Auto-detects columns from first row."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV file path (default: <input>.csv)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Determine output file
    output_file = args.output or f"{input_path.stem}.csv"

    total_records = 0
    failed = 0
    fieldnames = []

    try:
        # First pass: read first record to determine columns
        with open(args.file, "r") as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    first_record = json.loads(line)
                    fieldnames = get_column_names(first_record)
                    if not args.quiet:
                        print(f"Found {len(fieldnames)} column(s) from first record")
                    break
                except json.JSONDecodeError as e:
                    print(f"Error: First line is not valid JSON: {e}", file=sys.stderr)
                    sys.exit(1)

        if not fieldnames:
            print("Error: No valid records found in input file", file=sys.stderr)
            sys.exit(1)

        # Second pass: convert all records
        with open(args.file, "r") as infile, open(output_file, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, restval="")
            writer.writeheader()

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    flattened = flatten_dict(record)
                    writer.writerow(flattened)
                    total_records += 1

                except json.JSONDecodeError as e:
                    failed += 1
                    if not args.quiet:
                        print(f"Warning: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                except Exception as e:
                    failed += 1
                    if not args.quiet:
                        print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)

        if not args.quiet:
            print(f"✓ Conversion complete")
            print(f"  Records processed: {total_records}")
            print(f"  Columns: {len(fieldnames)}")
            print(f"  Failed: {failed}")
            print(f"  Output file: {output_file}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge two JSONL files (newline-delimited JSON) into one deduplicated JSONL file.

Usage:
  python combine_jsonl_files.py a.jsonl b.jsonl out.jsonl

Options:
  --key KEY    : unique key to dedupe on (default: file_id)

Behavior:
  - Reads both input files line-by-line (skips blank lines and invalid JSON).
  - Writes output as JSONL preserving order: entries from the first file, then
    entries from the second file that are not duplicates based on the key.
  - Deduplication uses the specified key (default `file_id`); if the key is missing for a
    record, falls back to `file_url` then to the JSON string of the record.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip invalid lines
                continue


def key_for_record(rec: Dict[str, Any], key: str) -> str:
    v = rec.get(key)
    if v:
        return str(v)
    v = rec.get("file_url") or rec.get("url") or rec.get("file_name")
    if v:
        return str(v)
    return json.dumps(rec, sort_keys=True)


def merge_jsonl(first_path: Path, second_path: Path, out_path: Path, key: str) -> int:
    seen = set()
    count = 0

    with open(out_path, "w") as out:
        for rec in iter_jsonl(first_path):
            k = key_for_record(rec, key)
            if k not in seen:
                seen.add(k)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

        for rec in iter_jsonl(second_path):
            k = key_for_record(rec, key)
            if k not in seen:
                seen.add(k)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine two JSONL files into one deduplicated JSONL file")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output")
    parser.add_argument("--key", default="file_id", help="Field to dedupe on (default: file_id)")
    args = parser.parse_args(argv)

    first_path = Path(args.first)
    second_path = Path(args.second)
    out_path = Path(args.output)

    if not first_path.exists():
        print(f"Missing file: {first_path}", file=sys.stderr)
        return 2
    if not second_path.exists():
        print(f"Missing file: {second_path}", file=sys.stderr)
        return 2

    total = merge_jsonl(first_path, second_path, out_path, args.key)
    print(f"Wrote {total} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

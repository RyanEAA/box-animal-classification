#!/usr/bin/env python3
"""Merge two JSON files containing lists of records into one deduplicated JSON list.

Usage:
  python combine_json_files.py first.json second.json output.json

Options:
  --key KEY     : unique key to dedupe on (default: file_id)

Behavior:
  - Loads both input files. If the top-level JSON is a list, uses it. If it's an object
    containing a list under a common key like 'items'/'records'/'images', that list is used.
  - Merges entries preserving order: entries from the first file come first, then the
    second file's entries that are not duplicates.
  - Deduplication uses the specified key (default `file_id`); if the key is missing for a
    record, falls back to `file_url` then to the JSON string of the record.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # If it's a dict, try to find a list value
    if isinstance(data, dict):
        for candidate in ("items", "records", "images", "data"):
            v = data.get(candidate)
            if isinstance(v, list):
                return v

    raise ValueError(f"Unsupported JSON structure in {path}; expected list or object with a list field")


def key_for_record(rec: Dict[str, Any], key: str) -> str:
    v = rec.get(key)
    if v:
        return str(v)
    v = rec.get("file_url") or rec.get("url") or rec.get("file_name")
    if v:
        return str(v)
    # Last resort: full JSON string
    return json.dumps(rec, sort_keys=True)


def merge_lists(a: List[Dict[str, Any]], b: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for rec in a:
        k = key_for_record(rec, key)
        if k not in seen:
            seen.add(k)
            out.append(rec)

    for rec in b:
        k = key_for_record(rec, key)
        if k not in seen:
            seen.add(k)
            out.append(rec)

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Combine two JSON list files into one deduplicated JSON list")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output")
    parser.add_argument("--key", default="file_id", help="Record field to use as unique key (default: file_id)")
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

    a = load_json_list(first_path)
    b = load_json_list(second_path)

    merged = merge_lists(a, b, args.key)

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

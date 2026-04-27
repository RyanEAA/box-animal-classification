#!/usr/bin/env python3
"""
Convert speciesnet_results.jsonl to CSV format.
Flattens nested animal/detection data so each row represents one animal per image.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def flatten_record(record: dict) -> list[dict]:
    """
    Convert a single JSONL record into one or more CSV rows.
    
    If an image has multiple animals, creates one row per animal.
    If an image has detections with bbox, includes that spatial data.
    
    Args:
        record: Single JSONL record (dict)
    
    Returns:
        List of flattened dicts (one per animal)
    """
    rows = []
    
    status = record.get("status")
    file_id = record.get("file_id")
    file_name = record.get("file_name")
    file_url = record.get("file_url")
    prediction = record.get("prediction")
    prediction_score = record.get("prediction_score")
    prediction_source = record.get("prediction_source")
    
    # Extract animals list
    animals = record.get("animals", [])
    detections = record.get("detections", [])
    
    # If no animals detected, still create one row with NULLs for animal fields
    if not animals:
        rows.append({
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "status": status,
            "animal_label": None,
            "animal_score": None,
            "animal_taxonomy": None,
            "detection_conf": None,
            "bbox_xmin": None,
            "bbox_ymin": None,
            "bbox_width": None,
            "bbox_height": None,
            "prediction": prediction,
            "prediction_score": prediction_score,
            "prediction_source": prediction_source,
        })
        return rows
    
    # Create one row per animal
    for animal_idx, animal in enumerate(animals):
        animal_label = animal.get("label")
        animal_score = animal.get("score")
        animal_taxonomy = animal.get("taxonomy")
        bbox = animal.get("bbox")  # [xmin, ymin, width, height] or None
        
        # Try to find matching detection for additional confidence score
        detection_conf = None
        if detections and animal_idx < len(detections):
            detection_conf = detections[animal_idx].get("conf")
        
        # Parse bbox if present
        bbox_xmin = None
        bbox_ymin = None
        bbox_width = None
        bbox_height = None
        if bbox and len(bbox) >= 4:
            bbox_xmin, bbox_ymin, bbox_width, bbox_height = bbox[:4]
        
        row = {
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "status": status,
            "animal_label": animal_label,
            "animal_score": animal_score,
            "animal_taxonomy": animal_taxonomy,
            "detection_conf": detection_conf,
            "bbox_xmin": bbox_xmin,
            "bbox_ymin": bbox_ymin,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "prediction": prediction,
            "prediction_score": prediction_score,
            "prediction_source": prediction_source,
        }
        rows.append(row)
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert speciesnet_results.jsonl to CSV format"
    )
    parser.add_argument(
        "--input-file",
        default="speciesnet_results.jsonl",
        help="Input JSONL file (default: speciesnet_results.jsonl)",
    )
    parser.add_argument(
        "--output-file",
        default="speciesnet_results.csv",
        help="Output CSV file (default: speciesnet_results.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Fieldnames for CSV (order matters)
    fieldnames = [
        "file_id",
        "file_name",
        "file_url",
        "status",
        "animal_label",
        "animal_score",
        "animal_taxonomy",
        "detection_conf",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_width",
        "bbox_height",
        "prediction",
        "prediction_score",
        "prediction_source",
    ]
    
    total_records = 0
    total_rows = 0
    failed = 0
    
    try:
        with open(args.input_file, "r") as infile, open(args.output_file, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    total_records += 1
                    
                    # Flatten the record into one or more rows
                    flat_rows = flatten_record(record)
                    writer.writerows(flat_rows)
                    total_rows += len(flat_rows)
                    
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
            print(f"  Input records: {total_records}")
            print(f"  Output rows: {total_rows}")
            print(f"  Failed: {failed}")
            print(f"  Output file: {args.output_file}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

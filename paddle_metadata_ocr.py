import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Set

import cv2
import requests
from dotenv import load_dotenv
from paddleocr import PaddleOCR


load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def log(message: str, quiet: bool = False):
    if not quiet:
        print(message)


def load_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_processed_file_ids(results_file: str) -> Set[str]:
    processed = set()
    if not os.path.exists(results_file):
        return processed

    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            file_id = str(record.get("file_id", "")).strip()
            status = record.get("status")
            if file_id and status == "ok":
                processed.add(file_id)

    return processed


def append_result(results_file: str, result: Dict):
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def flush_metadata_batch(metadata_file: str, batch: List[Dict]):
    """Append batch of metadata records to metadata output file."""
    if not batch:
        return
    with open(metadata_file, "a") as f:
        for record in batch:
            f.write(json.dumps(record) + "\n")


def crop_bottom_percent(image_path: str, percent: float = 0.065) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, _ = img.shape[:2]
    crop_h = max(1, int(h * percent))
    cropped = img[h - crop_h : h, :]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_crop.jpg")
    tmp_path = tmp.name
    tmp.close()

    ok = cv2.imwrite(tmp_path, cropped)
    if not ok:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise ValueError(f"Failed to write cropped image: {tmp_path}")

    return tmp_path


def extract_texts_from_prediction(prediction_result) -> List[str]:
    texts = []

    if isinstance(prediction_result, list):
        for item in prediction_result:
            if isinstance(item, dict) and "rec_texts" in item:
                rec_texts = item.get("rec_texts") or []
                for t in rec_texts:
                    if t is not None:
                        texts.append(str(t).strip())
    elif isinstance(prediction_result, dict):
        rec_texts = prediction_result.get("rec_texts") or []
        for t in rec_texts:
            if t is not None:
                texts.append(str(t).strip())

    return [t for t in texts if t]


def run_paddle_ocr(image_path: str, crop_percent: float) -> List[str]:
    cropped_path = crop_bottom_percent(image_path, crop_percent)
    try:
        result = ocr.predict(cropped_path)
        return extract_texts_from_prediction(result)
    finally:
        if os.path.exists(cropped_path):
            os.remove(cropped_path)


def parse_metadata_12345(texts: List[str]) -> Dict:
    data = {
        "temperature": None,
        "pressure": None,
        "camera_id": None,
        "date": None,
        "time": None,
    }

    for t in texts:
        t = t.strip()

        if re.match(r"\d+°?C", t):
            data["temperature"] = t
        elif re.match(r"\d+\.\d+", t):
            data["pressure"] = t
        elif "inhg" in t.lower():
            if data["pressure"]:
                data["pressure"] += " " + t
        elif "TRAILCAM" in t.upper():
            data["camera_id"] = t
        elif re.match(r"\d{2}/\d{2}/\d{4}", t):
            data["date"] = t
        elif re.match(r"\d{1,2}:\d{2}\s?(AM|PM)", t, re.IGNORECASE):
            data["time"] = t

    if data["temperature"] is None:
        data["temperature"] = texts[0].strip() if texts else None

    if data["date"] is None:
        data["date"] = texts[-2].strip() if len(texts) >= 2 else None

    if data["time"] is None:
        data["time"] = texts[-2].strip() if len(texts) >= 2 else None

    return data


def parse_metadata_678(texts: List[str]) -> Dict:
    data = {
        "temperature": None,
        "pressure": None,
        "camera_id": None,
        "date": None,
        "time": None,
    }

    for t in texts:
        t = t.strip()

        if re.fullmatch(r"\d+°?C", t):
            data["temperature"] = t
        elif re.fullmatch(r"\d{2}[-/]\d{2}[-/]\d{4}", t):
            data["date"] = t
        elif re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", t):
            data["time"] = t
        elif re.fullmatch(r"\d{1,2}:\d{2}\s?(AM|PM)", t, re.IGNORECASE):
            data["time"] = t
        elif "TRAILCAM" in t.upper():
            data["camera_id"] = t

    if data["temperature"] is None and len(texts) > 1:
        data["temperature"] = texts[1].strip()
    elif data["temperature"] is None and texts:
        data["temperature"] = texts[0].strip()

    if data["date"] is None and len(texts) >= 2:
        data["date"] = texts[-2].strip()

    if data["time"] is None and texts:
        data["time"] = texts[-1].strip()

    return data


def choose_parser(path_value: str):
    path_l = (path_value or "").lower()
    # Match patterns like /Cam1/, /cam2/, or just Cam1, cam3, etc. for cameras 1-5
    if re.search(r"\bcam[1-5]\b", path_l):
        return "parse_metadata_12345", parse_metadata_12345
    return "parse_metadata_678", parse_metadata_678


def download_image_from_record(record: Dict) -> str:
    if not ACCESS_TOKEN:
        raise ValueError("ACCESS_TOKEN is missing. Set it in your .env file.")

    file_name = record.get("file_name") or "image.jpg"
    suffix = Path(file_name).suffix or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()

    download_url = record.get("direct_download_url")
    file_id = str(record.get("file_id", "")).strip()
    if not download_url and file_id:
        download_url = f"https://api.box.com/2.0/files/{file_id}/content"

    if not download_url:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise ValueError("No direct_download_url or file_id found in record.")

    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(download_url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    try:
        with open(temp_path, "wb") as out:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out.write(chunk)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return temp_path


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR metadata extraction on images listed in box_images.json"
    )
    parser.add_argument(
        "--input-file",
        default=os.getenv("OUTPUT_FILE", "box_images.json"),
        help="Path to box image metadata JSON file.",
    )
    parser.add_argument(
        "--results-file",
        default=os.getenv("METADATA_RESULTS_FILE", "metadata_results.jsonl"),
        help="JSONL file where all extraction results (including errors) are appended.",
    )
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_OUTPUT_FILE", "metadata.jsonl"),
        help="JSONL file where successful metadata is written in batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("METADATA_BATCH_SIZE", "20")),
        help="Number of successful metadata records to batch before writing (default: 20).",
    )
    parser.add_argument(
        "--crop-percent",
        type=float,
        default=float(os.getenv("OCR_CROP_PERCENT", "0.065")),
        help="Bottom crop percentage used for OCR (default: 0.065).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of files to process (0 means no limit).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-run files that already have successful results in results file.",
    )
    args = parser.parse_args()

    quiet_mode = args.quiet or to_bool(os.getenv("QUIET_MODE", "false"))

    records = load_json_list(args.input_file)
    processed_ids = set() if args.reprocess else load_processed_file_ids(args.results_file)

    total = len(records)
    processed = 0
    skipped = 0
    failed = 0
    metadata_batch = []

    log(f"Loaded {total} records from {args.input_file}", quiet_mode)
    if processed_ids:
        log(f"Skipping {len(processed_ids)} already processed file_id values", quiet_mode)

    for record in records:
        if args.limit and processed >= args.limit:
            break

        file_id = str(record.get("file_id", "")).strip()
        file_name = record.get("file_name") or f"{file_id}.jpg"
        file_url = record.get("file_url") or record.get("web_url")
        path_value = record.get("path") or ""

        if not file_id:
            failed += 1
            append_result(
                args.results_file,
                {
                    "status": "error",
                    "error": "missing_file_id",
                    "source_record": record,
                },
            )
            continue

        if file_id in processed_ids:
            skipped += 1
            continue

        tmp_img_path = None
        try:
            log(f"Processing metadata file_id={file_id} name={file_name}", quiet_mode)
            tmp_img_path = download_image_from_record(record)
            texts = run_paddle_ocr(tmp_img_path, args.crop_percent)
            log(f"  OCR texts: {texts}", quiet_mode)
            parser_name, parser_func = choose_parser(path_value)
            metadata = parser_func(texts)

            result = {
                "status": "ok",
                "file_id": file_id,
                "file_name": file_name,
                "file_url": file_url,
                "path": path_value,
                "ocr_texts": texts,
                "metadata": metadata,
                "parser": parser_name,
            }
            append_result(args.results_file, result)

            # Add to metadata batch for periodic flushing
            metadata_entry = {
                "file_id": file_id,
                "file_name": file_name,
                "metadata": metadata,
                "parser": parser_name,
            }
            metadata_batch.append(metadata_entry)
            if len(metadata_batch) >= args.batch_size:
                flush_metadata_batch(args.metadata_file, metadata_batch)
                log(f"  flushed batch of {len(metadata_batch)} metadata records to {args.metadata_file}", quiet_mode)
                metadata_batch = []

            processed += 1
            processed_ids.add(file_id)
            log(
                f"  done file_id={file_id} | parser={parser_name} | "
                f"temp={metadata.get('temperature')} date={metadata.get('date')} time={metadata.get('time')}",
                quiet_mode,
            )

        except Exception as e:
            failed += 1
            append_result(
                args.results_file,
                {
                    "status": "error",
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_url": file_url,
                    "path": path_value,
                    "error": type(e).__name__,
                    "message": str(e),
                },
            )
            log(f"  failed file_id={file_id} error={type(e).__name__}", quiet_mode)

        finally:
            if tmp_img_path and os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)

    # Flush any remaining metadata batch
    if metadata_batch:
        flush_metadata_batch(args.metadata_file, metadata_batch)
        log(f"  flushed final batch of {len(metadata_batch)} metadata records to {args.metadata_file}", quiet_mode)

    log(
        "\nMetadata extraction complete | "
        f"processed={processed} | skipped={skipped} | failed={failed} | "
        f"results_file={args.results_file} | metadata_file={args.metadata_file}",
        quiet_mode,
    )


if __name__ == "__main__":
    main()
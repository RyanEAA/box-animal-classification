import argparse
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
from boxsdk import Client, OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth
from dotenv import find_dotenv, load_dotenv, set_key
from paddleocr import PaddleOCR


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ENV_PATH = find_dotenv(usecwd=True) or ".env"

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def store_tokens(access_token, refresh_token):
    set_key(ENV_PATH, "ACCESS_TOKEN", access_token)
    if refresh_token:
        set_key(ENV_PATH, "REFRESH_TOKEN", refresh_token)


def build_client() -> Client:
    if not ACCESS_TOKEN:
        raise ValueError("ACCESS_TOKEN is missing. Set it in your .env file.")

    if REFRESH_TOKEN and CLIENT_ID and CLIENT_SECRET:
        auth = OAuth2(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            access_token=ACCESS_TOKEN,
            refresh_token=REFRESH_TOKEN,
            store_tokens=store_tokens,
        )
    else:
        auth = DeveloperTokenAuth(get_new_token_callback=lambda: ACCESS_TOKEN)

    return Client(auth)


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
            if file_id and record.get("status") == "ok":
                processed.add(file_id)

    return processed


def append_result(results_file: str, result: Dict):
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def flush_metadata_batch(metadata_file: str, batch: List[Dict]):
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


def download_box_file_to_temp(client: Client, file_id: str, file_name: str) -> str:
    suffix = Path(file_name).suffix or ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name

    try:
        with temp_file:
            client.file(file_id).download_to(temp_file)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return temp_path


def download_batch_parallel(
    client: Client,
    records: List[dict],
    max_workers: int,
) -> Tuple[List[Tuple[int, dict, str]], List[Tuple[dict, Exception]]]:
    successes = []
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}

        for index, record in enumerate(records):
            file_id = str(record.get("file_id", "")).strip()
            file_name = record.get("file_name") or f"{file_id}.jpg"
            future = executor.submit(download_box_file_to_temp, client, file_id, file_name)
            future_map[future] = (index, record)

        for future in as_completed(future_map):
            index, record = future_map[future]
            try:
                temp_path = future.result()
                successes.append((index, record, temp_path))
            except Exception as e:
                failures.append((record, e))

    successes.sort(key=lambda item: item[0])
    return successes, failures


def cleanup_temp_files(paths: List[str]):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def process_downloaded_batch(
    batch_records: List[dict],
    batch_paths: List[str],
    results_file: str,
    crop_percent: float,
    quiet_mode: bool,
):
    for record, image_path in zip(batch_records, batch_paths):
        file_id = str(record.get("file_id", "")).strip()
        file_name = record.get("file_name") or f"{file_id}.jpg"
        file_url = record.get("file_url") or record.get("web_url")
        path_value = record.get("path") or ""

        try:
            texts = run_paddle_ocr(image_path, crop_percent)
            log(f"  OCR texts for file_id={file_id}: {texts}", quiet_mode)

            append_result(
                results_file,
                {
                    "status": "ok",
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_url": file_url,
                    "path": path_value,
                    "ocr_texts": texts,
                },
            )

            log(
                f"  done file_id={file_id} extracted {len(texts)} text lines",
                quiet_mode,
            )

        except Exception as e:
            append_result(
                results_file,
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


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR metadata extraction on Box images using batched downloads and OCR."
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
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "64")),
        help="Number of files to download and OCR per batch.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=int(os.getenv("DOWNLOAD_WORKERS", "32")),
        help="Number of parallel Box download workers.",
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

    client = build_client()

    total = len(records)
    processed = 0
    skipped = 0
    failed = 0
    batch_records = []

    log(f"Loaded {total} records from {args.input_file}", quiet_mode)
    if processed_ids:
        log(f"Skipping {len(processed_ids)} already processed file_id values", quiet_mode)
    log(f"Batch size: {args.batch_size}", quiet_mode)
    log(f"Download workers: {args.download_workers}", quiet_mode)

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

        batch_records.append(record)

        if len(batch_records) < args.batch_size:
            continue

        log(f"Downloading batch of {len(batch_records)} images...", quiet_mode)
        downloaded, download_failures = download_batch_parallel(
            client=client,
            records=batch_records,
            max_workers=args.download_workers,
        )

        for failed_record, error in download_failures:
            failed += 1
            failed_file_id = str(failed_record.get("file_id", "")).strip()
            failed_file_name = failed_record.get("file_name") or f"{failed_file_id}.jpg"
            failed_file_url = failed_record.get("file_url") or failed_record.get("web_url")
            failed_path = failed_record.get("path") or ""
            append_result(
                args.results_file,
                {
                    "status": "error",
                    "file_id": failed_file_id,
                    "file_name": failed_file_name,
                    "file_url": failed_file_url,
                    "path": failed_path,
                    "error": type(error).__name__,
                    "message": str(error),
                },
            )

        if not downloaded:
            batch_records = []
            continue

        batch_records = [record for _, record, _ in downloaded]
        batch_paths = [temp_path for _, _, temp_path in downloaded]

        log(f"Running OCR batch of {len(batch_paths)} images...", quiet_mode)
        try:
            process_downloaded_batch(
                batch_records=batch_records,
                batch_paths=batch_paths,
                results_file=args.results_file,
                crop_percent=args.crop_percent,
                quiet_mode=quiet_mode,
            )

            processed += len(batch_paths)
            for record in batch_records:
                processed_ids.add(str(record.get("file_id", "")).strip())

            log(f"Progress | processed={processed} skipped={skipped} failed={failed}", quiet_mode)

        finally:
            cleanup_temp_files(batch_paths)

        batch_records = []

    if batch_records:
        log(f"Downloading final batch of {len(batch_records)} images...", quiet_mode)
        downloaded, download_failures = download_batch_parallel(
            client=client,
            records=batch_records,
            max_workers=args.download_workers,
        )

        for failed_record, error in download_failures:
            failed += 1
            failed_file_id = str(failed_record.get("file_id", "")).strip()
            failed_file_name = failed_record.get("file_name") or f"{failed_file_id}.jpg"
            failed_file_url = failed_record.get("file_url") or failed_record.get("web_url")
            failed_path = failed_record.get("path") or ""
            append_result(
                args.results_file,
                {
                    "status": "error",
                    "file_id": failed_file_id,
                    "file_name": failed_file_name,
                    "file_url": failed_file_url,
                    "path": failed_path,
                    "error": type(error).__name__,
                    "message": str(error),
                },
            )

        if downloaded:
            batch_records = [record for _, record, _ in downloaded]
            batch_paths = [temp_path for _, _, temp_path in downloaded]

            log(f"Running final OCR batch of {len(batch_paths)} images...", quiet_mode)
            try:
                process_downloaded_batch(
                    batch_records=batch_records,
                    batch_paths=batch_paths,
                    results_file=args.results_file,
                    crop_percent=args.crop_percent,
                    quiet_mode=quiet_mode,
                )

                processed += len(batch_paths)
                for record in batch_records:
                    processed_ids.add(str(record.get("file_id", "")).strip())

            finally:
                cleanup_temp_files(batch_paths)

    log(
        "\nDone | "
        f"processed={processed} | skipped={skipped} | failed={failed} | "
        f"results_file={args.results_file}",
        quiet_mode,
    )


if __name__ == "__main__":
    main()
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

from boxsdk import Client, OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth
from boxsdk.exception import BoxAPIException
from boxsdk.exception import BoxOAuthException
from dotenv import find_dotenv, load_dotenv, set_key
from speciesnet import DEFAULT_MODEL, SpeciesNet


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ENV_PATH = find_dotenv(usecwd=True) or ".env"

EXCLUDED_LABELS = {"blank", "human", "vehicle", "empty"}


def store_tokens(access_token, refresh_token):
    set_key(ENV_PATH, "ACCESS_TOKEN", access_token)
    if refresh_token:
        set_key(ENV_PATH, "REFRESH_TOKEN", refresh_token)


def to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def parse_common_name(class_token: Optional[str]) -> Optional[str]:
    if not class_token:
        return None

    parts = str(class_token).split(";")
    label = parts[-1].strip() if parts else str(class_token).strip()
    return label or None


def extract_animals(prediction_entry: Dict, min_score: float, max_animals: int) -> List[Dict]:
    """
    Extract animals from both classifications and detections.
    
    Returns a list of dicts with:
    - label: animal species name
    - score: classification confidence score
    - taxonomy: full taxonomy string
    - bbox: bounding box [xmin, ymin, width, height] (normalized to [0.0, 1.0]), or None if not detected
    """
    # Build a map of detection boxes to potentially match with classifications
    detections = prediction_entry.get("detections") or []
    animal_detections = [
        d for d in detections
        if d.get("label") == "animal" or d.get("category") == "1"
    ]

    # Extract classifications
    classifications = prediction_entry.get("classifications") or {}
    classes = classifications.get("classes") or []
    scores = classifications.get("scores") or []

    animals = []
    for class_token, score in zip(classes, scores):
        label = parse_common_name(class_token)
        if not label:
            continue

        if label.lower() in EXCLUDED_LABELS:
            continue

        if score is None or float(score) < min_score:
            continue

        # Try to match with detection boxes (use first available or None)
        bbox = None
        if animal_detections and len(animals) < len(animal_detections):
            detection = animal_detections[len(animals)]
            bbox_raw = detection.get("bbox")
            if bbox_raw:
                bbox = [float(v) for v in bbox_raw]

        animals.append({
            "label": label,
            "score": float(score),
            "taxonomy": class_token,
            "bbox": bbox,  # [xmin, ymin, width, height] normalized to [0.0, 1.0]
        })

        if len(animals) >= max_animals:
            break

    return animals


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


def append_result(results_file: str, result: Dict):
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run SpeciesNet on images listed in box_images.json"
    )
    parser.add_argument(
        "--input-file",
        default=os.getenv("OUTPUT_FILE", "box_images.json"),
        help="Path to box image metadata JSON file.",
    )
    parser.add_argument(
        "--results-file",
        default=os.getenv("SPECIESNET_RESULTS_FILE", "speciesnet_results.jsonl"),
        help="JSONL file where classification results are appended.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SPECIESNET_MODEL", DEFAULT_MODEL),
        help="SpeciesNet model name to load.",
    )
    parser.add_argument(
        "--run-mode",
        default=os.getenv("SPECIESNET_RUN_MODE", "single_thread"),
        choices=["single_thread", "multi_thread", "multi_process"],
        help="Inference run mode.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.getenv("SPECIESNET_MIN_SCORE", "0.1")),
        help="Minimum score threshold for reporting animal labels.",
    )
    parser.add_argument(
        "--max-animals",
        type=int,
        default=int(os.getenv("SPECIESNET_MAX_ANIMALS", "5")),
        help="Maximum number of animal labels to keep per image.",
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
    model = SpeciesNet(args.model, components="all")

    total = len(records)
    processed = 0
    skipped = 0
    failed = 0

    log(f"Loaded {total} records from {args.input_file}", quiet_mode)
    if processed_ids:
        log(f"Skipping {len(processed_ids)} already processed file_id values", quiet_mode)

    for record in records:
        if args.limit and processed >= args.limit:
            break

        file_id = str(record.get("file_id", "")).strip()
        file_name = record.get("file_name") or f"{file_id}.jpg"
        file_url = record.get("file_url") or record.get("web_url")

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

        temp_path = None
        try:
            log(f"Processing file_id={file_id} name={file_name}", quiet_mode)
            temp_path = download_box_file_to_temp(client, file_id, file_name)

            prediction_dict = model.predict(
                filepaths=[temp_path],
                run_mode=args.run_mode,
                progress_bars=False,
            )

            predictions = (prediction_dict or {}).get("predictions", [])
            prediction_entry = predictions[0] if predictions else {}
            animals = extract_animals(prediction_entry, args.min_score, args.max_animals)

            # Extract detections for detailed spatial info
            detections_raw = prediction_entry.get("detections") or []
            detections_summary = []
            for det in detections_raw:
                if det.get("label") == "animal" or det.get("category") == "1":
                    detections_summary.append({
                        "category": det.get("category"),
                        "label": det.get("label"),
                        "conf": float(det.get("conf", 0)) if det.get("conf") is not None else None,
                        "bbox": det.get("bbox"),  # [xmin, ymin, width, height] normalized
                    })

            result = {
                "status": "ok",
                "file_id": file_id,
                "file_name": file_name,
                "file_url": file_url,
                "animals": animals,  # Top-N classified animals with optional bbox
                "detections": detections_summary,  # All animal detections with bbox and confidence
                "prediction": prediction_entry.get("prediction"),
                "prediction_score": prediction_entry.get("prediction_score"),
                "prediction_source": prediction_entry.get("prediction_source"),
            }
            append_result(args.results_file, result)

            processed += 1
            processed_ids.add(file_id)
            log(
                f"  done file_id={file_id} | animals_found={len(animals)} | "
                f"prediction={result['prediction']}",
                quiet_mode,
            )

        except (BoxOAuthException, BoxAPIException) as e:
            failed += 1
            append_result(
                args.results_file,
                {
                    "status": "error",
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_url": file_url,
                    "error": type(e).__name__,
                    "message": str(e),
                },
            )
            log(f"  failed file_id={file_id} error={type(e).__name__}", quiet_mode)

        except Exception as e:
            failed += 1
            append_result(
                args.results_file,
                {
                    "status": "error",
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_url": file_url,
                    "error": type(e).__name__,
                    "message": str(e),
                },
            )
            log(f"  failed file_id={file_id} error={type(e).__name__}", quiet_mode)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    log(
        "\nClassification complete | "
        f"processed={processed} | skipped={skipped} | failed={failed} | "
        f"results_file={args.results_file}",
        quiet_mode,
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

from boxsdk import Client, OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth
from boxsdk.exception import BoxAPIException, BoxOAuthException
from dotenv import find_dotenv, load_dotenv, set_key
from speciesnet import DEFAULT_MODEL, SpeciesNet


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ENV_PATH = find_dotenv(usecwd=True) or ".env"

EXCLUDED_LABELS = {"blank", "human", "vehicle", "empty"}


from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

def download_images_parallel(
    client,
    records,
    max_workers=32
):
    results = []

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_map = {}

        for record in records:

            file_id = str(
                record["file_id"]
            )

            file_name = (
                record.get("file_name")
                or f"{file_id}.jpg"
            )

            future = executor.submit(
                download_box_file_to_temp,
                client,
                file_id,
                file_name
            )

            future_map[future] = record

        for future in as_completed(
            future_map
        ):

            record = future_map[future]

            try:

                temp_path = future.result()

                results.append(
                    (
                        record,
                        temp_path
                    )
                )

            except Exception as e:

                print(
                    f"Download failed: "
                    f"{record['file_id']} "
                    f"{e}"
                )

    return results

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
            try:
                record = json.loads(line)
                if record.get("status") == "ok":
                    processed.add(str(record.get("file_id")))
            except json.JSONDecodeError:
                continue

    return processed


def append_result(results_file: str, result: Dict):
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def parse_common_name(class_token: Optional[str]) -> Optional[str]:
    if not class_token:
        return None

    parts = str(class_token).split(";")
    label = parts[-1].strip() if parts else str(class_token).strip()
    return label or None


def extract_animals(prediction_entry: Dict, min_score: float, max_animals: int) -> List[Dict]:
    detections = prediction_entry.get("detections") or []
    animal_detections = [
        d for d in detections
        if d.get("label") == "animal" or d.get("category") == "1"
    ]

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

        bbox = None
        if animal_detections and len(animals) < len(animal_detections):
            bbox_raw = animal_detections[len(animals)].get("bbox")
            if bbox_raw:
                bbox = [float(v) for v in bbox_raw]

        animals.append({
            "label": label,
            "score": float(score),
            "taxonomy": class_token,
            "bbox": bbox,
        })

        if len(animals) >= max_animals:
            break

    return animals


def summarize_detections(prediction_entry: Dict) -> List[Dict]:
    detections_raw = prediction_entry.get("detections") or []
    detections_summary = []

    for det in detections_raw:
        if det.get("label") == "animal" or det.get("category") == "1":
            detections_summary.append({
                "category": det.get("category"),
                "label": det.get("label"),
                "conf": float(det.get("conf", 0)) if det.get("conf") is not None else None,
                "bbox": det.get("bbox"),
            })

    return detections_summary


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


def process_batch(
    model: SpeciesNet,
    batch_records: List[dict],
    batch_paths: List[str],
    results_file: str,
    min_score: float,
    max_animals: int,
    run_mode: str,
):
    prediction_dict = model.predict(
        filepaths=batch_paths,
        run_mode=run_mode,
        progress_bars=False,
    )

    predictions = (prediction_dict or {}).get("predictions", [])

    for record, prediction_entry in zip(batch_records, predictions):
        file_id = str(record.get("file_id", "")).strip()
        file_name = record.get("file_name") or f"{file_id}.jpg"
        file_url = record.get("file_url") or record.get("web_url")

        animals = extract_animals(
            prediction_entry,
            min_score=min_score,
            max_animals=max_animals,
        )

        result = {
            "status": "ok",
            "file_id": file_id,
            "file_name": file_name,
            "file_url": file_url,
            "animals": animals,
            "detections": summarize_detections(prediction_entry),
            "prediction": prediction_entry.get("prediction"),
            "prediction_score": prediction_entry.get("prediction_score"),
            "prediction_source": prediction_entry.get("prediction_source"),
        }

        append_result(results_file, result)


def cleanup_temp_files(paths: List[str]):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def main():

    parser = argparse.ArgumentParser(
        description="Run SpeciesNet on Box images using batched inference."
    )

    parser.add_argument("--input-file", default="box_images.json")
    parser.add_argument("--results-file", default="speciesnet_results.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)

    parser.add_argument(
        "--run-mode",
        default="single_thread",
        choices=[
            "single_thread",
            "multi_thread",
            "multi_process",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--download-workers",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-animals",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
    )

    args = parser.parse_args()

    records = load_json_list(
        args.input_file
    )

    processed_ids = (
        set()
        if args.reprocess
        else load_processed_file_ids(
            args.results_file
        )
    )

    if args.limit:
        records = records[: args.limit]

    records = [
        r
        for r in records
        if str(
            r.get("file_id", "")
        ).strip()
        not in processed_ids
    ]

    print(
        f"Loaded {len(records)} records "
        f"to process"
    )

    client = build_client()

    model = SpeciesNet(
        args.model,
        components="all",
    )

    processed = 0
    failed = 0

    batch_size = args.batch_size

    for start_idx in range(
        0,
        len(records),
        batch_size,
    ):

        end_idx = min(
            start_idx + batch_size,
            len(records),
        )

        chunk = records[
            start_idx:end_idx
        ]

        print(
            "\n"
            f"Downloading batch "
            f"{start_idx:,}"
            f" - "
            f"{end_idx:,}"
        )

        downloaded = (
            download_images_parallel(
                client=client,
                records=chunk,
                max_workers=args.download_workers,
            )
        )

        if not downloaded:

            print(
                "No images downloaded "
                "successfully."
            )

            continue

        batch_records = [
            r
            for r, _ in downloaded
        ]

        batch_paths = [
            p
            for _, p in downloaded
        ]

        print(
            f"Running SpeciesNet "
            f"on {len(batch_paths)} images..."
        )

        try:

            process_batch(
                model=model,
                batch_records=batch_records,
                batch_paths=batch_paths,
                results_file=args.results_file,
                min_score=args.min_score,
                max_animals=args.max_animals,
                run_mode=args.run_mode,
            )

            processed += len(
                batch_paths
            )

            print(
                f"Processed "
                f"{processed:,} images"
            )

        except Exception as e:

            failed += len(
                batch_paths
            )

            print(
                "Batch failed:"
            )

            print(e)

        finally:

            cleanup_temp_files(
                batch_paths
            )

    print(
        "\nDone\n"
        f"Processed: "
        f"{processed:,}\n"
        f"Failed: "
        f"{failed:,}\n"
        f"Results: "
        f"{args.results_file}"
    )


if __name__ == "__main__":
    main()
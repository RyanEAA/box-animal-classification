import os
import argparse
from boxsdk import Client, OAuth2
from boxsdk.auth.developer_token_auth import DeveloperTokenAuth
from boxsdk.exception import BoxAPIException
from boxsdk.exception import BoxOAuthException
from dotenv import load_dotenv, find_dotenv, set_key
import json


# load variables from .env
load_dotenv()

# =========================
# 🔐 CONFIG
# =========================
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
ENV_PATH = find_dotenv(usecwd=True) or ".env"


def store_tokens(access_token, refresh_token):
    set_key(ENV_PATH, "ACCESS_TOKEN", access_token)
    if refresh_token:
        set_key(ENV_PATH, "REFRESH_TOKEN", refresh_token)

if not ACCESS_TOKEN:
    raise ValueError("ACCESS_TOKEN is missing. Set it in your .env file.")

if REFRESH_TOKEN and CLIENT_ID and CLIENT_SECRET:
    auth = OAuth2(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        access_token=ACCESS_TOKEN,
        refresh_token=REFRESH_TOKEN,
        store_tokens=store_tokens
    )
else:
    # Developer tokens are short-lived and not refreshable.
    auth = DeveloperTokenAuth(get_new_token_callback=lambda: ACCESS_TOKEN)

client = Client(auth)

ROOT_FOLDER_ID = os.getenv("ROOT_FOLDER_ID")  # Box folder ID to start from
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "box_images.json")


def to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


parser = argparse.ArgumentParser(description="Export Box image URLs from a folder tree.")
parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress traversal logs (enter/exit folders and per-file progress).",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=int(os.getenv("BATCH_SIZE", "100")),
    help="Number of new records to buffer before writing to box_images.json.",
)
args = parser.parse_args()

QUIET_MODE = args.quiet or to_bool(os.getenv("QUIET_MODE", "false"))
BATCH_SIZE = max(1, args.batch_size)


def log(message, quiet=False):
    if not quiet:
        print(message)


def load_existing_records(output_file):
    if not os.path.exists(output_file):
        return {}

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, list):
        return {}

    records = {}
    for item in data:
        if isinstance(item, dict) and item.get("file_id"):
            records[str(item["file_id"])] = item
    return records


def flush_pending(output_state, quiet=False, force=False):
    pending_records = output_state["pending_records"]
    if not pending_records and not force:
        return

    new_count = len(pending_records)
    for record in pending_records:
        file_id = str(record["file_id"])
        output_state["records_by_id"][file_id] = record

    if pending_records:
        output_state["pending_records"] = []
        output_state["batches_written"] += 1

    with open(output_state["output_file"], "w") as f:
        json.dump(list(output_state["records_by_id"].values()), f, indent=4)

    if new_count:
        log(
            f"Flushed batch {output_state['batches_written']}: "
            f"{new_count} new records written | "
            f"total logged={len(output_state['records_by_id'])}",
            quiet,
        )


def get_all_images(
    folder_id,
    parent_path="",
    seen_file_ids=None,
    stats=None,
    depth=0,
    quiet=False,
    output_state=None,
):
    if seen_file_ids is None:
        seen_file_ids = set()
    if stats is None:
        stats = {"files_searched": 0, "images_found": 0, "already_logged": 0}
    if output_state is None:
        raise ValueError("output_state is required")

    indent = "  " * depth
    folder_label = parent_path if parent_path else "/"
    log(f"{indent}Entering folder: {folder_label} (id={folder_id})", quiet)

    items = client.folder(folder_id=folder_id).get_items(limit=1000)
    for item in items:
        if item.type == "file":
            stats["files_searched"] += 1

            if item.name.lower().endswith(IMAGE_EXTENSIONS):
                if item.id in seen_file_ids:
                    stats["already_logged"] += 1
                    continue

                seen_file_ids.add(item.id)
                file = client.file(item.id).get(fields=["id", "name"])
                file_url = f"https://app.box.com/file/{file.id}"

                output_state["pending_records"].append({
                    "file_name": file.name,
                    "file_id": file.id,
                    "path": parent_path,
                    "file_url": file_url,

                    # These URLs are regular Box URLs and require auth.
                    "web_url": file_url,
                    "direct_download_url": f"https://api.box.com/2.0/files/{file.id}/content",
                    "preview_url": f"https://app.box.com/file/{file.id}/preview"
                })
                stats["images_found"] += 1
                log(
                    f"{indent}  Image matched: {file.name} | "
                    f"url={file_url} | "
                    f"files searched={stats['files_searched']} | "
                    f"images found={stats['images_found']}",
                    quiet,
                )

                if len(output_state["pending_records"]) >= output_state["batch_size"]:
                    flush_pending(output_state, quiet=quiet)

        elif item.type == "folder":
            get_all_images(
                item.id,
                f"{parent_path}/{item.name}",
                seen_file_ids,
                stats,
                depth + 1,
                quiet,
                output_state,
            )

    log(
        f"{indent}Exiting folder: {folder_label} | "
        f"files searched so far={stats['files_searched']} | "
        f"images found so far={stats['images_found']}",
        quiet,
    )


if not ROOT_FOLDER_ID:
    raise ValueError("ROOT_FOLDER_ID is missing. Set it in your .env file.")

existing_records = load_existing_records(OUTPUT_FILE)
output_state = {
    "output_file": OUTPUT_FILE,
    "batch_size": BATCH_SIZE,
    "records_by_id": existing_records,
    "pending_records": [],
    "batches_written": 0,
}
seen_file_ids = set(existing_records.keys())
stats = {"files_searched": 0, "images_found": 0, "already_logged": 0}

try:
    get_all_images(
        ROOT_FOLDER_ID,
        seen_file_ids=seen_file_ids,
        stats=stats,
        quiet=QUIET_MODE,
        output_state=output_state,
    )
    flush_pending(output_state, quiet=QUIET_MODE, force=True)
except BoxOAuthException as e:
    raise RuntimeError(
        "Box authentication failed. Your ACCESS_TOKEN is likely expired or invalid. "
        "Generate a new developer token, or provide CLIENT_ID/CLIENT_SECRET/REFRESH_TOKEN for auto-refresh."
    ) from e
except BoxAPIException as e:
    if e.status == 401:
        raise RuntimeError(
            "Box access was denied with 401 invalid_token. Your ACCESS_TOKEN is expired or invalid. "
            "Generate a fresh token (or configure REFRESH_TOKEN with CLIENT_ID/CLIENT_SECRET)."
        ) from e
    raise

print(
    f"✅ Crawl complete | new images logged={stats['images_found']} | "
    f"already logged skipped={stats['already_logged']} | "
    f"total records in {OUTPUT_FILE}={len(output_state['records_by_id'])}"
)

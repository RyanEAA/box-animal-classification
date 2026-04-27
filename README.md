# Box Animal Classification Pipeline

End-to-end pipeline for classifying animals in Box images with spatial bounding box data:

1. **OAuth Setup** – Authenticate with Box and store tokens (.env)
2. **Crawl Box** – Recursively traverse folders and export image metadata
3. **Run SpeciesNet** – Classify animals in each image, extract bounding boxes
4. **Export to CSV** – Flatten results for analysis, ML training, or database import

## Project Files

- `box-oauth-setup.py` – One-time OAuth bootstrap (gets ACCESS_TOKEN + REFRESH_TOKEN)
- `box-get-urls.py` – Crawls Box folders, exports image metadata + URLs with batch writes + de-duplication
- `box-run-speciesnet.py` – Downloads images, runs SpeciesNet classification, extracts animal labels + bounding boxes
- `jsonl-to-csv.py` – Converts JSONL results to CSV (one row per animal per image)
- `requirements.txt` – Python dependencies
- `.env` – Local credentials and config (not committed)
- `box_images.json` – Index of all crawled images (source data)
- `speciesnet_results.jsonl` – Animal classifications with spatial data (JSONL format)
- `speciesnet_results.csv` – Flattened results for analysis (CSV format)

## 1) Setup

Use your virtual environment and install dependencies:

- python -m venv .venv
- source .venv/bin/activate
- python -m pip install -r requirements.txt

## 2) Box App Configuration

In Box Developer Console for your app:

1. Enable OAuth 2.0
2. Set Redirect URI (example: http://localhost)
3. Copy these values:
- CLIENT_ID
- CLIENT_SECRET

## 3) Create .env

Create .env in the project root with:

```
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
REDIRECT_URI=http://localhost
ROOT_FOLDER_ID=your_box_folder_id
ACCESS_TOKEN=
REFRESH_TOKEN=
QUIET_MODE=false
BATCH_SIZE=100
OUTPUT_FILE=box_images.json
SPECIESNET_MODEL=
SPECIESNET_RUN_MODE=single_thread
SPECIESNET_MIN_SCORE=0.1
SPECIESNET_MAX_ANIMALS=5
```

Notes:

- **OAuth tokens**: ACCESS_TOKEN and REFRESH_TOKEN start empty; box-oauth-setup.py fills them after auth.
- **Box crawling**: ROOT_FOLDER_ID is your source folder; OUTPUT_FILE changes the crawl output name/path.
- **Crawler options**: BATCH_SIZE controls buffer size before each write (default 100); QUIET_MODE=true suppresses logs.
- **SpeciesNet options**: SPECIESNET_MODEL selects the model (leave empty for default); RUN_MODE can be single_thread/multi_thread/multi_process.
- **Filtering**: SPECIESNET_MIN_SCORE filters low-confidence predictions (default 0.1); SPECIESNET_MAX_ANIMALS limits top-N animals per image (default 5).

## 4) Run OAuth Bootstrap (one-time)

Run:

- python box-oauth-setup.py

What happens:

1. Script opens the Box consent URL in your browser.
2. You approve access.
3. Browser redirects to localhost and may show connection refused.
4. Copy the full URL from browser address bar.
5. Paste it into the script prompt.
6. Script exchanges code for tokens and writes ACCESS_TOKEN + REFRESH_TOKEN into .env.

## 5) Crawl Box Image URLs

Run with logs:

- python box-get-urls.py

Run quietly:

- python box-get-urls.py --quiet

Or use env quiet mode:

- QUIET_MODE=true python box-get-urls.py

Run with custom batch size:

- python box-get-urls.py --batch-size 25

Or use env batch size:

- BATCH_SIZE=25 python box-get-urls.py

What the crawler logs in verbose mode:

- Folder enter
- Folder exit
- Running counts of searched files and found images
- Matched image file URL
- Batch flush summaries while crawling

How output writing now works:

- Writes in batches during crawl (not only once at the end)
- Loads existing OUTPUT_FILE on startup
- Uses file_id de-duplication to skip files already logged in previous runs
- Prevents duplicate entries in box_images.json across reruns

## 6) Output Format: box_images.json

The crawler writes box_images.json. Each image record includes:

- `file_name` – Original filename from Box
- `file_id` – Stable unique identifier (best for de-duplication)
- `path` – Full folder path (e.g., /folder1/subfolder/image.jpg)
- `file_url` – Direct file URL via Box API
- `web_url` – Box web portal link (requires login)
- `direct_download_url` – Direct download endpoint (requires auth token)
- `preview_url` – Box preview link (requires login)

Output behavior:

- Existing records are preserved on rerun.
- New records are appended in-memory and flushed in batches.
- Records are keyed by **file_id** internally to avoid duplicates across reruns.

## 7) Run SpeciesNet Classification

Process each image in box_images.json through the SpeciesNet model:

```bash
# Default: process all images, output to speciesnet_results.jsonl
.venv/bin/python box-run-speciesnet.py

# Quiet mode (no logs)
.venv/bin/python box-run-speciesnet.py --quiet

# Process only first 10 images (useful for testing)
.venv/bin/python box-run-speciesnet.py --limit 10

# Reprocess all files (ignore prior results)
.venv/bin/python box-run-speciesnet.py --reprocess

# Filter by confidence threshold
.venv/bin/python box-run-speciesnet.py --min-score 0.5

# Limit top-N animals per image
.venv/bin/python box-run-speciesnet.py --max-animals 3

# Use multi-threaded inference
.venv/bin/python box-run-speciesnet.py --run-mode multi_thread

# Custom input/output files
.venv/bin/python box-run-speciesnet.py --input-file my_images.json --results-file my_results.jsonl
```

What happens:

1. Reads image metadata from box_images.json
2. Downloads each image to a temporary local file
3. Runs SpeciesNet inference (detector + classifier + ensemble)
4. Extracts animal labels with confidence scores and bounding box coordinates
5. Appends result to speciesnet_results.jsonl (JSONL format = one record per line)
6. **Deletes temp file** (guaranteed cleanup in finally block)
7. De-duplicates by file_id (skips files already successfully processed)

Output schema (speciesnet_results.jsonl):

```json
{
  "status": "ok",
  "file_id": "2123025883133",
  "file_name": "IMG_0001.JPG",
  "file_url": "https://app.box.com/file/2123025883133",
  "animals": [
    {
      "label": "striped skunk",
      "score": 0.9809,
      "taxonomy": "9282b7e4-f3b0-4ef3-9741-cae7f7bd346a;mammalia;carnivora;mephitidae;mephitis;mephitis;striped skunk",
      "bbox": [0.1, 0.2, 0.6, 0.7]
    }
  ],
  "detections": [
    {
      "category": "1",
      "label": "animal",
      "conf": 0.9809,
      "bbox": [0.1, 0.2, 0.6, 0.7]
    }
  ],
  "prediction": "9282b7e4-f3b0-4ef3-9741-cae7f7bd346a;...",
  "prediction_score": 0.9809,
  "prediction_source": "classifier"
}
```

### Bounding Box Format

All bounding boxes are normalized to `[0.0, 1.0]` range:

- `bbox[0]` = xmin (left edge, 0.0 = image left, 1.0 = image right)
- `bbox[1]` = ymin (top edge, 0.0 = image top, 1.0 = image bottom)
- `bbox[2]` = width (as fraction of image width)
- `bbox[3]` = height (as fraction of image height)

Example: `[0.1, 0.2, 0.6, 0.7]` = animal located at x:10%-70%, y:20%-90% of image.

## 8) Export to CSV

Convert JSONL results to CSV (one row per animal per image) for analysis and ML training:

```bash
# Default: read speciesnet_results.jsonl, write speciesnet_results.csv
.venv/bin/python jsonl-to-csv.py

# Custom input/output
.venv/bin/python jsonl-to-csv.py --input-file my_results.jsonl --output-file export.csv

# Quiet mode
.venv/bin/python jsonl-to-csv.py --quiet
```

CSV columns:

- `file_id`, `file_name`, `file_url` – Image identifiers
- `status` – "ok" (success) or "error" (failure)
- `animal_label`, `animal_score` – Species name and classification confidence
- `animal_taxonomy` – Full taxonomic ID
- `detection_conf` – Detector confidence score
- `bbox_xmin`, `bbox_ymin`, `bbox_width`, `bbox_height` – Normalized bounding box
- `prediction`, `prediction_score`, `prediction_source` – Ensemble result

CSV is ideal for:

- Importing into Excel/Google Sheets
- Pandas DataFrames and data analysis
- Training ML models (object detection, segmentation)
- Database import
- Sharing with team/collaborators

## 9) Recommended Next Stage: Download -> Process -> Delete

For your full pipeline (AI model + OCR + deletion), process one file at a time using file_id as the source of truth.

Suggested per-image flow:

1. Read one record from box_images.json
2. Download bytes using file_id via Box SDK
3. Run AI model inference (already done by box-run-speciesnet.py)
4. Run OCR (optional next phase)
5. Save result to local results file (JSONL/CSV/DB)
6. If and only if processing succeeded, delete the file from Box by file_id
7. Record status as processed

Why file_id is your unique key:

- Stable across renames and path changes
- More reliable than URL or path
- Fits delete operations directly (Box SDK: `client.file(file_id).delete()`)

## 10) Safety Recommendations Before Deleting

- Start with dry-run mode (no delete)
- Write a processing log with statuses: pending, processed, failed, deleted
- Only delete after successful AI + OCR + persistence
- Keep retry logic for transient API failures
- Consider moving to an archive folder first instead of immediate delete

## 11) Common Issues

**401 invalid_token:**

- Token expired/invalid
- Re-run `python box-oauth-setup.py` to refresh bootstrap tokens
- Ensure CLIENT_ID/CLIENT_SECRET/REDIRECT_URI match your Box app

**403 insufficient permissions:**

- Token user/app lacks access to folder or file actions
- Check app scopes and folder collaboration permissions

**No files found in crawl:**

- Verify ROOT_FOLDER_ID
- Verify image extensions in box-get-urls.py

**SpeciesNet inference fails:**

- Check temp disk space (images downloaded locally first)
- Verify SPECIESNET_MODEL name is valid
- Check GPU memory if using multi_process mode

**CSV has NULL bbox values:**

- Detector may not have confidence in bounding box for that detection
- This is expected for some images; use available box data for training

## 12) Current Script Behavior Notes

**box-get-urls.py** supports two auth modes:

1. OAuth refresh mode when REFRESH_TOKEN + CLIENT_ID + CLIENT_SECRET are present
2. Developer-token fallback mode otherwise

In OAuth refresh mode, refreshed tokens are written back to .env automatically.

Additional features:

- `--quiet` suppresses traversal/progress logs
- `--batch-size N` controls incremental write frequency (default 100)
- `QUIET_MODE` env var toggles quiet mode
- `BATCH_SIZE` env var sets default batch size
- `OUTPUT_FILE` env var changes output destination

**box-run-speciesnet.py** features:

- Downloads images to temp files and **guarantees cleanup** (even on error)
- De-duplicates by file_id; safe to rerun without reprocessing
- Supports `--limit`, `--reprocess`, `--min-score`, `--max-animals`, `--quiet` flags
- Appends to JSONL incrementally (resumable on interruption)
- Extracts both classifications and raw detection bounding boxes

**jsonl-to-csv.py** features:

- Flattens one-animal-per-row for analysis
- Expands nested bounding box arrays
- Handles images with multiple animals (one CSV row per animal)
- Includes error handling for malformed JSONL records

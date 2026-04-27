# Box Animal Classification Pipeline

This project currently does two things:

1. Gets Box OAuth tokens and stores them in .env
2. Crawls a Box folder tree and exports image URLs to box_images.json

You can then build your next stage to download each image, run AI model + OCR, and delete processed images.

## Project Files

- box-oauth-setup.py: One-time OAuth bootstrap helper (gets ACCESS_TOKEN + REFRESH_TOKEN)
- box-get-urls.py: Crawls Box folders and exports image metadata + URLs
- requirements.txt: Python dependencies
- .env: Local credentials and config (not committed)

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

CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
REDIRECT_URI=http://localhost
ROOT_FOLDER_ID=your_box_folder_id
ACCESS_TOKEN=
REFRESH_TOKEN=
QUIET_MODE=false
BATCH_SIZE=100
OUTPUT_FILE=box_images.json

Notes:

- ACCESS_TOKEN and REFRESH_TOKEN can start empty.
- box-oauth-setup.py will fill them in after auth.
- BATCH_SIZE controls how many new records are buffered before each write.
- OUTPUT_FILE lets you change the output JSON filename/path.

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

## 6) Output Format

The crawler writes box_images.json. Each image record includes:

- file_name
- file_id (best unique ID)
- path
- file_url
- web_url
- direct_download_url
- preview_url

Output behavior details:

- Existing records are preserved.
- New records are appended in-memory and flushed in batches.
- Records are keyed by file_id internally to avoid duplicate JSON entries.

Important URL behavior:

- web_url and preview_url are Box web links and typically require login.
- direct_download_url requires authorization token when requested.
- These are regular Box URLs, not public shared links.

## 7) Recommended Next Stage: Download -> Process -> Delete

For your plan (AI model + OCR + deletion), process one file at a time using file_id as the source of truth.

Suggested per-image flow:

1. Read one record from box_images.json
2. Download bytes using file_id via Box SDK
3. Run AI model inference
4. Run OCR
5. Save result to a local results file (JSONL/CSV/DB)
6. If and only if processing succeeded, delete the file from Box by file_id
7. Record status as processed

Why file_id should be your unique key:

- Stable across renames and path changes
- More reliable than URL or path
- Fits delete operations directly

## 8) Safety Recommendations Before Deleting

- Start with dry-run mode (no delete)
- Write a processing log with statuses: pending, processed, failed, deleted
- Only delete after successful AI + OCR + persistence
- Keep retry logic for transient API failures
- Consider moving to an archive folder first instead of immediate delete

## 9) Common Issues

401 invalid_token:

- Token expired/invalid
- Re-run python box-oauth-setup.py to refresh bootstrap tokens
- Ensure CLIENT_ID/CLIENT_SECRET/REDIRECT_URI match your Box app

403 insufficient permissions:

- Token user/app lacks access to folder or file actions
- Check app scopes and folder collaboration permissions

No files found:

- Verify ROOT_FOLDER_ID
- Verify image extensions in box-get-urls.py

## 10) Current Script Behavior Notes

box-get-urls.py supports two auth modes:

1. OAuth refresh mode when REFRESH_TOKEN + CLIENT_ID + CLIENT_SECRET are present
2. Developer-token fallback mode otherwise

In OAuth refresh mode, refreshed tokens are written back to .env automatically.

box-get-urls.py also supports:

- --quiet to suppress traversal/progress logs
- --batch-size N to control incremental write frequency
- QUIET_MODE env var as a quiet-mode toggle
- BATCH_SIZE env var for default batch size
- OUTPUT_FILE env var to change output destination

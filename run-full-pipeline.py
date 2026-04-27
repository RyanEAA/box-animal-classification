#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "box-oauth-setup.py",
    "box-get-urls.py",
    "box-run-speciesnet.py",
    "paddle_metadata_ocr.py",
]


def run_step(step_name: str, python_exe: str, cwd: Path) -> int:
    script_path = cwd / step_name
    if not script_path.exists():
        print(f"[ERROR] Missing script: {script_path}")
        return 1

    print("\n" + "=" * 72)
    print(f"Running: {step_name}")
    print("=" * 72)

    # Use -u for unbuffered output so logs stream live.
    cmd = [python_exe, "-u", str(script_path)]
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full Box -> SpeciesNet -> OCR pipeline in sequence."
    )
    parser.add_argument(
        "--skip-oauth",
        action="store_true",
        help="Skip box-oauth-setup.py (useful when tokens are already valid).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent
    python_exe = args.python

    if not Path(python_exe).exists():
        print(f"[ERROR] Python interpreter not found: {python_exe}")
        return 1

    print("Pipeline start")
    print(f"Workspace: {cwd}")
    print(f"Python: {python_exe}")

    steps = SCRIPT_ORDER[:]
    if args.skip_oauth:
        steps = [s for s in steps if s != "box-oauth-setup.py"]

    for step in steps:
        code = run_step(step, python_exe, cwd)
        if code != 0:
            print(f"\n[FAILED] {step} exited with code {code}")
            print("Stopping pipeline.")
            return code

    print("\n[SUCCESS] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

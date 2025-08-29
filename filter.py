#!/usr/bin/env python3
import os
import sys
import csv
import time
import shutil
from pathlib import Path

import pandas as pd

# --- CONFIG (no arguments needed) ---
CSV_PATH = Path("test-set-original-sample-with-families.backup_20250827_175621.csv")
DATASET_ROOT = Path("dataset")  # includes VOL001, VOL001-addition, VOL001-addition2, etc., recursively
TXT_EXTENSION = ".txt"
CHAR_LIMIT = 100  # "100 characters or fewer" => remove

def read_first_n_chars(fp: Path, n: int) -> int:
    """
    Return the number of characters in the first n+1 characters read (capped at n+1),
    so we can test <= n without reading entire file.
    """
    # Decode as text; ignore weird bytes to avoid choking on encoding glitches.
    # We only need to know if length <= n, so read n+1 characters max.
    try:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            s = f.read(n + 1)
        return len(s)
    except Exception:
        # If we can't read, treat it as very short so it's conservative for quality filtering.
        return 0

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH.resolve()}", file=sys.stderr)
        sys.exit(1)
    if not DATASET_ROOT.exists():
        print(f"ERROR: dataset folder not found at {DATASET_ROOT.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Load CSV
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    if "Control" not in df.columns:
        print("ERROR: 'Control' column not found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # Normalize Control values (strip whitespace); keep original column as strings
    df["Control"] = df["Control"].astype(str).str.strip()
    control_values = set(df["Control"].tolist())

    print(f"Loaded {len(df)} rows from {CSV_PATH.name}.")
    print(f"Unique Control values in CSV: {len(control_values)}")

    # Map: control -> found_short (bool), found_any (bool)
    found_short = {c: False for c in control_values}
    found_any = {c: False for c in control_values}

    # Walk dataset once; only consider files whose basename (without .txt) is in control set
    checked_count = 0
    matched_file_count = 0

    for root, dirs, files in os.walk(DATASET_ROOT):
        # Optional micro-optimization: skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if not name.endswith(TXT_EXTENSION):
                continue
            stem = name[:-len(TXT_EXTENSION)]
            if stem not in control_values:
                continue

            matched_file_count += 1
            fp = Path(root) / name
            found_any[stem] = True

            # If we've already determined this control is short, no need to re-check other duplicates
            if found_short[stem]:
                continue

            # Efficiently determine if the file has <= CHAR_LIMIT characters
            chars = read_first_n_chars(fp, CHAR_LIMIT)
            checked_count += 1
            if chars <= CHAR_LIMIT:
                found_short[stem] = True  # any short duplicate triggers removal

    # Compute which rows to drop
    controls_to_remove = {c for c, is_short in found_short.items() if is_short}
    not_found_controls = {c for c, seen in found_any.items() if not seen}

    remove_mask = df["Control"].isin(controls_to_remove)
    to_remove = int(remove_mask.sum())

    # Backup original, then overwrite with filtered
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = CSV_PATH.with_suffix(f".backup_{ts}.csv")
    shutil.copy2(CSV_PATH, backup_path)

    filtered_df = df.loc[~remove_mask].copy()
    # Overwrite original file
    filtered_df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)

    # Also save a separate filtered copy for convenience (optional)
    filtered_copy = CSV_PATH.with_name(CSV_PATH.stem + f"_filtered_{ts}.csv")
    filtered_df.to_csv(filtered_copy, index=False, quoting=csv.QUOTE_MINIMAL)

    # Summary
    print("\n--- Summary ---")
    print(f"Scanned dataset root: {DATASET_ROOT.resolve()}")
    print(f"Total CSV rows: {len(df)}")
    print(f"Controls in CSV: {len(control_values)}")
    print(f"Matching .txt files found in dataset: {matched_file_count}")
    print(f"Controls actually checked (deduped): {checked_count}")
    print(f"Rows removed (Control with <= {CHAR_LIMIT} chars file): {to_remove}")
    print(f"Rows remaining: {len(filtered_df)}")
    if not_found_controls:
        print(f"Controls with no matching .txt found in dataset: {len(not_found_controls)}")
        # If the list is very long, avoid flooding the console — print a small sample
        sample = list(sorted(not_found_controls))[:20]
        print("  Sample (up to 20):", ", ".join(sample))

    print(f"\nBackup of original CSV: {backup_path.name}")
    print(f"Filtered copy also saved as: {filtered_copy.name}")
    print("Done.")

if __name__ == "__main__":
    main()

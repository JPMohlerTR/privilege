#!/usr/bin/env python3
"""
Attorney-Client Privilege Analyzer (streaming approach to avoid file handle issues)

Key changes:
- Streaming CSV output generation (one row at a time)
- No batch loading of result files
- Explicit file handle management everywhere
- System resource monitoring
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
import gc
import resource
import shutil  # add this with the other imports


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('privilege_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe counters
progress_lock = Lock()
processed_count = 0
total_count = 0


def set_file_limits():
    """Attempt to increase file descriptor limits"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"Current file descriptor limits: soft={soft}, hard={hard}")
        # Try to increase soft limit to hard limit
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))
            new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"Increased soft limit to: {new_soft}")
    except Exception as e:
        logger.warning(f"Could not adjust file limits: {e}")


def call_o3_llm(prompt):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Error: OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        if not response.choices:
            raise ValueError("No choices returned in API response")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise


def clean_attorney_value(attorney_value):
    """Remove all occurrences of 'w/2' from the attorney value."""
    if pd.isna(attorney_value) or attorney_value == '':
        return None
    return str(attorney_value).replace('w/2', '')


def create_privilege_prompt(document_content, attorney_names=None):
    """Create the privilege determination prompt (kept as you wrote it)."""
    base_prompt = (
        "Is the following document attorney-client privileged? Defer toward thinking something is privileged if you are not sure, but you should have a good reason to call it privileged. Here is a list of reasons a document might be attorney-client privileged: "
        "1. it includes legal advice "
        "2. it includes a request for legal advice, which could include a request to draft, review, look at or comment upon specific documents or communications even if the document/communications being referenced does not appear to be a legal document "
        "3. it includes a request for information for the purpose of legal advice "
        "4. it involves providing information for the purpose of legal advice "
        "5. it includes both legal advice and a request for legal advice "
        "6. it includes or references discussions with attorneys seeking or providing legal advice "
        "7. it includes a request to a lawyer for clarification about rules or regulations. "
        "8. it includes discussion, references or is otherwise relevant in some way to some sort of dispute, litigation, arbitration, mediation or settlement of a litigation matter "
        "9. It includes documents in which Xcel is responding to a government investigation "
        "10. It includes emails and documents sent to or from internal or external counsel, experts, or consultants regarding the Smokehouse Creek Fire "
        "11. It includes emails and documents sent to or from internal or external counsel, experts, or consultants discussing potential causes of the Smokehouse Creek Fire, litigation preparation and potential liability related to the Smokehouse Creek Fire "
        "12. It contains communications that occurred after the fire related to the investigations conducted by Ryan \"Chance\" Hutson, Christopher R. Topliff, and/or Ismael Granda, who were acting at the direction of Xcel's lawyers after the Smokehouse Fire "
        "13. It includes documents created after the fire by Ryan \"Chance\" Hutson, Christopher R. Topliff, and/or Ismael Granda, who were acting at the direction of Xcel's lawyers, such as notes, in connection with their investigation into the Smokehouse Creek Fire "
        "14. It includes internal communications to gather facts and records relating to the Smokehouse Creek Fire "
        "15. It includes communications between Xcel's lawyers and its insurers updating its insurers on litigation relating to the Smokehouse Creek Fire or discussing with them Xcel's approach to such litigation/litigation strategy "
        "16. It includes presentations or minutes from a board or board committee meeting "
        "17. It is a legal agreement or settlement between parties, including draft agreements or settlements "
        "18. It relates to the process by which Xcel received claims from individuals and businesses harmed by the Smokehouse Creek Fire and paid those claims "
        "19. It relates to mediation with any individuals or businesses harmed by the Smokehouse Creek Fire, including any discussion of settlement amounts "
        "20. It contains legal analysis. "
        "-- Remember you should defer toward identifying the document as privileged as long as you have a good reason to think so. Keep a close eye on documents that refer to a law firm, or outside counsel/attorney, and especially if that law firm is Cravath, Swaine and Moore (these are often privileged). \n\n"
    )
    if attorney_names:
        base_prompt += f"Here are some names of attorneys (or possibly just the attorneys' website domain) who are involved in the case and mentioned in the document: {attorney_names}\n\n"

    base_prompt += (
        "If you decide privileged, respond with the character: Y\n"
        "If you decide not privileged, respond with the character: N\n\n"
        "only respond with one character, Y or N. Nothing else.\n\n"
        f"Document content:\n{document_content}"
    )
    return base_prompt


# ---------------- Fast helpers with explicit cleanup ----------------

def preload_processed_controls(output_dir: Path) -> set:
    """Read the run output folder once and return a set of already-processed control IDs."""
    processed = set()
    if not output_dir.exists():
        return processed

    try:
        with os.scandir(output_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if name.endswith(".txt"):
                    processed.add(name[:-4])  # strip .txt
    except FileNotFoundError:
        pass

    # Force cleanup
    gc.collect()
    return processed


def build_control_to_path_index(directory_path: Path, run_name: str) -> dict:
    """One-time walk of the dataset tree to map 'CONTROL' -> full path of source .txt."""
    index = {}
    run_dir = (directory_path / run_name).resolve()

    for root, dirs, files in os.walk(directory_path):
        # Skip the run output dir
        if Path(root).resolve() == run_dir:
            dirs[:] = []
            continue

        for fname in files:
            if not fname.endswith(".txt"):
                continue
            stem = fname[:-4]
            if stem not in index:
                index[stem] = str(Path(root) / fname)

    logger.info(f"Indexed {len(index):,} source .txt files under {directory_path}")

    # Force cleanup
    gc.collect()
    return index


def read_single_result_file(control_value: str, output_dir: Path) -> str:
    """Read a single result file and return its mapped classification for the CSV."""
    result_file = output_dir / f"{control_value}.txt"

    if not result_file.exists():
        # If no result file exists, we can't know—treat as not processed.
        # (When process_single_document runs, it will create NOT_FOUND or another marker.)
        return 'Not Processed'

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        upper = raw.upper()

        # Primary expected LLM outputs
        if upper.startswith('Y'):
            return 'Privileged'
        if upper.startswith('N'):
            return 'Not Privileged'

        # Our explicit markers (new + backward compatible)
        if upper in {'TOO_SHORT', 'DOCUMENT_TOO_SHORT'}:
            return 'TOO_SHORT'
        if upper in {'NOT_FOUND', 'FILE_NOT_FOUND'}:
            return 'NOT_FOUND'

        # Anything else is unknown
        if raw:
            logger.warning(f"Unexpected LLM/marker response for {control_value}: {raw[:80]}")
        return 'Unknown'

    except Exception as e:
        logger.error(f"Error reading result file for {control_value}: {e}")
        return 'Error'


def process_single_document(row_data):
    """Process a single document (parallel-safe)."""
    global processed_count, total_count

    control_value, attorney_value, directory_path, run_name, txt_path = row_data
    output_dir = Path(directory_path) / run_name
    output_file = output_dir / f"{control_value}.txt"

    try:
        # --- MISSING DOCUMENT HANDLING ---
        if not txt_path:
            logger.warning(f"Could not find .txt file for control value: {control_value}")
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("NOT_FOUND")
            except Exception as e:
                logger.error(f"Error writing NOT_FOUND marker for {control_value}: {e}")
            with progress_lock:
                processed_count += 1
            return control_value, "NOT_FOUND"
        # ---------------------------------

        # Read the document content (limit to first 300k chars)
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                document_content = f.read(300000)
        except Exception as e:
            logger.error(f"Error reading file {txt_path}: {e}")
            with progress_lock:
                processed_count += 1
            return control_value, "READ_ERROR"

        # --- SHORT DOCUMENT CHECK ---
        if len(document_content.strip()) < 100:
            logger.info(f"Skipping {control_value}: document too short")
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    # Write the new standardized marker
                    f.write("TOO_SHORT")
            except Exception as e:
                logger.error(f"Error writing TOO_SHORT marker for {control_value}: {e}")
            with progress_lock:
                processed_count += 1
            return control_value, "TOO_SHORT"
        # ----------------------------

        cleaned_attorney = clean_attorney_value(attorney_value)
        prompt = create_privilege_prompt(document_content, cleaned_attorney)

        try:
            llm_response = call_o3_llm(prompt).strip()

            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(llm_response)

            with progress_lock:
                processed_count += 1
            if processed_count % 100 == 0 or processed_count == total_count:
                logger.info(
                    f"Processed {processed_count}/{total_count} (latest: {control_value} -> {llm_response[:1]})"
                )

            return control_value, llm_response
        except Exception as e:
            logger.error(f"Error calling LLM for {control_value}: {e}")
            with progress_lock:
                processed_count += 1
            return control_value, "LLM_ERROR"

    except Exception as e:
        logger.error(f"Unexpected error processing {control_value}: {e}")
        with progress_lock:
            processed_count += 1
        return control_value, "UNEXPECTED_ERROR"


def create_output_csv_streaming(df, output_dir, output_csv_path):
    """Create output CSV by streaming through rows one at a time to avoid file handle issues."""
    logger.info("Creating output CSV using streaming approach...")

    # Write CSV header first
    df_sample = df.copy()
    df_sample['LLM_Privileged'] = 'placeholder'  # temporary column

    # Write just the header
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        df_sample.iloc[0:0].to_csv(f, index=False)  # Empty dataframe with just headers

    # Now stream through each row and append
    chunk_size = 1000
    processed_rows = 0

    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()

        # Process each row in the chunk
        llm_privileged_values = []
        for _, row in chunk.iterrows():
            control_value = str(row['Control'])
            llm_privileged = read_single_result_file(control_value, output_dir)
            llm_privileged_values.append(llm_privileged)

        chunk['LLM_Privileged'] = llm_privileged_values

        # Append to CSV (without header)
        with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
            chunk.to_csv(f, index=False, header=False)

        processed_rows += len(chunk)
        if processed_rows % 5000 == 0:
            logger.info(f"Streamed {processed_rows}/{len(df)} rows to CSV ({processed_rows / len(df) * 100:.1f}%)")
            gc.collect()  # Force cleanup periodically

    logger.info(f"Completed streaming all {len(df)} rows to CSV")


def main():
    parser = argparse.ArgumentParser(description='Analyze documents for attorney-client privilege (streaming approach)')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('directory_path', help='Path to the directory containing .txt files')
    parser.add_argument('run_name', help='Name of the run (will create a subdirectory)')
    parser.add_argument('--workers', type=int, default=100, help='Number of parallel threads (default: 100)')

    args = parser.parse_args()



    # Set file descriptor limits
    set_file_limits()

    csv_file = args.csv_file
    directory_path = Path(args.directory_path)
    run_name = args.run_name
    workers = args.workers

    for item in directory_path.iterdir():
        if item.is_dir() and item.name.startswith("priv-full"):
            logger.info(f"Deleting existing output folder: {item}")
            shutil.rmtree(item, ignore_errors=True)


    logger.info(f"Starting privilege analysis (streaming approach)")
    logger.info(f"CSV file: {csv_file}")
    logger.info(f"Directory path: {directory_path}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Threads: {workers}")

    # Read CSV
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded CSV with {len(df):,} rows")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return 1

    # Required columns
    if 'Control' not in df.columns:
        logger.error("CSV file must contain a 'Control' column")
        return 1
    if 'Attorney' not in df.columns:
        logger.warning("CSV has no 'Attorney' column; proceeding without")
        df['Attorney'] = ''

    # Preload already-processed controls from the run output dir
    output_dir = directory_path / run_name
    processed_controls = preload_processed_controls(output_dir)
    logger.info(f"Found {len(processed_controls):,} already-processed controls in {output_dir}")

    # Filter rows that still need processing
    df['Control'] = df['Control'].astype(str)
    todo_mask = ~df['Control'].isin(processed_controls)
    df_todo = df[todo_mask].copy()
    skipped = len(df) - len(df_todo)
    logger.info(f"Resume summary: {skipped:,} rows already done; {len(df_todo):,} remaining to process")

    # Build a one-time index of source files (fast O(1) lookup)
    control_to_path = build_control_to_path_index(directory_path, run_name)

    # Prepare row data only for remaining rows
    global total_count
    total_count = len(df_todo)

    row_data_list = []
    for _, row in df_todo.iterrows():
        control_value = str(row['Control'])
        attorney_value = row.get('Attorney', '')
        txt_path = control_to_path.get(control_value, None)
        row_data_list.append((control_value, attorney_value, str(directory_path), run_name, txt_path))

    results = {}

    if total_count > 0:
        logger.info(f"Processing {total_count:,} documents with {workers} threads...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_control = {
                executor.submit(process_single_document, row_data): row_data[0]
                for row_data in row_data_list
            }
            for future in as_completed(future_to_control):
                control_value = future_to_control[future]
                try:
                    control, result = future.result()
                    results[control] = result
                except Exception as e:
                    logger.error(f"Error processing {control_value}: {e}")
                    results[control_value] = "PROCESSING_ERROR"
    else:
        logger.info("Nothing new to process; all rows already completed in this run.")

    # Create output CSV using streaming approach (no bulk file loading)
    csv_path = Path(csv_file)
    output_csv_name = f"{csv_path.stem}_for_priv_run_{run_name}.csv"
    output_csv_path = csv_path.parent / output_csv_name

    try:
        create_output_csv_streaming(df, output_dir, output_csv_path)
        logger.info(f"Saved output CSV to: {output_csv_path}")
    except Exception as e:
        logger.error(f"Error creating output CSV: {e}")
        return 1

    # Optional stats if Answer column present
    if 'Answer' in df.columns:
        logger.info("Found Answer column - calculating accuracy statistics...")

        # Read the LLM_Privileged column back from the generated CSV for stats
        try:
            result_df = pd.read_csv(output_csv_path)

            answer_privileged = result_df[result_df['Answer'] == 'Privileged']
            total_answer_privileged = len(answer_privileged)
            both_privileged = len(
                result_df[(result_df['Answer'] == 'Privileged') & (result_df['LLM_Privileged'] == 'Privileged')])
            llm_false_positive = len(
                result_df[(result_df['Answer'] != 'Privileged') & (result_df['LLM_Privileged'] == 'Privileged')])
            answer_not_privileged = len(result_df[result_df['Answer'] == 'Not Privileged'])
            human_no_determination = len(result_df[~result_df['Answer'].isin(['Privileged', 'Not Privileged'])])
            llm_no_determination = len(result_df[~result_df['LLM_Privileged'].isin(['Privileged', 'Not Privileged'])])

            logger.info(
                f"Of the {total_answer_privileged} total Privileged docs, LLM found {both_privileged}. "
                f"LLM mistakenly identified {llm_false_positive} as Privileged out of the {answer_not_privileged} Not Privileged docs. "
                f"Humans failed to make a determination on {human_no_determination} docs; "
                f"LLM failed to make a determination on {llm_no_determination} docs"
            )
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
    else:
        logger.info("No Answer column found - skipping accuracy statistics")

    logger.info("Privilege analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

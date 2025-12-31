import os
import time
import requests
import pandas as pd
import sqlite3
import json
from dotenv import load_dotenv

# 1. Load configuration
load_dotenv()

API_KEY = os.getenv("ALLIUM_API_KEY")
QUERY_ID = os.getenv("ALLIUM_QUERY_ID")
DB_NAME = "allium_data.db"
TABLE_NAME = "query_results"

if not API_KEY or not QUERY_ID:
    raise ValueError("Please set ALLIUM_API_KEY and ALLIUM_QUERY_ID in your .env file")

HEADERS = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

# Configuration for Pagination
# WARNING: Increasing BATCH_LIMIT will increase memory usage for each batch.
# If you experience memory issues (e.g., "MemoryError"), reduce this value.
BATCH_LIMIT = 200000

def process_params_column(df):
    """
    Parses the 'params' column (which is a JSON string or dict)
    and flattens it into separate columns.
    """
    if 'params' not in df.columns:
        return df

    def parse_json(val):
        # Case 1: It's already a dictionary (API returned object)
        if isinstance(val, dict):
            return val
        # Case 2: It's a string (API returned serialized JSON)
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return {} # Return empty dict for unparseable strings
        return {} # Return empty dict for other types

    # 1. Expand the 'params' column into a new DataFrame of columns
    # This automatically creates columns: fileId, proofUrl, dlpId, etc.
    # Handle potential errors if parsing fails for some rows, resulting in non-dict values.
    expanded_data = df['params'].apply(parse_json).apply(pd.Series)

    # 2. Concatenate with original data
    df = pd.concat([df, expanded_data], axis=1)

    # 3. Drop the raw complex columns we don't need anymore
    df = df.drop(columns=['params', 'params_keys_mapping'], errors='ignore')

    return df

def clean_and_type_convert(df):
    """
    Cleans string artifacts, infers correct data types, and handles specific column types.
    This function aims to convert columns to more appropriate types where possible
    and suppresses the FutureWarning from pandas.
    """
    for col in df.columns:
        # 1. Strip quotes from all object columns
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip('"')

        # 2. Convert 'block_timestamp' to datetime
        if 'timestamp' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
        # 3. Attempt to convert other columns to numeric (Int64 or float)
        else:
            # Try converting to Int64 (Pandas nullable integer type) first
            # This is suitable for 'block_number', 'dlpId', 'proofIndex', 'score'
            # errors='coerce' turns non-numeric values into NaN
            try:
                temp_series = pd.to_numeric(df[col], errors='coerce')
                if not temp_series.isna().all(): # Only convert if there's at least one valid number
                    df[col] = temp_series.astype(pd.Int64Dtype())
                else: # If all are NaN, it means none were convertible to Int64, leave as original (string after stripping quotes)
                    pass
            except Exception: # If astype(pd.Int64Dtype()) fails for some reason (e.g., very large numbers needing float)
                # Try float conversion
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass # If float conversion also fails, leave as is (string/object)
    return df

def run_pagination_loop():
    offset = 0
    loop_count = 0
    total_rows_stored = 0

    print(f"--- Starting Sync for Query {QUERY_ID} ---")
    print(f"--- Fetching in batches of {BATCH_LIMIT} rows ---")

    # Use a 'with' statement for SQLite connection to ensure it's always closed
    # Enable WAL mode for potentially better performance and robustness with writes
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("PRAGMA journal_mode=WAL;") # Enable Write-Ahead Logging

            while True:
                print(f"\n[Batch {loop_count + 1}] Fetching offset {offset}...")

                # 1. Payload for API request
                payload = {
                    "parameters": {
                        "OFFSET": str(offset),
                        "LIMIT": str(BATCH_LIMIT)
                    },
                    "run_config": {
                        "limit": BATCH_LIMIT + 1000  # API's internal limit
                    }
                }

                # 2. Start Run (API call)
                start_url = f"https://api.allium.so/api/v1/explorer/queries/{QUERY_ID}/run-async"
                try:
                    response = requests.post(start_url, headers=HEADERS, json=payload)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    run_id = response.json()['run_id']
                except requests.exceptions.RequestException as e:
                    print(f"Error starting batch API request: {e}")
                    break

                # 3. Poll Status
                status_url = f"https://api.allium.so/api/v1/explorer/query-runs/{run_id}/status"
                timeout_start = time.time()
                API_POLL_TIMEOUT = 300 # 5 minutes timeout for status polling
                while True:
                    if time.time() - timeout_start > API_POLL_TIMEOUT:
                        print(f"Polling timed out after {API_POLL_TIMEOUT} seconds for run ID: {run_id}")
                        break # Break from status polling, effectively skipping this batch

                    status_resp = requests.get(status_url, headers=HEADERS)
                    status_data = status_resp.json()
                    status = status_data if isinstance(status_data, str) else status_data.get('status')

                    if status == 'success':
                        break
                    elif status == 'failed':
                        print(f"Batch query failed for run ID: {run_id}")
                        return # Exit the function on query failure
                    time.sleep(1)

                if status != 'success': # If polling timed out or failed
                    print("Skipping current batch due to API query status issue.")
                    offset += BATCH_LIMIT # Still increment offset to try next batch
                    loop_count += 1
                    continue

                # 4. Get Results
                results_url = f"https://api.allium.so/api/v1/explorer/query-runs/{run_id}/results"
                try:
                    results_resp = requests.get(results_url, headers=HEADERS)
                    results_resp.raise_for_status()
                    data = results_resp.json().get('data', [])
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching results for run ID {run_id}: {e}")
                    offset += BATCH_LIMIT # Increment offset to try next batch
                    loop_count += 1
                    continue # Continue to next batch

                row_count = len(data)
                print(f"   -> Received {row_count} rows.")

                if row_count == 0:
                    print("   -> No more data returned. Pagination complete.")
                    break # Exit the while loop

                # 5. Process Data with Pandas
                df = pd.DataFrame(data)

                # --- NEW STEP: Parse the JSON 'params' column ---
                df = process_params_column(df)

                # --- Clean types ---
                df = clean_and_type_convert(df)

                if df.empty:
                    print("   -> DataFrame became empty after processing/cleaning. Skipping save.")
                    offset += BATCH_LIMIT
                    loop_count += 1
                    continue

                # 6. Store to SQLite
                # 'replace' for the first batch to ensure a fresh table schema, 'append' for subsequent
                mode = 'replace' if loop_count == 0 else 'append'

                try:
                    df.to_sql(TABLE_NAME, conn, if_exists=mode, index=False)
                    total_rows_stored += len(df)
                    print(f"   -> Appended {len(df)} rows to DB. Total stored: {total_rows_stored}")
                except sqlite3.Error as e:
                    print(f"   -> Error saving to SQLite: {e}. Check disk space and file permissions.")
                    # Depending on the error, you might want to break or continue
                    # For a persistent disk I/O error, breaking is often necessary.
                    break
                except Exception as e:
                    print(f"   -> An unexpected error occurred while saving to SQLite: {e}")
                    break

                # Update loop variables
                offset += BATCH_LIMIT
                loop_count += 1

    except sqlite3.Error as e:
        print(f"A critical SQLite error occurred outside the batch loop: {e}")
        print("Please check your database file, disk space, and permissions.")
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")

    print(f"\nDone. Final total rows stored: {total_rows_stored}")

if __name__ == "__main__":
    run_pagination_loop()
#!/usr/bin/env python3
"""
Allium API Data Fetcher - Direct SQL Queries

Uses direct SQL queries instead of saved queries to avoid permission issues.
This approach works with the same API key used in the notebook.

Usage:
    python allium_get_data_direct_sql.py [dlps|2025]

Environment Variables:
    ALLIUM_API_KEY: Your Allium API key (required)
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import sqlite3
import json
from typing import Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("ALLIUM_API_KEY")
PROOF_ROOT_CONTRACT = '0x8C8788f98385F6ba1adD4234e551ABba0f82Cb7C'
DB_NAME = "allium_data.db"
TABLE_NAME = "query_results"
BATCH_LIMIT = 200000
API_POLL_TIMEOUT = 600  # 10 minutes

if not API_KEY:
    raise ValueError("ALLIUM_API_KEY environment variable is required")

HEADERS = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

BASE_URL = "https://api.allium.so/api/v1/explorer"


class AlliumAPIError(Exception):
    """Custom exception for Allium API errors"""
    pass


def process_params_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'params' column (JSON string or dict) and flatten into separate columns.
    
    Args:
        df: DataFrame with 'params' column
        
    Returns:
        DataFrame with flattened params
    """
    if 'params' not in df.columns:
        return df

    def parse_json(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    expanded_data = df['params'].apply(parse_json).apply(pd.Series)
    df = pd.concat([df, expanded_data], axis=1)
    df = df.drop(columns=['params', 'params_keys_mapping'], errors='ignore')
    return df


def clean_and_type_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean string artifacts, infer correct data types, and handle specific column types.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame with proper types
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip('"')

        if 'timestamp' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            temp_series = pd.to_numeric(df[col], errors='coerce')
            if not temp_series.isna().all():
                try:
                    df[col] = temp_series.astype(pd.Int64Dtype())
                except (OverflowError, ValueError):
                    df[col] = temp_series
    return df


def submit_query(sql_query: str) -> Optional[str]:
    """
    Submit SQL query to Allium API and return query ID.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        Query ID if successful, None otherwise
    """
    try:
        response = requests.post(
            f"{BASE_URL}/queries",
            headers=HEADERS,
            json={'sql': sql_query},
            timeout=30
        )
        response.raise_for_status()
        query_id = response.json().get('id')
        
        if not query_id:
            logger.error(f"No query ID returned. Response: {response.text}")
            return None
            
        return query_id
    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting query: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise AlliumAPIError(f"Failed to submit query: {e}")


def poll_query_results(query_id: str) -> Optional[list]:
    """
    Poll Allium API for query results.
    
    Args:
        query_id: Query ID to poll for
        
    Returns:
        List of results if available, None otherwise
    """
    timeout_start = time.time()
    
    while time.time() - timeout_start < API_POLL_TIMEOUT:
        try:
            response = requests.get(
                f"{BASE_URL}/queries/{query_id}/results",
                headers=HEADERS,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json().get('data')
                if data:
                    logger.info(f"Received {len(data):,} rows")
                    return data
                    
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error polling: {e}")
        
        time.sleep(2)
        
        elapsed = int(time.time() - timeout_start)
        if elapsed % 30 == 0:
            logger.info(f"Still waiting... ({elapsed}s elapsed)")
    
    logger.error(f"Polling timed out after {API_POLL_TIMEOUT} seconds")
    return None


def execute_sql_query_with_pagination(
    sql_query: str,
    query_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> int:
    """
    Execute SQL query with pagination using direct SQL.
    
    Args:
        sql_query: SQL query string with OFFSET_PLACEHOLDER and LIMIT_PLACEHOLDER
        query_name: Name for logging
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        Total number of rows stored
    """
    offset = 0
    loop_count = 0
    total_rows_stored = 0
    
    logger.info(f"{query_name}")
    logger.info(f"Fetching in batches of {BATCH_LIMIT:,} rows...")
    
    # Add date filters if provided
    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append(f"block_timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"block_timestamp < '{end_date}'")
        
        if conditions:
            date_filter = "WHERE " + " AND ".join(conditions)
            if "ORDER BY" in sql_query.upper():
                sql_query = sql_query.replace("ORDER BY", f"{date_filter} ORDER BY")
            else:
                sql_query += f" {date_filter}"
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            
            while True:
                logger.info(f"Batch {loop_count + 1}: Fetching offset {offset:,}...")
                
                # Replace placeholders in query
                paginated_query = sql_query.replace("OFFSET_PLACEHOLDER", str(offset))
                paginated_query = paginated_query.replace("LIMIT_PLACEHOLDER", str(BATCH_LIMIT))
                
                # Submit query
                query_id = submit_query(paginated_query)
                if not query_id:
                    break
                
                # Poll for results
                data = poll_query_results(query_id)
                if data is None:
                    break
                
                if not data:
                    logger.info("No more data. Pagination complete.")
                    break
                
                row_count = len(data)
                logger.info(f"Processing {row_count:,} rows...")
                
                # Process data
                df = pd.DataFrame(data)
                if df.empty:
                    logger.info("DataFrame is empty. Pagination complete.")
                    break
                
                df = process_params_column(df)
                df = clean_and_type_convert(df)
                
                if df.empty:
                    logger.warning("DataFrame became empty after processing. Skipping.")
                    offset += BATCH_LIMIT
                    loop_count += 1
                    continue
                
                # Store to SQLite
                mode = 'replace' if loop_count == 0 else 'append'
                df.to_sql(TABLE_NAME, conn, if_exists=mode, index=False)
                total_rows_stored += len(df)
                logger.info(f"Saved {len(df):,} rows to DB. Total: {total_rows_stored:,}")
                
                # Check if last batch
                if row_count < BATCH_LIMIT:
                    logger.info("Received fewer rows than requested. Pagination complete.")
                    break
                
                offset += BATCH_LIMIT
                loop_count += 1
                
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
    
    logger.info(f"Done. Total rows stored: {total_rows_stored:,}")
    return total_rows_stored


def fetch_complete_2025_proofs() -> int:
    """Fetch all 2025 proof data with pagination."""
    sql_query = """
    SELECT
        block_timestamp,
        transaction_hash,
        block_number,
        params
    FROM vana.decoded.logs
    WHERE address = LOWER('0x8C8788f98385F6ba1adD4234e551ABba0f82Cb7C')
      AND name = 'ProofAdded'
      AND block_timestamp >= '2025-01-01'
      AND block_timestamp < '2026-01-01'
    ORDER BY block_timestamp ASC NULLS LAST
    LIMIT LIMIT_PLACEHOLDER
    OFFSET OFFSET_PLACEHOLDER
    """
    
    return execute_sql_query_with_pagination(
        sql_query,
        "Complete 2025 Proof Data (Chronological)",
        start_date='2025-01-01',
        end_date='2026-01-01'
    )


def fetch_all_dlps_ranked() -> pd.DataFrame:
    """Fetch all DLPs ranked by submission count (aggregated query, no pagination)."""
    sql_query = """
    SELECT
        params['dlpId'] AS dlpid,
        COUNT(*) AS submission_count,
        COUNT(DISTINCT params['ownerAddress']) AS unique_owners,
        SUM(TRY_CAST(params['score'] AS DOUBLE) / 1e18) AS total_score,
        AVG(TRY_CAST(params['score'] AS DOUBLE) / 1e18) AS avg_score,
        MIN(block_timestamp) AS first_proof_date,
        MAX(block_timestamp) AS last_proof_date
    FROM vana.decoded.logs
    WHERE address = LOWER('0x8C8788f98385F6ba1adD4234e551ABba0f82Cb7C')
      AND name = 'ProofAdded'
    GROUP BY params['dlpId']
    ORDER BY submission_count DESC NULLS LAST
    """
    
    logger.info("Fetching ALL DLPs Ranked (Aggregated - No Pagination)")
    
    query_id = submit_query(sql_query)
    if not query_id:
        return pd.DataFrame()
    
    logger.info(f"Query ID: {query_id}")
    logger.info("Waiting for results...")
    
    # Poll for results
    for attempt in range(300):  # 10 minutes max
        time.sleep(2)
        
        try:
            response = requests.get(
                f"{BASE_URL}/queries/{query_id}/results",
                headers=HEADERS,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json().get('data')
                if data:
                    df = pd.DataFrame(data)
                    logger.info(f"Retrieved {len(df)} DLPs")
                    
                    # Save to CSV
                    os.makedirs('outputs/data', exist_ok=True)
                    output_path = 'outputs/data/allium_all_dlps_ranked.csv'
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved to {output_path}")
                    return df
                    
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error polling: {e}")
        
        if attempt % 15 == 0:
            logger.info(f"Still waiting... ({attempt * 2}s elapsed)")
    
    logger.error("Query timeout")
    return pd.DataFrame()


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("ALLIUM DATA FETCHER - Direct SQL Queries")
    logger.info("=" * 80)
    logger.info(f"API Key: {API_KEY[:10]}...")
    logger.info(f"Database: {DB_NAME}")
    logger.info(f"Table: {TABLE_NAME}")
    logger.info("=" * 80)
    
    os.makedirs('outputs/data', exist_ok=True)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "dlps":
            fetch_all_dlps_ranked()
        elif command == "2025":
            fetch_complete_2025_proofs()
        else:
            logger.error(f"Unknown command: {command}")
            print("Usage: python allium_get_data_direct_sql.py [dlps|2025]")
            print("  dlps  - Fetch all DLPs ranked (aggregated)")
            print("  2025  - Fetch complete 2025 proof data (paginated)")
            sys.exit(1)
    else:
        print("\nAvailable commands:")
        print("  python allium_get_data_direct_sql.py dlps   - Fetch all DLPs")
        print("  python allium_get_data_direct_sql.py 2025  - Fetch 2025 proofs")
        print("\nOr modify the script to call functions directly.")


if __name__ == "__main__":
    main()


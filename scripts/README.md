# Vana Year-in-Review Analytics Scripts

Production-ready Python scripts for analyzing Vana blockchain data.

## Overview

This directory contains cleaned, production-ready scripts for:
- Data fetching from Allium API and VanaScan API
- Proof of Contribution (POC) analytics
- DLP performance analysis
- Token transfer analysis
- Data flow analytics

## Scripts

### Data Fetching

- **`allium_get_data_direct_sql.py`** - Fetch data from Allium API using direct SQL queries
  - Usage: `python allium_get_data_direct_sql.py [dlps|2025]`
  - Requires: `ALLIUM_API_KEY` environment variable

- **`extract_github_from_imageurl.py`** - Extract GitHub repository information from proof data
  - Extracts GitHub URLs from `imageURL` fields in proof metadata
  - Trims URLs to base repository paths (removes /releases/ paths and file extensions)
  - Fetches repository metadata from GitHub API
  - Usage: `python extract_github_from_imageurl.py`
  - Requires: Proof data parquet file in `outputs/data/`

### Analytics

- **`run_poc_analytics.py`** - Comprehensive POC analytics from SQLite database
  - Analyzes DLP performance, contributor activity, temporal trends, and score distributions
  - Requires: `allium_data.db` SQLite database

### Utilities

- **`dlp_name_mapping.py`** - DLP ID to name mapping utility
  - Provides `get_dlp_name(dlp_id)` function
  - Source: Vana Analytics Dashboard

## Requirements

```bash
pip install pandas numpy requests python-dotenv sqlite3
```

## Environment Variables

- `ALLIUM_API_KEY` - Allium API key (required for data fetching)
- `VANA_SCAN` or `VANA_API_TOKEN` - VanaScan API key (optional)

## Data Sources

- **Allium API**: `https://api.allium.so/api/v1/explorer`
- **VanaScan API**: `https://vanascan.io/api`
- **SQLite Database**: `allium_data.db` (local storage)

## Output

All scripts output to `outputs/data/` directory:
- CSV files for analysis results
- JSON files for summary reports
- Parquet files for large datasets

## Code Quality

All scripts follow production standards:
- No emojis in output
- Proper error handling with specific exceptions
- Logging instead of print statements
- Type hints where appropriate
- Clean, maintainable code structure


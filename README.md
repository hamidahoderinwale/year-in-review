# Vana Network Analytics

This repository is a fork of [vana-com/year-in-review](https://github.com/vana-com/year-in-review.git) containing production-ready Python scripts for acquiring and analyzing data from the Vana blockchain network. The scripts focus on extracting Proof of Contribution (PoC) data, analyzing network activity, and generating comprehensive analytics reports.

## Repository Lineage

This repository is a fork of the original Vana year-in-review analytics repository. The scripts in this fork have been cleaned and optimized for production use:
- Removed emojis and improved logging
- Enhanced error handling
- Removed hardcoded API keys
- Added comprehensive documentation

## Overview

The Vana network is an EVM-compatible blockchain that enables user-owned data economies through Data Liquidity Pools (DLPs). Contributors submit data proofs that are verified and scored across multiple dimensions (authenticity, ownership, quality, completeness, uniqueness), with rewards distributed based on contribution metrics.

## Data Acquisition

### Primary Data Sources

The project uses two main APIs to acquire blockchain data:

1. **Allium API** - Real-time blockchain data streaming and SQL queries
   - Base URL: `https://api.allium.so/api/v1/explorer`
   - Provides direct SQL access to indexed blockchain data
   - Best for: Historical queries, event logs, proof data extraction
   - Authentication: API key via `ALLIUM_API_KEY` environment variable

2. **VanaScan API** - REST API for blockchain explorer data
   - Base URL: `https://vanascan.io/api`
   - Provides transaction, address, and contract data
   - Best for: Transaction history, token transfers, contract interactions
   - Authentication: API key via `VANA_SCAN` or `VANA_API_TOKEN` environment variable

### Data Storage

All acquired data is stored in a local SQLite database (`allium_data.db`) for efficient querying and analysis. The database schema includes:

- **Proof of Contribution Events**: `ProofAdded` events with scores, DLP IDs, contributor addresses
- **Transaction Data**: Block numbers, timestamps, transaction hashes
- **Token Transfers**: Transfer amounts, token addresses, sender/recipient addresses
- **Contract Interactions**: Function calls, event logs, contract addresses

## Scripts Overview

### Data Acquisition Scripts

#### `allium_get_data_direct_sql.py`

Fetches Proof of Contribution data from the Allium API using direct SQL queries. This script:

- Executes SQL queries against Allium's indexed blockchain data
- Handles pagination automatically for large result sets
- Stores results in SQLite database (`allium_data.db`)
- Processes and cleans data (type conversion, JSON parsing)

**Usage:**
```bash
# Fetch all 2025 proof data
python allium_get_data_direct_sql.py 2025

# Fetch aggregated DLP statistics
python allium_get_data_direct_sql.py dlps
```

**Key Features:**
- Automatic pagination (batches of 200,000 rows)
- Query result polling with timeout handling
- Data type inference and cleaning
- JSON parameter expansion

**Data Acquired:**
- Proof events with scores, DLP IDs, file IDs
- Contributor addresses and proof indices
- Block timestamps and transaction hashes
- IPFS proof URLs

#### `extract_github_from_imageurl.py`

Extracts GitHub repository information from proof data by parsing `imageURL` fields. This script:

- Extracts GitHub URLs from proof metadata (either directly from data or from IPFS content)
- Trims GitHub URLs to base repository URLs (removes `/releases/` paths and file extensions)
- Extracts repository owner, name, and full repository path
- Fetches additional repository metadata via GitHub API (stars, forks, languages, releases)

**Usage:**
```bash
python extract_github_from_imageurl.py
```

**Key Features:**
- Smart URL trimming: Removes `/releases/` paths and file extensions to get base repo URLs
- IPFS content fetching: Can extract imageURL from IPFS proof metadata if not in direct data
- GitHub API integration: Fetches repository metadata (description, stars, forks, languages, releases)
- DLP mapping: Links GitHub repositories to Data Liquidity Pools

**Data Extracted:**
- GitHub repository URLs (trimmed to base repo)
- Repository owner and name
- Full repository path (owner/repo)
- Repository metadata (stars, forks, languages, release counts)
- DLP associations

**Output Files:**
- `github_repos_from_imageurl.csv` - GitHub repository URLs and basic info
- `github_repo_metadata_from_imageurl.json` - Full repository metadata from GitHub API
- `github_repo_summary_from_imageurl.csv` - Summary table with key metrics

### Analytics Scripts

#### `run_poc_analytics.py`

Performs comprehensive analytics on Proof of Contribution data from the SQLite database. This script:

- Analyzes DLP performance metrics
- Examines contributor activity patterns
- Tracks temporal trends (daily/monthly)
- Calculates score distributions
- Generates summary reports

**Usage:**
```bash
python run_poc_analytics.py
```

**Analytics Generated:**
- **DLP Performance**: Proof counts, contributor counts, total/average scores per DLP
- **Contributor Activity**: Proof counts, DLP participation, score accumulation
- **Temporal Trends**: Daily and monthly proof submission patterns
- **Score Distribution**: Score ranges, percentiles, quality metrics

**Output Files:**
- `dlp_performance_analysis.csv` - DLP-level statistics
- `contributor_activity_analysis.csv` - Contributor-level statistics
- `daily_trends.csv` - Daily aggregation metrics
- `monthly_trends.csv` - Monthly aggregation metrics
- `score_distribution.csv` - Score bin distribution
- `poc_analytics_summary.json` - Comprehensive summary report

#### `comprehensive_data_flow_analytics.py`

Analyzes complete data flows throughout the platform using existing extracted datasets. This script:

- Tracks reward distribution flows (proofs → scores → payouts)
- Analyzes contributor balance changes
- Examines token transfer patterns
- Calculates network-level metrics

**Usage:**
```bash
python comprehensive_data_flow_analytics.py
```

**Data Flows Analyzed:**
1. **Reward Distribution Flow**: Complete path from proof submission to reward payout
2. **Contributor Balance Flow**: Net balance changes, received/sent amounts
3. **Token Transfer Flow**: Transfer patterns, token types, amounts
4. **Network Metrics**: Overall activity patterns and distributions

**Output Files:**
- `contributor_transfer_statistics.csv` - Transfer statistics per contributor
- `contributor_token_transfers_comprehensive.csv` - Detailed transfer data
- `contributor_balance_snapshot.csv` - Balance change analysis

### Data Enrichment Scripts

#### `extract_github_from_imageurl.py`

Extracts GitHub repository information from proof data. This script enriches proof data by:

- Parsing `imageURL` fields from proof metadata
- Extracting and normalizing GitHub repository URLs
- Fetching repository metadata from GitHub API
- Mapping repositories to DLPs

**Usage:**
```bash
python extract_github_from_imageurl.py
```

**Process:**
1. Loads proof data from parquet files or database
2. Extracts `imageURL` field (either directly or from IPFS content)
3. Trims GitHub URLs to base repository URLs
4. Extracts owner/repo information
5. Fetches metadata from GitHub API (optional)
6. Generates summary reports

**Output Files:**
- `github_repos_from_imageurl.csv` - Extracted GitHub URLs with DLP associations
- `github_repo_metadata_from_imageurl.json` - Full repository metadata
- `github_repo_summary_from_imageurl.csv` - Summary statistics

### Utility Scripts

#### `dlp_name_mapping.py`

Provides a mapping from DLP IDs to human-readable names. Used by analytics scripts to enrich data with DLP names.

**Usage:**
```python
from dlp_name_mapping import get_dlp_name

dlp_name = get_dlp_name(10)  # Returns "YKYR"
```

## Data Flow

```
┌─────────────────┐
│  Allium API     │
│  (SQL Queries)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQLite Database│
│  (allium_data.db)│
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│ POC Analytics   │  │ Data Flow       │
│ Script          │  │ Analytics       │
└────────┬────────┘  └────────┬────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│ CSV/JSON Reports│  │ Flow Analysis   │
│                 │  │ Reports         │
└─────────────────┘  └─────────────────┘
```

## Setup

### Prerequisites

- Python 3.8+
- SQLite3
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd vana-year
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API keys:**

Create a `.env` file in the project root:
```bash
ALLIUM_API_KEY=your_allium_api_key_here
VANA_SCAN=your_vanascan_api_key_here
```

Or set environment variables:
```bash
export ALLIUM_API_KEY="your_allium_api_key_here"
export VANA_SCAN="your_vanascan_api_key_here"
```

### Getting API Keys

- **Allium API**: Contact Allium support or check your Allium dashboard
- **VanaScan API**: Get from [VanaScan API Documentation](https://vanascan.io/api-docs)

## Usage Workflow

### Step 1: Acquire Data

Fetch Proof of Contribution data from Allium API:

```bash
python scripts/allium_get_data_direct_sql.py 2025
```

This will:
- Query Allium API for all 2025 proof events
- Store data in `allium_data.db`
- Process and clean the data automatically

### Step 1.5: Extract GitHub URLs (Optional)

Extract GitHub repository information from proof metadata:

```bash
python extract_github_from_imageurl.py
```

This will:
- Extract GitHub URLs from `imageURL` fields in proof data
- Trim URLs to base repository paths
- Fetch repository metadata from GitHub API
- Generate GitHub repository summary reports

### Step 2: Run Analytics

Analyze the acquired data:

```bash
python scripts/run_poc_analytics.py
```

This generates:
- DLP performance metrics
- Contributor activity statistics
- Temporal trend analysis
- Score distribution reports

### Step 3: Analyze Data Flows

Examine complete data flows (requires token transfer data):

```bash
python scripts/comprehensive_data_flow_analytics.py
```

## Output Structure

All outputs are saved to the `outputs/` directory:

```
outputs/
├── data/
│   ├── dlp_performance_analysis.csv
│   ├── contributor_activity_analysis.csv
│   ├── daily_trends.csv
│   ├── monthly_trends.csv
│   ├── score_distribution.csv
│   ├── poc_analytics_summary.json
│   ├── contributor_transfer_statistics.csv
│   ├── contributor_balance_snapshot.csv
│   ├── github_repos_from_imageurl.csv
│   ├── github_repo_metadata_from_imageurl.json
│   └── github_repo_summary_from_imageurl.csv
└── (chart outputs excluded)
```

## Key Metrics Analyzed

### Proof of Contribution Metrics

- **Total Proofs**: Number of proof events submitted
- **Unique Contributors**: Number of unique wallet addresses
- **DLP Distribution**: Proof counts per Data Liquidity Pool
- **Score Distribution**: Quality score ranges and percentiles
- **Temporal Patterns**: Daily/monthly submission trends

### Contributor Metrics

- **Activity Levels**: Proof counts per contributor
- **DLP Participation**: Number of DLPs each contributor participates in
- **Score Accumulation**: Total and average scores per contributor
- **Retention**: Repeat submission patterns

### Network Metrics

- **Reward Distribution**: Token transfer patterns
- **Balance Flows**: Net balance changes per contributor
- **Transfer Patterns**: Token movement analysis
- **Network Health**: Overall activity indicators

### GitHub Repository Analysis

- **Repository Extraction**: GitHub URLs extracted from proof metadata
- **DLP-Repository Mapping**: Links between Data Liquidity Pools and GitHub repositories
- **Repository Metadata**: Stars, forks, languages, release information
- **Code Analysis**: Repository activity and contribution patterns

## Data Schema

### Proof of Contribution Events

- `block_number`: Block number where proof was added
- `block_timestamp`: Unix timestamp of the block
- `transaction_hash`: Transaction hash
- `dlpId`: Data Liquidity Pool ID
- `fileId`: Unique file identifier
- `ownerAddress`: Contributor wallet address
- `proofIndex`: Index of the proof
- `proofUrl`: IPFS URL for proof metadata
- `score`: Proof quality score (0.0-1.0, stored as wei, divided by 1e18)

### Token Transfers

- `transaction_hash`: Transaction hash
- `timestamp`: Transfer timestamp
- `token_address`: Token contract address
- `from_address`: Sender address
- `to_address`: Recipient address
- `amount_normalized`: Transfer amount (normalized by decimals)
- `token_symbol`: Token symbol
- `dlp_id`: Associated DLP ID (if mapped)

## Technical Details

### Data Processing

- **Type Conversion**: Automatic inference of data types (integers, floats, timestamps)
- **JSON Parsing**: Expansion of nested JSON parameters into columns
- **Pagination**: Automatic handling of large result sets
- **Error Handling**: Robust error handling with retry logic

### Performance

- **Batch Processing**: Data processed in batches of 200,000 rows
- **Connection Pooling**: Efficient API request handling
- **Caching**: Query results cached in SQLite for fast subsequent access
- **Parallel Processing**: Support for concurrent operations where applicable

## Documentation

- **Vana Documentation**: https://docs.vana.org
- **Allium Documentation**: https://docs.allium.so
- **VanaScan API Docs**: https://vanascan.io/api-docs
- **Proof of Contribution**: https://docs.vana.org/docs/proof-of-contribution-1
- **DLP Rewards**: https://docs.vana.org/docs/dlp-rewards-dlp-performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License is a permissive open-source license that allows for:
- Commercial use
- Modification
- Distribution
- Private use

The only requirement is that the license and copyright notice be included in copies of the software.

#!/usr/bin/env python3
"""
Comprehensive POC Analytics - Using Current Data

Run analytics on existing database without waiting for IPFS fetching.
Analyzes proof of contribution data from SQLite database.
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DatabaseNotFoundError(Exception):
    """Raised when database file cannot be found"""
    pass


def find_database() -> str:
    """
    Find the SQLite database file.
    
    Returns:
        Path to database file
        
    Raises:
        DatabaseNotFoundError: If database not found
    """
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    db_paths = [
        base_dir / 'allium_data.db',
        Path('allium_data.db'),
        Path.cwd() / 'allium_data.db',
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            logger.info(f"Found database: {db_path}")
            return str(db_path)
    
    raise DatabaseNotFoundError(
        "Database not found. Searched in: " + ", ".join(str(p) for p in db_paths)
    )


def load_data() -> pd.DataFrame:
    """
    Load POC data from database.
    
    Returns:
        DataFrame with POC data
    """
    db_path = find_database()
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            block_number,
            block_timestamp,
            dlpId as dlpid,
            fileId as fileid,
            ownerAddress as owneraddress,
            proofIndex as proofindex,
            proofUrl as proofurl,
            CAST(score AS REAL) / 1e18 as score
        FROM query_results
        WHERE score IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamp
    if 'block_timestamp' in df.columns:
        df['block_timestamp'] = pd.to_datetime(df['block_timestamp'], unit='s', errors='coerce')
        df['date'] = df['block_timestamp'].dt.date
    
    logger.info(f"Loaded {len(df):,} records from database")
    return df


def analyze_dlp_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze DLP performance metrics.
    
    Args:
        df: DataFrame with POC data
        
    Returns:
        DataFrame with DLP statistics
    """
    logger.info("Analyzing DLP performance...")
    
    dlp_stats = df.groupby('dlpid').agg({
        'proofindex': 'count',
        'owneraddress': 'nunique',
        'fileid': 'nunique',
        'score': ['sum', 'mean', 'std', 'min', 'max']
    }).round(6)
    
    dlp_stats.columns = [
        'proof_count', 'contributors', 'files', 
        'total_score', 'avg_score', 'std_score', 
        'min_score', 'max_score'
    ]
    dlp_stats = dlp_stats.sort_values('total_score', ascending=False)
    
    logger.info(f"Top 10 DLPs by total score:")
    for idx, row in dlp_stats.head(10).iterrows():
        logger.info(f"  DLP {idx}: {row['total_score']:.2f} total score, "
                   f"{row['proof_count']:,} proofs, {row['contributors']:,} contributors")
    
    output_file = OUTPUT_DIR / 'dlp_performance_analysis.csv'
    dlp_stats.to_csv(output_file)
    logger.info(f"Saved to: {output_file}")
    
    return dlp_stats


def analyze_contributor_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze contributor activity patterns.
    
    Args:
        df: DataFrame with POC data
        
    Returns:
        DataFrame with contributor statistics
    """
    logger.info("Analyzing contributor activity...")
    
    contributor_stats = df.groupby('owneraddress').agg({
        'proofindex': 'count',
        'dlpid': 'nunique',
        'fileid': 'nunique',
        'score': ['sum', 'mean']
    }).round(6)
    
    contributor_stats.columns = [
        'proof_count', 'dlps_contributed', 'files', 
        'total_score', 'avg_score'
    ]
    contributor_stats = contributor_stats.sort_values('total_score', ascending=False)
    
    logger.info(f"Total contributors: {len(contributor_stats):,}")
    logger.info(f"Contributors with 1 proof: {(contributor_stats['proof_count'] == 1).sum():,}")
    logger.info(f"Contributors with 10+ proofs: {(contributor_stats['proof_count'] >= 10).sum():,}")
    logger.info(f"Contributors with 100+ proofs: {(contributor_stats['proof_count'] >= 100).sum():,}")
    logger.info(f"Contributors with 1000+ proofs: {(contributor_stats['proof_count'] >= 1000).sum():,}")
    
    output_file = OUTPUT_DIR / 'contributor_activity_analysis.csv'
    contributor_stats.to_csv(output_file)
    logger.info(f"Saved to: {output_file}")
    
    return contributor_stats


def analyze_temporal_trends(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Analyze temporal trends.
    
    Args:
        df: DataFrame with POC data
        
    Returns:
        Tuple of (daily_stats, monthly_stats) or None if no date information
    """
    logger.info("Analyzing temporal trends...")
    
    if 'date' not in df.columns:
        logger.warning("No date information available")
        return None
    
    daily_stats = df.groupby('date').agg({
        'proofindex': 'count',
        'owneraddress': 'nunique',
        'dlpid': 'nunique',
        'score': 'sum'
    })
    daily_stats.columns = ['proofs', 'contributors', 'dlps', 'daily_score']
    
    # Monthly aggregation
    df['year_month'] = df['block_timestamp'].dt.to_period('M')
    monthly_stats = df.groupby('year_month').agg({
        'proofindex': 'count',
        'owneraddress': 'nunique',
        'dlpid': 'nunique',
        'score': 'sum'
    })
    monthly_stats.columns = ['proofs', 'contributors', 'dlps', 'monthly_score']
    
    daily_file = OUTPUT_DIR / 'daily_trends.csv'
    monthly_file = OUTPUT_DIR / 'monthly_trends.csv'
    daily_stats.to_csv(daily_file)
    monthly_stats.to_csv(monthly_file)
    
    logger.info(f"Saved daily trends to: {daily_file}")
    logger.info(f"Saved monthly trends to: {monthly_file}")
    
    return daily_stats, monthly_stats


def analyze_score_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Analyze score distribution.
    
    Args:
        df: DataFrame with POC data
        
    Returns:
        Series with score bin counts
    """
    logger.info("Analyzing score distribution...")
    
    logger.info(f"Total Score: {df['score'].sum():,.2f}")
    logger.info(f"Mean Score: {df['score'].mean():.6f}")
    logger.info(f"Median Score: {df['score'].median():.6f}")
    logger.info(f"Std Deviation: {df['score'].std():.6f}")
    logger.info(f"Min Score: {df['score'].min():.6f}")
    logger.info(f"Max Score: {df['score'].max():.6f}")
    
    # Score bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df['score_bin'] = pd.cut(df['score'], bins=bins, include_lowest=True)
    bin_counts = df['score_bin'].value_counts().sort_index()
    
    for bin_range, count in bin_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {bin_range}: {count:>10,} ({pct:>5.2f}%)")
    
    output_file = OUTPUT_DIR / 'score_distribution.csv'
    bin_counts.to_csv(output_file)
    logger.info(f"Saved to: {output_file}")
    
    return bin_counts


def generate_summary_report(
    df: pd.DataFrame,
    dlp_stats: pd.DataFrame,
    contributor_stats: pd.DataFrame
) -> dict:
    """
    Generate comprehensive summary report.
    
    Args:
        df: Full DataFrame
        dlp_stats: DLP statistics DataFrame
        contributor_stats: Contributor statistics DataFrame
        
    Returns:
        Dictionary with summary report
    """
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'data_summary': {
            'total_records': len(df),
            'unique_dlps': df['dlpid'].nunique(),
            'unique_contributors': df['owneraddress'].nunique(),
            'unique_files': df['fileid'].nunique(),
            'total_score': float(df['score'].sum()),
            'avg_score': float(df['score'].mean()),
        },
        'top_dlps': dlp_stats.head(10).to_dict('index'),
        'top_contributors': contributor_stats.head(10).to_dict('index'),
    }
    
    report_file = OUTPUT_DIR / 'poc_analytics_summary.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Summary report saved to: {report_file}")
    return report


def main():
    """Main analytics function."""
    logger.info("=" * 80)
    logger.info("POC DATA ANALYTICS - USING CURRENT DATA")
    logger.info("=" * 80)
    logger.info(f"Analysis started: {datetime.now()}")
    
    try:
        # Load data
        df = load_data()
        
        # Run analyses
        dlp_stats = analyze_dlp_performance(df)
        contributor_stats = analyze_contributor_activity(df)
        temporal = analyze_temporal_trends(df)
        score_dist = analyze_score_distribution(df)
        
        # Generate summary
        report = generate_summary_report(df, dlp_stats, contributor_stats)
        
        logger.info("=" * 80)
        logger.info("ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output files saved to: {OUTPUT_DIR}/")
        logger.info("  - dlp_performance_analysis.csv")
        logger.info("  - contributor_activity_analysis.csv")
        logger.info("  - daily_trends.csv")
        logger.info("  - monthly_trends.csv")
        logger.info("  - score_distribution.csv")
        logger.info("  - poc_analytics_summary.json")
        
    except DatabaseNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


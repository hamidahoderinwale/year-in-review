#!/usr/bin/env python3
"""
DLP Submission Distribution Analysis

Generate individual PNG charts with proper spacing and readability.
Creates 6 publication-ready charts analyzing DLP submission distributions.
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
ALLIUM_API_KEY = os.getenv('ALLIUM_API_KEY')
ALLIUM_BASE_URL = 'https://api.allium.so/api/v1/explorer'
PROOF_ROOT_CONTRACT = '0x8C8788f98385F6ba1adD4234e551ABba0f82Cb7C'

if not ALLIUM_API_KEY:
    raise ValueError("ALLIUM_API_KEY environment variable is required")

# Setup output directories
CHART_DIR = Path('outputs/charts/dlp_analysis')
DATA_DIR = Path('outputs/data')
CHART_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


class AlliumAPIError(Exception):
    """Custom exception for Allium API errors"""
    pass


def execute_allium_query(sql_query: str, query_name: str) -> pd.DataFrame:
    """
    Execute Allium query and return DataFrame.
    
    Args:
        sql_query: SQL query string
        query_name: Name for logging
        
    Returns:
        DataFrame with results
    """
    logger.info(f"{query_name}")
    
    try:
        response = requests.post(
            f"{ALLIUM_BASE_URL}/queries",
            headers={'X-API-Key': ALLIUM_API_KEY, 'Content-Type': 'application/json'},
            json={'sql': sql_query},
            timeout=30
        )
        response.raise_for_status()
        query_id = response.json().get('id')
        
        if not query_id:
            raise AlliumAPIError("No query ID returned")
        
        logger.info(f"Query ID: {query_id}")
        
        # Poll for results
        for attempt in range(150):  # 5 minutes max
            time.sleep(2)
            result = requests.get(
                f"{ALLIUM_BASE_URL}/queries/{query_id}/results",
                headers={'X-API-Key': ALLIUM_API_KEY},
                timeout=10
            )
            
            if result.status_code == 200:
                data = result.json().get('data')
                if data:
                    df = pd.DataFrame(data)
                    logger.info(f"Retrieved {len(df):,} records")
                    return df
            
            if attempt % 15 == 0:
                logger.info(f"Still waiting... ({attempt * 2}s elapsed)")
        
        logger.error("Query timeout")
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error executing query: {e}")
        raise AlliumAPIError(f"Failed to execute query: {e}")


def generate_chart_1_top_20_dlps(df_dlps: pd.DataFrame, total_submissions: int):
    """Generate Chart 1: Top 20 DLPs by Submission Count"""
    logger.info("=" * 80)
    logger.info("CHART 1: Top 20 DLPs by Submission Count")
    logger.info("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    top_20 = df_dlps.head(20).sort_values('submission_count')
    colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(top_20)))
    
    bars = ax.barh(range(len(top_20)), top_20['submission_count'], 
                   color=colors, alpha=0.85, edgecolor='darkgray', linewidth=0.5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_20.iterrows()):
        value = row['submission_count']
        ax.text(value + total_submissions*0.01, i, f'{value:,.0f}', 
               va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f'DLP "{dlp}"' for dlp in top_20['dlpId']], fontsize=10)
    ax.set_xlabel('Submission Count', fontweight='bold', fontsize=12)
    ax.set_title('Top 20 DLPs by Submission Count', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = CHART_DIR / '1_top_20_dlps_bar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_chart_2_market_share(df_dlps: pd.DataFrame, total_submissions: int):
    """Generate Chart 2: Market Share Distribution"""
    logger.info("=" * 80)
    logger.info("CHART 2: Market Share Distribution")
    logger.info("=" * 80)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_5 = df_dlps.head(5)
    top_5_total = top_5['submission_count'].sum()
    others_total = total_submissions - top_5_total
    
    labels = [f'DLP "{dlp}"' for dlp in top_5['dlpId']] + ['Others (All Other DLPs)']
    sizes = list(top_5['submission_count']) + [others_total]
    percentages = [(s/total_submissions*100) for s in sizes]
    
    colors_pie = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd', '#c8d6e5']
    explode = [0.05 if i < 3 else 0 for i in range(len(sizes))]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                      colors=colors_pie, explode=explode,
                                      startangle=90, textprops={'fontsize': 11})
    
    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Add legend with actual counts
    legend_labels = [f'{label}: {size:,} ({pct:.1f}%)' 
                     for label, size, pct in zip(labels, sizes, percentages)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    ax.set_title('Market Share Distribution\nTop 5 DLPs vs Others', 
                 fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    output_path = CHART_DIR / '2_market_share_pie.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_chart_3_log_scale(df_dlps: pd.DataFrame):
    """Generate Chart 3: Submission Distribution Across All DLPs (Log Scale)"""
    logger.info("=" * 80)
    logger.info("CHART 3: Submission Distribution Across All DLPs (Log Scale)")
    logger.info("=" * 80)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = range(len(df_dlps))
    colors_gradient = plt.cm.Blues_r(np.linspace(0.4, 0.9, len(df_dlps)))
    
    bars = ax.bar(x_pos, df_dlps['submission_count'], 
                  color=colors_gradient, alpha=0.85, edgecolor='navy', linewidth=0.3)
    
    # Add labels for top 3
    for i, (idx, row) in enumerate(df_dlps.head(3).iterrows()):
        ax.text(i, row['submission_count'] * 1.15, 
               f'DLP "{row["dlpId"]}"\n{row["submission_count"]:,}',
               ha='center', fontsize=9, fontweight='bold')
    
    ax.set_yscale('log')
    ax.set_xlabel('DLP Rank (by submission count)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Submission Count (log scale)', fontweight='bold', fontsize=12)
    ax.set_title('Submission Distribution Across All DLPs (Log Scale)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation for scale
    ax.text(0.98, 0.98, 'Log scale shows distribution\nacross all DLPs', 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = CHART_DIR / '3_all_dlps_log_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_chart_4_cumulative_share(df_dlps: pd.DataFrame):
    """Generate Chart 4: Cumulative Submission Share"""
    logger.info("=" * 80)
    logger.info("CHART 4: Cumulative Submission Share")
    logger.info("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = range(1, len(df_dlps) + 1)
    ax.plot(x_pos, df_dlps['cumulative_percentage'], 
            color='#e74c3c', linewidth=3, marker='o', markersize=5, 
            markerfacecolor='white', markeredgewidth=2, markeredgecolor='#e74c3c')
    
    # Add reference lines
    ax.axhline(y=50, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7, label='50% of submissions')
    ax.axhline(y=80, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label='80% of submissions')
    
    # Find where we cross 50% and 80%
    dlps_for_50 = (df_dlps['cumulative_percentage'] >= 50).idxmax() + 1
    dlps_for_80 = (df_dlps['cumulative_percentage'] >= 80).idxmax() + 1
    
    ax.axvline(x=dlps_for_50, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(x=dlps_for_80, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Annotations
    ax.text(dlps_for_50, 50, f'  {dlps_for_50} DLPs', 
            fontsize=10, fontweight='bold', va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.text(dlps_for_80, 80, f'  {dlps_for_80} DLPs', 
            fontsize=10, fontweight='bold', va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    ax.set_xlabel('Number of DLPs', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative % of Submissions', fontweight='bold', fontsize=12)
    ax.set_title('Cumulative Submission Share\nHow many DLPs account for X% of activity?', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.set_ylim([0, 105])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = CHART_DIR / '4_cumulative_share.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")
    
    return dlps_for_50, dlps_for_80


def generate_chart_5_distribution_by_range(df_dlps: pd.DataFrame):
    """Generate Chart 5: DLP Distribution by Submission Range"""
    logger.info("=" * 80)
    logger.info("CHART 5: DLP Distribution by Submission Range")
    logger.info("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define submission ranges
    bins = [0, 100, 1000, 10000, 100000, 1000000, float('inf')]
    labels = ['0-100', '100-1K', '1K-10K', '10K-100K', '100K-1M', '1M+']
    df_dlps['submission_range'] = pd.cut(df_dlps['submission_count'], 
                                          bins=bins, labels=labels, right=False)
    
    range_counts = df_dlps['submission_range'].value_counts().sort_index()
    
    colors_hist = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#e67e22']
    bars = ax.bar(range(len(range_counts)), range_counts.values, 
                  color=colors_hist[:len(range_counts)], alpha=0.85, 
                  edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (idx, value) in enumerate(range_counts.items()):
        ax.text(i, value + 0.3, str(value), 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(range_counts)))
    ax.set_xticklabels(range_counts.index, fontsize=11)
    ax.set_xlabel('Submission Count Range', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of DLPs', fontweight='bold', fontsize=12)
    ax.set_title('DLP Distribution by Submission Range\nHow many DLPs fall into each activity tier?', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = CHART_DIR / '5_distribution_by_range.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_chart_6_top_10_table(df_dlps: pd.DataFrame):
    """Generate Chart 6: Top 10 DLP Breakdown Table"""
    logger.info("=" * 80)
    logger.info("CHART 6: Top 10 DLP Breakdown Table")
    logger.info("=" * 80)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    top_10 = df_dlps.head(10).copy()
    top_10['Rank'] = range(1, 11)
    top_10['DLP ID'] = top_10['dlpId'].apply(lambda x: f'"{x}"')
    top_10['Submissions'] = top_10['submission_count'].apply(lambda x: f'{x:,}')
    top_10['% of Total'] = top_10['percentage'].apply(lambda x: f'{x:.2f}%')
    
    table_data = top_10[['Rank', 'DLP ID', 'Submissions', '% of Total']].values
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=['Rank', 'DLP ID', 'Submissions', '% of Total'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.25, 0.30, 0.30])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows with alternating colors
    for i in range(1, 11):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(color)
            if j == 0:  # Rank column
                cell.set_text_props(weight='bold')
    
    # Highlight top 3
    for i in range(1, 4):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor('#ffe5e5')
            cell.set_edgecolor('#e74c3c')
            cell.set_linewidth(2)
    
    ax.set_title('Top 10 DLP Breakdown\nDetailed Statistics', 
                 fontweight='bold', fontsize=16, pad=30)
    
    plt.tight_layout()
    output_path = CHART_DIR / '6_top_10_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("DLP SUBMISSION DISTRIBUTION ANALYSIS")
    logger.info("Individual Chart Generation with Enhanced Readability")
    logger.info("=" * 80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch DLP data
    query_dlp_comprehensive = f"""
    SELECT
        params['dlpId'] AS dlpId,
        COUNT(*) AS submission_count,
        COUNT(DISTINCT params['ownerAddress']) AS unique_contributors
    FROM
        vana.decoded.logs
    WHERE
        address = LOWER('{PROOF_ROOT_CONTRACT}')
        AND name = 'ProofAdded'
    GROUP BY
        1
    ORDER BY
        submission_count DESC NULLS LAST
    """
    
    df_dlps = execute_allium_query(query_dlp_comprehensive, "Fetching All DLP Data")
    
    if df_dlps.empty:
        logger.error("No data retrieved. Exiting.")
        sys.exit(1)
    
    # Clean data
    df_dlps['dlpId'] = df_dlps['dlpId'].astype(str).str.strip('"').astype(int)
    df_dlps['submission_count'] = df_dlps['submission_count'].astype(int)
    df_dlps['unique_contributors'] = df_dlps['unique_contributors'].astype(int)
    
    # Calculate percentages
    total_submissions = df_dlps['submission_count'].sum()
    df_dlps['percentage'] = (df_dlps['submission_count'] / total_submissions * 100).round(2)
    df_dlps['cumulative_percentage'] = df_dlps['percentage'].cumsum()
    
    logger.info(f"Total DLPs: {len(df_dlps)}")
    logger.info(f"Total Submissions: {total_submissions:,}")
    
    # Generate all charts
    generate_chart_1_top_20_dlps(df_dlps, total_submissions)
    generate_chart_2_market_share(df_dlps, total_submissions)
    generate_chart_3_log_scale(df_dlps)
    dlps_for_50, dlps_for_80 = generate_chart_4_cumulative_share(df_dlps)
    generate_chart_5_distribution_by_range(df_dlps)
    generate_chart_6_top_10_table(df_dlps)
    
    # Export data
    logger.info("=" * 80)
    logger.info("EXPORTING DATA")
    logger.info("=" * 80)
    
    analysis_output = DATA_DIR / 'dlp_comprehensive_analysis.csv'
    df_dlps.to_csv(analysis_output, index=False)
    logger.info(f"Saved: {analysis_output}")
    
    # Summary statistics
    summary = {
        'Total DLPs': len(df_dlps),
        'Total Submissions': total_submissions,
        'Top DLP': f'DLP {df_dlps.iloc[0]["dlpId"]}',
        'Top DLP Submissions': df_dlps.iloc[0]['submission_count'],
        'Top DLP Market Share': f'{df_dlps.iloc[0]["percentage"]:.2f}%',
        'DLPs for 50% share': dlps_for_50,
        'DLPs for 80% share': dlps_for_80,
        'Median Submissions': df_dlps['submission_count'].median(),
        'Mean Submissions': df_dlps['submission_count'].mean()
    }
    
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['Value']
    summary_output = DATA_DIR / 'dlp_summary_statistics.csv'
    summary_df.to_csv(summary_output)
    logger.info(f"Saved: {summary_output}")
    
    logger.info("=" * 80)
    logger.info("ALL CHARTS GENERATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Generated 6 individual charts:")
    logger.info("  1. Top 20 DLPs Bar Chart")
    logger.info("  2. Market Share Pie Chart")
    logger.info("  3. All DLPs Log Scale Distribution")
    logger.info("  4. Cumulative Share Line Chart")
    logger.info("  5. Distribution by Range Histogram")
    logger.info("  6. Top 10 Table")
    logger.info(f"Location: {CHART_DIR}/")
    logger.info(f"Data: {DATA_DIR}/dlp_*.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


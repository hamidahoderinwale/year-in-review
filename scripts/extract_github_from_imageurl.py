#!/usr/bin/env python3
"""
Extract GitHub Repository Information from imageURL

Fetches imageURL from IPFS content (if stored there), trims GitHub URLs correctly
(removes everything after file extensions or before /releases/), and extracts
repository owner, name, and metadata.
"""

import os
import sys
import time
import json
import re
import logging
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_API_BASE = "https://api.github.com"

IPFS_GATEWAYS = [
    'https://ipfs.io/ipfs/',
    'https://cloudflare-ipfs.com/ipfs/',
    'https://dweb.link/ipfs/',
    'https://gateway.ipfs.io/ipfs/',
]


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors"""
    pass


def trim_github_url(url: str) -> Optional[str]:
    """
    Trim GitHub URL to base repository URL.
    
    Removes everything after file extensions (.py, .tar.gz, etc.) 
    or everything before /releases/ to get base repo URL.
    
    Examples:
    - "https://github.com/dfusionai/private-social-lens-satya-proof-py/releases/download/v23/psl-proof-23.tar.gz"
      -> "https://github.com/dfusionai/private-social-lens-satya-proof-py"
    
    - "https://github.com/owner/repo/blob/main/file.py"
      -> "https://github.com/owner/repo"
    
    - "https://github.com/owner/repo/tree/v1.2.0"
      -> "https://github.com/owner/repo"
    
    Args:
        url: GitHub URL to trim
        
    Returns:
        Trimmed base repository URL or None if invalid
    """
    if not url or pd.isna(url):
        return None
    
    url_str = str(url).strip()
    
    # Check if it's a GitHub URL
    if 'github.com' not in url_str.lower():
        return None
    
    # Pattern: https://github.com/owner/repo/...
    github_pattern = r'github\.com/([^/]+)/([^/]+)'
    match = re.search(github_pattern, url_str, re.IGNORECASE)
    
    if not match:
        return None
    
    owner = match.group(1)
    repo = match.group(2)
    
    # If URL contains /releases/, trim everything after /releases/
    if '/releases/' in url_str:
        releases_idx = url_str.find('/releases/')
        # Base URL is everything before /releases/
        base_url = url_str[:releases_idx]
        # Extract just the repo part (owner/repo)
        repo_match = re.search(r'github\.com/([^/]+)/([^/]+)', base_url, re.IGNORECASE)
        if repo_match:
            return f"https://github.com/{repo_match.group(1)}/{repo_match.group(2)}"
        return f"https://github.com/{owner}/{repo}"
    
    # If URL contains file extensions in path (e.g., /blob/main/file.py or /tree/branch/file.tar.gz)
    # Extract base repo URL
    base_url = f"https://github.com/{owner}/{repo}"
    
    # Remove any path after the repo name
    repo_end_pattern = r'(github\.com/[^/]+/[^/]+)'
    repo_match = re.search(repo_end_pattern, url_str, re.IGNORECASE)
    if repo_match:
        return f"https://{repo_match.group(1)}"
    
    return base_url


def extract_github_repo_info(url: str) -> Optional[Dict]:
    """
    Extract GitHub repository information from URL.
    
    Args:
        url: GitHub URL
        
    Returns:
        Dictionary with owner, repo, and full repo path, or None if invalid
    """
    trimmed_url = trim_github_url(url)
    if not trimmed_url:
        return None
    
    # Pattern: https://github.com/owner/repo
    github_pattern = r'github\.com/([^/]+)/([^/]+)'
    match = re.search(github_pattern, trimmed_url, re.IGNORECASE)
    
    if not match:
        return None
    
    owner = match.group(1)
    repo = match.group(2).rstrip('/')
    
    return {
        'original_url': url,
        'trimmed_url': trimmed_url,
        'owner': owner,
        'repo': repo,
        'full_repo': f"{owner}/{repo}",
    }


def fetch_ipfs_content(ipfs_hash: str, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch content from IPFS hash.
    
    Args:
        ipfs_hash: IPFS hash
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with 'type' and 'data' keys, or None if fetch fails
    """
    hash_clean = str(ipfs_hash).strip().replace('"', '').replace("'", "")
    
    for gateway in IPFS_GATEWAYS:
        try:
            url = f"{gateway}{hash_clean}"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    data = response.json()
                    return {'type': 'json', 'data': data}
                except (json.JSONDecodeError, ValueError):
                    # If not JSON, return text
                    return {'type': 'text', 'data': response.text}
        except requests.exceptions.RequestException:
            continue
    return None


def extract_imageurl_from_ipfs(ipfs_content: Dict) -> Optional[str]:
    """
    Extract imageURL from IPFS content.
    
    Args:
        ipfs_content: IPFS content dictionary
        
    Returns:
        imageURL string or None if not found
    """
    if not ipfs_content:
        return None
    
    if ipfs_content.get('type') == 'json':
        data = ipfs_content.get('data', {})
        # Check various possible field names
        for field in ['imageURL', 'image_url', 'imageUrl', 'image']:
            if field in data:
                return data[field]
    
    elif ipfs_content.get('type') == 'text':
        text = ipfs_content.get('data', '')
        # Try to find imageURL in text (could be JSON string)
        try:
            json_data = json.loads(text)
            for field in ['imageURL', 'image_url', 'imageUrl', 'image']:
                if field in json_data:
                    return json_data[field]
        except (json.JSONDecodeError, ValueError):
            # Try regex search for GitHub URLs
            github_match = re.search(r'https?://github\.com/[^\s"\'<>]+', text)
            if github_match:
                return github_match.group(0)
    
    return None


def fetch_github_repo_info(owner: str, repo: str) -> Optional[Dict]:
    """
    Fetch GitHub repository information via API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        Dictionary with repository metadata or error information
    """
    try:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'description': data.get('description'),
                'stars': data.get('stargazers_count', 0),
                'forks': data.get('forks_count', 0),
                'watchers': data.get('watchers_count', 0),
                'language': data.get('language'),
                'languages_url': data.get('languages_url'),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'pushed_at': data.get('pushed_at'),
                'default_branch': data.get('default_branch'),
                'open_issues': data.get('open_issues_count', 0),
                'license': data.get('license', {}).get('name') if data.get('license') else None,
                'topics': data.get('topics', []),
                'archived': data.get('archived', False),
                'private': data.get('private', False),
                'size': data.get('size', 0),
                'has_wiki': data.get('has_wiki', False),
                'has_pages': data.get('has_pages', False),
            }
        elif response.status_code == 404:
            return {'error': 'Repository not found'}
        else:
            return {'error': f'API error: {response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def fetch_repo_languages(owner: str, repo: str) -> Optional[Dict]:
    """
    Fetch repository language statistics.
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        Dictionary mapping languages to byte counts, or None if fetch fails
    """
    try:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/languages"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def fetch_repo_releases(owner: str, repo: str, max_releases: int = 10) -> List[Dict]:
    """
    Fetch repository releases.
    
    Args:
        owner: Repository owner
        repo: Repository name
        max_releases: Maximum number of releases to fetch
        
    Returns:
        List of release dictionaries
    """
    try:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases"
        response = requests.get(url, params={'per_page': max_releases}, timeout=10)
        if response.status_code == 200:
            releases = response.json()
            return [
                {
                    'tag_name': r.get('tag_name'),
                    'name': r.get('name'),
                    'published_at': r.get('published_at'),
                    'prerelease': r.get('prerelease', False),
                    'draft': r.get('draft', False),
                }
                for r in releases
            ]
    except requests.exceptions.RequestException:
        pass
    return []


def analyze_github_from_imageurl() -> Tuple[Optional[pd.DataFrame], List[Dict]]:
    """
    Analyze GitHub repositories from imageURL in proof data.
    
    Returns:
        Tuple of (GitHub URLs DataFrame, repository metadata list)
    """
    logger.info("=" * 80)
    logger.info("EXTRACTING GITHUB REPOSITORY INFO FROM imageURL")
    logger.info("=" * 80)
    
    # Load proof data - try multiple possible paths
    script_dir = Path(__file__).parent
    possible_paths = [
        Path('outputs/data/allium_proof_contributions_combined.parquet'),
        script_dir / 'outputs/data/allium_proof_contributions_combined.parquet',
        Path.cwd() / 'outputs/data/allium_proof_contributions_combined.parquet',
    ]
    
    filepath = None
    for path in possible_paths:
        if path.exists():
            filepath = path
            break
    
    if not filepath:
        logger.error("File not found. Tried:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        return None, []
    
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df):,} proofs")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check if imageURL column exists directly
    if 'imageURL' in df.columns or 'image_url' in df.columns:
        image_col = 'imageURL' if 'imageURL' in df.columns else 'image_url'
        logger.info(f"Found {image_col} column directly in data")
        df[image_col] = df[image_col].astype(str).str.strip().str.replace('"', '', regex=False)
    else:
        # Need to fetch from IPFS
        logger.info("imageURL not found in columns, fetching from IPFS...")
        if 'proofurl' not in df.columns:
            logger.error("No proofurl column found to fetch IPFS content")
            return None, []
        
        # Sample a subset for testing (can be expanded)
        sample_size = min(1000, len(df))
        logger.info(f"Sampling {sample_size:,} proofs to extract imageURL from IPFS...")
        
        df_sample = df.head(sample_size).copy()
        image_urls = []
        
        for idx, row in df_sample.iterrows():
            if pd.notna(row.get('proofurl')):
                ipfs_content = fetch_ipfs_content(row['proofurl'])
                image_url = extract_imageurl_from_ipfs(ipfs_content)
                image_urls.append(image_url)
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1:,} proofs...")
                time.sleep(0.1)  # Rate limiting
            else:
                image_urls.append(None)
        
        df_sample['imageURL'] = image_urls
        df = df_sample
        image_col = 'imageURL'
        logger.info(f"Extracted imageURL from {df[image_col].notna().sum():,} proofs")
    
    # Extract GitHub repo info from imageURL
    logger.info(f"Extracting GitHub repository information from {image_col}...")
    github_repos = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(image_col)):
            repo_info = extract_github_repo_info(row[image_col])
            if repo_info:
                repo_info['dlp_id'] = row.get('dlpid', None)
                repo_info['score'] = row.get('score', None)
                repo_info['proof_index'] = idx
                github_repos.append(repo_info)
    
    logger.info(f"Found {len(github_repos):,} proofs with GitHub URLs in imageURL")
    
    if not github_repos:
        logger.warning("No GitHub URLs found in imageURL")
        return None, []
    
    # Create DataFrame
    github_df = pd.DataFrame(github_repos)
    
    # Get unique repositories
    unique_repos = github_df[['owner', 'repo', 'full_repo']].drop_duplicates()
    logger.info(f"Found {len(unique_repos):,} unique GitHub repositories")
    
    # Fetch GitHub API data for each unique repo
    logger.info("Fetching GitHub repository data...")
    repo_data_list = []
    
    for _, repo_row in unique_repos.iterrows():
        owner = repo_row['owner']
        repo = repo_row['repo']
        full_repo = repo_row['full_repo']
        
        logger.info(f"Fetching: {full_repo}")
        repo_info = fetch_github_repo_info(owner, repo)
        
        if repo_info and 'error' not in repo_info:
            repo_data = {
                'owner': owner,
                'repo': repo,
                'full_repo': full_repo,
                **repo_info
            }
            
            # Fetch languages
            languages = fetch_repo_languages(owner, repo)
            if languages:
                repo_data['languages'] = languages
                repo_data['primary_language'] = max(languages.items(), key=lambda x: x[1])[0] if languages else None
                repo_data['total_code_bytes'] = sum(languages.values())
            
            # Fetch releases
            releases = fetch_repo_releases(owner, repo)
            repo_data['releases_count'] = len(releases)
            repo_data['latest_release'] = releases[0]['tag_name'] if releases else None
            
            repo_data_list.append(repo_data)
        else:
            error_msg = repo_info.get('error', 'Unknown') if repo_info else 'Unknown'
            logger.warning(f"Error fetching {full_repo}: {error_msg}")
        
        time.sleep(0.5)  # Rate limiting for GitHub API
    
    # Save results
    if github_repos:
        github_urls_file = OUTPUT_DIR / 'github_repos_from_imageurl.csv'
        github_df.to_csv(github_urls_file, index=False)
        logger.info(f"GitHub URL extraction saved to: {github_urls_file}")
    
    if repo_data_list:
        repo_data_file = OUTPUT_DIR / 'github_repo_metadata_from_imageurl.json'
        with open(repo_data_file, 'w') as f:
            json.dump(repo_data_list, f, indent=2, default=str)
        logger.info(f"GitHub repository metadata saved to: {repo_data_file}")
        
        # Create summary table
        repo_summary_df = pd.DataFrame([
            {
                'full_repo': r['full_repo'],
                'description': r.get('description', ''),
                'language': r.get('language', 'N/A'),
                'primary_language': r.get('primary_language', 'N/A'),
                'stars': r.get('stars', 0),
                'forks': r.get('forks', 0),
                'releases_count': r.get('releases_count', 0),
                'latest_release': r.get('latest_release', 'N/A'),
                'created_at': r.get('created_at', ''),
                'updated_at': r.get('updated_at', ''),
                'topics': ', '.join(r.get('topics', [])),
            }
            for r in repo_data_list
        ])
        summary_file = OUTPUT_DIR / 'github_repo_summary_from_imageurl.csv'
        repo_summary_df.to_csv(summary_file, index=False)
        logger.info(f"Repository summary saved to: {summary_file}")
        
        # Print summary
        logger.info("GITHUB REPOSITORY SUMMARY:")
        logger.info("-" * 120)
        logger.info(f"{'Repository':<50} {'Language':<15} {'Stars':<10} {'Releases':<10} {'Latest Release':<20}")
        logger.info("-" * 120)
        for repo in repo_data_list[:20]:
            repo_name = repo['full_repo'][:48]
            lang = repo.get('language', 'N/A')[:13]
            stars = repo.get('stars', 0)
            releases = repo.get('releases_count', 0)
            latest = repo.get('latest_release', 'N/A')[:18]
            logger.info(f"{repo_name:<50} {lang:<15} {stars:<10} {releases:<10} {latest:<20}")
        
        # Analyze what data can be surfaced
        logger.info("DATA THAT CAN BE SURFACED FROM GITHUB REPOSITORIES:")
        logger.info("-" * 80)
        
        # Language distribution
        languages = [r.get('language') for r in repo_data_list if r.get('language')]
        if languages:
            lang_counts = pd.Series(languages).value_counts()
            logger.info("1. Language Distribution:")
            for lang, count in lang_counts.head(10).items():
                logger.info(f"   {lang}: {count} repositories")
        
        # Topics/keywords
        all_topics = []
        for r in repo_data_list:
            all_topics.extend(r.get('topics', []))
        if all_topics:
            topic_counts = pd.Series(all_topics).value_counts()
            logger.info("2. Common Topics/Keywords:")
            for topic, count in topic_counts.head(10).items():
                logger.info(f"   {topic}: {count} repositories")
        
        # Release activity
        repos_with_releases = [r for r in repo_data_list if r.get('releases_count', 0) > 0]
        logger.info("3. Release Activity:")
        logger.info(f"   Repositories with releases: {len(repos_with_releases)}/{len(repo_data_list)}")
        if repos_with_releases:
            avg_releases = sum(r.get('releases_count', 0) for r in repos_with_releases) / len(repos_with_releases)
            logger.info(f"   Average releases per repo: {avg_releases:.1f}")
        
        # Activity metrics
        logger.info("4. Activity Metrics:")
        total_stars = sum(r.get('stars', 0) for r in repo_data_list)
        total_forks = sum(r.get('forks', 0) for r in repo_data_list)
        logger.info(f"   Total stars: {total_stars:,}")
        logger.info(f"   Total forks: {total_forks:,}")
        logger.info(f"   Average stars per repo: {total_stars / len(repo_data_list):.1f}")
        logger.info(f"   Average forks per repo: {total_forks / len(repo_data_list):.1f}")
    
    return github_df, repo_data_list


def main():
    """Main execution function."""
    try:
        github_df, repo_data = analyze_github_from_imageurl()
        if github_df is not None:
            logger.info("GitHub extraction completed successfully")
        else:
            logger.error("GitHub extraction failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


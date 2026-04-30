"""
quota_tracker.py - API usage tracking and quota monitoring

Pure business logic for tracking:
- API call counts (YouTube + Gemini)
- Daily quota usage (persists to disk)
- Quota cost estimation

This module is Streamlit-agnostic and can be used independently.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List


# ============================================================================
# KEY FINGERPRINT HELPER
# ============================================================================

def key_fingerprint(api_key: str) -> str:
    """
    Return an 8-character hex fingerprint of an API key for per-key quota tracking.

    Args:
        api_key: The API key to fingerprint

    Returns:
        Empty string if api_key is falsy, otherwise first 8 hex chars of SHA-256 hash
    """
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]


# ============================================================================
# QUOTA COST CONSTANTS
# ============================================================================

YOUTUBE_QUOTA_COSTS = {
    'search': 100,
    'channels': 1,
    'videos': 1,
    'playlistItems': 1
}

GEMINI_COSTS = {
    'flash': {'input': 0.0001, 'output': 0.0004},
}

# Default quota cache file path
DEFAULT_QUOTA_CACHE_FILE = '.quota_cache.json'


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DebugData:
    """Per-search API tracking data."""
    # API call counters (this search only)
    youtube_search_calls: int = 0
    youtube_channel_calls: int = 0
    youtube_video_calls: int = 0
    youtube_playlist_calls: int = 0
    gemini_summary_calls: int = 0
    gemini_outreach_calls: int = 0
    gemini_similarity_calls: int = 0
    gemini_relevance_calls: int = 0

    # Timing data (in seconds)
    timings: Dict[str, float] = field(default_factory=lambda: {
        'search': 0.0,
        'channel_stats': 0.0,
        'video_details': 0.0,
        'similarity_ranking': 0.0,
        'ai_generation': 0.0,
        'total': 0.0
    })

    # Detailed similarity scores (for all channels)
    similarity_details: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for session state storage."""
        return {
            'youtube_search_calls': self.youtube_search_calls,
            'youtube_channel_calls': self.youtube_channel_calls,
            'youtube_video_calls': self.youtube_video_calls,
            'youtube_playlist_calls': self.youtube_playlist_calls,
            'gemini_summary_calls': self.gemini_summary_calls,
            'gemini_outreach_calls': self.gemini_outreach_calls,
            'gemini_similarity_calls': self.gemini_similarity_calls,
            'gemini_relevance_calls': self.gemini_relevance_calls,
            'timings': self.timings.copy(),
            'similarity_details': self.similarity_details.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DebugData':
        """Create from dictionary."""
        return cls(
            youtube_search_calls=data.get('youtube_search_calls', 0),
            youtube_channel_calls=data.get('youtube_channel_calls', 0),
            youtube_video_calls=data.get('youtube_video_calls', 0),
            youtube_playlist_calls=data.get('youtube_playlist_calls', 0),
            gemini_summary_calls=data.get('gemini_summary_calls', 0),
            gemini_outreach_calls=data.get('gemini_outreach_calls', 0),
            gemini_similarity_calls=data.get('gemini_similarity_calls', 0),
            gemini_relevance_calls=data.get('gemini_relevance_calls', 0),
            timings=data.get('timings', {
                'search': 0.0,
                'channel_stats': 0.0,
                'video_details': 0.0,
                'similarity_ranking': 0.0,
                'ai_generation': 0.0,
                'total': 0.0
            }),
            similarity_details=data.get('similarity_details', []),
        )


@dataclass
class DailyQuota:
    """Daily quota tracking data."""
    date: str  # YYYY-MM-DD format in Pacific Time
    youtube_calls: int = 0
    gemini_calls: int = 0
    youtube_units: int = 0
    gemini_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'date': self.date,
            'youtube_calls': self.youtube_calls,
            'gemini_calls': self.gemini_calls,
            'youtube_units': self.youtube_units,
            'gemini_cost_usd': self.gemini_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DailyQuota':
        """Create from dictionary."""
        return cls(
            date=data.get('date', ''),
            youtube_calls=data.get('youtube_calls', 0),
            gemini_calls=data.get('gemini_calls', 0),
            youtube_units=data.get('youtube_units', 0),
            gemini_cost_usd=data.get('gemini_cost_usd', 0.0),
        )


# ============================================================================
# QUOTA CALCULATION FUNCTIONS
# ============================================================================

def calculate_youtube_quota_used(data: dict) -> int:
    """
    Estimate YouTube quota units consumed.

    Args:
        data: Debug data dict with API call counts

    Returns:
        Total quota units used
    """
    total = 0
    total += data.get('youtube_search_calls', 0) * YOUTUBE_QUOTA_COSTS['search']
    total += data.get('youtube_channel_calls', 0) * YOUTUBE_QUOTA_COSTS['channels']
    total += data.get('youtube_video_calls', 0) * YOUTUBE_QUOTA_COSTS['videos']
    total += data.get('youtube_playlist_calls', 0) * YOUTUBE_QUOTA_COSTS['playlistItems']
    return total


def calculate_gemini_cost_estimate(data: dict) -> float:
    """
    Estimate Gemini API cost in USD.

    Args:
        data: Debug data dict with Gemini call counts

    Returns:
        Estimated cost in USD
    """
    summary_calls = data.get('gemini_summary_calls', 0)
    outreach_calls = data.get('gemini_outreach_calls', 0)
    similarity_calls = data.get('gemini_similarity_calls', 0)
    relevance_calls = data.get('gemini_relevance_calls', 0)

    # Estimate: ~500 tokens input + 200 tokens output per call
    cost_per_call = (500 * GEMINI_COSTS['flash']['input'] + 200 * GEMINI_COSTS['flash']['output']) / 1000

    total = (summary_calls + outreach_calls + similarity_calls + relevance_calls) * cost_per_call
    return total


def get_total_youtube_calls(data: dict) -> int:
    """Calculate total YouTube API calls from debug data."""
    return (
        data.get('youtube_search_calls', 0) +
        data.get('youtube_channel_calls', 0) +
        data.get('youtube_video_calls', 0) +
        data.get('youtube_playlist_calls', 0)
    )


def get_total_gemini_calls(data: dict) -> int:
    """Calculate total Gemini API calls from debug data."""
    return (
        data.get('gemini_summary_calls', 0) +
        data.get('gemini_outreach_calls', 0) +
        data.get('gemini_similarity_calls', 0) +
        data.get('gemini_relevance_calls', 0)
    )


# ============================================================================
# DATE/TIME UTILITIES
# ============================================================================

def get_current_date_pt() -> str:
    """
    Get current date in Pacific Time (YYYY-MM-DD format).

    Uses proper DST-aware timezone handling via zoneinfo (Python 3.9+).
    Falls back to UTC-8 if zoneinfo is unavailable.
    """
    try:
        from zoneinfo import ZoneInfo
        pt_tz = ZoneInfo("America/Los_Angeles")
        pt_time = datetime.now(pt_tz)
    except ImportError:
        # Fallback for systems without zoneinfo
        # Note: This doesn't handle DST correctly
        pt_offset = timedelta(hours=-8)
        pt_time = datetime.now(timezone.utc) + pt_offset

    return pt_time.strftime('%Y-%m-%d')


def get_next_reset_time() -> str:
    """
    Get human-readable time until quota resets (midnight PT).

    Uses proper DST-aware timezone handling via zoneinfo (Python 3.9+).
    Falls back to UTC-8 if zoneinfo is unavailable.
    """
    try:
        from zoneinfo import ZoneInfo
        pt_tz = ZoneInfo("America/Los_Angeles")
        pt_now = datetime.now(pt_tz)
    except ImportError:
        # Fallback for systems without zoneinfo
        pt_offset = timedelta(hours=-8)
        pt_now = datetime.now(timezone.utc) + pt_offset

    # Calculate midnight PT
    midnight_pt = pt_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    time_until = midnight_pt - pt_now

    hours = int(time_until.total_seconds() // 3600)

    if hours < 1:
        return "in less than 1 hour"
    elif hours == 1:
        return "in 1 hour"
    else:
        return f"in {hours} hours"


# ============================================================================
# FILE PERSISTENCE
# ============================================================================

def _empty_bucket() -> dict:
    """Return a fresh per-key quota bucket."""
    return {
        'youtube_calls': 0,
        'gemini_calls': 0,
        'youtube_units': 0,
        'gemini_cost_usd': 0.0,
    }


def load_daily_quota(filepath: str = DEFAULT_QUOTA_CACHE_FILE, key_fingerprint: str = "") -> dict:
    """
    Load daily quota from disk, or create new if expired/missing.

    Args:
        filepath: Path to quota cache file
        key_fingerprint: If provided, returns the per-key bucket for this fingerprint.
                         If falsy, returns the legacy top-level dict (old behavior).

    Returns:
        Daily quota dict. When key_fingerprint is falsy: top-level dict with keys
        date, youtube_calls, gemini_calls, youtube_units, gemini_cost_usd.
        When key_fingerprint is truthy: per-key bucket dict with the same quota keys
        (no date field).
    """
    if not key_fingerprint:
        # Legacy path: return top-level dict
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Check if it's from today (Pacific Time)
                saved_date = data.get('date', '')
                today = get_current_date_pt()

                if saved_date == today:
                    return data

            except Exception:
                pass  # Fall through to create new

        # Create new daily quota
        return {
            'date': get_current_date_pt(),
            'youtube_calls': 0,
            'gemini_calls': 0,
            'youtube_units': 0,
            'gemini_cost_usd': 0.0
        }

    # Per-key path
    file_data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                file_data = json.load(f)
        except Exception:
            file_data = {}

    # Reset top-level date if stale
    today = get_current_date_pt()
    if file_data.get('date', '') != today:
        file_data = {'date': today, 'by_key': {}}

    # Return (or create) the bucket for this fingerprint
    return file_data.setdefault('by_key', {}).setdefault(key_fingerprint, _empty_bucket())


def save_daily_quota(data: dict, filepath: str = DEFAULT_QUOTA_CACHE_FILE, key_fingerprint: str = "") -> bool:
    """
    Save daily quota to disk.

    Args:
        data: Daily quota dict (top-level or per-key bucket)
        filepath: Path to quota cache file
        key_fingerprint: If provided, updates only the per-key bucket for this
                         fingerprint and writes back the whole file.
                         If falsy, writes data directly (old behavior).

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if not key_fingerprint:
            # Legacy path: write data as-is
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True

        # Per-key path: load existing file, update only our bucket, write back
        file_data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    file_data = json.load(f)
            except Exception:
                file_data = {}

        # Preserve the top-level date
        today = get_current_date_pt()
        if file_data.get('date', '') != today:
            file_data = {'date': today, 'by_key': {}}

        if 'by_key' not in file_data:
            file_data['by_key'] = {}

        file_data['by_key'][key_fingerprint] = data

        with open(filepath, 'w') as f:
            json.dump(file_data, f, indent=2)
        return True
    except Exception:
        return False


# ============================================================================
# TRACKING FUNCTIONS (stateless - operate on passed data)
# ============================================================================

def track_api_call(data: dict, api_name: str) -> dict:
    """
    Increment counter for specific API.

    Args:
        data: Debug data dict
        api_name: One of 'youtube_search', 'youtube_channel', 'youtube_video',
                  'youtube_playlist', 'gemini_summary', 'gemini_outreach',
                  'gemini_similarity', 'gemini_relevance'

    Returns:
        Updated debug data dict
    """
    key = f'{api_name}_calls'
    if key in data:
        data[key] += 1
    return data


def track_timing(data: dict, operation_name: str, elapsed: float) -> dict:
    """
    Add timing data for an operation.

    Args:
        data: Debug data dict
        operation_name: Key in timings dict
        elapsed: Time elapsed in seconds

    Returns:
        Updated debug data dict
    """
    if 'timings' not in data:
        data['timings'] = {}
    if operation_name in data['timings']:
        data['timings'][operation_name] += elapsed
    return data


def track_similarity_scores(data: dict, channels_with_scores: list) -> dict:
    """
    Store detailed similarity breakdowns for channels.

    Args:
        data: Debug data dict
        channels_with_scores: List of dicts with 'channel_title', 'similarity' keys

    Returns:
        Updated debug data dict
    """
    simplified = []
    for ch in channels_with_scores[:20]:
        similarity = ch.get('similarity', {})
        simplified.append({
            'channel': ch.get('channel_title', 'Unknown'),
            'total_score': similarity.get('total_score', 0),
            'breakdown': similarity.get('breakdown', {}),
            'reasons': similarity.get('match_reasons', [])
        })

    data['similarity_details'] = simplified
    return data


def accumulate_to_daily_quota(
    debug_data: dict,
    daily_quota: dict,
    filepath: str = DEFAULT_QUOTA_CACHE_FILE,
    key_fingerprint: str = ""
) -> dict:
    """
    Add current search's API usage to the daily total and save.

    Args:
        debug_data: Current search debug data dict
        daily_quota: Daily quota dict
        filepath: Path to quota cache file
        key_fingerprint: If provided, saves under the per-key bucket for this fingerprint.

    Returns:
        Updated daily quota dict
    """
    # Calculate this search's usage
    youtube_units = calculate_youtube_quota_used(debug_data)
    gemini_cost = calculate_gemini_cost_estimate(debug_data)
    total_youtube_calls = get_total_youtube_calls(debug_data)
    total_gemini_calls = get_total_gemini_calls(debug_data)

    # Add to daily total
    daily_quota['youtube_calls'] += total_youtube_calls
    daily_quota['gemini_calls'] += total_gemini_calls
    daily_quota['youtube_units'] += youtube_units
    daily_quota['gemini_cost_usd'] += gemini_cost

    # Save to disk
    save_daily_quota(daily_quota, filepath, key_fingerprint)

    return daily_quota


def create_empty_debug_data() -> dict:
    """Create a new empty debug data dict."""
    return {
        'youtube_search_calls': 0,
        'youtube_channel_calls': 0,
        'youtube_video_calls': 0,
        'youtube_playlist_calls': 0,
        'gemini_summary_calls': 0,
        'gemini_outreach_calls': 0,
        'gemini_similarity_calls': 0,
        'gemini_relevance_calls': 0,
        'timings': {
            'search': 0.0,
            'channel_stats': 0.0,
            'video_details': 0.0,
            'similarity_ranking': 0.0,
            'ai_generation': 0.0,
            'total': 0.0
        },
        'similarity_details': []
    }

"""
cache_layer.py - Centralized caching layer for CCSeeker

All Streamlit cache decorators and wrappers are centralized here.
This module provides cached versions of core API functions.

Cache TTLs:
- Channel stats: 7 days (rarely changes)
- Search results: 3 days (moderate freshness)
- Video details: 24 hours (via smart_cache.py)

Design notes:
- These functions use @st.cache_data for Streamlit caching
- They delegate to core functions for actual API calls
- The CacheFunctionsAdapter class implements the CacheFunctions protocol
  expected by core/pipeline.py
"""

import os
import streamlit as st
from typing import Callable
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core functions
try:
    from ..core import (
        get_channel_stats as _get_channel_stats_core,
        search_channels_hybrid as _search_channels_hybrid_core,
    )
except ImportError:
    from core import (
        get_channel_stats as _get_channel_stats_core,
        search_channels_hybrid as _search_channels_hybrid_core,
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Cache TTL constants
CACHE_TTL_CHANNEL_STATS = 604800  # 7 days
CACHE_TTL_SEARCH = 259200         # 3 days
CACHE_TTL_VIDEO = 86400           # 24 hours (used by smart_cache.py)


# ============================================================================
# YOUTUBE SERVICE (cached)
# ============================================================================

@st.cache_resource(show_spinner=False)
def _get_youtube():
    """Create and cache a YouTube Data API client."""
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is not configured")
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)


def _get_api_tracker() -> Callable[[str], None] | None:
    """Get API call tracker callback if debug mode is enabled."""
    try:
        from .. import debug_tracker
    except ImportError:
        import debug_tracker

    if st.session_state.get('debug_mode', False):
        return debug_tracker.track_api_call
    return None


# ============================================================================
# CACHED FUNCTIONS
# ============================================================================

@st.cache_data(ttl=CACHE_TTL_CHANNEL_STATS)  # 7 days
def get_channel_stats_cached(channel_ids_tuple: tuple) -> list[dict]:
    """
    Fetch channel statistics with 7-day caching.

    Args:
        channel_ids_tuple: Tuple of channel IDs (must be tuple for cache key)

    Returns:
        List of channel stats dicts

    Notes:
        - Input is normalized (sorted, deduplicated) to maximize cache hits
        - Uses core.youtube_api.get_channel_stats for actual API call
    """
    # Normalize order to maximize cache hits and avoid duplicate fetches
    channel_ids = tuple(sorted(set(channel_ids_tuple)))
    if not channel_ids:
        return []

    youtube = _get_youtube()
    result = _get_channel_stats_core(
        youtube_service=youtube,
        channel_ids=list(channel_ids),
        on_api_call=_get_api_tracker(),
    )
    return result.stats


def get_video_details_cached(channel_ids_tuple: tuple, max_videos: int = 10) -> list[dict]:
    """
    Fetch video details with per-channel caching (via smart_cache.py).

    Args:
        channel_ids_tuple: Tuple of channel IDs
        max_videos: Maximum videos per channel

    Returns:
        List of video detail dicts

    Notes:
        - Delegates to smart_cache.py which has 24-hour per-channel caching
        - Tracking happens inside smart_cache.py
    """
    try:
        from ..smart_cache import get_video_details_smart
    except ImportError:
        from smart_cache import get_video_details_smart

    youtube = _get_youtube()

    # Normalize order to improve cache hits downstream and deduplicate
    channel_ids_norm = tuple(sorted(set(channel_ids_tuple)))
    if not channel_ids_norm:
        return []

    # Get channel stats to get uploads playlist IDs
    stats = get_channel_stats_cached(channel_ids_norm)
    channel_data_full = []

    for stat in stats:
        channel_data_full.append({
            'channel_id': stat['channel_id'],
            'uploads_playlist_id': stat['uploads_playlist_id']
        })

    # Smart caching handles tracking internally
    return get_video_details_smart(youtube, channel_data_full, max_videos)


@st.cache_data(ttl=CACHE_TTL_SEARCH)  # 3 days
def _search_channels_hybrid_cached(
    query: str,
    region_code: str,
    max_videos: int,
    max_channels: int,
) -> tuple[list[dict], list[str]]:
    """
    Internal cached hybrid search. Returns (channels, warnings).

    Separated from the public function to allow caching without
    displaying warnings on cache hits.
    """
    youtube = _get_youtube()
    result = _search_channels_hybrid_core(
        youtube_service=youtube,
        query=query,
        region_code=region_code,
        max_videos=max_videos,
        max_channels=max_channels,
        on_api_call=_get_api_tracker(),
    )
    return result.channels, result.warnings


def search_channels_hybrid_cached(
    query: str,
    region_code: str,
    max_videos: int = 100,
    max_channels: int = 50,
) -> list[dict]:
    """
    Hybrid search: Find channels by VIDEO content (primary) + channel names (secondary).

    Args:
        query: Search query
        region_code: YouTube region code (e.g., 'US')
        max_videos: Max videos to search per term
        max_channels: Max channels to return

    Returns:
        List[dict] with keys 'channel_id', 'channel_title', 'match_score'

    Notes:
        - Results cached for ~3 days
        - Warnings are displayed via st.warning on first call (not on cache hits)
    """
    channels, warnings = _search_channels_hybrid_cached(
        query, region_code, max_videos, max_channels
    )

    # Display any warnings that occurred (only on first call, not cache hits)
    # Note: This relies on Streamlit's behavior where cached functions
    # don't re-execute, so warnings only show on actual API calls
    for warning in warnings:
        st.warning(warning)

    return channels


@st.cache_data(ttl=CACHE_TTL_SEARCH)  # 3 days
def search_channels_cached(
    query: str,
    region_code: str,
    max_videos: int = 100,
    cache_bust: str = "v2-no-early-cap",
) -> list[dict]:
    """
    Cached multi-term channel search.

    Args:
        query: Comma-separated search terms (e.g., "manga, anime")
        region_code: YouTube region code
        max_videos: Max videos per term
        cache_bust: String to invalidate cache when search logic changes

    Returns:
        List of channel dicts with 'channel_id', 'channel_title', 'match_score'

    Notes:
        - Query terms are normalized (sorted, deduplicated) for stable cache keys
        - Multi-term queries search each term separately and merge results
        - cache_bust is only used to affect the cache key, not in logic
    """
    # Normalize query terms for a stable cache key (order-agnostic)
    terms = [t.strip() for t in (query or "").split(',') if t.strip()]
    if not terms:
        return []

    canonical_terms = sorted(set(terms), key=str.lower)

    # For single term, use hybrid search directly
    if len(terms) == 1:
        return search_channels_hybrid_cached(
            terms[0], region_code, max_videos
        )

    # Multiple terms: search each, then merge
    # Limit to 2 terms (enforced here, not in UI)
    terms_to_search = canonical_terms[:2]

    all_channels = {}  # {channel_id: {'title': str, 'total_score': int}}

    for term in terms_to_search:
        results = search_channels_hybrid_cached(term, region_code, max_videos)
        for channel in results:
            channel_id = channel['channel_id']
            if channel_id not in all_channels:
                all_channels[channel_id] = {
                    'title': channel['channel_title'],
                    'total_score': 0
                }
            all_channels[channel_id]['total_score'] += channel['match_score']

    # Convert back to list format and sort by score
    merged = [
        {
            'channel_id': ch_id,
            'channel_title': data['title'],
            'match_score': data['total_score']
        }
        for ch_id, data in sorted(
            all_channels.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
    ]

    return merged


# ============================================================================
# CACHE FUNCTIONS ADAPTER
# ============================================================================

class CacheFunctionsAdapter:
    """
    Adapter class that implements the CacheFunctions protocol
    expected by core/pipeline.py.

    This bridges the Streamlit-cached functions in this module
    to the pure pipeline that expects a cache interface.
    """

    def get_channel_stats_cached(self, channel_ids: tuple) -> list[dict]:
        return get_channel_stats_cached(channel_ids)

    def get_video_details_cached(self, channel_ids: tuple, max_videos: int) -> list[dict]:
        return get_video_details_cached(channel_ids, max_videos)

    def search_channels_cached(self, query: str, region: str, max_videos: int) -> list[dict]:
        return search_channels_cached(query, region, max_videos, cache_bust="mt-search-v2")

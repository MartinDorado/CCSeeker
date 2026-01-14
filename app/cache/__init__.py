"""
Cache layer for CCSeeker.

This module provides Streamlit-cached wrappers around core API functions.
All caching configuration (TTLs, cache keys) is centralized here.
"""

from .cache_layer import (
    # Cache TTL constants
    CACHE_TTL_CHANNEL_STATS,
    CACHE_TTL_SEARCH,
    CACHE_TTL_VIDEO,
    # Cached functions
    get_channel_stats_cached,
    get_video_details_cached,
    search_channels_cached,
    search_channels_hybrid_cached,
    # Adapter for pipeline
    CacheFunctionsAdapter,
)

__all__ = [
    # Constants
    "CACHE_TTL_CHANNEL_STATS",
    "CACHE_TTL_SEARCH",
    "CACHE_TTL_VIDEO",
    # Functions
    "get_channel_stats_cached",
    "get_video_details_cached",
    "search_channels_cached",
    "search_channels_hybrid_cached",
    # Adapter
    "CacheFunctionsAdapter",
]

"""
CCSeeker Core Module
====================

Pure business logic functions extracted from main.py for better testability
and separation of concerns.

Modules:
- query_utils: Query validation, URL parsing, channel ID resolution
- relevance: Keyword relevance scoring for channels
- youtube_api: YouTube Data API wrapper functions
- gemini_api: Gemini AI API wrapper functions
- pipeline: Search pipeline orchestration

These modules are Streamlit-agnostic and can be unit tested independently.
"""

from .query_utils import (
    MAX_SEARCH_TERMS,
    validate_and_truncate_query,
    extract_identifier_from_url,
    resolve_channel_id,
    strip_outer_quotes,
)

from .relevance import calculate_keyword_relevance

from .youtube_api import (
    # Result types
    SearchResult,
    ChannelStatsResult,
    VideoDetailsResult,
    # Functions
    search_channels_hybrid,
    search_channels_multi_term,
    get_channel_stats,
    get_video_details,
)

from .gemini_api import (
    # Result types
    OutreachDraft,
    SummaryResult,
    # Functions
    generate_ai_relevance_score,
    generate_summary,
    generate_outreach_drafts,
)

from .pipeline import (
    # Result types
    PipelineResult,
    PipelineConfig,
    # Functions
    run_search_pipeline,
)

__all__ = [
    # Constants
    "MAX_SEARCH_TERMS",
    # Query utilities
    "validate_and_truncate_query",
    "extract_identifier_from_url",
    "resolve_channel_id",
    "strip_outer_quotes",
    # Relevance
    "calculate_keyword_relevance",
    # YouTube API result types
    "SearchResult",
    "ChannelStatsResult",
    "VideoDetailsResult",
    # YouTube API functions
    "search_channels_hybrid",
    "search_channels_multi_term",
    "get_channel_stats",
    "get_video_details",
    # Gemini API result types
    "OutreachDraft",
    "SummaryResult",
    # Gemini API functions
    "generate_ai_relevance_score",
    "generate_summary",
    "generate_outreach_drafts",
    # Pipeline
    "PipelineResult",
    "PipelineConfig",
    "run_search_pipeline",
]

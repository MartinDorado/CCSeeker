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
- scoring_version: Centralized scoring weights and version signatures
- seed_topics: Seed channel topic extraction and profiling

These modules are Streamlit-agnostic and can be unit tested independently.
"""

from .query_utils import (
    MAX_SEARCH_TERMS,
    validate_and_truncate_query,
    extract_identifier_from_url,
    resolve_channel_id,
    strip_outer_quotes,
    build_seed_query,
)

from .scoring_version import (
    SCORING_VERSION,
    KEYWORD_WEIGHTS,
    SEED_WEIGHTS,
    CHANNEL_FEEDBACK_REASONS,
    VALID_RATINGS,
    get_scoring_version,
    is_version_compatible,
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

from .seed_topics import (
    # Result types
    SeedProfile,
    SeedAnalysisResult,
    # Functions
    analyze_seed_channel,
    detect_language,
    calculate_subscriber_tier,
)

from .similarity import (
    # Callback interface
    SimilarityCallbacks,
    # Similarity metrics
    jaccard_similarity,
    overlap_count,
    # Subscriber utilities
    get_subscriber_similarity,
    is_within_tier_range,
    # Scoring functions
    calculate_similarity_score,
    gemini_similarity_analysis,
    calculate_final_score,
    # Batch operations
    rank_channels_by_similarity,
    filter_by_subscriber_range,
    # Explanation
    generate_match_explanation,
)

__all__ = [
    # Constants
    "MAX_SEARCH_TERMS",
    "SCORING_VERSION",
    # Query utilities
    "validate_and_truncate_query",
    "extract_identifier_from_url",
    "resolve_channel_id",
    "strip_outer_quotes",
    "build_seed_query",
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
    # Scoring version
    "KEYWORD_WEIGHTS",
    "SEED_WEIGHTS",
    "CHANNEL_FEEDBACK_REASONS",
    "VALID_RATINGS",
    "get_scoring_version",
    "is_version_compatible",
    # Seed topics
    "SeedProfile",
    "SeedAnalysisResult",
    "analyze_seed_channel",
    "detect_language",
    "calculate_subscriber_tier",
    # Similarity module
    "SimilarityCallbacks",
    "jaccard_similarity",
    "overlap_count",
    "get_subscriber_similarity",
    "is_within_tier_range",
    "calculate_similarity_score",
    "gemini_similarity_analysis",
    "calculate_final_score",
    "rank_channels_by_similarity",
    "filter_by_subscriber_range",
    "generate_match_explanation",
]

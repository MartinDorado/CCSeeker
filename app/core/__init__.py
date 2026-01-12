"""
CCSeeker Core Module
====================

Pure business logic functions extracted from main.py for better testability
and separation of concerns.

Modules:
- query_utils: Query validation, URL parsing, channel ID resolution
- relevance: Keyword relevance scoring for channels

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
]

"""
query_utils.py - Query validation and URL parsing utilities

Pure functions for:
- Search query validation and truncation
- YouTube URL parsing (channel IDs, handles)
- Channel ID resolution

These functions are Streamlit-agnostic and can be unit tested independently.
"""

import re
from googleapiclient.errors import HttpError

# ============================================================================
# CONSTANTS
# ============================================================================

MAX_SEARCH_TERMS = 2  # Maximum comma-separated terms allowed in search queries


# ============================================================================
# QUERY VALIDATION
# ============================================================================

def validate_and_truncate_query(query: str) -> tuple[str, bool]:
    """
    Validate and auto-truncate search query to MAX_SEARCH_TERMS.

    Args:
        query: Comma-separated search terms

    Returns:
        tuple: (truncated_query, was_truncated)

    Examples:
        >>> validate_and_truncate_query("manga, anime")
        ("manga, anime", False)

        >>> validate_and_truncate_query("manga, anime, gaming, tech")
        ("manga, anime", True)

        >>> validate_and_truncate_query("")
        ("", False)
    """
    if not query or not query.strip():
        return "", False

    # Split by comma and clean whitespace
    terms = [t.strip() for t in query.split(',') if t.strip()]

    # Check if within limit
    if len(terms) <= MAX_SEARCH_TERMS:
        return query, False

    # Truncate to first MAX_SEARCH_TERMS
    truncated = terms[:MAX_SEARCH_TERMS]
    truncated_query = ", ".join(truncated)

    return truncated_query, True


# ============================================================================
# URL PARSING
# ============================================================================

def extract_identifier_from_url(url: str) -> str | None:
    """
    Extract a channel identifier from common YouTube URL formats.

    Supported formats:
    - https://www.youtube.com/channel/UC...  -> Returns UC...
    - https://www.youtube.com/@handle        -> Returns handle (without @)

    Args:
        url: YouTube URL or other string

    Returns:
        Channel ID (starting with 'UC...') or handle (e.g., 'SomeCreator'),
        or None if no match found.

    Note:
        Resolution to an actual channel ID happens in `resolve_channel_id()`.

    Examples:
        >>> extract_identifier_from_url("https://www.youtube.com/channel/UC123abc")
        "UC123abc"

        >>> extract_identifier_from_url("https://www.youtube.com/@MrBeast")
        "MrBeast"

        >>> extract_identifier_from_url("invalid-url")
        None
    """
    patterns = [
        r'(?:youtube\.com/channel/)([^/?&]+)',
        r'(?:youtube\.com/@)([^/?&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def resolve_channel_id(youtube_service, user_input: str) -> str | None:
    """
    Accept any user input (URL, handle, or raw UC... ID) and normalize it to a canonical channel ID.

    Resolution strategy:
    1. If input starts with "UC" and is 20+ chars, assume it's already a channel ID
    2. Try parsing as URL to extract identifier
    3. If still a handle/name, query YouTube API to resolve it

    Args:
        youtube_service: Authenticated YouTube Data API client
        user_input: Channel URL, @handle, or UC... ID

    Returns:
        Canonical channel ID (e.g., "UCX6OQ3DkcsbYNE6H8uQQuVA") or None if resolution fails.

    Examples:
        >>> resolve_channel_id(youtube, "UCX6OQ3DkcsbYNE6H8uQQuVA")
        "UCX6OQ3DkcsbYNE6H8uQQuVA"

        >>> resolve_channel_id(youtube, "https://www.youtube.com/@MrBeast")
        "UCX6OQ3DkcsbYNE6H8uQQuVA"  # (actual MrBeast channel ID)

        >>> resolve_channel_id(youtube, "@MrBeast")
        "UCX6OQ3DkcsbYNE6H8uQQuVA"
    """
    if not user_input:
        return None

    s = user_input.strip()

    # Direct ID - already in canonical format
    if s.startswith("UC") and len(s) >= 20:
        return s

    # Try parsing URL
    ident = extract_identifier_from_url(s) or s
    if ident.startswith("UC"):
        return ident

    # Treat as handle or name: remove leading '@'
    handle = ident[1:] if ident.startswith("@") else ident

    try:
        response = youtube_service.search().list(
            q=handle,
            part="id",
            type="channel",
            maxResults=1
        ).execute()
        items = response.get("items", [])
        if items:
            return items[0]["id"]["channelId"]
        return None
    except HttpError:
        return None


# ============================================================================
# STRING UTILITIES
# ============================================================================

def strip_outer_quotes(s: str) -> str:
    """
    Remove outer quotes (single or double) from a string.

    Args:
        s: Input string, possibly surrounded by quotes

    Returns:
        String with outer quotes removed, or original if no quotes

    Examples:
        >>> strip_outer_quotes('"hello world"')
        "hello world"

        >>> strip_outer_quotes("'single quotes'")
        "single quotes"

        >>> strip_outer_quotes("no quotes")
        "no quotes"

        >>> strip_outer_quotes("")
        ""
    """
    s = (s or "").strip()
    if len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1].strip()
    return s


# Keep the original name as an alias for backward compatibility during refactoring
_strip_outer_quotes = strip_outer_quotes


# ============================================================================
# SEED QUERY BUILDER
# ============================================================================

def _is_redundant(candidate: str, existing: list[str]) -> bool:
    """Return True if candidate is a substring of (or contains) any existing term."""
    c = candidate.lower()
    return any(c in t.lower() or t.lower() in c for t in existing)


def build_seed_query(profile: dict, max_terms: int = 2) -> str:
    """
    Build a YouTube search query from a seed channel profile.

    Term priority:
    1. primary_keywords — NLP bigrams/unigrams from video titles.
    2. common_tags — aggregated video tags; last-resort padding.

    Multi-word terms are double-quoted. Redundant terms (substring overlap) are
    skipped when padding so the same concept is not repeated.

    Args:
        profile: Seed channel profile dict (from SeedProfile.to_dict()).
        max_terms: Maximum number of search terms to include (default 2).

    Returns:
        Comma-separated search query string, e.g. '"machine learning", python'.
    """
    # Prefer Gemini-generated query when available
    suggestion = profile.get("seed_query_suggestion", "").strip()
    if suggestion:
        return suggestion

    terms: list[str] = []

    # 1. Pad with primary_keywords
    for kw in profile.get("primary_keywords", []):
        if len(terms) >= max_terms:
            break
        kw = kw.strip()
        if kw and not _is_redundant(kw, terms):
            terms.append(kw)

    # 2. Pad with common_tags
    for tag in profile.get("common_tags", []):
        if len(terms) >= max_terms:
            break
        tag = tag.strip()
        if tag and not _is_redundant(tag, terms):
            terms.append(tag)

    quoted = [f'"{t}"' if " " in t else t for t in terms]
    return ", ".join(quoted)

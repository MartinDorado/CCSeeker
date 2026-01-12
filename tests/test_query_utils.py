"""
Tests for core.query_utils module

Tests cover:
- Query validation and truncation
- YouTube URL parsing (channel IDs, handles)
- String utility functions
"""

import pytest
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.query_utils import (
    MAX_SEARCH_TERMS,
    validate_and_truncate_query,
    extract_identifier_from_url,
    strip_outer_quotes,
)


class TestValidateAndTruncateQuery:
    """Tests for validate_and_truncate_query function."""

    def test_empty_query_returns_empty(self):
        """Empty or whitespace-only queries return empty string."""
        assert validate_and_truncate_query("") == ("", False)
        assert validate_and_truncate_query("   ") == ("", False)
        assert validate_and_truncate_query(None) == ("", False)

    def test_single_term_not_truncated(self):
        """Single term queries are not truncated."""
        result, truncated = validate_and_truncate_query("manga")
        assert result == "manga"
        assert truncated is False

    def test_two_terms_not_truncated(self):
        """Two term queries (at the limit) are not truncated."""
        result, truncated = validate_and_truncate_query("manga, anime")
        assert result == "manga, anime"
        assert truncated is False

    def test_three_terms_truncated_to_two(self):
        """Three+ term queries are truncated to first two terms."""
        result, truncated = validate_and_truncate_query("manga, anime, gaming")
        assert result == "manga, anime"
        assert truncated is True

    def test_four_terms_truncated_to_two(self):
        """Four term queries are truncated to first two terms."""
        result, truncated = validate_and_truncate_query("manga, anime, gaming, tech")
        assert result == "manga, anime"
        assert truncated is True

    def test_whitespace_handling(self):
        """Extra whitespace around terms is handled correctly."""
        result, truncated = validate_and_truncate_query("  manga  ,   anime  ")
        assert result == "  manga  ,   anime  "  # Original returned if within limit
        assert truncated is False

    def test_max_search_terms_constant(self):
        """MAX_SEARCH_TERMS constant is 2."""
        assert MAX_SEARCH_TERMS == 2


class TestExtractIdentifierFromUrl:
    """Tests for extract_identifier_from_url function."""

    def test_channel_url_extracts_id(self):
        """Channel URLs return the channel ID."""
        url = "https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA"
        assert extract_identifier_from_url(url) == "UCX6OQ3DkcsbYNE6H8uQQuVA"

    def test_channel_url_with_query_params(self):
        """Channel URLs with query parameters still extract ID."""
        url = "https://www.youtube.com/channel/UC123abc?feature=subscribe"
        assert extract_identifier_from_url(url) == "UC123abc"

    def test_handle_url_extracts_handle(self):
        """@handle URLs return the handle (without @)."""
        url = "https://www.youtube.com/@MrBeast"
        assert extract_identifier_from_url(url) == "MrBeast"

    def test_handle_url_with_query_params(self):
        """@handle URLs with query parameters still extract handle."""
        url = "https://www.youtube.com/@SomeCreator?sub_confirmation=1"
        assert extract_identifier_from_url(url) == "SomeCreator"

    def test_invalid_url_returns_none(self):
        """Invalid URLs return None."""
        assert extract_identifier_from_url("not-a-url") is None
        assert extract_identifier_from_url("https://google.com") is None
        assert extract_identifier_from_url("https://www.youtube.com/watch?v=abc") is None

    def test_empty_url_returns_none(self):
        """Empty strings return None."""
        assert extract_identifier_from_url("") is None


class TestStripOuterQuotes:
    """Tests for strip_outer_quotes function."""

    def test_double_quotes_stripped(self):
        """Double quotes are removed from both ends."""
        assert strip_outer_quotes('"hello world"') == "hello world"

    def test_single_quotes_stripped(self):
        """Single quotes are removed from both ends."""
        assert strip_outer_quotes("'hello world'") == "hello world"

    def test_no_quotes_unchanged(self):
        """Strings without outer quotes remain unchanged."""
        assert strip_outer_quotes("hello world") == "hello world"

    def test_mismatched_quotes_unchanged(self):
        """Mismatched quotes are not stripped."""
        assert strip_outer_quotes('"hello world\'') == '"hello world\''
        assert strip_outer_quotes("'hello world\"") == "'hello world\""

    def test_empty_string(self):
        """Empty strings return empty."""
        assert strip_outer_quotes("") == ""

    def test_none_returns_empty(self):
        """None returns empty string."""
        assert strip_outer_quotes(None) == ""

    def test_whitespace_preserved_inside_quotes(self):
        """Whitespace inside quotes is preserved (after trimming)."""
        assert strip_outer_quotes('"  spaced  "') == "spaced"

    def test_single_character_quotes(self):
        """Single quote characters are not stripped."""
        assert strip_outer_quotes('"') == '"'
        assert strip_outer_quotes("'") == "'"

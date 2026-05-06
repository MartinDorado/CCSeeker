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
    build_seed_query,
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


class TestBuildSeedQuery:
    """Tests for build_seed_query function."""

    # ------------------------------------------------------------------
    # primary_keywords as primary source
    # ------------------------------------------------------------------

    def test_two_primary_keywords_used_directly(self):
        profile = {
            "primary_keywords": ["machine learning tutorials", "python data science"],
            "common_tags": [],
        }
        assert build_seed_query(profile) == '"machine learning tutorials", "python data science"'

    def test_single_word_primary_keyword_not_quoted(self):
        profile = {"primary_keywords": ["python", "automation"], "common_tags": []}
        assert build_seed_query(profile) == "python, automation"

    def test_primary_keywords_mixed_quoting(self):
        profile = {"primary_keywords": ["machine learning", "python"], "common_tags": []}
        assert build_seed_query(profile) == '"machine learning", python'

    def test_primary_keywords_truncated_to_max_terms(self):
        profile = {"primary_keywords": ["a", "b", "c", "d"], "common_tags": []}
        assert build_seed_query(profile) == "a, b"

    def test_max_terms_override(self):
        profile = {"primary_keywords": ["a", "b", "c"], "common_tags": []}
        assert build_seed_query(profile, max_terms=1) == "a"
        assert build_seed_query(profile, max_terms=3) == "a, b, c"

    # ------------------------------------------------------------------
    # Fallback to common_tags
    # ------------------------------------------------------------------

    def test_primary_keywords_falls_back_to_primary_keywords(self):
        profile = {
            "primary_keywords": ["data science", "machine learning"],
            "common_tags": ["python"],
        }
        assert build_seed_query(profile) == '"data science", "machine learning"'

    def test_common_tags_used_when_primary_keywords_exhausted(self):
        profile = {"primary_keywords": ["anime"], "common_tags": ["manga", "japan"]}
        assert build_seed_query(profile) == "anime, manga"

    def test_common_tags_redundancy_skipped(self):
        profile = {
            "primary_keywords": ["anime reviews"],
            "common_tags": ["anime", "japan"],
        }
        result = build_seed_query(profile)
        assert "anime reviews" in result or '"anime reviews"' in result
        assert result.count("anime") == 1
        assert "japan" in result

    def test_padding_skips_redundant_term(self):
        """common_tags substring of an existing term are skipped."""
        profile = {
            "primary_keywords": ["machine learning tutorials"],
            "common_tags": ["machine learning", "unrelated topic"],
        }
        result = build_seed_query(profile)
        assert '"machine learning tutorials"' in result
        assert "machine learning," not in result
        assert "unrelated topic" in result

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_profile_returns_empty_string(self):
        assert build_seed_query({}) == ""

    def test_whitespace_only_terms_ignored(self):
        profile = {"primary_keywords": ["  ", "python tutorials"], "common_tags": []}
        result = build_seed_query(profile)
        assert '"python tutorials"' in result
        assert "  " not in result

    # ------------------------------------------------------------------
    # seed_query_suggestion priority
    # ------------------------------------------------------------------

    def test_seed_query_suggestion_takes_priority(self):
        """Gemini-generated suggestion overrides NLP keywords entirely."""
        profile = {
            "seed_query_suggestion": '"vegan cooking", plant-based',
            "primary_keywords": ["machine learning", "python"],
            "common_tags": ["ai", "data"],
        }
        assert build_seed_query(profile) == '"vegan cooking", plant-based'

    def test_empty_suggestion_falls_back_to_nlp(self):
        """Empty seed_query_suggestion triggers NLP fallback."""
        profile = {
            "seed_query_suggestion": "",
            "primary_keywords": ["anime", "manga"],
            "common_tags": [],
        }
        assert build_seed_query(profile) == "anime, manga"

    def test_absent_suggestion_falls_back_to_nlp(self):
        """Missing seed_query_suggestion key triggers NLP fallback."""
        profile = {"primary_keywords": ["anime"], "common_tags": ["manga"]}
        assert build_seed_query(profile) == "anime, manga"

    def test_whitespace_only_suggestion_falls_back_to_nlp(self):
        """Whitespace-only suggestion is treated as absent."""
        profile = {
            "seed_query_suggestion": "   ",
            "primary_keywords": ["anime"],
            "common_tags": ["manga"],
        }
        assert build_seed_query(profile) == "anime, manga"

    def test_suggestion_ignores_max_terms(self):
        """Suggestion is returned as-is regardless of max_terms."""
        profile = {
            "seed_query_suggestion": '"a", "b", "c"',
            "primary_keywords": [],
            "common_tags": [],
        }
        assert build_seed_query(profile, max_terms=1) == '"a", "b", "c"'

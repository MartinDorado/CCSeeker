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
    # Helpers
    # ------------------------------------------------------------------

    def _profile(
        self,
        topic_emphasis=None,
        primary_keywords=None,
        common_tags=None,
        niche_confidence="high",
    ) -> dict:
        niche = {}
        if topic_emphasis is not None:
            niche = {
                "niche": "test niche",
                "audience": "developers",
                "style": "tutorial",
                "topic_emphasis": topic_emphasis,
                "tone": "clear",
                "confidence": niche_confidence,
            }
        return {
            "transcript_niche_summary": niche,
            "primary_keywords": primary_keywords or [],
            "common_tags": common_tags or [],
        }

    # ------------------------------------------------------------------
    # Transcript topic_emphasis as primary source
    # ------------------------------------------------------------------

    def test_two_topic_emphasis_used_directly(self):
        """When topic_emphasis has 2+ items they become the full query."""
        profile = self._profile(
            topic_emphasis=["machine learning tutorials", "python data science"],
            primary_keywords=["beginner coding"],
        )
        assert build_seed_query(profile) == '"machine learning tutorials", "python data science"'

    def test_single_word_topic_emphasis_not_quoted(self):
        """Single-word topic_emphasis terms are not wrapped in quotes."""
        profile = self._profile(
            topic_emphasis=["python", "automation"],
        )
        assert build_seed_query(profile) == "python, automation"

    def test_topic_emphasis_mixed_quoting(self):
        """Multi-word terms are quoted; single-word terms are not."""
        profile = self._profile(
            topic_emphasis=["machine learning", "python"],
        )
        assert build_seed_query(profile) == '"machine learning", python'

    def test_topic_emphasis_truncated_to_max_terms(self):
        """Only max_terms (default 2) terms are taken from topic_emphasis."""
        profile = self._profile(
            topic_emphasis=["a", "b", "c", "d"],
        )
        assert build_seed_query(profile) == "a, b"

    def test_max_terms_override(self):
        """max_terms parameter controls the limit."""
        profile = self._profile(
            topic_emphasis=["a", "b", "c"],
        )
        assert build_seed_query(profile, max_terms=1) == "a"
        assert build_seed_query(profile, max_terms=3) == "a, b, c"

    # ------------------------------------------------------------------
    # Padding with primary_keywords
    # ------------------------------------------------------------------

    def test_one_topic_emphasis_padded_with_primary_keyword(self):
        """A single topic_emphasis item is padded to max_terms with primary_keywords."""
        profile = self._profile(
            topic_emphasis=["machine learning"],
            primary_keywords=["python", "data science"],
        )
        result = build_seed_query(profile)
        assert result == '"machine learning", python'

    def test_padding_skips_redundant_primary_keyword(self):
        """Primary keyword that is a substring of a topic_emphasis term is skipped."""
        profile = self._profile(
            topic_emphasis=["machine learning tutorials"],
            primary_keywords=["machine learning", "unrelated topic"],
        )
        result = build_seed_query(profile)
        # "machine learning" is redundant with "machine learning tutorials"
        assert '"machine learning tutorials"' in result
        assert "machine learning," not in result
        assert "unrelated topic" in result

    def test_topic_emphasis_substring_of_primary_keyword_also_skipped(self):
        """Primary keyword that contains a topic_emphasis term is also skipped."""
        profile = self._profile(
            topic_emphasis=["python"],
            primary_keywords=["advanced python programming", "django"],
        )
        result = build_seed_query(profile)
        # "advanced python programming" contains "python" so it's redundant
        assert "python," in result or result.startswith("python")
        assert "advanced python programming" not in result
        assert "django" in result

    # ------------------------------------------------------------------
    # Fallback to primary_keywords (no transcripts)
    # ------------------------------------------------------------------

    def test_empty_niche_summary_falls_back_to_primary_keywords(self):
        """Empty transcript_niche_summary falls back to primary_keywords."""
        profile = {
            "transcript_niche_summary": {},
            "primary_keywords": ["data science", "machine learning"],
            "common_tags": ["python"],
        }
        assert build_seed_query(profile) == '"data science", "machine learning"'

    def test_missing_niche_summary_key_falls_back_to_primary_keywords(self):
        """Missing transcript_niche_summary key falls back to primary_keywords."""
        profile = {
            "primary_keywords": ["data science", "machine learning"],
            "common_tags": ["python"],
        }
        assert build_seed_query(profile) == '"data science", "machine learning"'

    def test_empty_topic_emphasis_list_falls_back_to_primary_keywords(self):
        """Empty topic_emphasis list inside a niche summary falls back to primary_keywords."""
        profile = self._profile(
            topic_emphasis=[],
            primary_keywords=["anime reviews", "manga"],
        )
        assert build_seed_query(profile) == '"anime reviews", manga'

    # ------------------------------------------------------------------
    # Fallback to common_tags
    # ------------------------------------------------------------------

    def test_common_tags_used_when_primary_keywords_exhausted(self):
        """common_tags pad the query when primary_keywords are insufficient."""
        profile = {
            "transcript_niche_summary": {},
            "primary_keywords": ["anime"],
            "common_tags": ["manga", "japan"],
        }
        assert build_seed_query(profile) == "anime, manga"

    def test_common_tags_redundancy_skipped(self):
        """Redundant common_tags are skipped."""
        profile = {
            "transcript_niche_summary": {},
            "primary_keywords": ["anime reviews"],
            "common_tags": ["anime", "japan"],
        }
        result = build_seed_query(profile)
        # "anime" is redundant with "anime reviews"
        assert "anime reviews" in result or '"anime reviews"' in result
        assert result.count("anime") == 1
        assert "japan" in result

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_profile_returns_empty_string(self):
        """A fully empty profile returns an empty string."""
        assert build_seed_query({}) == ""

    def test_whitespace_only_terms_ignored(self):
        """Whitespace-only entries in topic_emphasis are ignored."""
        profile = self._profile(
            topic_emphasis=["  ", "python tutorials"],
            primary_keywords=["data"],
        )
        result = build_seed_query(profile)
        assert '"python tutorials"' in result
        assert "  " not in result

    def test_niche_confidence_does_not_affect_query(self):
        """topic_emphasis is used regardless of niche confidence level."""
        for confidence in ("high", "medium", "low"):
            profile = self._profile(
                topic_emphasis=["python tutorials"],
                niche_confidence=confidence,
            )
            assert '"python tutorials"' in build_seed_query(profile)

    def test_none_niche_summary_treated_as_empty(self):
        """None value for transcript_niche_summary falls back gracefully."""
        profile = {
            "transcript_niche_summary": None,
            "primary_keywords": ["python"],
            "common_tags": [],
        }
        assert build_seed_query(profile) == "python"

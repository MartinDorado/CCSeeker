"""
Tests for core.relevance module

Tests cover:
- Keyword relevance scoring for channels
- Edge cases (empty DataFrames, empty queries)
- Weighting behavior
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.relevance import calculate_keyword_relevance


class TestCalculateKeywordRelevance:
    """Tests for calculate_keyword_relevance function."""

    def test_empty_dataframe_returns_empty(self):
        """Empty DataFrame returns empty result."""
        df = pd.DataFrame()
        result = calculate_keyword_relevance(df, "manga")
        assert result.empty
        assert list(result.columns) == ['channel_id', 'relevance_score']

    def test_empty_query_returns_empty(self):
        """Empty query returns empty result."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['Manga Review'],
            'video_tags': [['manga']]
        })
        result = calculate_keyword_relevance(df, "")
        assert result.empty

    def test_whitespace_query_returns_empty(self):
        """Whitespace-only query returns empty result."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['Manga Review'],
            'video_tags': [['manga']]
        })
        result = calculate_keyword_relevance(df, "   ")
        assert result.empty

    def test_single_term_match_in_title(self):
        """Single term matching in title scores correctly."""
        df = pd.DataFrame({
            'channel_id': ['UC1', 'UC2'],
            'video_title': ['Manga Review Episode 1', 'Gaming Stream'],
            'video_tags': [[], []]
        })
        result = calculate_keyword_relevance(df, "manga")

        # UC1 should have relevance (manga in title)
        # UC2 should have no relevance
        uc1_score = result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0]
        uc2_score = result[result['channel_id'] == 'UC2']['relevance_score'].iloc[0]

        assert uc1_score > 0
        assert uc2_score == 0

    def test_single_term_match_in_tags(self):
        """Single term matching in tags scores correctly."""
        df = pd.DataFrame({
            'channel_id': ['UC1', 'UC2'],
            'video_title': ['Some Video', 'Another Video'],
            'video_tags': [['manga', 'anime'], ['gaming']]
        })
        result = calculate_keyword_relevance(df, "manga")

        uc1_score = result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0]
        uc2_score = result[result['channel_id'] == 'UC2']['relevance_score'].iloc[0]

        assert uc1_score > 0
        assert uc2_score == 0

    def test_multiple_videos_per_channel_averaged(self):
        """Scores are averaged across multiple videos per channel."""
        df = pd.DataFrame({
            'channel_id': ['UC1', 'UC1', 'UC1'],
            'video_title': ['Manga Review', 'Cooking Tutorial', 'Manga Chapter 2'],
            'video_tags': [[], [], []]
        })
        result = calculate_keyword_relevance(df, "manga")

        # With title_weight=2, tags_weight=1:
        # - Title match with no tag match = 2/(2+1) = 0.667 per video
        # - 2 out of 3 videos match = (0.667 + 0 + 0.667) / 3 = 0.444
        uc1_score = result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0]
        assert 0.4 < uc1_score < 0.5  # ~0.444

    def test_comma_separated_terms(self):
        """Comma-separated terms are treated as OR."""
        df = pd.DataFrame({
            'channel_id': ['UC1', 'UC2', 'UC3'],
            'video_title': ['Manga Review', 'Anime News', 'Gaming Stream'],
            'video_tags': [[], [], []]
        })
        result = calculate_keyword_relevance(df, "manga, anime")

        uc1_score = result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0]
        uc2_score = result[result['channel_id'] == 'UC2']['relevance_score'].iloc[0]
        uc3_score = result[result['channel_id'] == 'UC3']['relevance_score'].iloc[0]

        assert uc1_score > 0  # manga matches
        assert uc2_score > 0  # anime matches
        assert uc3_score == 0  # neither matches

    def test_case_insensitive_matching(self):
        """Matching is case-insensitive."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['MANGA REVIEW'],
            'video_tags': [[]]
        })
        result = calculate_keyword_relevance(df, "manga")

        assert result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0] > 0

    def test_word_boundary_matching(self):
        """Word boundaries prevent partial matches (e.g., 'man' shouldn't match 'manga')."""
        df = pd.DataFrame({
            'channel_id': ['UC1', 'UC2'],
            'video_title': ['The man walks', 'Manga review'],
            'video_tags': [[], []]
        })
        result = calculate_keyword_relevance(df, "man")

        uc1_score = result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0]
        uc2_score = result[result['channel_id'] == 'UC2']['relevance_score'].iloc[0]

        assert uc1_score > 0  # "man" matches
        assert uc2_score == 0  # "manga" should NOT match "man" due to word boundary

    def test_custom_weights(self):
        """Custom title and tag weights affect scoring."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['Some Video'],
            'video_tags': [['manga']]
        })

        # With default weights (title=2, tags=1), tags-only match = 1/3 = 0.33
        result_default = calculate_keyword_relevance(df, "manga")

        # With equal weights (title=1, tags=1), tags-only match = 1/2 = 0.5
        result_equal = calculate_keyword_relevance(df, "manga", title_weight=1.0, tags_weight=1.0)

        default_score = result_default[result_default['channel_id'] == 'UC1']['relevance_score'].iloc[0]
        equal_score = result_equal[result_equal['channel_id'] == 'UC1']['relevance_score'].iloc[0]

        assert equal_score > default_score

    def test_quoted_terms_handled(self):
        """Quoted terms have quotes stripped."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['Manga Review'],
            'video_tags': [[]]
        })
        result = calculate_keyword_relevance(df, '"manga"')

        assert result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0] > 0

    def test_missing_video_tags_column(self):
        """DataFrame without video_tags column still works."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['Manga Review']
        })
        result = calculate_keyword_relevance(df, "manga")

        # Should still score based on title
        assert result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0] > 0

    def test_none_values_in_tags(self):
        """None values in tags are handled gracefully."""
        df = pd.DataFrame({
            'channel_id': ['UC1'],
            'video_title': ['Video'],
            'video_tags': [[None, 'manga', None]]
        })
        result = calculate_keyword_relevance(df, "manga")

        # Should still find 'manga' in tags
        assert result[result['channel_id'] == 'UC1']['relevance_score'].iloc[0] > 0

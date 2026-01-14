"""
Tests for core.gemini_api module

Tests cover:
- AI relevance scoring
- Summary generation
- Outreach draft generation
- Error handling and edge cases

All tests use mocked Gemini API clients for isolation.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.gemini_api import (
    OutreachDraft,
    SummaryResult,
    generate_ai_relevance_score,
    generate_summary,
    generate_outreach_drafts,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_model():
    """Create a mock Gemini model."""
    return Mock()


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for summary/outreach tests."""
    return pd.DataFrame({
        'channel_title': ['Channel A', 'Channel B', 'Channel C'],
        'subscribers': [100000, 50000, 25000],
        'country': ['US', 'UK', 'ES'],
        'relevance_score': ['85%', '70%', '55%'],
        'engagement_rate': ['3.5%', '2.1%', '4.2%'],
    })


@pytest.fixture
def sample_df_seed_mode():
    """Create a sample DataFrame for seed-based search mode."""
    return pd.DataFrame({
        'channel_title': ['Channel A', 'Channel B'],
        'subscribers': [100000, 50000],
        'country': ['US', 'UK'],
        'relevance_score': ['85%', '70%'],
        'engagement_rate': ['3.5%', '2.1%'],
        'similarity_score': [92, 78],
        'similarity': [
            {'total_score': 92, 'match_reasons': ['Similar tags', 'Same size']},
            {'total_score': 78, 'match_reasons': ['Topic overlap']},
        ],
    })


# ============================================================================
# generate_ai_relevance_score TESTS
# ============================================================================

class TestGenerateAIRelevanceScore:
    """Tests for AI relevance scoring function."""

    def test_empty_video_titles_returns_zero(self, mock_model):
        """Channel with no video titles returns 0.0 score."""
        channel_data = {'channel_title': 'Test', 'video_titles': []}

        score = generate_ai_relevance_score(mock_model, channel_data, "manga")

        assert score == 0.0

    def test_missing_video_titles_returns_zero(self, mock_model):
        """Channel without video_titles key returns 0.0 score."""
        channel_data = {'channel_title': 'Test'}

        score = generate_ai_relevance_score(mock_model, channel_data, "manga")

        assert score == 0.0

    def test_valid_score_normalized(self, mock_model):
        """Valid score from model is normalized to 0.0-1.0 range."""
        mock_model.generate_content.return_value = Mock(text="8")
        channel_data = {
            'channel_title': 'MangaFan',
            'video_titles': ['Manga Review', 'Best Manga 2024']
        }

        score = generate_ai_relevance_score(mock_model, channel_data, "manga")

        assert score == 0.8  # 8 / 10 = 0.8

    def test_score_capped_at_ten(self, mock_model):
        """Scores above 10 are capped at 1.0."""
        mock_model.generate_content.return_value = Mock(text="15")
        channel_data = {
            'channel_title': 'Test',
            'video_titles': ['Video 1']
        }

        score = generate_ai_relevance_score(mock_model, channel_data, "test")

        assert score == 1.0

    def test_zero_score_returns_zero(self, mock_model):
        """Zero score from model returns 0.0."""
        mock_model.generate_content.return_value = Mock(text="0")
        channel_data = {
            'channel_title': 'Test',
            'video_titles': ['Video 1']
        }

        score = generate_ai_relevance_score(mock_model, channel_data, "test")

        assert score == 0.0

    def test_api_error_returns_zero(self, mock_model):
        """API error returns 0.0 score."""
        mock_model.generate_content.side_effect = Exception("API Error")
        channel_data = {
            'channel_title': 'Test',
            'video_titles': ['Video 1']
        }

        score = generate_ai_relevance_score(mock_model, channel_data, "test")

        assert score == 0.0

    def test_invalid_response_returns_zero(self, mock_model):
        """Non-numeric response returns 0.0 score."""
        mock_model.generate_content.return_value = Mock(text="This channel is relevant")
        channel_data = {
            'channel_title': 'Test',
            'video_titles': ['Video 1']
        }

        score = generate_ai_relevance_score(mock_model, channel_data, "test")

        assert score == 0.0

    def test_extracts_first_number_from_text(self, mock_model):
        """Extracts first number from mixed text response."""
        mock_model.generate_content.return_value = Mock(text="I rate this a 7 out of 10")
        channel_data = {
            'channel_title': 'Test',
            'video_titles': ['Video 1']
        }

        score = generate_ai_relevance_score(mock_model, channel_data, "test")

        assert score == 0.7

    def test_uses_first_10_titles_only(self, mock_model):
        """Only first 10 video titles are sent to model."""
        mock_model.generate_content.return_value = Mock(text="5")
        channel_data = {
            'channel_title': 'Test',
            'video_titles': [f'Video {i}' for i in range(20)]
        }

        generate_ai_relevance_score(mock_model, channel_data, "test")

        # Check the prompt only includes 10 titles
        call_args = mock_model.generate_content.call_args[0][0]
        # Count how many "- Video" appear in the prompt (should be 10)
        assert call_args.count('Video 9') == 1  # 10th video (0-indexed)
        assert 'Video 10' not in call_args  # 11th video should not be included


# ============================================================================
# generate_summary TESTS
# ============================================================================

class TestGenerateSummary:
    """Tests for summary generation function."""

    def test_successful_summary_generation(self, mock_model, sample_df):
        """Successful summary is returned correctly."""
        mock_model.generate_content.return_value = Mock(
            text="These are the top channels for your search..."
        )

        result = generate_summary(mock_model, sample_df, "manga")

        assert isinstance(result, SummaryResult)
        assert result.text == "These are the top channels for your search..."
        assert result.error is None

    def test_api_error_returns_error_message(self, mock_model, sample_df):
        """API error is captured in result.error."""
        mock_model.generate_content.side_effect = Exception("Rate limit exceeded")

        result = generate_summary(mock_model, sample_df, "manga")

        assert result.error is not None
        assert "Rate limit exceeded" in result.error
        assert result.text == ""

    def test_uses_top_5_channels_only(self, mock_model):
        """Only top 5 channels are sent to model."""
        df = pd.DataFrame({
            'channel_title': [f'Channel {i}' for i in range(10)],
            'subscribers': [i * 1000 for i in range(10)],
            'country': ['US'] * 10,
            'relevance_score': ['50%'] * 10,
            'engagement_rate': ['2.0%'] * 10,
        })
        mock_model.generate_content.return_value = Mock(text="Summary")

        generate_summary(mock_model, df, "test")

        call_args = mock_model.generate_content.call_args[0][0]
        assert 'Channel 4' in call_args  # 5th channel
        assert 'Channel 5' not in call_args  # 6th channel not included

    def test_keyword_mode_prompt(self, mock_model, sample_df):
        """Keyword mode uses appropriate prompt."""
        mock_model.generate_content.return_value = Mock(text="Summary")

        generate_summary(mock_model, sample_df, "manga")

        call_args = mock_model.generate_content.call_args[0][0]
        assert 'query "manga"' in call_args.lower() or '"manga"' in call_args

    def test_seed_mode_prompt(self, mock_model, sample_df_seed_mode):
        """Seed mode uses appropriate prompt with seed channel reference."""
        mock_model.generate_content.return_value = Mock(text="Summary")

        generate_summary(
            mock_model, sample_df_seed_mode, "manga",
            seed_channel_name="MangaMaster"
        )

        call_args = mock_model.generate_content.call_args[0][0]
        assert 'MangaMaster' in call_args
        assert 'similar' in call_args.lower()

    def test_seed_mode_includes_similarity_score(self, mock_model, sample_df_seed_mode):
        """Seed mode prompt includes similarity score data."""
        mock_model.generate_content.return_value = Mock(text="Summary")

        generate_summary(
            mock_model, sample_df_seed_mode, "manga",
            seed_channel_name="MangaMaster"
        )

        call_args = mock_model.generate_content.call_args[0][0]
        assert 'Similarity Score' in call_args

    def test_api_callback_invoked(self, mock_model, sample_df):
        """on_api_call callback is invoked."""
        mock_model.generate_content.return_value = Mock(text="Summary")
        tracker = Mock()

        generate_summary(mock_model, sample_df, "manga", on_api_call=tracker)

        tracker.assert_called_once_with('gemini_summary')


# ============================================================================
# generate_outreach_drafts TESTS
# ============================================================================

class TestGenerateOutreachDrafts:
    """Tests for outreach draft generation function."""

    def test_empty_dataframe_returns_empty_list(self, mock_model):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame()

        result = generate_outreach_drafts(mock_model, df, "manga")

        assert result == []

    def test_none_dataframe_returns_empty_list(self, mock_model):
        """None DataFrame returns empty list."""
        result = generate_outreach_drafts(mock_model, None, "manga")

        assert result == []

    def test_missing_channel_title_column_returns_empty(self, mock_model):
        """DataFrame without channel_title column returns empty list."""
        df = pd.DataFrame({'other_column': ['value']})

        result = generate_outreach_drafts(mock_model, df, "manga")

        assert result == []

    def test_successful_draft_generation(self, mock_model):
        """Successful drafts are returned correctly."""
        mock_model.generate_content.return_value = Mock(
            text="Hi! I love your manga content..."
        )
        df = pd.DataFrame({'channel_title': ['MangaFan', 'AnimeWorld']})

        result = generate_outreach_drafts(mock_model, df, "manga", limit=2)

        assert len(result) == 2
        assert isinstance(result[0], OutreachDraft)
        assert result[0].channel_title == 'MangaFan'
        assert result[0].draft_text == "Hi! I love your manga content..."

    def test_limit_parameter_respected(self, mock_model):
        """limit parameter controls number of drafts generated."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']})

        result = generate_outreach_drafts(mock_model, df, "test", limit=2)

        assert len(result) == 2
        assert mock_model.generate_content.call_count == 2

    def test_empty_channel_titles_filtered(self, mock_model):
        """Empty or whitespace channel titles are filtered."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Valid', '', '  ', 'AlsoValid']})

        result = generate_outreach_drafts(mock_model, df, "test", limit=10)

        assert len(result) == 2
        assert result[0].channel_title == 'Valid'
        assert result[1].channel_title == 'AlsoValid'

    def test_duplicate_channels_deduplicated(self, mock_model):
        """Duplicate channel titles are removed."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Channel', 'Channel', 'Channel']})

        result = generate_outreach_drafts(mock_model, df, "test", limit=10)

        assert len(result) == 1

    def test_english_language_default(self, mock_model):
        """English language instruction is used by default."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Test']})

        generate_outreach_drafts(mock_model, df, "test", language="en")

        call_args = mock_model.generate_content.call_args[0][0]
        assert 'English' in call_args

    def test_spanish_language_option(self, mock_model):
        """Spanish language instruction is used when specified."""
        mock_model.generate_content.return_value = Mock(text="Borrador")
        df = pd.DataFrame({'channel_title': ['Test']})

        generate_outreach_drafts(mock_model, df, "test", language="es")

        call_args = mock_model.generate_content.call_args[0][0]
        assert 'Spanish' in call_args

    def test_api_error_retries(self, mock_model):
        """API errors trigger retries."""
        mock_model.generate_content.side_effect = [
            Exception("First fail"),
            Exception("Second fail"),
            Mock(text="Success!"),
        ]
        df = pd.DataFrame({'channel_title': ['Test']})

        result = generate_outreach_drafts(mock_model, df, "test", retries=2)

        assert len(result) == 1
        assert result[0].draft_text == "Success!"
        assert mock_model.generate_content.call_count == 3

    def test_persistent_error_includes_error_message(self, mock_model):
        """Persistent API error includes error in draft text."""
        mock_model.generate_content.side_effect = Exception("API Down")
        df = pd.DataFrame({'channel_title': ['TestChannel']})

        result = generate_outreach_drafts(mock_model, df, "test", retries=0)

        assert len(result) == 1
        assert "Error generating draft" in result[0].draft_text
        assert "TestChannel" in result[0].draft_text

    def test_code_block_markdown_stripped(self, mock_model):
        """Markdown code block wrapping is stripped."""
        mock_model.generate_content.return_value = Mock(
            text="```\nHello there!\n```"
        )
        df = pd.DataFrame({'channel_title': ['Test']})

        result = generate_outreach_drafts(mock_model, df, "test")

        assert "```" not in result[0].draft_text
        assert "Hello there!" in result[0].draft_text

    def test_api_callback_invoked_per_draft(self, mock_model):
        """on_api_call is invoked for each draft generated."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Ch1', 'Ch2', 'Ch3']})
        tracker = Mock()

        generate_outreach_drafts(
            mock_model, df, "test", limit=3, on_api_call=tracker
        )

        assert tracker.call_count == 3
        tracker.assert_called_with('gemini_outreach')

    def test_query_referenced_in_prompt(self, mock_model):
        """Original query is referenced in the prompt."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Test']})

        generate_outreach_drafts(mock_model, df, "manga enthusiasts")

        call_args = mock_model.generate_content.call_args[0][0]
        assert "manga enthusiasts" in call_args

    def test_empty_query_uses_fallback(self, mock_model):
        """Empty query uses fallback text."""
        mock_model.generate_content.return_value = Mock(text="Draft")
        df = pd.DataFrame({'channel_title': ['Test']})

        generate_outreach_drafts(mock_model, df, "")

        call_args = mock_model.generate_content.call_args[0][0]
        assert "my audience's interests" in call_args

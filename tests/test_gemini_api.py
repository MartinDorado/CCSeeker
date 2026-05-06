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
    generate_seed_query,
    _build_relevance_prompt,
    _parse_query_alternatives,
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


# ============================================================================
# generate_seed_query TESTS
# ============================================================================

class TestGenerateSeedQuery:
    """Tests for Gemini-powered seed query generation (returns list of alternatives)."""

    _NUMBERED = "1. \"vegan cooking\", plant-based\n2. healthy recipes, vegan food\n3. \"plant-based diet\", nutrition"

    def _call(self, mock_model, **kwargs):
        defaults = dict(
            channel_description="A channel about vegan cooking and healthy plant-based recipes.",
            recent_titles=["Vegan tacos", "Best smoothies", "Plant-based meal prep"],
            topic_categories=["Food & Drink"],
            channel_keywords=["vegan", "plant-based"],
            language="en",
            gemini_model=mock_model,
        )
        defaults.update(kwargs)
        return generate_seed_query(**defaults)

    def test_returns_list_of_alternatives(self, mock_model):
        """Returns a list with all valid alternatives parsed from numbered output."""
        mock_model.generate_content.return_value = Mock(text=self._NUMBERED)
        result = self._call(mock_model)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == '"vegan cooking", plant-based'
        assert result[1] == "healthy recipes, vegan food"
        assert result[2] == '"plant-based diet", nutrition'

    def test_none_model_returns_empty_list(self):
        """None model returns empty list without calling Gemini."""
        result = generate_seed_query(
            channel_description="desc",
            recent_titles=[],
            topic_categories=[],
            channel_keywords=[],
            language="en",
            gemini_model=None,
        )
        assert result == []

    def test_api_exception_returns_empty_list(self, mock_model):
        """Exception from Gemini returns empty list (triggers NLP fallback)."""
        mock_model.generate_content.side_effect = Exception("Rate limit")
        assert self._call(mock_model) == []

    def test_invalid_alternatives_excluded(self, mock_model):
        """Candidates with disallowed characters are dropped; valid ones kept."""
        mock_model.generate_content.return_value = Mock(
            text='1. vegan cooking\n2. <script>alert(1)</script>\n3. plant-based'
        )
        result = self._call(mock_model)
        assert "vegan cooking" in result
        assert "plant-based" in result
        assert not any("<script>" in r for r in result)

    def test_all_invalid_returns_empty_list(self, mock_model):
        """All invalid candidates → empty list."""
        mock_model.generate_content.return_value = Mock(
            text='1. <bad>\n2. {injection}\n3. ' + 'x' * 121
        )
        assert self._call(mock_model) == []

    def test_count_respected(self, mock_model):
        """count parameter controls how many alternatives are requested and returned."""
        mock_model.generate_content.return_value = Mock(text=self._NUMBERED)
        result = generate_seed_query(
            channel_description="vegan channel",
            recent_titles=[],
            topic_categories=[],
            channel_keywords=[],
            language="en",
            gemini_model=mock_model,
            count=2,
        )
        assert len(result) <= 2

    def test_markdown_code_block_stripped(self, mock_model):
        """Markdown code block wrapping is stripped before parsing."""
        mock_model.generate_content.return_value = Mock(
            text='```\n1. vegan cooking\n2. plant-based\n```'
        )
        result = self._call(mock_model)
        assert "vegan cooking" in result
        assert "plant-based" in result

    def test_xml_delimiters_present_in_prompt(self, mock_model):
        """Channel data is wrapped in XML tags in the prompt."""
        mock_model.generate_content.return_value = Mock(text="1. vegan")
        self._call(mock_model, channel_description="test description")
        prompt = mock_model.generate_content.call_args[0][0]
        assert "<channel_description>" in prompt
        assert "</channel_description>" in prompt
        assert "test description" in prompt

    def test_description_truncated_to_300_chars(self, mock_model):
        """Long descriptions are truncated to 300 chars in the prompt."""
        mock_model.generate_content.return_value = Mock(text="1. vegan")
        self._call(mock_model, channel_description="x" * 500)
        prompt = mock_model.generate_content.call_args[0][0]
        assert "x" * 300 in prompt
        assert "x" * 301 not in prompt

    def test_only_first_5_titles_used(self, mock_model):
        """Only first 5 recent titles are included in the prompt."""
        mock_model.generate_content.return_value = Mock(text="1. cooking")
        self._call(mock_model, recent_titles=[f"Title {i}" for i in range(10)])
        prompt = mock_model.generate_content.call_args[0][0]
        assert "Title 4" in prompt
        assert "Title 5" not in prompt

    def test_empty_fields_do_not_crash(self, mock_model):
        """Empty optional fields produce a valid prompt and return list."""
        mock_model.generate_content.return_value = Mock(text="1. cooking")
        result = generate_seed_query(
            channel_description="",
            recent_titles=[],
            topic_categories=[],
            channel_keywords=[],
            language="en",
            gemini_model=mock_model,
        )
        assert result == ["cooking"]


class TestParseQueryAlternatives:
    """Tests for the _parse_query_alternatives helper."""

    def test_numbered_format_parsed(self):
        text = "1. vegan cooking\n2. plant-based recipes\n3. healthy food"
        assert _parse_query_alternatives(text, 3) == [
            "vegan cooking", "plant-based recipes", "healthy food"
        ]

    def test_count_limits_results(self):
        text = "1. vegan\n2. plant-based\n3. healthy"
        assert len(_parse_query_alternatives(text, 2)) == 2

    def test_invalid_lines_skipped(self):
        text = "1. vegan cooking\n2. <bad>\n3. plant-based"
        result = _parse_query_alternatives(text, 3)
        assert "vegan cooking" in result
        assert "plant-based" in result
        assert not any("<" in r for r in result)

    def test_fallback_to_plain_lines(self):
        """When no numbered items found, falls back to plain line splitting."""
        text = "vegan cooking\nplant-based recipes"
        result = _parse_query_alternatives(text, 3)
        assert "vegan cooking" in result
        assert "plant-based recipes" in result

    def test_empty_input_returns_empty(self):
        assert _parse_query_alternatives("", 3) == []


# ============================================================================
# _build_relevance_prompt / enriched prompt TESTS
# ============================================================================

class TestBuildRelevancePrompt:
    """Tests for the enriched keyword-mode relevance prompt builder."""

    def _base_channel_data(self, **kwargs):
        data = {
            'channel_title': 'Test Channel',
            'video_titles': ['Video 1', 'Video 2'],
        }
        data.update(kwargs)
        return data

    def test_channel_description_included_when_present(self):
        """Channel description appears in prompt when provided."""
        cd = self._base_channel_data(channel_description='We cover DIY home projects.')
        prompt = _build_relevance_prompt(cd, 'DIY')
        assert 'We cover DIY home projects.' in prompt

    def test_channel_description_omitted_when_empty(self):
        """Description section is omitted when channel_description is empty."""
        cd = self._base_channel_data(channel_description='')
        prompt = _build_relevance_prompt(cd, 'DIY')
        assert 'Description:' not in prompt

    def test_topic_categories_included_when_present(self):
        """Topic categories appear in prompt when provided."""
        cd = self._base_channel_data(topic_categories=['Cooking', 'Food'])
        prompt = _build_relevance_prompt(cd, 'cooking')
        assert 'Cooking' in prompt
        assert 'Food' in prompt

    def test_topic_categories_omitted_when_empty(self):
        """Topic section is omitted when topic_categories is empty."""
        cd = self._base_channel_data(topic_categories=[])
        prompt = _build_relevance_prompt(cd, 'test')
        assert 'topic categories' not in prompt.lower()

    def test_channel_keywords_included_when_present(self):
        """Channel keywords appear in prompt when provided."""
        cd = self._base_channel_data(channel_keywords=['diy', 'woodworking', 'tools'])
        prompt = _build_relevance_prompt(cd, 'woodworking')
        assert 'woodworking' in prompt

    def test_channel_keywords_omitted_when_missing(self):
        """Channel keywords section is omitted when field is absent."""
        cd = self._base_channel_data()  # no channel_keywords key
        prompt = _build_relevance_prompt(cd, 'test')
        assert 'Channel keywords' not in prompt

    def test_video_description_excerpt_appended_to_title(self):
        """Video description excerpt is appended to the title line when available."""
        cd = self._base_channel_data(
            video_descriptions=['How to make bread at home', ''],
        )
        prompt = _build_relevance_prompt(cd, 'baking')
        assert 'How to make bread at home' in prompt

    def test_video_description_omitted_when_empty_string(self):
        """Empty description excerpt is not appended."""
        cd = self._base_channel_data(
            video_titles=['Title Only'],
            video_descriptions=[''],
        )
        prompt = _build_relevance_prompt(cd, 'test')
        assert 'Title Only —' not in prompt  # separator not added for empty desc

    def test_prompt_under_3kb_with_large_inputs(self):
        """Prompt is capped at ~3 KB even with very long description."""
        long_desc = 'x' * 2000
        cd = self._base_channel_data(
            channel_description=long_desc,
            topic_categories=['Topic A', 'Topic B'],
            channel_keywords=['kw'] * 20,
            video_descriptions=['A long description excerpt. ' * 5] * 10,
        )
        prompt = _build_relevance_prompt(cd, 'query')
        assert len(prompt) <= 3500  # generous headroom above 3 KB limit

    def test_backward_compatible_with_titles_only(self, mock_model):
        """Old callers that only pass channel_title + video_titles still work."""
        mock_model.generate_content.return_value = Mock(text="7")
        cd = {'channel_title': 'OldStyle', 'video_titles': ['V1', 'V2']}

        score = generate_ai_relevance_score(mock_model, cd, 'query')

        assert 0.0 <= score <= 1.0

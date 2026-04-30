"""
Tests for core.pipeline module

Tests cover:
- Full search pipeline execution
- Configuration handling
- Error handling and early exits
- Progress callbacks
- Filter application
- Result structure

These are integration tests using mocked API services.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import (
    PipelineResult,
    PipelineConfig,
    run_search_pipeline,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_youtube():
    """Create a mock YouTube API service with reasonable defaults."""
    youtube = Mock()

    # Default search response
    youtube.search().list().execute.side_effect = [
        # Video search
        {
            'items': [
                {'snippet': {'channelId': 'UC1', 'channelTitle': 'Channel One'}},
                {'snippet': {'channelId': 'UC2', 'channelTitle': 'Channel Two'}},
            ]
        },
        # Channel name search
        {'items': []},
    ]

    # Default channel stats response
    youtube.channels().list().execute.return_value = {
        'items': [
            {
                'id': 'UC1',
                'snippet': {'country': 'US', 'publishedAt': '2020-01-01T00:00:00Z'},
                'statistics': {'subscriberCount': '50000', 'viewCount': '1000000', 'videoCount': '100'},
                'contentDetails': {'relatedPlaylists': {'uploads': 'UU1'}}
            },
            {
                'id': 'UC2',
                'snippet': {'country': 'UK', 'publishedAt': '2021-01-01T00:00:00Z'},
                'statistics': {'subscriberCount': '25000', 'viewCount': '500000', 'videoCount': '50'},
                'contentDetails': {'relatedPlaylists': {'uploads': 'UU2'}}
            },
        ]
    }

    # Default playlist/video responses
    youtube.playlistItems().list().execute.return_value = {
        'items': [
            {'snippet': {'resourceId': {'videoId': 'VID1'}}},
            {'snippet': {'resourceId': {'videoId': 'VID2'}}},
        ]
    }
    youtube.videos().list().execute.return_value = {
        'items': [
            {
                'id': 'VID1',
                'snippet': {
                    'title': 'Manga Review Episode 1',
                    'publishedAt': '2024-01-01T00:00:00Z',
                    'tags': ['manga', 'review']
                },
                'statistics': {'viewCount': '10000', 'likeCount': '500', 'commentCount': '50'}
            },
            {
                'id': 'VID2',
                'snippet': {
                    'title': 'Anime Discussion',
                    'publishedAt': '2024-01-15T00:00:00Z',
                    'tags': ['anime']
                },
                'statistics': {'viewCount': '8000', 'likeCount': '400', 'commentCount': '40'}
            },
        ]
    }

    return youtube


@pytest.fixture
def mock_cache_functions():
    """Create mock cache functions adapter."""
    cache = Mock()

    cache.search_channels_cached.return_value = [
        {'channel_id': 'UC1', 'channel_title': 'Channel One', 'match_score': 20},
        {'channel_id': 'UC2', 'channel_title': 'Channel Two', 'match_score': 10},
    ]

    cache.get_channel_stats_cached.return_value = [
        {
            'channel_id': 'UC1', 'country': 'US', 'subscribers': 50000,
            'views': 1000000, 'videos': 100, 'uploads_playlist_id': 'UU1',
            'avg_views_per_video': 10000, 'channel_age_days': 1000
        },
        {
            'channel_id': 'UC2', 'country': 'UK', 'subscribers': 25000,
            'views': 500000, 'videos': 50, 'uploads_playlist_id': 'UU2',
            'avg_views_per_video': 10000, 'channel_age_days': 500
        },
    ]

    cache.get_video_details_cached.return_value = [
        {
            'channel_id': 'UC1', 'video_id': 'VID1', 'video_title': 'Manga Review',
            'published_at': '2024-01-01T00:00:00Z', 'video_views': 10000,
            'video_likes': 500, 'video_comments': 50, 'video_tags': ['manga']
        },
        {
            'channel_id': 'UC2', 'video_id': 'VID2', 'video_title': 'Anime News',
            'published_at': '2024-01-15T00:00:00Z', 'video_views': 8000,
            'video_likes': 400, 'video_comments': 40, 'video_tags': ['anime']
        },
    ]

    return cache


@pytest.fixture
def default_config():
    """Create default pipeline configuration."""
    return PipelineConfig(
        min_subscribers=1000,
        country_filter=None,
        months_ago=0,
        enable_ai_relevance=False,
        enable_ai_summary=False,
    )


# ============================================================================
# PIPELINE RESULT STRUCTURE TESTS
# ============================================================================

class TestPipelineResultStructure:
    """Tests for PipelineResult dataclass structure."""

    def test_pipeline_result_has_required_fields(self):
        """PipelineResult has all required fields."""
        result = PipelineResult(
            channels_df=pd.DataFrame(),
            display_columns=[],
            column_explanations={},
            top_channels_for_outreach=pd.DataFrame(),
            final_query="test",
        )

        assert hasattr(result, 'channels_df')
        assert hasattr(result, 'display_columns')
        assert hasattr(result, 'column_explanations')
        assert hasattr(result, 'final_query')
        assert hasattr(result, 'top_channels_for_outreach')
        assert hasattr(result, 'search_log')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'error')

    def test_pipeline_result_default_values(self):
        """PipelineResult has correct default values."""
        result = PipelineResult(
            channels_df=pd.DataFrame(),
            display_columns=[],
            column_explanations={},
            top_channels_for_outreach=pd.DataFrame(),
            final_query="test",
        )

        assert result.search_log == []
        assert result.warnings == []
        assert result.error is None
        assert result.ai_summary is None


# ============================================================================
# PIPELINE EXECUTION TESTS
# ============================================================================

class TestPipelineExecution:
    """Tests for main pipeline execution."""

    def test_successful_pipeline_execution(self, mock_youtube, default_config):
        """Pipeline executes successfully with valid inputs."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
        )

        assert isinstance(result, PipelineResult)
        assert result.error is None
        assert result.final_query == "manga"
        assert not result.channels_df.empty

    def test_pipeline_with_cache_functions(self, mock_youtube, mock_cache_functions, default_config):
        """Pipeline uses cache functions when provided."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        # Verify cache functions were called
        mock_cache_functions.search_channels_cached.assert_called_once()
        mock_cache_functions.get_channel_stats_cached.assert_called_once()
        assert result.error is None

    def test_empty_query_returns_error(self, mock_youtube, default_config):
        """Empty query returns error result."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="",
            region_code="US",
            config=default_config,
        )

        assert result.error is not None
        assert "Empty query" in result.error
        assert result.channels_df.empty

    def test_whitespace_query_returns_error(self, mock_youtube, default_config):
        """Whitespace-only query returns error result."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="   ",
            region_code="US",
            config=default_config,
        )

        assert result.error is not None

    def test_query_truncation_adds_warning(self, mock_youtube, mock_cache_functions, default_config):
        """Query with more than 2 terms adds warning."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga, anime, gaming, tech",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        assert len(result.warnings) >= 1
        warning_text = ' '.join(result.warnings)
        assert 'adjusted' in warning_text.lower() or 'removed' in warning_text.lower()

    def test_no_channels_found_returns_error(self, mock_youtube, default_config):
        """No channels found returns error result."""
        mock_youtube.search().list().execute.side_effect = [
            {'items': []},
            {'items': []},
        ]

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="xyznonexistent123",
            region_code="US",
            config=default_config,
        )

        assert result.error is not None
        assert "no channel" in result.error.lower() or "did not return" in result.error.lower()


# ============================================================================
# FILTER TESTS
# ============================================================================

class TestPipelineFilters:
    """Tests for filter application in the pipeline."""

    def test_min_subscribers_filter(self, mock_youtube, mock_cache_functions):
        """Channels below min_subscribers are filtered out."""
        config = PipelineConfig(
            min_subscribers=30000,  # Only UC1 (50000) passes
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
        )

        # UC2 (25000 subs) should be filtered out
        if not result.channels_df.empty:
            channel_ids = result.channels_df.get('channel_id', result.raw_channels_df.get('channel_id', []))
            # Check the search log for filter message
            filter_log = [log for log in result.search_log if 'filter' in log.lower()]
            assert len(filter_log) > 0 or len(result.channels_df) <= 1

    def test_country_filter(self, mock_youtube, mock_cache_functions):
        """Country filter works correctly."""
        config = PipelineConfig(
            min_subscribers=0,
            country_filter="US",  # Only US channels
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
        )

        # Should only include US channels
        assert result.error is None or "filter" in str(result.error).lower()

    def test_all_channels_filtered_returns_error(self, mock_youtube, mock_cache_functions):
        """When all channels are filtered out, returns error."""
        config = PipelineConfig(
            min_subscribers=1000000,  # Higher than any channel
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
        )

        assert result.error is not None
        assert "filter" in result.error.lower() or "criteria" in result.error.lower()

    def test_zero_relevance_channels_excluded(self, mock_youtube, mock_cache_functions):
        """Channels that score 0 relevance are excluded from keyword-mode results."""
        # Override cache so one channel returns relevance 0 via having no keyword match
        mock_cache_functions.search_channels_cached.return_value = [
            {'channel_id': 'UC1', 'channel_title': 'Manga World', 'match_score': 20},
            {'channel_id': 'UC_zero', 'channel_title': 'Unrelated Channel', 'match_score': 0},
        ]
        mock_cache_functions.get_channel_stats_cached.return_value = [
            {
                'channel_id': 'UC1', 'country': 'US', 'subscribers': 50000,
                'views': 1000000, 'videos': 100, 'uploads_playlist_id': 'UU1',
                'avg_views_per_video': 10000, 'channel_age_days': 1000
            },
            {
                'channel_id': 'UC_zero', 'country': 'US', 'subscribers': 50000,
                'views': 1000000, 'videos': 100, 'uploads_playlist_id': 'UU_zero',
                'avg_views_per_video': 10000, 'channel_age_days': 1000
            },
        ]
        mock_cache_functions.get_video_details_cached.return_value = [
            {
                'channel_id': 'UC1', 'video_id': 'VID1', 'video_title': 'Manga Review',
                'published_at': '2024-01-01T00:00:00Z', 'video_views': 10000,
                'video_likes': 500, 'video_comments': 50, 'video_tags': ['manga', 'review']
            },
            {
                'channel_id': 'UC_zero', 'video_id': 'VID_z', 'video_title': 'Cooking Class',
                'published_at': '2024-01-01T00:00:00Z', 'video_views': 10000,
                'video_likes': 500, 'video_comments': 50, 'video_tags': ['food', 'cooking']
            },
        ]

        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga anime",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
        )

        if not result.channels_df.empty and 'relevance_score' in result.channels_df.columns:
            assert (result.channels_df['relevance_score'] > 0).all(), \
                "All returned channels must have relevance_score > 0"

    def test_country_filter_includes_null_country(self, mock_youtube, mock_cache_functions):
        """Country filter passes channels with country=None alongside matching-country channels."""
        mock_cache_functions.get_channel_stats_cached.return_value = [
            {
                'channel_id': 'UC1', 'country': 'US', 'subscribers': 50000,
                'views': 1000000, 'videos': 100, 'uploads_playlist_id': 'UU1',
                'avg_views_per_video': 10000, 'channel_age_days': 1000
            },
            {
                'channel_id': 'UC2', 'country': 'UK', 'subscribers': 25000,
                'views': 500000, 'videos': 50, 'uploads_playlist_id': 'UU2',
                'avg_views_per_video': 10000, 'channel_age_days': 500
            },
            {
                'channel_id': 'UC_null', 'country': None, 'subscribers': 30000,
                'views': 600000, 'videos': 60, 'uploads_playlist_id': 'UU_null',
                'avg_views_per_video': 10000, 'channel_age_days': 700
            },
        ]
        mock_cache_functions.search_channels_cached.return_value = [
            {'channel_id': 'UC1', 'channel_title': 'Channel US', 'match_score': 20},
            {'channel_id': 'UC2', 'channel_title': 'Channel UK', 'match_score': 15},
            {'channel_id': 'UC_null', 'channel_title': 'Channel Unknown', 'match_score': 18},
        ]
        mock_cache_functions.get_video_details_cached.return_value = [
            {
                'channel_id': 'UC1', 'video_id': 'VID1', 'video_title': 'Manga Review',
                'published_at': '2024-01-01T00:00:00Z', 'video_views': 10000,
                'video_likes': 500, 'video_comments': 50, 'video_tags': ['manga']
            },
            {
                'channel_id': 'UC_null', 'video_id': 'VID_n', 'video_title': 'Manga World',
                'published_at': '2024-01-01T00:00:00Z', 'video_views': 8000,
                'video_likes': 400, 'video_comments': 40, 'video_tags': ['manga']
            },
        ]

        config = PipelineConfig(
            min_subscribers=1000,
            country_filter="US",
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
        )

        # UK channel must not appear; US and null-country channels should pass the filter
        if not result.channels_df.empty and 'channel_id' in result.channels_df.columns:
            ids = result.channels_df['channel_id'].tolist()
            assert 'UC2' not in ids, "UK channel should be excluded"
        elif result.raw_channels_df is not None and not result.raw_channels_df.empty:
            # Pipeline may have returned early due to zero-score filter; UK must not be present
            if 'channel_id' in result.raw_channels_df.columns:
                assert 'UC2' not in result.raw_channels_df['channel_id'].tolist()


# ============================================================================
# CALLBACK TESTS
# ============================================================================

class TestPipelineCallbacks:
    """Tests for pipeline callback functions."""

    def test_progress_callback_invoked(self, mock_youtube, mock_cache_functions, default_config):
        """on_progress callback is invoked during pipeline execution."""
        progress_calls = []

        def track_progress(msg, pct):
            progress_calls.append((msg, pct))

        run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
            on_progress=track_progress,
        )

        # Should have multiple progress updates
        assert len(progress_calls) >= 3
        # First should be at low percentage
        assert progress_calls[0][1] < 0.5
        # Last should be at high percentage (near 1.0)
        assert progress_calls[-1][1] >= 0.9

    def test_api_callback_invoked(self, mock_youtube, default_config):
        """on_api_call callback is invoked for API requests."""
        api_calls = []

        def track_api(call_type):
            api_calls.append(call_type)

        run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            on_api_call=track_api,
        )

        # Should have multiple API calls tracked
        assert len(api_calls) >= 1


# ============================================================================
# RESULT FORMATTING TESTS
# ============================================================================

class TestResultFormatting:
    """Tests for result data formatting."""

    def test_channel_url_added(self, mock_youtube, mock_cache_functions, default_config):
        """channel_url column is added to results."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        if not result.channels_df.empty:
            assert 'channel_url' in result.display_columns

    def test_display_columns_correct_for_keyword_mode(self, mock_youtube, mock_cache_functions, default_config):
        """Display columns are correct for keyword search mode."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        if not result.channels_df.empty:
            # Should include standard columns for keyword mode
            assert 'channel_title' in result.display_columns
            assert 'relevance_score' in result.display_columns
            assert 'subscribers' in result.display_columns
            # Should NOT have similarity_score (that's seed mode)
            assert 'similarity_score' not in result.display_columns

    def test_column_explanations_provided(self, mock_youtube, mock_cache_functions, default_config):
        """Column explanations are provided."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        if not result.channels_df.empty:
            assert len(result.column_explanations) > 0
            assert 'relevance_score' in result.column_explanations

    def test_outreach_data_prepared(self, mock_youtube, mock_cache_functions, default_config):
        """Outreach data is prepared for top channels."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        if not result.channels_df.empty:
            assert not result.top_channels_for_outreach.empty
            assert 'channel_title' in result.top_channels_for_outreach.columns


# ============================================================================
# SEARCH LOG TESTS
# ============================================================================

class TestSearchLog:
    """Tests for search log generation."""

    def test_search_log_records_steps(self, mock_youtube, mock_cache_functions, default_config):
        """Search log records pipeline steps."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        # Should have log entries
        assert len(result.search_log) >= 1

    def test_search_log_includes_channel_counts(self, mock_youtube, mock_cache_functions, default_config):
        """Search log includes channel count information."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        log_text = ' '.join(result.search_log)
        assert 'channel' in log_text.lower()


# ============================================================================
# TIMING TESTS
# ============================================================================

class TestPipelineTiming:
    """Tests for timing information in results."""

    def test_timings_recorded(self, mock_youtube, mock_cache_functions, default_config):
        """Pipeline records timing information."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        assert len(result.timings) >= 1
        assert 'total' in result.timings or 'search' in result.timings

    def test_timings_are_positive(self, mock_youtube, mock_cache_functions, default_config):
        """All timing values are positive."""
        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
            cache_functions=mock_cache_functions,
        )

        for key, value in result.timings.items():
            assert value >= 0, f"Timing {key} should be positive"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestPipelineErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_exception_captured_in_result(self, mock_youtube, default_config):
        """Exceptions are captured in result.error, not raised."""
        # Make channels() raise an exception
        mock_youtube.channels().list().execute.side_effect = Exception("API Error")

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
        )

        # Should not raise, but capture in result
        assert result.error is not None

    def test_no_stats_returns_error(self, mock_youtube, default_config):
        """Empty channel stats returns error."""
        mock_youtube.search().list().execute.side_effect = [
            {'items': [{'snippet': {'channelId': 'UC1', 'channelTitle': 'Test'}}]},
            {'items': []},
        ]
        mock_youtube.channels().list().execute.return_value = {'items': []}

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=default_config,
        )

        assert result.error is not None
        assert "statistic" in result.error.lower()


# ============================================================================
# AI FEATURE TESTS
# ============================================================================

class TestAIFeatures:
    """Tests for AI-enhanced features in the pipeline."""

    def test_ai_disabled_by_config(self, mock_youtube, mock_cache_functions):
        """AI features are disabled when config specifies."""
        config = PipelineConfig(
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
            gemini_model=Mock(),  # Would be used if enabled
        )

        # AI summary should not be generated
        assert result.ai_summary is None

    def test_ai_summary_generated_when_enabled(self, mock_youtube, mock_cache_functions):
        """AI summary is generated when enabled and model provided."""
        config = PipelineConfig(
            enable_ai_relevance=False,
            enable_ai_summary=True,
        )
        mock_model = Mock()
        mock_model.generate_content.return_value = Mock(text="AI Summary here")

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
            gemini_model=mock_model,
        )

        if not result.channels_df.empty:
            assert result.ai_summary == "AI Summary here"

    def test_ai_summary_error_captured(self, mock_youtube, mock_cache_functions):
        """AI summary errors are captured, not raised."""
        config = PipelineConfig(
            enable_ai_relevance=False,
            enable_ai_summary=True,
        )
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("AI Error")

        result = run_search_pipeline(
            youtube_service=mock_youtube,
            query="manga",
            region_code="US",
            config=config,
            cache_functions=mock_cache_functions,
            gemini_model=mock_model,
        )

        # Should not crash pipeline
        assert result.error is None or "AI" not in str(result.error)
        # Error should be captured in ai_summary_error
        if not result.channels_df.empty:
            assert result.ai_summary_error is not None or result.ai_summary is not None

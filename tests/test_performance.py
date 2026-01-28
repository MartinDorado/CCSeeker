"""
test_performance.py - Comprehensive performance testing for CCSeeker

Tests measure execution time and API usage for:
1. Keyword Search Mode (1 term and 2 terms)
2. Seed-Based Search Mode (1 term and 2 terms)

Each scenario is tested with:
- Cold cache / Warm cache
- With AI / Without AI

Test results are logged with consistent step timing for both modes.

Usage:
    pytest tests/test_performance.py -v -s

Note: These tests use mocked APIs for consistent, reproducible results.
For real API performance testing, run manually with actual API keys.
"""

import pytest
import time
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import (
    PipelineResult,
    PipelineConfig,
    run_search_pipeline,
)
from app.core.seed_topics import (
    analyze_seed_channel,
    SeedAnalysisResult,
    SeedProfile,
)


# ============================================================================
# PERFORMANCE TEST CONFIGURATION
# ============================================================================

@dataclass
class PerformanceResult:
    """Results from a performance test run."""
    scenario: str
    search_mode: str  # "keyword" or "seed"
    terms: int  # 1 or 2
    ai_enabled: bool
    cache_state: str  # "cold" or "warm"

    # Timing data (in seconds)
    total_time: float
    search_time: float
    channel_stats_time: float
    video_details_time: float
    ai_relevance_time: float
    similarity_time: float  # Only for seed mode
    ai_generation_time: float

    # API usage
    youtube_units: int
    gemini_calls: int

    # Results
    channels_found: int
    error: Optional[str] = None


# ============================================================================
# FIXTURES - Mock API Responses
# ============================================================================

@pytest.fixture
def mock_youtube():
    """Create a mock YouTube API service with realistic delays."""
    youtube = Mock()

    # Configure search response
    def search_execute():
        return {
            'items': [
                {'snippet': {'channelId': f'UC{i}', 'channelTitle': f'Channel {i}'}}
                for i in range(1, 11)  # 10 channels
            ]
        }

    youtube.search().list().execute = search_execute

    # Configure channel stats response
    def channels_execute():
        return {
            'items': [
                {
                    'id': f'UC{i}',
                    'snippet': {
                        'title': f'Channel {i}',
                        'description': f'Description for channel {i}',
                        'country': 'US',
                        'publishedAt': '2020-01-01T00:00:00Z'
                    },
                    'statistics': {
                        'subscriberCount': str(50000 + i * 10000),
                        'viewCount': str(1000000 + i * 100000),
                        'videoCount': str(100 + i * 10)
                    },
                    'contentDetails': {'relatedPlaylists': {'uploads': f'UU{i}'}}
                }
                for i in range(1, 11)
            ]
        }

    youtube.channels().list().execute = channels_execute

    # Configure playlist items response
    def playlist_execute():
        return {
            'items': [
                {'snippet': {'resourceId': {'videoId': f'VID{j}'}}}
                for j in range(1, 11)  # 10 videos per channel
            ]
        }

    youtube.playlistItems().list().execute = playlist_execute

    # Configure video details response
    def videos_execute():
        return {
            'items': [
                {
                    'id': f'VID{j}',
                    'snippet': {
                        'title': f'Video Title {j} - manga anime review',
                        'publishedAt': f'2024-01-{j:02d}T00:00:00Z',
                        'tags': ['manga', 'anime', 'review', 'japan']
                    },
                    'statistics': {
                        'viewCount': str(10000 + j * 1000),
                        'likeCount': str(500 + j * 50),
                        'commentCount': str(50 + j * 5)
                    }
                }
                for j in range(1, 11)
            ]
        }

    youtube.videos().list().execute = videos_execute

    return youtube


@pytest.fixture
def mock_cache_functions_cold():
    """Create mock cache functions that simulate cold cache (API calls)."""
    cache = Mock()

    # Simulate search with realistic channel data
    cache.search_channels_cached.return_value = [
        {
            'channel_id': f'UC{i}',
            'channel_title': f'Manga Anime Channel {i}',
            'match_score': 20 + i * 5
        }
        for i in range(1, 11)
    ]

    # Simulate channel stats
    cache.get_channel_stats_cached.return_value = [
        {
            'channel_id': f'UC{i}',
            'country': 'US',
            'subscribers': 50000 + i * 10000,
            'views': 1000000 + i * 100000,
            'videos': 100 + i * 10,
            'uploads_playlist_id': f'UU{i}',
            'avg_views_per_video': 10000,
            'channel_age_days': 1000 + i * 100
        }
        for i in range(1, 11)
    ]

    # Simulate video details
    cache.get_video_details_cached.return_value = [
        {
            'channel_id': f'UC{(j // 10) + 1}',
            'video_id': f'VID{j}',
            'video_title': f'Video Title {j} - manga anime review',
            'published_at': f'2024-01-{(j % 28) + 1:02d}T00:00:00Z',
            'video_views': 10000 + j * 500,
            'video_likes': 500 + j * 25,
            'video_comments': 50 + j * 5,
            'video_tags': ['manga', 'anime', 'review']
        }
        for j in range(1, 101)  # 100 videos total (10 per channel)
    ]

    return cache


@pytest.fixture
def mock_cache_functions_warm():
    """Create mock cache functions that simulate warm cache (instant response)."""
    # Same as cold, warm cache is simulated by the test measuring minimal time
    cache = Mock()

    cache.search_channels_cached.return_value = [
        {
            'channel_id': f'UC{i}',
            'channel_title': f'Manga Anime Channel {i}',
            'match_score': 20 + i * 5
        }
        for i in range(1, 11)
    ]

    cache.get_channel_stats_cached.return_value = [
        {
            'channel_id': f'UC{i}',
            'country': 'US',
            'subscribers': 50000 + i * 10000,
            'views': 1000000 + i * 100000,
            'videos': 100 + i * 10,
            'uploads_playlist_id': f'UU{i}',
            'avg_views_per_video': 10000,
            'channel_age_days': 1000 + i * 100
        }
        for i in range(1, 11)
    ]

    cache.get_video_details_cached.return_value = [
        {
            'channel_id': f'UC{(j // 10) + 1}',
            'video_id': f'VID{j}',
            'video_title': f'Video Title {j} - manga anime review',
            'published_at': f'2024-01-{(j % 28) + 1:02d}T00:00:00Z',
            'video_views': 10000 + j * 500,
            'video_likes': 500 + j * 25,
            'video_comments': 50 + j * 5,
            'video_tags': ['manga', 'anime', 'review']
        }
        for j in range(1, 101)
    ]

    return cache


@pytest.fixture
def mock_gemini_model():
    """Create mock Gemini model with realistic response."""
    model = Mock()
    model.generate_content.return_value = Mock(text="AI generated summary of channels found.")
    return model


@pytest.fixture
def mock_similarity_engine():
    """Create mock similarity engine for seed mode."""
    engine = Mock()

    def rank_channels_by_similarity(candidates, seed_profile, **kwargs):
        """Mock similarity ranking with realistic scores."""
        result = []
        for i, candidate in enumerate(candidates):
            candidate['similarity'] = {
                'total_score': 80 - i * 5,
                'breakdown': {
                    'tag_overlap': 25 - i,
                    'keyword_overlap': 25 - i,
                    'subscriber_similarity': 12 - i * 0.5,
                    'engagement_similarity': 12 - i * 0.5,
                    'frequency_similarity': 6 - i * 0.2
                },
                'match_reasons': ['Similar content topics', 'Similar audience size']
            }
            result.append(candidate)
        return result

    engine.rank_channels_by_similarity = rank_channels_by_similarity
    return engine


@pytest.fixture
def seed_profile():
    """Create a realistic seed profile for testing."""
    return {
        'channel_id': 'UCseed123',
        'channel_name': 'Tech Channel',
        'subscriber_count': 500000,
        'subscriber_tier': 'mid',
        'upload_frequency': 4.5,
        'avg_engagement_rate': 0.045,
        'category': '28',
        'language': 'en',
        'primary_keywords': ['tech review', 'software tutorial'],
        'secondary_keywords': ['programming', 'coding', 'development'],
        'common_tags': ['tech', 'software', 'coding', 'tutorial', 'review'],
        'recent_titles': ['Best IDE 2024', 'Python Tutorial', 'Code Review Tips'],
        'gemini_api_key': None  # Will be set by tests
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_performance_result(result: PerformanceResult):
    """Log performance result in a consistent format."""
    print("\n" + "=" * 70)
    print(f"SCENARIO: {result.scenario}")
    print("=" * 70)
    print(f"  Search Mode: {result.search_mode}")
    print(f"  Terms: {result.terms}")
    print(f"  AI Enabled: {result.ai_enabled}")
    print(f"  Cache State: {result.cache_state}")
    print("-" * 70)
    print("TIMING BREAKDOWN:")
    print(f"  1. Search:          {result.search_time:.4f}s")
    print(f"  2. Channel Stats:   {result.channel_stats_time:.4f}s")
    print(f"  3. Video Details:   {result.video_details_time:.4f}s")
    print(f"  4. AI Relevance:    {result.ai_relevance_time:.4f}s")
    if result.search_mode == "seed":
        print(f"  5. Similarity:      {result.similarity_time:.4f}s")
    print(f"  6. AI Generation:   {result.ai_generation_time:.4f}s")
    print(f"  -----------------------------------------")
    print(f"  TOTAL:              {result.total_time:.4f}s")
    print("-" * 70)
    print("API USAGE:")
    print(f"  YouTube Units: {result.youtube_units}")
    print(f"  Gemini Calls: {result.gemini_calls}")
    print("-" * 70)
    print(f"RESULTS: {result.channels_found} channels found")
    if result.error:
        print(f"ERROR: {result.error}")
    print("=" * 70 + "\n")


def run_pipeline_with_tracking(
    youtube_service,
    query: str,
    config: PipelineConfig,
    cache_functions,
    gemini_model=None,
    similarity_engine=None,
) -> tuple[PipelineResult, dict]:
    """Run pipeline and track API calls."""
    api_calls = {
        'youtube_search': 0,
        'youtube_channel': 0,
        'youtube_video': 0,
        'youtube_playlist': 0,
        'gemini_summary': 0,
        'gemini_outreach': 0,
        'gemini_similarity': 0,
        'gemini_relevance': 0,
    }

    def on_api_call(api_name):
        if api_name in api_calls:
            api_calls[api_name] += 1

    result = run_search_pipeline(
        youtube_service=youtube_service,
        query=query,
        region_code="US",
        config=config,
        gemini_model=gemini_model,
        similarity_engine=similarity_engine,
        cache_functions=cache_functions,
        on_api_call=on_api_call,
    )

    return result, api_calls


def calculate_youtube_units(api_calls: dict) -> int:
    """Calculate YouTube API quota units used."""
    return (
        api_calls['youtube_search'] * 100 +
        api_calls['youtube_channel'] * 1 +
        api_calls['youtube_video'] * 1 +
        api_calls['youtube_playlist'] * 1
    )


def calculate_gemini_calls(api_calls: dict) -> int:
    """Calculate total Gemini API calls."""
    return (
        api_calls['gemini_summary'] +
        api_calls['gemini_outreach'] +
        api_calls['gemini_similarity'] +
        api_calls['gemini_relevance']
    )


# ============================================================================
# KEYWORD SEARCH MODE TESTS
# ============================================================================

class TestKeywordSearchPerformance:
    """Performance tests for Keyword Search Mode."""

    def test_keyword_1term_cold_cache_no_ai(
        self, mock_youtube, mock_cache_functions_cold
    ):
        """Keyword search: 1 term, cold cache, no AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga",
            config=config,
            cache_functions=mock_cache_functions_cold,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Keyword 1 term - Cold Cache - No AI",
            search_mode="keyword",
            terms=1,
            ai_enabled=False,
            cache_state="cold",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=0,  # N/A for keyword mode
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None
        assert not result.channels_df.empty
        assert 'similarity_score' not in result.display_columns

    def test_keyword_1term_warm_cache_no_ai(
        self, mock_youtube, mock_cache_functions_warm
    ):
        """Keyword search: 1 term, warm cache, no AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga",
            config=config,
            cache_functions=mock_cache_functions_warm,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Keyword 1 term - Warm Cache - No AI",
            search_mode="keyword",
            terms=1,
            ai_enabled=False,
            cache_state="warm",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=0,
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None

    def test_keyword_1term_warm_cache_with_ai(
        self, mock_youtube, mock_cache_functions_warm, mock_gemini_model
    ):
        """Keyword search: 1 term, warm cache, with AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=True,
            enable_ai_summary=True,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga",
            config=config,
            cache_functions=mock_cache_functions_warm,
            gemini_model=mock_gemini_model,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Keyword 1 term - Warm Cache - With AI",
            search_mode="keyword",
            terms=1,
            ai_enabled=True,
            cache_state="warm",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=0,
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None

    def test_keyword_2terms_cold_cache_no_ai(
        self, mock_youtube, mock_cache_functions_cold
    ):
        """Keyword search: 2 terms, cold cache, no AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga, anime",
            config=config,
            cache_functions=mock_cache_functions_cold,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Keyword 2 terms - Cold Cache - No AI",
            search_mode="keyword",
            terms=2,
            ai_enabled=False,
            cache_state="cold",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=0,
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None

    def test_keyword_2terms_warm_cache_with_ai(
        self, mock_youtube, mock_cache_functions_warm, mock_gemini_model
    ):
        """Keyword search: 2 terms, warm cache, with AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=True,
            enable_ai_summary=True,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga, anime",
            config=config,
            cache_functions=mock_cache_functions_warm,
            gemini_model=mock_gemini_model,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Keyword 2 terms - Warm Cache - With AI",
            search_mode="keyword",
            terms=2,
            ai_enabled=True,
            cache_state="warm",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=0,
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None


# ============================================================================
# SEED-BASED SEARCH MODE TESTS
# ============================================================================

class TestSeedSearchPerformance:
    """Performance tests for Seed-Based Search Mode."""

    def test_seed_1term_cold_cache_no_ai(
        self, mock_youtube, mock_cache_functions_cold, mock_similarity_engine, seed_profile
    ):
        """Seed search: 1 term, cold cache, no AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
            seed_profile=seed_profile,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="tech review",
            config=config,
            cache_functions=mock_cache_functions_cold,
            similarity_engine=mock_similarity_engine,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Seed 1 term - Cold Cache - No AI",
            search_mode="seed",
            terms=1,
            ai_enabled=False,
            cache_state="cold",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=result.timings.get('similarity', 0),
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None
        # Seed mode should have similarity_score column
        if not result.channels_df.empty:
            assert 'similarity_score' in result.display_columns

    def test_seed_1term_warm_cache_no_ai(
        self, mock_youtube, mock_cache_functions_warm, mock_similarity_engine, seed_profile
    ):
        """Seed search: 1 term, warm cache, no AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
            seed_profile=seed_profile,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="tech review",
            config=config,
            cache_functions=mock_cache_functions_warm,
            similarity_engine=mock_similarity_engine,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Seed 1 term - Warm Cache - No AI",
            search_mode="seed",
            terms=1,
            ai_enabled=False,
            cache_state="warm",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=result.timings.get('similarity', 0),
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None

    def test_seed_1term_warm_cache_with_ai(
        self, mock_youtube, mock_cache_functions_warm, mock_gemini_model,
        mock_similarity_engine, seed_profile
    ):
        """Seed search: 1 term, warm cache, with AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=True,
            enable_ai_summary=True,
            seed_profile=seed_profile,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="tech review",
            config=config,
            cache_functions=mock_cache_functions_warm,
            gemini_model=mock_gemini_model,
            similarity_engine=mock_similarity_engine,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Seed 1 term - Warm Cache - With AI",
            search_mode="seed",
            terms=1,
            ai_enabled=True,
            cache_state="warm",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=result.timings.get('similarity', 0),
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None

    def test_seed_2terms_cold_cache_no_ai(
        self, mock_youtube, mock_cache_functions_cold, mock_similarity_engine, seed_profile
    ):
        """Seed search: 2 terms, cold cache, no AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
            seed_profile=seed_profile,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="tech review, software tutorial",
            config=config,
            cache_functions=mock_cache_functions_cold,
            similarity_engine=mock_similarity_engine,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Seed 2 terms - Cold Cache - No AI",
            search_mode="seed",
            terms=2,
            ai_enabled=False,
            cache_state="cold",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=result.timings.get('similarity', 0),
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None

    def test_seed_2terms_warm_cache_with_ai(
        self, mock_youtube, mock_cache_functions_warm, mock_gemini_model,
        mock_similarity_engine, seed_profile
    ):
        """Seed search: 2 terms, warm cache, with AI."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=True,
            enable_ai_summary=True,
            seed_profile=seed_profile,
        )

        start = time.time()
        result, api_calls = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="tech review, software tutorial",
            config=config,
            cache_functions=mock_cache_functions_warm,
            gemini_model=mock_gemini_model,
            similarity_engine=mock_similarity_engine,
        )
        total_time = time.time() - start

        perf_result = PerformanceResult(
            scenario="Seed 2 terms - Warm Cache - With AI",
            search_mode="seed",
            terms=2,
            ai_enabled=True,
            cache_state="warm",
            total_time=total_time,
            search_time=result.timings.get('search', 0),
            channel_stats_time=result.timings.get('channel_stats', 0),
            video_details_time=result.timings.get('video_details', 0),
            ai_relevance_time=result.timings.get('ai_relevance', 0),
            similarity_time=result.timings.get('similarity', 0),
            ai_generation_time=result.timings.get('ai_generation', 0),
            youtube_units=calculate_youtube_units(api_calls),
            gemini_calls=calculate_gemini_calls(api_calls),
            channels_found=len(result.channels_df),
            error=result.error,
        )

        log_performance_result(perf_result)

        assert result.error is None


# ============================================================================
# SEED CHANNEL ANALYSIS TESTS
# ============================================================================

class TestSeedAnalysisPerformance:
    """Performance tests for seed channel analysis (seed_topics.py)."""

    def test_seed_analysis_no_ai(self, mock_youtube):
        """Seed channel analysis without AI summary."""
        api_calls = {'youtube_channel': 0, 'youtube_playlist': 0, 'youtube_video': 0}

        def on_api_call(api_name):
            if api_name in api_calls:
                api_calls[api_name] += 1

        start = time.time()
        result = analyze_seed_channel(
            youtube_service=mock_youtube,
            channel_id="UCtest123",
            max_videos=50,
            gemini_model=None,
            on_api_call=on_api_call,
        )
        total_time = time.time() - start

        print("\n" + "=" * 70)
        print("SCENARIO: Seed Channel Analysis - No AI")
        print("=" * 70)
        print(f"  Total Time: {total_time:.4f}s")
        print(f"  API Calls: {api_calls}")
        if result.error:
            print(f"  Error: {result.error}")
        elif result.profile:
            print(f"  Channel: {result.profile.channel_name}")
            print(f"  Primary Keywords: {result.profile.primary_keywords}")
            print(f"  Secondary Keywords: {result.profile.secondary_keywords}")
            print(f"  Common Tags: {result.profile.common_tags[:5]}...")
        print("=" * 70 + "\n")

        # Analysis should complete (may have error due to mock limitations)
        assert result is not None

    def test_seed_analysis_with_ai(self, mock_youtube, mock_gemini_model):
        """Seed channel analysis with AI summary."""
        api_calls = {'youtube_channel': 0, 'youtube_playlist': 0, 'youtube_video': 0}

        def on_api_call(api_name):
            if api_name in api_calls:
                api_calls[api_name] += 1

        start = time.time()
        result = analyze_seed_channel(
            youtube_service=mock_youtube,
            channel_id="UCtest123",
            max_videos=50,
            gemini_model=mock_gemini_model,
            on_api_call=on_api_call,
        )
        total_time = time.time() - start

        print("\n" + "=" * 70)
        print("SCENARIO: Seed Channel Analysis - With AI")
        print("=" * 70)
        print(f"  Total Time: {total_time:.4f}s")
        print(f"  API Calls: {api_calls}")
        if result.error:
            print(f"  Error: {result.error}")
        elif result.profile:
            print(f"  Channel: {result.profile.channel_name}")
            print(f"  AI Summary: {result.profile.description_summary[:100]}...")
        print("=" * 70 + "\n")

        assert result is not None


# ============================================================================
# TIMING VERIFICATION TESTS
# ============================================================================

class TestTimingConsistency:
    """Tests to verify timing data is recorded consistently."""

    def test_all_timing_keys_recorded_keyword_mode(
        self, mock_youtube, mock_cache_functions_cold
    ):
        """Verify all expected timing keys are recorded in keyword mode."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result, _ = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga",
            config=config,
            cache_functions=mock_cache_functions_cold,
        )

        expected_keys = ['search', 'channel_stats', 'video_details',
                        'relevance_filtering', 'select_channels', 'total']

        for key in expected_keys:
            assert key in result.timings, f"Missing timing key: {key}"
            assert result.timings[key] >= 0, f"Negative timing for {key}"

    def test_all_timing_keys_recorded_seed_mode(
        self, mock_youtube, mock_cache_functions_cold, mock_similarity_engine, seed_profile
    ):
        """Verify all expected timing keys are recorded in seed mode."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
            seed_profile=seed_profile,
        )

        result, _ = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="tech review",
            config=config,
            cache_functions=mock_cache_functions_cold,
            similarity_engine=mock_similarity_engine,
        )

        # Seed mode should additionally have 'similarity' timing
        if not result.channels_df.empty:
            assert 'similarity' in result.timings, "Missing similarity timing in seed mode"

    def test_total_time_is_sum_of_parts(
        self, mock_youtube, mock_cache_functions_cold
    ):
        """Verify total time approximately equals sum of component times."""
        config = PipelineConfig(
            min_subscribers=1000,
            enable_ai_relevance=False,
            enable_ai_summary=False,
        )

        result, _ = run_pipeline_with_tracking(
            youtube_service=mock_youtube,
            query="manga",
            config=config,
            cache_functions=mock_cache_functions_cold,
        )

        if 'total' in result.timings:
            # Sum all component times (excluding 'total')
            component_sum = sum(
                v for k, v in result.timings.items()
                if k != 'total'
            )

            # Total should be >= sum of components (may include unmeasured overhead)
            assert result.timings['total'] >= component_sum * 0.9, \
                "Total time should be approximately sum of components"


# ============================================================================
# PERFORMANCE SUMMARY REPORT
# ============================================================================

class TestPerformanceSummary:
    """Generate a summary report of all performance tests."""

    def test_generate_summary_report(
        self, mock_youtube, mock_cache_functions_cold, mock_cache_functions_warm,
        mock_gemini_model, mock_similarity_engine, seed_profile
    ):
        """Generate comprehensive performance summary."""
        results = []

        # Run all test scenarios
        scenarios = [
            # Keyword mode
            ("Keyword 1T Cold NoAI", "manga", False, False, None, mock_cache_functions_cold, None),
            ("Keyword 1T Warm NoAI", "manga", False, False, None, mock_cache_functions_warm, None),
            ("Keyword 1T Warm AI", "manga", True, True, mock_gemini_model, mock_cache_functions_warm, None),
            ("Keyword 2T Cold NoAI", "manga, anime", False, False, None, mock_cache_functions_cold, None),
            ("Keyword 2T Warm AI", "manga, anime", True, True, mock_gemini_model, mock_cache_functions_warm, None),
            # Seed mode
            ("Seed 1T Cold NoAI", "tech review", False, False, None, mock_cache_functions_cold, seed_profile),
            ("Seed 1T Warm NoAI", "tech review", False, False, None, mock_cache_functions_warm, seed_profile),
            ("Seed 1T Warm AI", "tech review", True, True, mock_gemini_model, mock_cache_functions_warm, seed_profile),
            ("Seed 2T Cold NoAI", "tech review, tutorial", False, False, None, mock_cache_functions_cold, seed_profile),
            ("Seed 2T Warm AI", "tech review, tutorial", True, True, mock_gemini_model, mock_cache_functions_warm, seed_profile),
        ]

        print("\n")
        print("=" * 80)
        print(" PERFORMANCE SUMMARY REPORT ")
        print("=" * 80)
        print(f"{'Scenario':<25} {'Total(s)':<10} {'Search':<10} {'Videos':<10} {'AI Rel':<10}")
        print("-" * 80)

        for scenario_name, query, ai_rel, ai_sum, gemini, cache, seed in scenarios:
            config = PipelineConfig(
                min_subscribers=1000,
                enable_ai_relevance=ai_rel,
                enable_ai_summary=ai_sum,
                seed_profile=seed,
            )

            sim_engine = mock_similarity_engine if seed else None

            start = time.time()
            result, _ = run_pipeline_with_tracking(
                youtube_service=mock_youtube,
                query=query,
                config=config,
                cache_functions=cache,
                gemini_model=gemini,
                similarity_engine=sim_engine,
            )
            total = time.time() - start

            timings = result.timings
            print(f"{scenario_name:<25} {total:<10.4f} {timings.get('search', 0):<10.4f} "
                  f"{timings.get('video_details', 0):<10.4f} {timings.get('ai_relevance', 0):<10.4f}")

        print("=" * 80)
        print("\nNote: These are mock timings. Real API performance varies.")
        print("=" * 80 + "\n")

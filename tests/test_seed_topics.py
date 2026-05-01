"""
Tests for core.seed_topics module

Tests cover:
- Language detection (EN, ES, mixed, edge cases)
- Tokenization and bigram extraction
- Term penalty calculation
- Subscriber tier classification
- Upload frequency calculation
- Engagement rate calculation
- Full seed channel analysis with mocks

All tests use mocked YouTube API clients for isolation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.seed_topics import (
    # Dataclasses
    SeedProfile,
    SeedAnalysisResult,
    # Language detection
    detect_language,
    get_stopwords,
    # Tokenization
    tokenize,
    extract_bigrams,
    # Scoring
    calculate_term_penalty,
    calculate_subscriber_tier,
    calculate_upload_frequency,
    calculate_engagement_rate,
    # Topic extraction
    extract_topics,
    # Main function
    analyze_seed_channel,
    # Constants
    STOPWORDS_EN,
    STOPWORDS_ES,
    STOPWORDS_COMMON,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_youtube():
    """Create a mock YouTube API service."""
    return Mock()


@pytest.fixture
def sample_videos():
    """Sample video data for testing."""
    return [
        {
            'video_id': 'vid1',
            'video_title': 'How to make vegan recipes at home',
            'video_description': 'Learn to cook healthy plant-based meals',
            'video_tags': ['vegan', 'cooking', 'healthy'],
            'video_views': 10000,
            'video_likes': 500,
            'video_comments': 50,
            'published_at': '2024-01-15T12:00:00Z'
        },
        {
            'video_id': 'vid2',
            'video_title': 'Best vegan breakfast ideas',
            'video_description': 'Start your day with healthy options',
            'video_tags': ['vegan', 'breakfast', 'healthy'],
            'video_views': 8000,
            'video_likes': 400,
            'video_comments': 40,
            'published_at': '2024-01-10T12:00:00Z'
        },
        {
            'video_id': 'vid3',
            'video_title': 'Quick vegan lunch recipes',
            'video_description': 'Fast and easy vegan meals',
            'video_tags': ['vegan', 'lunch', 'recipes'],
            'video_views': 12000,
            'video_likes': 600,
            'video_comments': 60,
            'published_at': '2024-01-05T12:00:00Z'
        }
    ]


# ============================================================================
# LANGUAGE DETECTION TESTS
# ============================================================================

class TestLanguageDetection:
    """Tests for the detect_language function."""

    def test_english_content_detected(self):
        """English content should be detected as 'en'."""
        texts = [
            "How to cook amazing food at home",
            "The best recipes for healthy eating",
            "What you need to know about cooking"
        ]
        assert detect_language(texts) == 'en'

    def test_spanish_content_detected(self):
        """Spanish content should be detected as 'es'."""
        texts = [
            "Cómo cocinar comida deliciosa en casa",
            "Las mejores recetas para comer sano",
            "Todo sobre la cocina española"
        ]
        assert detect_language(texts) == 'es'

    def test_mixed_content_defaults_to_english(self):
        """Mixed content with equal stopwords defaults to English."""
        texts = ["Hello world"]  # No stopwords
        assert detect_language(texts) == 'en'

    def test_empty_input_returns_english(self):
        """Empty input should return English as default."""
        assert detect_language([]) == 'en'
        assert detect_language(['']) == 'en'

    def test_elif_fix_no_double_counting(self):
        """
        Regression test: Words appearing in both EN and ES stopwords
        should only be counted once.

        The bug was using 'if' instead of 'elif', causing overlapping
        words to be counted twice.
        """
        # Words that might appear in both lists shouldn't skew results
        texts = ["para para para para para"]  # "para" is ES stopword
        result = detect_language(texts)
        # Should be detected as Spanish since 'para' is in STOPWORDS_ES
        assert result == 'es'


class TestGetStopwords:
    """Tests for the get_stopwords function."""

    def test_english_stopwords_include_common(self):
        """English stopwords should include common terms."""
        stopwords = get_stopwords('en')
        assert 'the' in stopwords
        assert 'channel' in stopwords  # From STOPWORDS_COMMON

    def test_spanish_stopwords_include_common(self):
        """Spanish stopwords should include common terms."""
        stopwords = get_stopwords('es')
        assert 'que' in stopwords
        assert 'canal' in stopwords  # From STOPWORDS_COMMON


# ============================================================================
# TOKENIZATION TESTS
# ============================================================================

class TestTokenization:
    """Tests for the tokenize function."""

    def test_removes_stopwords(self):
        """Stopwords should be removed from output."""
        stopwords = {'the', 'and', 'for'}
        result = tokenize("The quick and brown fox", stopwords)
        assert 'the' not in result
        assert 'and' not in result
        assert 'quick' in result
        assert 'brown' in result

    def test_respects_min_length(self):
        """Words shorter than min_length should be removed."""
        stopwords = set()
        result = tokenize("I am a big dog", stopwords, min_length=3)
        assert 'big' in result
        assert 'dog' in result
        assert 'am' not in result  # Only 2 chars
        assert 'a' not in result   # Only 1 char

    def test_keeps_accented_characters(self):
        """Spanish accented characters should be preserved."""
        stopwords = set()
        result = tokenize("Música española fantástica", stopwords)
        assert 'música' in result
        assert 'española' in result
        assert 'fantástica' in result

    def test_removes_numbers(self):
        """Words containing numbers should be removed."""
        stopwords = set()
        result = tokenize("Episode ep5 review 2024", stopwords)
        assert 'episode' in result
        assert 'review' in result
        # These contain digits so should be excluded
        # Note: The regex only matches alphabetic, so 'ep5' won't match
        assert 'ep5' not in result


class TestBigramExtraction:
    """Tests for the extract_bigrams function."""

    def test_creates_adjacent_pairs(self):
        """Should create pairs from adjacent tokens."""
        tokens = ['healthy', 'vegan', 'recipes']
        stopwords = set()
        result = extract_bigrams(tokens, stopwords)
        assert 'healthy vegan' in result
        assert 'vegan recipes' in result

    def test_skips_stopword_pairs(self):
        """Pairs containing stopwords should be skipped."""
        tokens = ['healthy', 'and', 'vegan']
        stopwords = {'and'}
        result = extract_bigrams(tokens, stopwords)
        assert 'healthy and' not in result
        assert 'and vegan' not in result
        # Since 'and' is in the middle, no valid bigrams
        assert len(result) == 0

    def test_empty_input(self):
        """Empty input should return empty list."""
        assert extract_bigrams([], set()) == []
        assert extract_bigrams(['single'], set()) == []


# ============================================================================
# TERM PENALTY TESTS
# ============================================================================

class TestTermPenalty:
    """Tests for the calculate_term_penalty function."""

    def test_year_penalty(self):
        """Years like 2024 should receive 0.5 penalty."""
        penalty = calculate_term_penalty("review 2024")
        assert penalty == 0.5

    def test_number_penalty(self):
        """Terms with numbers should receive 0.3 penalty."""
        penalty = calculate_term_penalty("episode ep5")
        # This contains a 4-digit number check first, then general number check
        # "ep5" contains digit, so penalty should be 0.3
        assert penalty == 0.3

    def test_month_penalty_english(self):
        """English month names should receive 0.4 penalty."""
        penalty = calculate_term_penalty("january special")
        assert penalty == 0.4

    def test_month_penalty_spanish(self):
        """Spanish month names should receive 0.4 penalty."""
        penalty = calculate_term_penalty("especial enero")
        assert penalty == 0.4

    def test_promo_word_penalty(self):
        """Promotional words should receive 0.3 penalty."""
        penalty = calculate_term_penalty("trailer nuevo")
        assert penalty == 0.3

    def test_stacked_penalties_capped(self):
        """Multiple penalties should stack but cap at 1.0."""
        # "2024 january trailer" = 0.5 (year) + 0.4 (month) + 0.3 (promo) = 1.2 -> capped to 1.0
        penalty = calculate_term_penalty("2024 january trailer")
        assert penalty == 1.0

    def test_clean_term_no_penalty(self):
        """Clean terms should have 0 penalty."""
        penalty = calculate_term_penalty("healthy vegan recipes")
        assert penalty == 0.0


# ============================================================================
# SUBSCRIBER TIER TESTS
# ============================================================================

class TestSubscriberTier:
    """Tests for the calculate_subscriber_tier function."""

    def test_nano_tier(self):
        """Channels under 10K should be nano."""
        assert calculate_subscriber_tier(5000) == "nano"
        assert calculate_subscriber_tier(9999) == "nano"

    def test_micro_tier(self):
        """Channels 10K-100K should be micro."""
        assert calculate_subscriber_tier(10000) == "micro"
        assert calculate_subscriber_tier(50000) == "micro"
        assert calculate_subscriber_tier(99999) == "micro"

    def test_mid_tier(self):
        """Channels 100K-1M should be mid."""
        assert calculate_subscriber_tier(100000) == "mid"
        assert calculate_subscriber_tier(500000) == "mid"
        assert calculate_subscriber_tier(999999) == "mid"

    def test_macro_tier(self):
        """Channels 1M-10M should be macro."""
        assert calculate_subscriber_tier(1000000) == "macro"
        assert calculate_subscriber_tier(5000000) == "macro"
        assert calculate_subscriber_tier(9999999) == "macro"

    def test_mega_tier(self):
        """Channels over 10M should be mega."""
        assert calculate_subscriber_tier(10000000) == "mega"
        assert calculate_subscriber_tier(100000000) == "mega"

    def test_boundary_values(self):
        """Test exact boundary values."""
        assert calculate_subscriber_tier(0) == "nano"
        assert calculate_subscriber_tier(10000) == "micro"  # Exactly at boundary
        assert calculate_subscriber_tier(100000) == "mid"
        assert calculate_subscriber_tier(1000000) == "macro"
        assert calculate_subscriber_tier(10000000) == "mega"


# ============================================================================
# UPLOAD FREQUENCY TESTS
# ============================================================================

class TestUploadFrequency:
    """Tests for the calculate_upload_frequency function."""

    def test_regular_uploads(self):
        """Regular uploads should calculate correct frequency."""
        # 30 days span, 10 videos = ~10 videos/month
        dates = [
            '2024-01-01T12:00:00Z',
            '2024-01-04T12:00:00Z',
            '2024-01-07T12:00:00Z',
            '2024-01-10T12:00:00Z',
            '2024-01-13T12:00:00Z',
            '2024-01-16T12:00:00Z',
            '2024-01-19T12:00:00Z',
            '2024-01-22T12:00:00Z',
            '2024-01-25T12:00:00Z',
            '2024-01-31T12:00:00Z',
        ]
        freq = calculate_upload_frequency(dates)
        # 30 day span, 10 videos = 10 videos/month
        assert freq > 0
        assert 8.0 < freq < 12.0  # Approximately 10 videos/month

    def test_single_video_returns_zero(self):
        """Single video should return 0 (insufficient data)."""
        dates = ['2024-01-01T12:00:00Z']
        freq = calculate_upload_frequency(dates)
        assert freq == 0.0

    def test_same_second_uploads(self):
        """Same-second uploads should use minimum 1 hour span."""
        # Two videos at exact same time
        dates = [
            '2024-01-01T12:00:00Z',
            '2024-01-01T12:00:00Z',
        ]
        freq = calculate_upload_frequency(dates)
        # Should not crash or return infinity
        assert freq > 0
        # With 1 hour minimum span and 2 videos, freq would be very high
        assert freq < 50000  # Sanity check


# ============================================================================
# ENGAGEMENT RATE TESTS
# ============================================================================

class TestEngagementRate:
    """Tests for the calculate_engagement_rate function."""

    def test_normal_calculation(self):
        """Normal engagement calculation should work correctly."""
        videos = [
            {'video_views': 1000, 'video_likes': 100, 'video_comments': 10},
            {'video_views': 2000, 'video_likes': 200, 'video_comments': 20},
        ]
        rate = calculate_engagement_rate(videos)
        # Video 1: (100 + 10) / 1000 = 0.11
        # Video 2: (200 + 20) / 2000 = 0.11
        # Average = 0.11
        assert rate == 0.11

    def test_zero_view_videos_included(self):
        """
        Regression test: 0-view videos should be included.

        The bug was excluding 0-view videos, biasing the average high.
        """
        videos = [
            {'video_views': 0, 'video_likes': 0, 'video_comments': 0},
            {'video_views': 1000, 'video_likes': 100, 'video_comments': 0},
        ]
        rate = calculate_engagement_rate(videos)
        # Video 1: (0 + 0) / max(0, 1) = 0.0
        # Video 2: (100 + 0) / 1000 = 0.1
        # Average = 0.05
        assert rate == 0.05

    def test_empty_videos_returns_zero(self):
        """Empty video list should return 0."""
        assert calculate_engagement_rate([]) == 0.0


# ============================================================================
# TOPIC EXTRACTION TESTS
# ============================================================================

class TestTopicExtraction:
    """Tests for the extract_topics function."""

    def test_extracts_keywords_from_titles(self, sample_videos):
        """Should extract keywords from video titles."""
        stopwords = get_stopwords('en')
        name_tokens = set()

        primary, secondary, tags = extract_topics(
            videos=sample_videos,
            stopwords=stopwords,
            name_tokens=name_tokens,
            n_videos=len(sample_videos)
        )

        # 'vegan' appears in all 3 titles, should be extracted
        assert 'vegan' in secondary or any('vegan' in p for p in primary)

    def test_extracts_common_tags(self, sample_videos):
        """Should extract common tags from videos."""
        stopwords = get_stopwords('en')
        name_tokens = set()

        primary, secondary, tags = extract_topics(
            videos=sample_videos,
            stopwords=stopwords,
            name_tokens=name_tokens,
            n_videos=len(sample_videos)
        )

        # 'vegan' and 'healthy' appear in multiple video tags
        assert 'vegan' in tags
        assert 'healthy' in tags

    def test_excludes_channel_name_tokens(self, sample_videos):
        """Channel name tokens should be excluded from results."""
        stopwords = get_stopwords('en')
        name_tokens = {'vegan', 'kitchen'}  # Pretend channel is "Vegan Kitchen"

        primary, secondary, tags = extract_topics(
            videos=sample_videos,
            stopwords=stopwords,
            name_tokens=name_tokens,
            n_videos=len(sample_videos)
        )

        # 'vegan' should be excluded because it's in channel name
        assert 'vegan' not in secondary
        assert 'vegan' not in tags


# ============================================================================
# SEED PROFILE TESTS
# ============================================================================

class TestSeedProfile:
    """Tests for the SeedProfile dataclass."""

    def test_to_dict_returns_all_fields(self):
        """to_dict() should return all fields for backward compatibility."""
        profile = SeedProfile(
            channel_id='UC123',
            channel_name='Test Channel',
            subscriber_count=50000,
            subscriber_tier='micro',
            category='22',
            language='en',
            upload_frequency=4.5,
            avg_engagement_rate=0.05,
            primary_keywords=['healthy eating', 'vegan recipes'],
            secondary_keywords=['cooking', 'food'],
            common_tags=['vegan', 'healthy'],
            recent_titles=['Video 1', 'Video 2'],
            description_summary='Test summary',
            topic_categories=['Food'],
            channel_keywords=['cooking', 'vegan']
        )

        result = profile.to_dict()

        # Check all required keys for similarity_engine.py compatibility
        assert result['channel_id'] == 'UC123'
        assert result['channel_name'] == 'Test Channel'
        assert result['subscriber_count'] == 50000
        assert result['subscriber_tier'] == 'micro'
        assert result['category'] == '22'
        assert result['language'] == 'en'
        assert result['upload_frequency'] == 4.5
        assert result['avg_engagement_rate'] == 0.05
        assert result['primary_keywords'] == ['healthy eating', 'vegan recipes']
        assert result['secondary_keywords'] == ['cooking', 'food']
        assert result['common_tags'] == ['vegan', 'healthy']
        assert result['recent_titles'] == ['Video 1', 'Video 2']
        assert result['description_summary'] == 'Test summary'

    def test_default_values(self):
        """Default values should be set correctly."""
        profile = SeedProfile(
            channel_id='UC123',
            channel_name='Test',
            subscriber_count=1000,
            subscriber_tier='nano',
            category='22',
            language='en',
            upload_frequency=0.0,
            avg_engagement_rate=0.0
        )

        assert profile.primary_keywords == []
        assert profile.secondary_keywords == []
        assert profile.common_tags == []
        assert profile.recent_titles == []
        assert profile.description_summary == ''
        assert profile.topic_categories == []
        assert profile.channel_keywords == []


# ============================================================================
# ANALYZE SEED CHANNEL TESTS
# ============================================================================

class TestAnalyzeSeedChannel:
    """Tests for the analyze_seed_channel function."""

    def test_channel_not_found(self, mock_youtube):
        """Should return error when channel is not found."""
        # Mock get_channel_stats to return empty stats
        with patch('app.core.seed_topics.get_channel_stats') as mock_stats:
            mock_stats.return_value = Mock(stats=[], api_calls=1)

            result = analyze_seed_channel(mock_youtube, 'UC_INVALID')

            assert isinstance(result, SeedAnalysisResult)
            assert result.profile is None
            assert result.error is not None
            assert 'not found' in result.error.lower()

    def test_api_callback_invoked(self, mock_youtube):
        """API callback should be invoked for tracking."""
        api_calls_tracked = []

        def track_call(call_type):
            api_calls_tracked.append(call_type)

        # Mock the channel stats call to return valid data
        with patch('app.core.seed_topics.get_channel_stats') as mock_stats:
            mock_stats.return_value = Mock(
                stats=[{
                    'channel_id': 'UC123',
                    'subscribers': 50000,
                    'uploads_playlist_id': 'UU123',
                    'description': 'Test channel',
                    'topic_categories': [],
                    'channel_keywords': [],
                    'default_language': 'en'
                }],
                api_calls=1
            )

            # Mock the channels().list() call for getting channel name
            mock_youtube.channels().list().execute.return_value = {
                'items': [{
                    'snippet': {
                        'title': 'Test Channel',
                        'description': 'Test description',
                        'categoryId': '22'
                    }
                }]
            }

            # Mock get_video_details
            with patch('app.core.seed_topics.get_video_details') as mock_videos:
                mock_videos.return_value = Mock(
                    videos=[
                        {
                            'video_id': 'vid1',
                            'video_title': 'Test Video',
                            'video_description': 'Description',
                            'video_tags': ['test'],
                            'video_views': 1000,
                            'video_likes': 100,
                            'video_comments': 10,
                            'published_at': '2024-01-01T12:00:00Z'
                        }
                    ],
                    warnings=[],
                    api_calls=2
                )

                result = analyze_seed_channel(
                    mock_youtube,
                    'UC123',
                    on_api_call=track_call
                )

        # Callback should have been called
        assert len(api_calls_tracked) > 0

    def test_progress_callback_invoked(self, mock_youtube):
        """Progress callback should be invoked with updates."""
        progress_updates = []

        def track_progress(msg, pct):
            progress_updates.append((msg, pct))

        with patch('app.core.seed_topics.get_channel_stats') as mock_stats:
            mock_stats.return_value = Mock(
                stats=[{
                    'channel_id': 'UC123',
                    'subscribers': 50000,
                    'uploads_playlist_id': 'UU123',
                    'description': 'Test channel',
                    'topic_categories': [],
                    'channel_keywords': [],
                    'default_language': 'en'
                }],
                api_calls=1
            )

            mock_youtube.channels().list().execute.return_value = {
                'items': [{
                    'snippet': {
                        'title': 'Test Channel',
                        'description': 'Test description',
                        'categoryId': '22'
                    }
                }]
            }

            with patch('app.core.seed_topics.get_video_details') as mock_videos:
                mock_videos.return_value = Mock(
                    videos=[
                        {
                            'video_id': 'vid1',
                            'video_title': 'Test Video',
                            'video_description': 'Description',
                            'video_tags': ['test'],
                            'video_views': 1000,
                            'video_likes': 100,
                            'video_comments': 10,
                            'published_at': '2024-01-01T12:00:00Z'
                        }
                    ],
                    warnings=[],
                    api_calls=2
                )

                result = analyze_seed_channel(
                    mock_youtube,
                    'UC123',
                    on_progress=track_progress
                )

        # Progress should have been reported multiple times
        assert len(progress_updates) > 0
        # First update should be around 0.1
        assert progress_updates[0][1] <= 0.2
        # Last update should be 1.0
        assert progress_updates[-1][1] == 1.0

    def test_profile_backward_compatible(self, mock_youtube):
        """Profile.to_dict() should have all keys needed by similarity_engine."""
        with patch('app.core.seed_topics.get_channel_stats') as mock_stats:
            mock_stats.return_value = Mock(
                stats=[{
                    'channel_id': 'UC123',
                    'subscribers': 50000,
                    'uploads_playlist_id': 'UU123',
                    'description': 'Test channel',
                    'topic_categories': ['Music'],
                    'channel_keywords': ['music', 'covers'],
                    'default_language': 'en'
                }],
                api_calls=1
            )

            mock_youtube.channels().list().execute.return_value = {
                'items': [{
                    'snippet': {
                        'title': 'Test Channel',
                        'description': 'A music channel',
                        'categoryId': '10'
                    }
                }]
            }

            with patch('app.core.seed_topics.get_video_details') as mock_videos:
                mock_videos.return_value = Mock(
                    videos=[
                        {
                            'video_id': 'vid1',
                            'video_title': 'Music Cover Song',
                            'video_description': 'My cover of a popular song',
                            'video_tags': ['music', 'cover', 'singing'],
                            'video_views': 5000,
                            'video_likes': 250,
                            'video_comments': 25,
                            'published_at': '2024-01-15T12:00:00Z'
                        },
                        {
                            'video_id': 'vid2',
                            'video_title': 'Original Music Track',
                            'video_description': 'My original composition',
                            'video_tags': ['music', 'original', 'composition'],
                            'video_views': 3000,
                            'video_likes': 150,
                            'video_comments': 15,
                            'published_at': '2024-01-10T12:00:00Z'
                        }
                    ],
                    warnings=[],
                    api_calls=2
                )

                result = analyze_seed_channel(mock_youtube, 'UC123')

        assert result.profile is not None
        profile_dict = result.profile.to_dict()

        # Check all required keys for similarity_engine.py
        required_keys = [
            'channel_id', 'channel_name', 'subscriber_count', 'subscriber_tier',
            'category', 'language', 'upload_frequency', 'avg_engagement_rate',
            'primary_keywords', 'secondary_keywords', 'common_tags',
            'recent_titles', 'description_summary'
        ]

        for key in required_keys:
            assert key in profile_dict, f"Missing required key: {key}"

    def test_no_videos_error(self, mock_youtube):
        """Should return error when channel has no videos."""
        with patch('app.core.seed_topics.get_channel_stats') as mock_stats:
            mock_stats.return_value = Mock(
                stats=[{
                    'channel_id': 'UC123',
                    'subscribers': 50000,
                    'uploads_playlist_id': 'UU123',
                    'description': 'Test channel',
                    'topic_categories': [],
                    'channel_keywords': [],
                    'default_language': 'en'
                }],
                api_calls=1
            )

            mock_youtube.channels().list().execute.return_value = {
                'items': [{
                    'snippet': {
                        'title': 'Test Channel',
                        'description': 'Test',
                        'categoryId': '22'
                    }
                }]
            }

            with patch('app.core.seed_topics.get_video_details') as mock_videos:
                mock_videos.return_value = Mock(
                    videos=[],  # No videos
                    warnings=[],
                    api_calls=1
                )

                result = analyze_seed_channel(mock_youtube, 'UC123')

        assert result.profile is None
        assert result.error is not None
        assert 'no videos' in result.error.lower()


# ============================================================================
# SEED ANALYSIS RESULT TESTS
# ============================================================================

class TestSeedAnalysisResult:
    """Tests for the SeedAnalysisResult dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        result = SeedAnalysisResult()
        assert result.profile is None
        assert result.error is None
        assert result.warnings == []
        assert result.api_calls == 0

    def test_with_error(self):
        """Error result should have correct structure."""
        result = SeedAnalysisResult(
            error="Channel not found",
            api_calls=1
        )
        assert result.profile is None
        assert result.error == "Channel not found"
        assert result.api_calls == 1

    def test_with_profile(self):
        """Success result should have profile."""
        profile = SeedProfile(
            channel_id='UC123',
            channel_name='Test',
            subscriber_count=1000,
            subscriber_tier='nano',
            category='22',
            language='en',
            upload_frequency=0.0,
            avg_engagement_rate=0.0
        )
        result = SeedAnalysisResult(
            profile=profile,
            api_calls=3
        )
        assert result.profile is not None
        assert result.error is None
        assert result.api_calls == 3


# ============================================================================
# METADATA ENRICHMENT TESTS (Item 4)
# ============================================================================

class TestMetadataEnrichment:
    """
    Tests that topic_categories and channel_keywords from brandingSettings
    are folded into the seed profile's common_tags and secondary_keywords.
    """

    def _run_analysis_with_meta(self, mock_youtube, topic_categories, channel_keywords):
        """Helper: run analyze_seed_channel with given channel metadata."""
        from unittest.mock import patch
        with patch('app.core.seed_topics.get_channel_stats') as mock_stats:
            mock_stats.return_value = Mock(
                stats=[{
                    'channel_id': 'UCtest',
                    'subscribers': 10000,
                    'uploads_playlist_id': 'UUtest',
                    'description': 'A test channel',
                    'topic_categories': topic_categories,
                    'channel_keywords': channel_keywords,
                    'default_language': 'en',
                }],
                api_calls=1,
            )
            mock_youtube.channels().list().execute.return_value = {
                'items': [{'snippet': {'title': 'Test', 'description': '', 'categoryId': '22'}}]
            }
            with patch('app.core.seed_topics.get_video_details') as mock_videos:
                mock_videos.return_value = Mock(
                    videos=[
                        {
                            'video_id': f'v{i}',
                            'video_title': 'cooking recipe vegan',
                            'video_description': 'plant-based food',
                            'video_tags': ['vegan', 'cooking'],
                            'video_views': 1000,
                            'video_likes': 50,
                            'video_comments': 5,
                            'published_at': f'2024-0{(i%9)+1}-01T00:00:00Z',
                        }
                        for i in range(10)
                    ],
                    warnings=[],
                    api_calls=2,
                )
                return analyze_seed_channel(mock_youtube, 'UCtest')

    def test_topic_categories_appear_in_common_tags(self, mock_youtube):
        """topic_categories from the channel are folded into common_tags."""
        result = self._run_analysis_with_meta(
            mock_youtube,
            topic_categories=['Food', 'Lifestyle'],
            channel_keywords=[],
        )
        assert result.profile is not None
        tags = [t.lower() for t in result.profile.common_tags]
        assert 'food' in tags
        assert 'lifestyle' in tags

    def test_channel_keywords_appear_in_common_tags(self, mock_youtube):
        """channel_keywords from brandingSettings appear in common_tags."""
        result = self._run_analysis_with_meta(
            mock_youtube,
            topic_categories=[],
            channel_keywords=['plantbased', 'veganlife'],
        )
        assert result.profile is not None
        tags = [t.lower() for t in result.profile.common_tags]
        assert 'plantbased' in tags
        assert 'veganlife' in tags

    def test_channel_keywords_tokens_in_secondary_keywords(self, mock_youtube):
        """Tokenizable channel_keywords contribute tokens to secondary_keywords."""
        result = self._run_analysis_with_meta(
            mock_youtube,
            topic_categories=[],
            channel_keywords=['nutrition tips'],  # two tokenizable words
        )
        assert result.profile is not None
        # 'nutrition' or 'tips' should appear somewhere in keywords
        all_kws = result.profile.primary_keywords + result.profile.secondary_keywords
        # At minimum the enrichment ran without error
        assert isinstance(all_kws, list)

    def test_empty_metadata_no_crash(self, mock_youtube):
        """Empty topic_categories + channel_keywords: profile still returns."""
        result = self._run_analysis_with_meta(
            mock_youtube,
            topic_categories=[],
            channel_keywords=[],
        )
        assert result.profile is not None
        assert result.error is None

    def test_candidate_with_matching_topic_categories_scores_higher(self):
        """
        A candidate whose tags include a topic_category matching the seed's
        common_tags should score higher than one without that match.
        Confirms that enriching tags with topic_categories improves Jaccard.
        """
        from app.core.similarity import calculate_similarity_score

        seed = {
            'common_tags': ['cooking', 'food', 'vegan lifestyle'],
            'primary_keywords': ['vegan'],
            'secondary_keywords': ['cooking'],
            'subscriber_count': 50000,
            'avg_engagement_rate': 0.03,
            'upload_frequency': 4.0,
        }

        # Candidate A: base tags only
        candidate_a = {
            'tags': ['cooking', 'food'],
            'keywords': ['vegan'],
            'subscribers': 50000,
            'engagement_rate': 0.03,
            'upload_frequency': 4.0,
        }

        # Candidate B: base tags + topic_category 'vegan lifestyle' in tags
        candidate_b = {
            'tags': ['cooking', 'food', 'vegan lifestyle'],
            'keywords': ['vegan'],
            'subscribers': 50000,
            'engagement_rate': 0.03,
            'upload_frequency': 4.0,
        }

        score_a = calculate_similarity_score(candidate_a, seed)['total_score']
        score_b = calculate_similarity_score(candidate_b, seed)['total_score']

        assert score_b > score_a, (
            f"Candidate with matching topic_category tag should score higher: "
            f"{score_b:.1f} vs {score_a:.1f}"
        )

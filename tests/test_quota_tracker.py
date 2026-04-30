"""
Unit tests for app/analytics/quota_tracker.py

Tests cover:
- Quota cost constants
- DebugData dataclass
- DailyQuota dataclass
- Quota calculation functions
- Date/time utilities
- File persistence
- Tracking functions
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, Mock
from datetime import datetime, timezone, timedelta

from app.analytics.quota_tracker import (
    # Constants
    YOUTUBE_QUOTA_COSTS,
    GEMINI_COSTS,
    DEFAULT_QUOTA_CACHE_FILE,
    # Dataclasses
    DebugData,
    DailyQuota,
    # Quota calculation
    calculate_youtube_quota_used,
    calculate_gemini_cost_estimate,
    get_total_youtube_calls,
    get_total_gemini_calls,
    # Date/time
    get_current_date_pt,
    get_next_reset_time,
    # File persistence
    load_daily_quota,
    save_daily_quota,
    # Tracking functions
    track_api_call,
    track_timing,
    track_similarity_scores,
    accumulate_to_daily_quota,
    create_empty_debug_data,
    # BYOK helpers
    key_fingerprint,
)


# ============================================================================
# TEST: Constants
# ============================================================================

class TestQuotaCostConstants:
    """Tests for quota cost constants."""

    def test_youtube_quota_costs_defined(self):
        """YouTube quota costs should be defined for all API types."""
        assert 'search' in YOUTUBE_QUOTA_COSTS
        assert 'channels' in YOUTUBE_QUOTA_COSTS
        assert 'videos' in YOUTUBE_QUOTA_COSTS
        assert 'playlistItems' in YOUTUBE_QUOTA_COSTS

    def test_youtube_search_is_100_units(self):
        """Search API should cost 100 units."""
        assert YOUTUBE_QUOTA_COSTS['search'] == 100

    def test_youtube_data_apis_are_1_unit(self):
        """Data APIs should cost 1 unit each."""
        assert YOUTUBE_QUOTA_COSTS['channels'] == 1
        assert YOUTUBE_QUOTA_COSTS['videos'] == 1
        assert YOUTUBE_QUOTA_COSTS['playlistItems'] == 1

    def test_gemini_costs_defined(self):
        """Gemini costs should be defined for flash model."""
        assert 'flash' in GEMINI_COSTS
        assert 'input' in GEMINI_COSTS['flash']
        assert 'output' in GEMINI_COSTS['flash']


# ============================================================================
# TEST: Key Fingerprint Helper
# ============================================================================

class TestKeyFingerprint:
    """Tests for key_fingerprint helper function."""

    def test_empty_key_returns_empty(self):
        """Empty string should return empty string."""
        assert key_fingerprint("") == ""

    def test_none_key_returns_empty(self):
        """None should return empty string."""
        assert key_fingerprint(None) == ""

    def test_returns_8_char_hex(self):
        """Non-empty key should return 8-character hex string."""
        result = key_fingerprint("somekey")
        assert isinstance(result, str)
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_key_same_fingerprint(self):
        """Same input should always produce the same fingerprint."""
        assert key_fingerprint("myapikey") == key_fingerprint("myapikey")

    def test_different_keys_different_fingerprints(self):
        """Distinct keys should produce distinct fingerprints."""
        fp1 = key_fingerprint("key-alpha")
        fp2 = key_fingerprint("key-beta")
        assert fp1 != fp2


# ============================================================================
# TEST: DebugData Dataclass
# ============================================================================

class TestDebugData:
    """Tests for DebugData dataclass."""

    def test_default_values_are_zero(self):
        """Default counters should be zero."""
        data = DebugData()
        assert data.youtube_search_calls == 0
        assert data.youtube_channel_calls == 0
        assert data.youtube_video_calls == 0
        assert data.youtube_playlist_calls == 0
        assert data.gemini_summary_calls == 0
        assert data.gemini_outreach_calls == 0
        assert data.gemini_similarity_calls == 0
        assert data.gemini_relevance_calls == 0

    def test_default_timings_are_zero(self):
        """Default timings should be zero."""
        data = DebugData()
        assert data.timings['search'] == 0.0
        assert data.timings['total'] == 0.0

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all fields."""
        data = DebugData(youtube_search_calls=5, gemini_summary_calls=3)
        result = data.to_dict()
        assert result['youtube_search_calls'] == 5
        assert result['gemini_summary_calls'] == 3
        assert 'timings' in result
        assert 'similarity_details' in result

    def test_from_dict_restores_values(self):
        """from_dict should restore all values."""
        original = DebugData(youtube_search_calls=5, gemini_summary_calls=3)
        dict_data = original.to_dict()
        restored = DebugData.from_dict(dict_data)
        assert restored.youtube_search_calls == 5
        assert restored.gemini_summary_calls == 3

    def test_from_dict_handles_missing_keys(self):
        """from_dict should handle missing keys with defaults."""
        data = DebugData.from_dict({})
        assert data.youtube_search_calls == 0
        assert data.gemini_summary_calls == 0


# ============================================================================
# TEST: DailyQuota Dataclass
# ============================================================================

class TestDailyQuota:
    """Tests for DailyQuota dataclass."""

    def test_default_values(self):
        """Default values should be zero."""
        quota = DailyQuota(date='2024-01-15')
        assert quota.youtube_calls == 0
        assert quota.gemini_calls == 0
        assert quota.youtube_units == 0
        assert quota.gemini_cost_usd == 0.0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        quota = DailyQuota(date='2024-01-15', youtube_calls=10, youtube_units=150)
        result = quota.to_dict()
        assert result['date'] == '2024-01-15'
        assert result['youtube_calls'] == 10
        assert result['youtube_units'] == 150

    def test_from_dict(self):
        """from_dict should restore values."""
        data = {'date': '2024-01-15', 'youtube_calls': 20, 'gemini_cost_usd': 0.05}
        quota = DailyQuota.from_dict(data)
        assert quota.date == '2024-01-15'
        assert quota.youtube_calls == 20
        assert quota.gemini_cost_usd == 0.05


# ============================================================================
# TEST: Quota Calculation Functions
# ============================================================================

class TestCalculateYoutubeQuotaUsed:
    """Tests for calculate_youtube_quota_used function."""

    def test_empty_data_returns_zero(self):
        """Empty data should return 0 units."""
        result = calculate_youtube_quota_used({})
        assert result == 0

    def test_single_search_costs_100(self):
        """One search call should cost 100 units."""
        result = calculate_youtube_quota_used({'youtube_search_calls': 1})
        assert result == 100

    def test_data_apis_cost_one_each(self):
        """Data API calls should cost 1 unit each."""
        result = calculate_youtube_quota_used({
            'youtube_channel_calls': 5,
            'youtube_video_calls': 10,
            'youtube_playlist_calls': 3
        })
        assert result == 18  # 5 + 10 + 3

    def test_combined_calculation(self):
        """Combined usage should sum correctly."""
        result = calculate_youtube_quota_used({
            'youtube_search_calls': 2,
            'youtube_channel_calls': 10,
            'youtube_video_calls': 5,
            'youtube_playlist_calls': 3
        })
        # 2*100 + 10*1 + 5*1 + 3*1 = 218
        assert result == 218


class TestCalculateGeminiCostEstimate:
    """Tests for calculate_gemini_cost_estimate function."""

    def test_empty_data_returns_zero(self):
        """Empty data should return $0."""
        result = calculate_gemini_cost_estimate({})
        assert result == 0.0

    def test_positive_calls_return_positive_cost(self):
        """Positive calls should return positive cost."""
        result = calculate_gemini_cost_estimate({
            'gemini_summary_calls': 5,
            'gemini_outreach_calls': 3
        })
        assert result > 0

    def test_more_calls_higher_cost(self):
        """More calls should result in higher cost."""
        few_calls = calculate_gemini_cost_estimate({'gemini_summary_calls': 1})
        many_calls = calculate_gemini_cost_estimate({'gemini_summary_calls': 10})
        assert many_calls > few_calls


class TestGetTotalYoutubeCalls:
    """Tests for get_total_youtube_calls function."""

    def test_empty_data_returns_zero(self):
        """Empty data should return 0."""
        assert get_total_youtube_calls({}) == 0

    def test_sums_all_youtube_calls(self):
        """Should sum all YouTube API types."""
        result = get_total_youtube_calls({
            'youtube_search_calls': 2,
            'youtube_channel_calls': 5,
            'youtube_video_calls': 10,
            'youtube_playlist_calls': 3
        })
        assert result == 20


class TestGetTotalGeminiCalls:
    """Tests for get_total_gemini_calls function."""

    def test_empty_data_returns_zero(self):
        """Empty data should return 0."""
        assert get_total_gemini_calls({}) == 0

    def test_sums_all_gemini_calls(self):
        """Should sum all Gemini call types."""
        result = get_total_gemini_calls({
            'gemini_summary_calls': 1,
            'gemini_outreach_calls': 3,
            'gemini_similarity_calls': 5,
            'gemini_relevance_calls': 10
        })
        assert result == 19


# ============================================================================
# TEST: Date/Time Utilities
# ============================================================================

class TestGetCurrentDatePt:
    """Tests for get_current_date_pt function."""

    def test_returns_valid_date_format(self):
        """Should return YYYY-MM-DD format."""
        result = get_current_date_pt()
        assert len(result) == 10
        assert result[4] == '-'
        assert result[7] == '-'

    def test_returns_parseable_date(self):
        """Result should be parseable as a date."""
        result = get_current_date_pt()
        datetime.strptime(result, '%Y-%m-%d')  # Should not raise


class TestGetNextResetTime:
    """Tests for get_next_reset_time function."""

    def test_returns_human_readable_string(self):
        """Should return human-readable time string."""
        result = get_next_reset_time()
        assert 'hour' in result

    def test_result_contains_in_prefix(self):
        """Result should start with 'in'."""
        result = get_next_reset_time()
        assert result.startswith('in ')


# ============================================================================
# TEST: File Persistence
# ============================================================================

class TestLoadDailyQuota:
    """Tests for load_daily_quota function."""

    def test_nonexistent_file_returns_new_quota(self):
        """Missing file should return new daily quota."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            result = load_daily_quota(filepath)
            assert result['youtube_calls'] == 0
            assert result['gemini_calls'] == 0

    def test_stale_date_returns_new_quota(self):
        """File from different day should return new quota."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            # Save old quota
            old_quota = {'date': '2020-01-01', 'youtube_calls': 100}
            with open(filepath, 'w') as f:
                json.dump(old_quota, f)

            result = load_daily_quota(filepath)
            assert result['youtube_calls'] == 0  # Should be reset

    def test_current_date_loads_existing(self):
        """File from today should load existing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            # Save today's quota
            today_quota = {
                'date': get_current_date_pt(),
                'youtube_calls': 50,
                'gemini_calls': 10
            }
            with open(filepath, 'w') as f:
                json.dump(today_quota, f)

            result = load_daily_quota(filepath)
            assert result['youtube_calls'] == 50
            assert result['gemini_calls'] == 10


class TestSaveDailyQuota:
    """Tests for save_daily_quota function."""

    def test_creates_file(self):
        """Should create quota file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            data = {'date': '2024-01-15', 'youtube_calls': 10}
            result = save_daily_quota(data, filepath)
            assert result is True
            assert os.path.exists(filepath)

    def test_saves_correct_data(self):
        """Should save correct data to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            data = {'date': '2024-01-15', 'youtube_calls': 25, 'gemini_cost_usd': 0.05}
            save_daily_quota(data, filepath)

            with open(filepath, 'r') as f:
                loaded = json.load(f)
            assert loaded['youtube_calls'] == 25
            assert loaded['gemini_cost_usd'] == 0.05


# ============================================================================
# TEST: Tracking Functions
# ============================================================================

class TestTrackApiCall:
    """Tests for track_api_call function."""

    def test_increments_youtube_search(self):
        """Should increment youtube_search_calls."""
        data = {'youtube_search_calls': 0}
        result = track_api_call(data, 'youtube_search')
        assert result['youtube_search_calls'] == 1

    def test_increments_gemini_summary(self):
        """Should increment gemini_summary_calls."""
        data = {'gemini_summary_calls': 5}
        result = track_api_call(data, 'gemini_summary')
        assert result['gemini_summary_calls'] == 6

    def test_unknown_api_does_nothing(self):
        """Unknown API should not modify data."""
        data = {'youtube_search_calls': 0}
        result = track_api_call(data, 'unknown_api')
        assert result == data


class TestTrackTiming:
    """Tests for track_timing function."""

    def test_adds_timing_to_existing(self):
        """Should add to existing timing value."""
        data = {'timings': {'search': 1.0}}
        result = track_timing(data, 'search', 0.5)
        assert result['timings']['search'] == 1.5

    def test_creates_timings_dict_if_missing(self):
        """Should create timings dict if missing."""
        data = {}
        result = track_timing(data, 'search', 1.0)
        assert 'timings' in result


class TestTrackSimilarityScores:
    """Tests for track_similarity_scores function."""

    def test_stores_simplified_scores(self):
        """Should store simplified similarity data."""
        data = {}
        channels = [
            {
                'channel_title': 'Test Channel',
                'similarity': {
                    'total_score': 75.5,
                    'breakdown': {'tag_score': 20},
                    'match_reasons': ['High overlap']
                }
            }
        ]
        result = track_similarity_scores(data, channels)
        assert len(result['similarity_details']) == 1
        assert result['similarity_details'][0]['channel'] == 'Test Channel'
        assert result['similarity_details'][0]['total_score'] == 75.5

    def test_limits_to_20_channels(self):
        """Should limit to 20 channels max."""
        data = {}
        channels = [{'channel_title': f'Channel {i}', 'similarity': {}} for i in range(30)]
        result = track_similarity_scores(data, channels)
        assert len(result['similarity_details']) == 20


class TestAccumulateToDailyQuota:
    """Tests for accumulate_to_daily_quota function."""

    def test_adds_to_daily_totals(self):
        """Should add current usage to daily totals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            debug_data = {
                'youtube_search_calls': 2,
                'youtube_channel_calls': 10,
                'gemini_summary_calls': 1
            }
            daily_quota = {
                'date': get_current_date_pt(),
                'youtube_calls': 5,
                'gemini_calls': 0,
                'youtube_units': 100,
                'gemini_cost_usd': 0.0
            }
            result = accumulate_to_daily_quota(debug_data, daily_quota, filepath)

            # Should add 12 YouTube calls (2 search + 10 channel)
            assert result['youtube_calls'] == 17  # 5 + 12
            # Should add 1 Gemini call
            assert result['gemini_calls'] == 1


class TestCreateEmptyDebugData:
    """Tests for create_empty_debug_data function."""

    def test_returns_dict_with_all_fields(self):
        """Should return dict with all required fields."""
        result = create_empty_debug_data()
        assert 'youtube_search_calls' in result
        assert 'gemini_summary_calls' in result
        assert 'timings' in result
        assert 'similarity_details' in result

    def test_all_counters_are_zero(self):
        """All counters should be zero."""
        result = create_empty_debug_data()
        assert result['youtube_search_calls'] == 0
        assert result['gemini_summary_calls'] == 0
        assert result['timings']['total'] == 0.0


# ============================================================================
# TEST: Per-Key Quota Isolation
# ============================================================================

class TestPerKeyQuota:
    """Tests for per-key quota bucketing via load/save_daily_quota with fingerprint."""

    def test_fingerprint_bucket_isolated(self):
        """Saving under fingerprint A should not affect fingerprint B's bucket."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            fp_a = key_fingerprint("key-a")
            fp_b = key_fingerprint("key-b")

            # Load and mutate bucket for fingerprint A
            bucket_a = load_daily_quota(filepath, fp_a)
            bucket_a['youtube_units'] = 50
            save_daily_quota(bucket_a, filepath, fp_a)

            # Load bucket for fingerprint B — should be untouched
            bucket_b = load_daily_quota(filepath, fp_b)
            assert bucket_b['youtube_units'] == 0

    def test_fingerprint_round_trip(self):
        """Saved per-key bucket values should be reloaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            fp = key_fingerprint("roundtrip-key")

            bucket = load_daily_quota(filepath, fp)
            bucket['youtube_calls'] = 7
            bucket['gemini_calls'] = 3
            bucket['youtube_units'] = 210
            bucket['gemini_cost_usd'] = 0.05
            save_daily_quota(bucket, filepath, fp)

            reloaded = load_daily_quota(filepath, fp)
            assert reloaded['youtube_calls'] == 7
            assert reloaded['gemini_calls'] == 3
            assert reloaded['youtube_units'] == 210
            assert reloaded['gemini_cost_usd'] == 0.05

    def test_no_fingerprint_uses_legacy_path(self):
        """load_daily_quota without fingerprint should return a flat dict with youtube_calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'quota.json')
            result = load_daily_quota(filepath)
            assert 'youtube_calls' in result
            assert 'date' in result

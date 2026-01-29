"""
Unit tests for app/core/similarity.py

Tests cover:
- Similarity metrics (jaccard_similarity, overlap_count)
- Subscriber tier utilities (get_subscriber_similarity, is_within_tier_range)
- Main similarity scoring (calculate_similarity_score)
- Gemini-enhanced similarity (gemini_similarity_analysis)
- Combined scoring (calculate_final_score)
- Batch ranking (rank_channels_by_similarity)
- Subscriber filtering (filter_by_subscriber_range)
- Explanation generation (generate_match_explanation)
- Callback interface (SimilarityCallbacks)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from app.core.similarity import (
    SimilarityCallbacks,
    jaccard_similarity,
    overlap_count,
    get_subscriber_similarity,
    is_within_tier_range,
    calculate_similarity_score,
    gemini_similarity_analysis,
    calculate_final_score,
    rank_channels_by_similarity,
    filter_by_subscriber_range,
    generate_match_explanation,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_seed_profile():
    """Standard seed profile for testing."""
    return {
        'channel_id': 'UC_seed_123',
        'channel_name': 'Seed Channel',
        'subscriber_count': 100000,
        'common_tags': ['python', 'programming', 'tutorial', 'coding', 'developer'],
        'primary_keywords': ['python', 'programming'],
        'secondary_keywords': ['tutorial', 'coding'],
        'avg_engagement_rate': 0.05,
        'upload_frequency': 4.0,
        'recent_titles': ['Python Basics', 'Advanced Python'],
        'description_summary': 'A programming channel',
    }


@pytest.fixture
def sample_candidate():
    """Standard candidate channel for testing."""
    return {
        'channel_id': 'UC_candidate_456',
        'channel_name': 'Candidate Channel',
        'channel_title': 'Candidate Channel',
        'subscribers': 80000,
        'tags': ['python', 'programming', 'web', 'django'],
        'keywords': ['python', 'web', 'django'],
        'engagement_rate': 0.045,
        'upload_frequency': 3.5,
        'recent_titles': ['Django Tutorial', 'Python Web Dev'],
    }


@pytest.fixture
def mock_callbacks():
    """Mock callbacks for testing."""
    return SimilarityCallbacks(
        on_info=Mock(),
        on_warning=Mock(),
        on_success=Mock(),
        on_api_call=Mock(),
        debug_mode=True
    )


# ============================================================================
# TEST: SimilarityCallbacks
# ============================================================================

class TestSimilarityCallbacks:
    """Tests for SimilarityCallbacks dataclass."""

    def test_default_callbacks_are_none(self):
        """Default callbacks should be None."""
        callbacks = SimilarityCallbacks()
        assert callbacks.on_info is None
        assert callbacks.on_warning is None
        assert callbacks.on_success is None
        assert callbacks.on_api_call is None
        assert callbacks.debug_mode is False

    def test_callbacks_can_be_set(self):
        """Callbacks can be set to callable functions."""
        on_info = Mock()
        on_warning = Mock()
        callbacks = SimilarityCallbacks(
            on_info=on_info,
            on_warning=on_warning,
            debug_mode=True
        )
        assert callbacks.on_info is on_info
        assert callbacks.on_warning is on_warning
        assert callbacks.debug_mode is True

    def test_callbacks_are_callable(self, mock_callbacks):
        """Callbacks should be callable."""
        mock_callbacks.on_info("test message")
        mock_callbacks.on_info.assert_called_once_with("test message")


# ============================================================================
# TEST: Similarity Metrics
# ============================================================================

class TestJaccardSimilarity:
    """Tests for jaccard_similarity function."""

    def test_identical_sets_return_one(self):
        """Identical sets should have similarity of 1.0."""
        result = jaccard_similarity({'a', 'b', 'c'}, {'a', 'b', 'c'})
        assert result == 1.0

    def test_disjoint_sets_return_zero(self):
        """Completely different sets should have similarity of 0.0."""
        result = jaccard_similarity({'a', 'b'}, {'c', 'd'})
        assert result == 0.0

    def test_partial_overlap(self):
        """Partial overlap should return correct Jaccard coefficient."""
        # J({a,b,c}, {b,c,d}) = |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        result = jaccard_similarity({'a', 'b', 'c'}, {'b', 'c', 'd'})
        assert result == 0.5

    def test_empty_sets_return_zero(self):
        """Empty sets should return 0.0 (avoid division by zero)."""
        result = jaccard_similarity(set(), set())
        assert result == 0.0

    def test_one_empty_set_returns_zero(self):
        """One empty set should return 0.0."""
        result = jaccard_similarity({'a', 'b'}, set())
        assert result == 0.0

    def test_accepts_list_input(self):
        """Should convert lists to sets."""
        result = jaccard_similarity(['a', 'b', 'c'], ['b', 'c', 'd'])
        assert result == 0.5

    def test_handles_duplicates_in_input(self):
        """Duplicates should be ignored (set behavior)."""
        result = jaccard_similarity(['a', 'a', 'b'], ['a', 'b', 'b'])
        assert result == 1.0


class TestOverlapCount:
    """Tests for overlap_count function."""

    def test_identical_sets_return_full_count(self):
        """Identical sets should return count of all items."""
        result = overlap_count({'a', 'b', 'c'}, {'a', 'b', 'c'})
        assert result == 3

    def test_disjoint_sets_return_zero(self):
        """Disjoint sets should return 0."""
        result = overlap_count({'a', 'b'}, {'c', 'd'})
        assert result == 0

    def test_partial_overlap_returns_correct_count(self):
        """Partial overlap should return count of common items."""
        result = overlap_count({'a', 'b', 'c'}, {'b', 'c', 'd'})
        assert result == 2

    def test_empty_sets_return_zero(self):
        """Empty sets should return 0."""
        result = overlap_count(set(), set())
        assert result == 0


# ============================================================================
# TEST: Subscriber Tier Utilities
# ============================================================================

class TestGetSubscriberSimilarity:
    """Tests for get_subscriber_similarity function."""

    def test_same_size_returns_one(self):
        """Same subscriber count should return 1.0."""
        result = get_subscriber_similarity(100000, 100000)
        assert result == 1.0

    def test_half_size_returns_half(self):
        """Half subscriber count should return 0.5."""
        result = get_subscriber_similarity(50000, 100000)
        assert result == 0.5

    def test_double_size_returns_half(self):
        """Double subscriber count should return 0.5 (symmetric)."""
        result = get_subscriber_similarity(200000, 100000)
        assert result == 0.5

    def test_tenth_size_returns_tenth(self):
        """10x difference should return 0.1."""
        result = get_subscriber_similarity(10000, 100000)
        assert result == 0.1

    def test_zero_seed_returns_zero(self):
        """Zero seed subscribers should return 0.0."""
        result = get_subscriber_similarity(100000, 0)
        assert result == 0.0

    def test_zero_candidate_returns_zero(self):
        """Zero candidate subscribers should return 0.0."""
        result = get_subscriber_similarity(0, 100000)
        assert result == 0.0

    def test_both_zero_returns_zero(self):
        """Both zero should return 0.0."""
        result = get_subscriber_similarity(0, 0)
        assert result == 0.0


class TestIsWithinTierRange:
    """Tests for is_within_tier_range function."""

    def test_same_count_within_range(self):
        """Same count should be within range."""
        assert is_within_tier_range(100000, 100000, 0.5) is True

    def test_just_below_upper_bound(self):
        """Just below upper bound should be within range."""
        # 100K seed, 0.5 tolerance = 50K-150K range
        assert is_within_tier_range(149000, 100000, 0.5) is True

    def test_above_upper_bound(self):
        """Above upper bound should be outside range."""
        assert is_within_tier_range(160000, 100000, 0.5) is False

    def test_just_above_lower_bound(self):
        """Just above lower bound should be within range."""
        assert is_within_tier_range(51000, 100000, 0.5) is True

    def test_below_lower_bound(self):
        """Below lower bound should be outside range."""
        assert is_within_tier_range(40000, 100000, 0.5) is False

    def test_narrow_tolerance(self):
        """Narrow tolerance (0.3) should have tighter range."""
        # 100K seed, 0.3 tolerance = 70K-130K range
        assert is_within_tier_range(75000, 100000, 0.3) is True
        assert is_within_tier_range(60000, 100000, 0.3) is False


# ============================================================================
# TEST: Main Similarity Scoring
# ============================================================================

class TestCalculateSimilarityScore:
    """Tests for calculate_similarity_score function."""

    def test_returns_total_score(self, sample_seed_profile, sample_candidate):
        """Should return a total_score field."""
        result = calculate_similarity_score(sample_candidate, sample_seed_profile)
        assert 'total_score' in result
        assert isinstance(result['total_score'], float)

    def test_returns_match_reasons(self, sample_seed_profile, sample_candidate):
        """Should return a match_reasons list."""
        result = calculate_similarity_score(sample_candidate, sample_seed_profile)
        assert 'match_reasons' in result
        assert isinstance(result['match_reasons'], list)

    def test_score_in_valid_range(self, sample_seed_profile, sample_candidate):
        """Score should be between 0 and 100."""
        result = calculate_similarity_score(sample_candidate, sample_seed_profile)
        assert 0 <= result['total_score'] <= 100

    def test_debug_mode_includes_breakdown(self, sample_seed_profile, sample_candidate):
        """Debug mode should include breakdown."""
        result = calculate_similarity_score(sample_candidate, sample_seed_profile, debug=True)
        assert 'breakdown' in result
        assert 'tag_score' in result['breakdown']
        assert 'keyword_score' in result['breakdown']
        assert 'subscriber_score' in result['breakdown']
        assert 'engagement_score' in result['breakdown']
        assert 'frequency_score' in result['breakdown']

    def test_no_breakdown_without_debug(self, sample_seed_profile, sample_candidate):
        """Without debug mode, breakdown should not be included."""
        result = calculate_similarity_score(sample_candidate, sample_seed_profile, debug=False)
        assert 'breakdown' not in result

    def test_empty_tags_scores_zero_for_tags(self, sample_seed_profile):
        """Empty tags should give zero tag score."""
        candidate = {'tags': [], 'keywords': [], 'subscribers': 100000,
                     'engagement_rate': 0.05, 'upload_frequency': 4.0}
        result = calculate_similarity_score(candidate, sample_seed_profile, debug=True)
        assert result['breakdown']['tag_score'] == 0.0

    def test_identical_channels_high_score(self, sample_seed_profile):
        """Identical data should give high score."""
        candidate = {
            'tags': sample_seed_profile['common_tags'].copy(),
            'keywords': sample_seed_profile['primary_keywords'] + sample_seed_profile['secondary_keywords'],
            'subscribers': sample_seed_profile['subscriber_count'],
            'engagement_rate': sample_seed_profile['avg_engagement_rate'],
            'upload_frequency': sample_seed_profile['upload_frequency'],
        }
        result = calculate_similarity_score(candidate, sample_seed_profile)
        assert result['total_score'] >= 90  # Should be very high

    def test_completely_different_channel_low_score(self, sample_seed_profile):
        """Completely different data should give low score."""
        candidate = {
            'tags': ['cooking', 'recipes', 'food'],
            'keywords': ['cooking', 'baking'],
            'subscribers': 1000,  # Very different
            'engagement_rate': 0.5,  # Very different
            'upload_frequency': 30.0,  # Very different
        }
        result = calculate_similarity_score(candidate, sample_seed_profile)
        assert result['total_score'] < 30  # Should be low

    def test_handles_missing_candidate_fields(self, sample_seed_profile):
        """Should handle missing fields gracefully."""
        candidate = {}  # Empty candidate
        result = calculate_similarity_score(candidate, sample_seed_profile)
        assert result['total_score'] >= 0  # Should not crash

    def test_non_list_tags_handled(self, sample_seed_profile):
        """Non-list tags should be handled gracefully."""
        candidate = {'tags': 'not a list', 'keywords': None, 'subscribers': 100000}
        result = calculate_similarity_score(candidate, sample_seed_profile)
        assert result['total_score'] >= 0  # Should not crash


# ============================================================================
# TEST: Gemini-Enhanced Similarity
# ============================================================================

class TestGeminiSimilarityAnalysis:
    """Tests for gemini_similarity_analysis function."""

    def test_no_api_key_returns_zero(self, sample_seed_profile, sample_candidate):
        """Without API key, should return zero score."""
        result = gemini_similarity_analysis(sample_candidate, sample_seed_profile, "")
        assert result['gemini_score'] == 0
        assert result['gemini_reason'] == 'Gemini not configured'

    def test_none_api_key_returns_zero(self, sample_seed_profile, sample_candidate):
        """None API key should return zero score."""
        result = gemini_similarity_analysis(sample_candidate, sample_seed_profile, None)
        assert result['gemini_score'] == 0

    @patch('app.core.similarity.genai')
    def test_successful_analysis(self, mock_genai, sample_seed_profile, sample_candidate):
        """Successful API call should return parsed score."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Score: 8/10\nReason: Very similar content style and topics"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        result = gemini_similarity_analysis(
            sample_candidate, sample_seed_profile, "test_api_key"
        )
        assert result['gemini_score'] == 8
        assert 'similar' in result['gemini_reason'].lower()

    @patch('app.core.similarity.genai')
    def test_api_error_returns_zero(self, mock_genai, sample_seed_profile, sample_candidate, mock_callbacks):
        """API error should return zero score and call warning callback."""
        mock_genai.configure.side_effect = Exception("API Error")

        result = gemini_similarity_analysis(
            sample_candidate, sample_seed_profile, "test_api_key", mock_callbacks
        )
        assert result['gemini_score'] == 0
        assert 'error' in result['gemini_reason'].lower()
        mock_callbacks.on_warning.assert_called()

    @patch('app.core.similarity.genai')
    def test_tracks_api_call_in_debug_mode(self, mock_genai, sample_seed_profile, sample_candidate, mock_callbacks):
        """Should track API call when debug mode is enabled."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Score: 7/10\nReason: Similar topics"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        gemini_similarity_analysis(
            sample_candidate, sample_seed_profile, "test_api_key", mock_callbacks
        )
        mock_callbacks.on_api_call.assert_called_with('gemini_similarity')


# ============================================================================
# TEST: Combined Scoring
# ============================================================================

class TestCalculateFinalScore:
    """Tests for calculate_final_score function."""

    def test_without_api_key_uses_algorithmic_only(self, sample_seed_profile, sample_candidate):
        """Without API key, should use algorithmic score only."""
        result = calculate_final_score(sample_candidate, sample_seed_profile)
        assert result['total_score'] == result['algorithmic_score']
        assert result['gemini_score'] == 0
        assert result['gemini_reason'] == 'Not configured'

    def test_returns_all_expected_fields(self, sample_seed_profile, sample_candidate):
        """Should return all expected fields."""
        result = calculate_final_score(sample_candidate, sample_seed_profile)
        assert 'total_score' in result
        assert 'algorithmic_score' in result
        assert 'gemini_score' in result
        assert 'match_reasons' in result
        assert 'gemini_reason' in result

    def test_debug_includes_breakdown(self, sample_seed_profile, sample_candidate):
        """Debug mode should include breakdown."""
        result = calculate_final_score(sample_candidate, sample_seed_profile, debug=True)
        assert 'breakdown' in result

    @patch('app.core.similarity.gemini_similarity_analysis')
    def test_blends_scores_with_api_key(self, mock_gemini, sample_seed_profile, sample_candidate):
        """With API key and successful Gemini, should blend scores."""
        mock_gemini.return_value = {'gemini_score': 10, 'gemini_reason': 'Perfect match'}

        result = calculate_final_score(
            sample_candidate, sample_seed_profile, gemini_api_key="test_key"
        )
        # Score should be blended (80% algo + 20% gemini)
        # Gemini score 10 = 100 on normalized scale
        assert result['gemini_score'] == 10
        assert result['total_score'] != result['algorithmic_score']

    @patch('app.core.similarity.gemini_similarity_analysis')
    def test_gemini_zero_falls_back_to_algorithmic(self, mock_gemini, sample_seed_profile, sample_candidate):
        """If Gemini returns 0, should use algorithmic score only."""
        mock_gemini.return_value = {'gemini_score': 0, 'gemini_reason': 'Error'}

        result = calculate_final_score(
            sample_candidate, sample_seed_profile, gemini_api_key="test_key"
        )
        assert result['total_score'] == result['algorithmic_score']

    def test_normalizes_channel_title_to_channel_name(self, sample_seed_profile):
        """Should normalize channel_title to channel_name."""
        candidate = {'channel_title': 'Test Channel', 'tags': [], 'keywords': []}
        result = calculate_final_score(candidate, sample_seed_profile)
        assert 'channel_name' in candidate  # Should be added


# ============================================================================
# TEST: Batch Ranking
# ============================================================================

class TestRankChannelsBySimilarity:
    """Tests for rank_channels_by_similarity function."""

    def test_returns_sorted_list(self, sample_seed_profile):
        """Should return candidates sorted by score descending."""
        candidates = [
            {'channel_id': '1', 'tags': ['python'], 'keywords': ['python'],
             'subscribers': 100000, 'engagement_rate': 0.05, 'upload_frequency': 4.0},
            {'channel_id': '2', 'tags': ['python', 'programming', 'tutorial'],
             'keywords': ['python', 'programming'], 'subscribers': 100000,
             'engagement_rate': 0.05, 'upload_frequency': 4.0},
        ]
        result = rank_channels_by_similarity(candidates, sample_seed_profile)

        # Second candidate should rank higher (more tag overlap)
        assert result[0]['channel_id'] == '2'

    def test_adds_similarity_field(self, sample_seed_profile, sample_candidate):
        """Should add similarity field to each candidate."""
        candidates = [sample_candidate]
        result = rank_channels_by_similarity(candidates, sample_seed_profile)

        assert 'similarity' in result[0]
        assert 'total_score' in result[0]['similarity']

    def test_empty_candidates_returns_empty(self, sample_seed_profile, mock_callbacks):
        """Empty candidates should return empty list."""
        result = rank_channels_by_similarity([], sample_seed_profile, callbacks=mock_callbacks)
        assert result == []
        mock_callbacks.on_warning.assert_called()  # Should warn about no candidates

    def test_callbacks_invoked(self, sample_seed_profile, sample_candidate, mock_callbacks):
        """Callbacks should be invoked during ranking."""
        candidates = [sample_candidate]
        rank_channels_by_similarity(candidates, sample_seed_profile, callbacks=mock_callbacks)

        mock_callbacks.on_info.assert_called()  # Start message
        mock_callbacks.on_success.assert_called()  # Completion message

    @patch('app.core.similarity.calculate_final_score')
    def test_gemini_limit_respected(self, mock_final, sample_seed_profile):
        """Gemini analysis should only apply to top N candidates."""
        mock_final.return_value = {
            'total_score': 50, 'algorithmic_score': 50,
            'gemini_score': 5, 'gemini_reason': 'Test',
            'match_reasons': []
        }

        candidates = [
            {'channel_id': str(i), 'tags': [], 'keywords': [],
             'subscribers': 100000, 'engagement_rate': 0.05, 'upload_frequency': 4.0}
            for i in range(15)
        ]

        rank_channels_by_similarity(
            candidates, sample_seed_profile,
            gemini_api_key="test_key", gemini_limit=5
        )

        # Should only call calculate_final_score 5 times
        assert mock_final.call_count == 5


# ============================================================================
# TEST: Subscriber Filtering
# ============================================================================

class TestFilterBySubscriberRange:
    """Tests for filter_by_subscriber_range function."""

    def test_filters_outside_range(self):
        """Should filter out candidates outside range."""
        candidates = [
            {'channel_id': '1', 'subscribers': 75000},
            {'channel_id': '2', 'subscribers': 200000},  # Outside
            {'channel_id': '3', 'subscribers': 125000},
        ]
        result = filter_by_subscriber_range(candidates, 100000, 0.5)

        assert len(result) == 2
        assert all(c['subscribers'] <= 150000 for c in result)
        assert all(c['subscribers'] >= 50000 for c in result)

    def test_empty_candidates_returns_empty(self):
        """Empty candidates should return empty list."""
        result = filter_by_subscriber_range([], 100000, 0.5)
        assert result == []

    def test_all_outside_range_returns_empty(self):
        """All outside range should return empty list."""
        candidates = [
            {'channel_id': '1', 'subscribers': 10000},
            {'channel_id': '2', 'subscribers': 500000},
        ]
        result = filter_by_subscriber_range(candidates, 100000, 0.2)
        assert result == []

    def test_callbacks_invoked(self, mock_callbacks):
        """Should invoke on_info callback."""
        candidates = [{'channel_id': '1', 'subscribers': 100000}]
        filter_by_subscriber_range(candidates, 100000, 0.5, mock_callbacks)
        mock_callbacks.on_info.assert_called()


# ============================================================================
# TEST: Explanation Generation
# ============================================================================

class TestGenerateMatchExplanation:
    """Tests for generate_match_explanation function."""

    def test_includes_score_header(self, sample_candidate, sample_seed_profile):
        """Should include score in header."""
        sample_candidate['similarity'] = {
            'total_score': 75.5,
            'match_reasons': ['Test reason'],
            'breakdown': {}
        }
        result = generate_match_explanation(sample_candidate, sample_seed_profile)
        assert '75.5/100' in result

    def test_includes_match_reasons(self, sample_candidate, sample_seed_profile):
        """Should include match reasons."""
        sample_candidate['similarity'] = {
            'total_score': 75.5,
            'match_reasons': ['Strong tag overlap', 'Similar audience'],
            'breakdown': {}
        }
        result = generate_match_explanation(sample_candidate, sample_seed_profile)
        assert 'Strong tag overlap' in result
        assert 'Similar audience' in result

    def test_detailed_includes_breakdown(self, sample_candidate, sample_seed_profile):
        """Detailed mode should include breakdown."""
        sample_candidate['similarity'] = {
            'total_score': 75.5,
            'match_reasons': [],
            'breakdown': {
                'tag_score': 20.0,
                'common_tags': 5,
                'keyword_score': 15.0,
                'common_keywords': 3,
                'subscriber_score': 12.0,
                'engagement_score': 10.0,
                'frequency_score': 6.0,
            }
        }
        result = generate_match_explanation(sample_candidate, sample_seed_profile, detailed=True)
        assert 'Tags' in result
        assert 'Keywords' in result
        assert 'Audience Size' in result

    def test_non_detailed_excludes_breakdown(self, sample_candidate, sample_seed_profile):
        """Non-detailed mode should exclude breakdown."""
        sample_candidate['similarity'] = {
            'total_score': 75.5,
            'match_reasons': [],
            'breakdown': {'tag_score': 20.0}
        }
        result = generate_match_explanation(sample_candidate, sample_seed_profile, detailed=False)
        assert 'Score Breakdown' not in result

    def test_includes_gemini_insight(self, sample_candidate, sample_seed_profile):
        """Should include Gemini insight if available."""
        sample_candidate['similarity'] = {
            'total_score': 75.5,
            'match_reasons': [],
            'gemini_score': 8,
            'gemini_reason': 'Similar content style'
        }
        result = generate_match_explanation(sample_candidate, sample_seed_profile)
        assert 'AI Analysis' in result
        assert 'Similar content style' in result

    def test_handles_missing_similarity(self, sample_seed_profile):
        """Should handle candidate without similarity field."""
        candidate = {'channel_id': 'test'}
        result = generate_match_explanation(candidate, sample_seed_profile)
        assert '0.0/100' in result  # Default score

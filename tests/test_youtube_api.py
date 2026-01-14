"""
Tests for core.youtube_api module

Tests cover:
- Hybrid channel search (video + channel name)
- Multi-term search with OR logic
- Channel statistics fetching
- Video details fetching
- Error handling and edge cases

All tests use mocked YouTube API clients for isolation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from googleapiclient.errors import HttpError

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.youtube_api import (
    SearchResult,
    ChannelStatsResult,
    VideoDetailsResult,
    search_channels_hybrid,
    search_channels_multi_term,
    get_channel_stats,
    get_video_details,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_youtube():
    """Create a mock YouTube API service."""
    return Mock()


@pytest.fixture
def mock_http_error():
    """Create a mock HttpError for error testing."""
    resp = Mock()
    resp.status = 403
    return HttpError(resp, b'Quota exceeded')


# ============================================================================
# search_channels_hybrid TESTS
# ============================================================================

class TestSearchChannelsHybrid:
    """Tests for the hybrid channel search function."""

    def test_empty_search_returns_empty_list(self, mock_youtube):
        """Search with no results returns empty channels list."""
        # Mock video search returning no items
        mock_youtube.search().list().execute.return_value = {'items': []}

        result = search_channels_hybrid(mock_youtube, "nonexistent", "US")

        assert isinstance(result, SearchResult)
        assert result.channels == []
        assert result.warnings == []
        assert result.api_calls >= 1

    def test_video_search_extracts_channel_ids(self, mock_youtube):
        """Video search correctly extracts channel IDs and titles."""
        mock_youtube.search().list().execute.side_effect = [
            # Video search response
            {
                'items': [
                    {'snippet': {'channelId': 'UC1', 'channelTitle': 'Channel One'}},
                    {'snippet': {'channelId': 'UC1', 'channelTitle': 'Channel One'}},
                    {'snippet': {'channelId': 'UC2', 'channelTitle': 'Channel Two'}},
                ]
            },
            # Channel search response
            {'items': []},
        ]

        result = search_channels_hybrid(mock_youtube, "manga", "US", max_videos=50)

        assert len(result.channels) == 2
        # UC1 has 2 video matches (2*10=20), UC2 has 1 (1*10=10)
        assert result.channels[0]['channel_id'] == 'UC1'
        assert result.channels[0]['match_score'] == 20
        assert result.channels[1]['channel_id'] == 'UC2'
        assert result.channels[1]['match_score'] == 10

    def test_channel_name_match_adds_bonus(self, mock_youtube):
        """Channel name match adds 5 bonus points."""
        mock_youtube.search().list().execute.side_effect = [
            # Video search - one video from UC1
            {'items': [{'snippet': {'channelId': 'UC1', 'channelTitle': 'MangaFan'}}]},
            # Channel search - UC1 also found by name
            {'items': [{'id': {'channelId': 'UC1'}, 'snippet': {'title': 'MangaFan'}}]},
        ]

        result = search_channels_hybrid(mock_youtube, "manga", "US", max_videos=50)

        # 1 video match (10 points) + name match bonus (5 points) = 15
        assert len(result.channels) == 1
        assert result.channels[0]['match_score'] == 15

    def test_channel_only_in_name_search(self, mock_youtube):
        """Channels found only in name search get score 5."""
        mock_youtube.search().list().execute.side_effect = [
            # Video search - empty
            {'items': []},
            # Channel search - UC1 found by name only
            {'items': [{'id': {'channelId': 'UC1'}, 'snippet': {'title': 'MangaReviews'}}]},
        ]

        result = search_channels_hybrid(mock_youtube, "manga", "US", max_videos=50)

        assert len(result.channels) == 1
        assert result.channels[0]['match_score'] == 5  # Name match only

    def test_video_search_error_returns_warning(self, mock_youtube, mock_http_error):
        """Video search API error is captured as warning."""
        mock_youtube.search().list().execute.side_effect = [
            mock_http_error,  # Video search fails
            {'items': []},   # Channel search succeeds
        ]

        result = search_channels_hybrid(mock_youtube, "manga", "US")

        assert len(result.warnings) == 1
        assert "Video search error" in result.warnings[0]

    def test_channel_search_error_returns_warning(self, mock_youtube, mock_http_error):
        """Channel name search API error is captured as warning."""
        mock_youtube.search().list().execute.side_effect = [
            {'items': [{'snippet': {'channelId': 'UC1', 'channelTitle': 'MangaFan'}}]},
            mock_http_error,  # Channel search fails
        ]

        result = search_channels_hybrid(mock_youtube, "manga", "US", max_videos=50)

        assert len(result.warnings) == 1
        assert "Channel name search failed" in result.warnings[0]
        # Should still have video results
        assert len(result.channels) == 1

    def test_region_code_passed_to_api(self, mock_youtube):
        """Region code is passed to both video and channel searches."""
        mock_youtube.search().list().execute.return_value = {'items': []}

        search_channels_hybrid(mock_youtube, "manga", "ES", max_videos=50)

        # Verify list() was called with regionCode in kwargs
        # The mock chain is search().list(**kwargs).execute()
        calls = mock_youtube.search().list.call_args_list
        region_codes_found = [
            call.kwargs.get('regionCode') for call in calls
            if call.kwargs.get('regionCode')
        ]
        assert 'ES' in region_codes_found, f"Expected 'ES' in region codes: {calls}"

    def test_api_callback_invoked(self, mock_youtube):
        """on_api_call callback is called for each API request."""
        mock_youtube.search().list().execute.side_effect = [
            {'items': []},
            {'items': []},
        ]
        tracker = Mock()

        search_channels_hybrid(mock_youtube, "manga", "US", on_api_call=tracker)

        # Should be called at least twice (video search + channel search)
        assert tracker.call_count >= 2

    def test_pagination_fetches_multiple_pages(self, mock_youtube):
        """Search continues fetching until max_videos reached."""
        # First page returns 50 items with a nextPageToken
        first_page = {
            'items': [{'snippet': {'channelId': f'UC{i}', 'channelTitle': f'Channel {i}'}}
                      for i in range(50)],
            'nextPageToken': 'page2'
        }
        # Second page returns 30 more
        second_page = {
            'items': [{'snippet': {'channelId': f'UC{i+50}', 'channelTitle': f'Channel {i+50}'}}
                      for i in range(30)],
        }
        mock_youtube.search().list().execute.side_effect = [
            first_page, second_page, {'items': []}
        ]

        result = search_channels_hybrid(mock_youtube, "manga", "US", max_videos=80)

        # Should have fetched from both pages
        assert len(result.channels) == 80

    def test_results_sorted_by_match_score(self, mock_youtube):
        """Results are sorted by match_score descending."""
        mock_youtube.search().list().execute.side_effect = [
            {
                'items': [
                    {'snippet': {'channelId': 'UC_LOW', 'channelTitle': 'Low'}},
                    {'snippet': {'channelId': 'UC_HIGH', 'channelTitle': 'High'}},
                    {'snippet': {'channelId': 'UC_HIGH', 'channelTitle': 'High'}},
                    {'snippet': {'channelId': 'UC_HIGH', 'channelTitle': 'High'}},
                ]
            },
            {'items': []},
        ]

        result = search_channels_hybrid(mock_youtube, "test", "US", max_videos=50)

        assert result.channels[0]['channel_id'] == 'UC_HIGH'  # 3 matches = 30
        assert result.channels[1]['channel_id'] == 'UC_LOW'   # 1 match = 10


# ============================================================================
# search_channels_multi_term TESTS
# ============================================================================

class TestSearchChannelsMultiTerm:
    """Tests for multi-term (comma-separated) search."""

    def test_single_term_uses_hybrid_search(self, mock_youtube):
        """Single term delegates to hybrid search."""
        mock_youtube.search().list().execute.side_effect = [
            {'items': [{'snippet': {'channelId': 'UC1', 'channelTitle': 'Test'}}]},
            {'items': []},
        ]

        result = search_channels_multi_term(mock_youtube, "manga", "US")

        assert len(result.channels) == 1

    def test_empty_query_returns_empty(self, mock_youtube):
        """Empty query returns empty result."""
        result = search_channels_multi_term(mock_youtube, "", "US")

        assert result.channels == []
        assert result.api_calls == 0

    def test_whitespace_query_returns_empty(self, mock_youtube):
        """Whitespace-only query returns empty result."""
        result = search_channels_multi_term(mock_youtube, "   ", "US")

        assert result.channels == []

    def test_two_terms_merged_correctly(self, mock_youtube):
        """Two search terms are merged with OR logic."""
        mock_youtube.search().list().execute.side_effect = [
            # First term "manga" finds UC1
            {'items': [{'snippet': {'channelId': 'UC1', 'channelTitle': 'MangaFan'}}]},
            {'items': []},
            # Second term "anime" finds UC1 and UC2
            {'items': [
                {'snippet': {'channelId': 'UC1', 'channelTitle': 'MangaFan'}},
                {'snippet': {'channelId': 'UC2', 'channelTitle': 'AnimeFan'}},
            ]},
            {'items': []},
        ]

        result = search_channels_multi_term(mock_youtube, "manga, anime", "US", max_videos_per_term=50)

        # UC1 should have highest score (found in both searches)
        assert result.channels[0]['channel_id'] == 'UC1'
        assert result.channels[0]['match_score'] == 20  # 10 + 10

    def test_three_terms_truncated_to_two(self, mock_youtube):
        """More than 2 terms results in warning and truncation."""
        mock_youtube.search().list().execute.return_value = {'items': []}

        result = search_channels_multi_term(
            mock_youtube, "manga, anime, gaming", "US"
        )

        assert len(result.warnings) == 1
        assert "limited to 2 terms" in result.warnings[0].lower()

    def test_max_channels_limits_results(self, mock_youtube):
        """max_channels parameter limits final result count."""
        mock_youtube.search().list().execute.side_effect = [
            {'items': [{'snippet': {'channelId': f'UC{i}', 'channelTitle': f'Ch{i}'}}
                       for i in range(20)]},
            {'items': []},
        ]

        result = search_channels_multi_term(
            mock_youtube, "test", "US", max_channels=5
        )

        assert len(result.channels) == 5

    def test_progress_callback_invoked(self, mock_youtube):
        """on_progress callback is called during multi-term search."""
        mock_youtube.search().list().execute.return_value = {'items': []}
        progress = Mock()

        search_channels_multi_term(
            mock_youtube, "manga, anime", "US", on_progress=progress
        )

        assert progress.call_count >= 1


# ============================================================================
# get_channel_stats TESTS
# ============================================================================

class TestGetChannelStats:
    """Tests for channel statistics fetching."""

    def test_empty_list_returns_empty(self, mock_youtube):
        """Empty channel list returns empty stats."""
        result = get_channel_stats(mock_youtube, [])

        assert result.stats == []
        assert result.api_calls == 0

    def test_single_channel_returns_stats(self, mock_youtube):
        """Single channel stats are correctly extracted."""
        mock_youtube.channels().list().execute.return_value = {
            'items': [{
                'id': 'UC1',
                'snippet': {
                    'country': 'US',
                    'publishedAt': '2020-01-01T00:00:00Z'
                },
                'statistics': {
                    'subscriberCount': '1000',
                    'viewCount': '50000',
                    'videoCount': '100'
                },
                'contentDetails': {
                    'relatedPlaylists': {'uploads': 'UU1'}
                }
            }]
        }

        result = get_channel_stats(mock_youtube, ['UC1'])

        assert len(result.stats) == 1
        assert result.stats[0]['channel_id'] == 'UC1'
        assert result.stats[0]['subscribers'] == 1000
        assert result.stats[0]['views'] == 50000
        assert result.stats[0]['videos'] == 100
        assert result.stats[0]['country'] == 'US'
        assert result.stats[0]['uploads_playlist_id'] == 'UU1'
        assert result.stats[0]['avg_views_per_video'] == 500

    def test_channels_batched_by_50(self, mock_youtube):
        """Channels are fetched in batches of 50."""
        channel_ids = [f'UC{i}' for i in range(75)]

        mock_youtube.channels().list().execute.side_effect = [
            {'items': [
                {'id': f'UC{i}',
                 'snippet': {'country': 'US'},
                 'statistics': {'subscriberCount': '0', 'viewCount': '0', 'videoCount': '0'},
                 'contentDetails': {'relatedPlaylists': {'uploads': f'UU{i}'}}}
                for i in range(50)
            ]},
            {'items': [
                {'id': f'UC{i}',
                 'snippet': {'country': 'US'},
                 'statistics': {'subscriberCount': '0', 'viewCount': '0', 'videoCount': '0'},
                 'contentDetails': {'relatedPlaylists': {'uploads': f'UU{i}'}}}
                for i in range(50, 75)
            ]},
        ]

        result = get_channel_stats(mock_youtube, channel_ids)

        assert len(result.stats) == 75
        assert result.api_calls == 2

    def test_channel_without_uploads_excluded(self, mock_youtube):
        """Channels without uploads playlist are excluded."""
        mock_youtube.channels().list().execute.return_value = {
            'items': [{
                'id': 'UC1',
                'snippet': {'country': 'US'},
                'statistics': {'subscriberCount': '0', 'viewCount': '0', 'videoCount': '0'},
                'contentDetails': {'relatedPlaylists': {}}  # No uploads
            }]
        }

        result = get_channel_stats(mock_youtube, ['UC1'])

        assert result.stats == []

    def test_api_callback_invoked(self, mock_youtube):
        """on_api_call callback is called for each batch."""
        mock_youtube.channels().list().execute.return_value = {
            'items': [{
                'id': 'UC1',
                'snippet': {},
                'statistics': {'subscriberCount': '0', 'viewCount': '0', 'videoCount': '0'},
                'contentDetails': {'relatedPlaylists': {'uploads': 'UU1'}}
            }]
        }
        tracker = Mock()

        get_channel_stats(mock_youtube, ['UC1'], on_api_call=tracker)

        tracker.assert_called_with('youtube_channel')


# ============================================================================
# get_video_details TESTS
# ============================================================================

class TestGetVideoDetails:
    """Tests for video details fetching."""

    def test_empty_channel_list_returns_empty(self, mock_youtube):
        """Empty channel list returns empty videos."""
        result = get_video_details(mock_youtube, [], 10)

        assert result.videos == []
        assert result.api_calls == 0

    def test_channel_without_playlist_skipped(self, mock_youtube):
        """Channels without uploads_playlist_id are skipped."""
        result = get_video_details(
            mock_youtube,
            [{'channel_id': 'UC1'}],  # No uploads_playlist_id
            10
        )

        assert result.videos == []

    def test_single_channel_videos_fetched(self, mock_youtube):
        """Videos are correctly fetched for a single channel."""
        mock_youtube.playlistItems().list().execute.return_value = {
            'items': [
                {'snippet': {'resourceId': {'videoId': 'VID1'}}},
                {'snippet': {'resourceId': {'videoId': 'VID2'}}},
            ]
        }
        mock_youtube.videos().list().execute.return_value = {
            'items': [
                {
                    'id': 'VID1',
                    'snippet': {
                        'title': 'Video One',
                        'publishedAt': '2024-01-01T00:00:00Z',
                        'tags': ['tag1', 'tag2']
                    },
                    'statistics': {
                        'viewCount': '1000',
                        'likeCount': '100',
                        'commentCount': '10'
                    }
                },
                {
                    'id': 'VID2',
                    'snippet': {
                        'title': 'Video Two',
                        'publishedAt': '2024-01-02T00:00:00Z',
                    },
                    'statistics': {
                        'viewCount': '2000',
                        'likeCount': '200',
                        'commentCount': '20'
                    }
                }
            ]
        }

        result = get_video_details(
            mock_youtube,
            [{'channel_id': 'UC1', 'uploads_playlist_id': 'UU1'}],
            10
        )

        assert len(result.videos) == 2
        assert result.videos[0]['video_id'] == 'VID1'
        assert result.videos[0]['video_title'] == 'Video One'
        assert result.videos[0]['video_views'] == 1000
        assert result.videos[0]['video_likes'] == 100
        assert result.videos[0]['video_comments'] == 10
        assert result.videos[0]['video_tags'] == ['tag1', 'tag2']
        assert result.videos[1]['video_tags'] == []  # Missing tags = empty list

    def test_playlist_error_adds_warning(self, mock_youtube, mock_http_error):
        """Playlist fetch error adds warning but continues."""
        mock_youtube.playlistItems().list().execute.side_effect = mock_http_error

        result = get_video_details(
            mock_youtube,
            [{'channel_id': 'UC1', 'uploads_playlist_id': 'UU1', 'channel_title': 'Test'}],
            10
        )

        assert len(result.warnings) == 1
        assert "Could not fetch videos" in result.warnings[0]

    def test_multiple_channels_processed(self, mock_youtube):
        """Multiple channels are processed independently."""
        mock_youtube.playlistItems().list().execute.side_effect = [
            {'items': [{'snippet': {'resourceId': {'videoId': 'VID1'}}}]},
            {'items': [{'snippet': {'resourceId': {'videoId': 'VID2'}}}]},
        ]
        mock_youtube.videos().list().execute.side_effect = [
            {'items': [{'id': 'VID1', 'snippet': {'title': 'V1', 'publishedAt': '2024-01-01T00:00:00Z'},
                        'statistics': {'viewCount': '0', 'likeCount': '0', 'commentCount': '0'}}]},
            {'items': [{'id': 'VID2', 'snippet': {'title': 'V2', 'publishedAt': '2024-01-01T00:00:00Z'},
                        'statistics': {'viewCount': '0', 'likeCount': '0', 'commentCount': '0'}}]},
        ]

        result = get_video_details(
            mock_youtube,
            [
                {'channel_id': 'UC1', 'uploads_playlist_id': 'UU1'},
                {'channel_id': 'UC2', 'uploads_playlist_id': 'UU2'},
            ],
            10
        )

        assert len(result.videos) == 2
        assert result.api_calls == 4  # 2 playlist calls + 2 video calls

    def test_max_videos_respected(self, mock_youtube):
        """max_videos_per_channel parameter is respected."""
        # Return more videos than max
        mock_youtube.playlistItems().list().execute.return_value = {
            'items': [{'snippet': {'resourceId': {'videoId': f'VID{i}'}}}
                      for i in range(20)]
        }
        mock_youtube.videos().list().execute.return_value = {
            'items': [
                {'id': f'VID{i}', 'snippet': {'title': f'V{i}', 'publishedAt': '2024-01-01T00:00:00Z'},
                 'statistics': {'viewCount': '0', 'likeCount': '0', 'commentCount': '0'}}
                for i in range(5)
            ]
        }

        result = get_video_details(
            mock_youtube,
            [{'channel_id': 'UC1', 'uploads_playlist_id': 'UU1'}],
            max_videos_per_channel=5
        )

        # Should only fetch 5 videos
        # Check the call was made with maxResults=5
        calls = mock_youtube.playlistItems().list.call_args_list
        assert any(call.kwargs.get('maxResults') == 5 for call in calls)

    def test_api_callback_invoked(self, mock_youtube):
        """on_api_call is called for playlist and video fetches."""
        mock_youtube.playlistItems().list().execute.return_value = {
            'items': [{'snippet': {'resourceId': {'videoId': 'VID1'}}}]
        }
        mock_youtube.videos().list().execute.return_value = {
            'items': [{'id': 'VID1', 'snippet': {'title': 'V1', 'publishedAt': '2024-01-01T00:00:00Z'},
                       'statistics': {'viewCount': '0', 'likeCount': '0', 'commentCount': '0'}}]
        }
        tracker = Mock()

        get_video_details(
            mock_youtube,
            [{'channel_id': 'UC1', 'uploads_playlist_id': 'UU1'}],
            10,
            on_api_call=tracker
        )

        # Should track both playlist and video calls
        call_types = [call[0][0] for call in tracker.call_args_list]
        assert 'youtube_playlist' in call_types
        assert 'youtube_video' in call_types

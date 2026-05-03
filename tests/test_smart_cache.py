"""
Unit tests for app/cache/smart_cache.py.

Covers:
- ChannelVideoCache._make_cache_key  (pure helper)
- ChannelVideoCache.get_channel_videos  (YouTube API wrapper + field parsing)
- get_video_details_smart  (orchestration: aggregation, quota tracking, cache-hit detection)
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


@pytest.fixture(autouse=True)
def clear_st_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yt_mock(video_ids=None, video_items=None):
    """Return a mock YouTube service with preset playlist + video responses."""
    if video_ids is None:
        video_ids = ["vid1", "vid2"]
    if video_items is None:
        video_items = [
            {
                "id": "vid1",
                "snippet": {
                    "title": "Video One",
                    "description": "desc1",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "tags": ["anime", "manga"],
                },
                "statistics": {"viewCount": "1000", "likeCount": "50", "commentCount": "10"},
                "contentDetails": {"duration": "PT5M30S"},
            },
            {
                "id": "vid2",
                "snippet": {
                    "title": "Video Two",
                    "description": "desc2",
                    "publishedAt": "2024-01-02T00:00:00Z",
                    # no 'tags' key — tests default-to-empty-list behaviour
                },
                "statistics": {"viewCount": "2000", "likeCount": "100", "commentCount": "20"},
                "contentDetails": {"duration": "PT1M"},
            },
        ]

    yt = MagicMock()
    playlist_response = {
        "items": [
            {"snippet": {"resourceId": {"videoId": vid}}} for vid in video_ids
        ]
    }
    yt.playlistItems.return_value.list.return_value.execute.return_value = playlist_response
    yt.videos.return_value.list.return_value.execute.return_value = {"items": video_items}
    return yt


# ---------------------------------------------------------------------------
# ChannelVideoCache._make_cache_key
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    def test_includes_channel_id(self):
        from cache.smart_cache import ChannelVideoCache
        assert "UC001" in ChannelVideoCache._make_cache_key("UC001", 10)

    def test_includes_max_videos(self):
        from cache.smart_cache import ChannelVideoCache
        assert "20" in ChannelVideoCache._make_cache_key("UC001", 20)

    def test_different_channels_differ(self):
        from cache.smart_cache import ChannelVideoCache
        assert (ChannelVideoCache._make_cache_key("UC001", 10)
                != ChannelVideoCache._make_cache_key("UC002", 10))

    def test_different_max_videos_differ(self):
        from cache.smart_cache import ChannelVideoCache
        assert (ChannelVideoCache._make_cache_key("UC001", 5)
                != ChannelVideoCache._make_cache_key("UC001", 15))

    def test_key_has_v2_prefix(self):
        from cache.smart_cache import ChannelVideoCache
        assert "v2" in ChannelVideoCache._make_cache_key("UC001", 10)


# ---------------------------------------------------------------------------
# ChannelVideoCache.get_channel_videos
# ---------------------------------------------------------------------------

class TestGetChannelVideos:
    # Each test uses a unique channel_id so Streamlit's in-process cache
    # never serves a result from a previous test (cache is also cleared by
    # the autouse fixture, but unique keys are a cheap extra safety net).

    def test_returns_two_element_tuple(self):
        from cache.smart_cache import ChannelVideoCache
        result = ChannelVideoCache.get_channel_videos(
            "CK_T01", "UU_T01", 10, "fp", _make_yt_mock()
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_was_fresh_is_true(self):
        from cache.smart_cache import ChannelVideoCache
        _, was_fresh = ChannelVideoCache.get_channel_videos(
            "CK_T02", "UU_T02", 10, "fp", _make_yt_mock()
        )
        assert was_fresh is True

    def test_returns_expected_video_count(self):
        from cache.smart_cache import ChannelVideoCache
        videos, _ = ChannelVideoCache.get_channel_videos(
            "CK_T03", "UU_T03", 10, "fp", _make_yt_mock()
        )
        assert len(videos) == 2

    def test_channel_id_injected_into_each_video(self):
        from cache.smart_cache import ChannelVideoCache
        videos, _ = ChannelVideoCache.get_channel_videos(
            "CK_T04", "UU_T04", 10, "fp", _make_yt_mock()
        )
        assert all(v["channel_id"] == "CK_T04" for v in videos)

    def test_video_fields_populated(self):
        from cache.smart_cache import ChannelVideoCache
        videos, _ = ChannelVideoCache.get_channel_videos(
            "CK_T05", "UU_T05", 10, "fp", _make_yt_mock()
        )
        v = next(v for v in videos if v["video_id"] == "vid1")
        assert v["video_title"] == "Video One"
        assert v["video_views"] == 1000
        assert v["video_likes"] == 50
        assert v["video_comments"] == 10
        assert v["video_tags"] == ["anime", "manga"]

    def test_tags_default_to_empty_list_when_absent(self):
        from cache.smart_cache import ChannelVideoCache
        videos, _ = ChannelVideoCache.get_channel_videos(
            "CK_T06", "UU_T06", 10, "fp", _make_yt_mock()
        )
        v = next(v for v in videos if v["video_id"] == "vid2")
        assert v["video_tags"] == []

    def test_duration_seconds_parsed_from_iso8601(self):
        from cache.smart_cache import ChannelVideoCache
        videos, _ = ChannelVideoCache.get_channel_videos(
            "CK_T07", "UU_T07", 10, "fp", _make_yt_mock()
        )
        v = next(v for v in videos if v["video_id"] == "vid1")
        assert v["duration_seconds"] == 330  # PT5M30S = 300 + 30

    def test_empty_playlist_returns_empty_list(self):
        from cache.smart_cache import ChannelVideoCache
        yt = MagicMock()
        yt.playlistItems.return_value.list.return_value.execute.return_value = {"items": []}
        videos, was_fresh = ChannelVideoCache.get_channel_videos(
            "CK_T08", "UU_T08", 10, "fp", yt
        )
        assert videos == [] and was_fresh is True

    def test_http_error_returns_empty_list(self):
        from cache.smart_cache import ChannelVideoCache
        from googleapiclient.errors import HttpError
        yt = MagicMock()
        yt.playlistItems.return_value.list.return_value.execute.side_effect = HttpError(
            resp=MagicMock(status=403), content=b"quota exceeded"
        )
        videos, was_fresh = ChannelVideoCache.get_channel_videos(
            "CK_T09", "UU_T09", 10, "fp", yt
        )
        assert videos == [] and was_fresh is True


# ---------------------------------------------------------------------------
# get_video_details_smart
# ---------------------------------------------------------------------------

class TestGetVideoDetailsSmart:
    def test_empty_channel_list_returns_empty(self):
        from cache.smart_cache import get_video_details_smart
        assert get_video_details_smart(MagicMock(), [], max_videos=10) == []

    def test_aggregates_videos_from_multiple_channels(self):
        from cache.smart_cache import get_video_details_smart, ChannelVideoCache
        channels = [
            {"channel_id": "UC001", "uploads_playlist_id": "UU001"},
            {"channel_id": "UC002", "uploads_playlist_id": "UU002"},
        ]
        ch1_vids = [{"video_id": "v1"}, {"video_id": "v2"}]
        ch2_vids = [{"video_id": "v3"}]

        def _side_effect(ch_id, up_id, max_v, fp, yt):
            return (ch1_vids if ch_id == "UC001" else ch2_vids), True

        with patch.object(ChannelVideoCache, "get_channel_videos", side_effect=_side_effect):
            result = get_video_details_smart(MagicMock(), channels, max_videos=10)

        assert len(result) == 3
        assert {v["video_id"] for v in result} == {"v1", "v2", "v3"}

    def test_channel_with_no_videos_skipped(self):
        from cache.smart_cache import get_video_details_smart, ChannelVideoCache
        channels = [{"channel_id": "UC001", "uploads_playlist_id": "UU001"}]
        with patch.object(ChannelVideoCache, "get_channel_videos", return_value=([], True)):
            assert get_video_details_smart(MagicMock(), channels, max_videos=10) == []

    def test_on_api_call_triggered_for_fresh_call(self):
        from cache.smart_cache import get_video_details_smart, ChannelVideoCache
        channels = [{"channel_id": "UC001", "uploads_playlist_id": "UU001"}]
        on_api_call = MagicMock()
        fake_videos = [{"video_id": "v1"}]

        with patch.object(ChannelVideoCache, "get_channel_videos", return_value=(fake_videos, True)), \
             patch("cache.smart_cache.time.time", side_effect=[0.0, 0.3]):  # 300ms > 50ms threshold
            get_video_details_smart(
                MagicMock(), channels, max_videos=10,
                debug_mode=True, on_api_call=on_api_call,
            )

        assert on_api_call.call_count == 2
        on_api_call.assert_any_call("youtube_playlist")
        on_api_call.assert_any_call("youtube_video")

    def test_on_api_call_not_triggered_for_cache_hit(self):
        from cache.smart_cache import get_video_details_smart, ChannelVideoCache
        channels = [{"channel_id": "UC001", "uploads_playlist_id": "UU001"}]
        on_api_call = MagicMock()
        fake_videos = [{"video_id": "v1"}]

        with patch.object(ChannelVideoCache, "get_channel_videos", return_value=(fake_videos, True)), \
             patch("cache.smart_cache.time.time", side_effect=[0.0, 0.01]):  # 10ms < threshold
            get_video_details_smart(
                MagicMock(), channels, max_videos=10,
                debug_mode=True, on_api_call=on_api_call,
            )

        on_api_call.assert_not_called()

    def test_on_api_call_not_triggered_when_debug_mode_false(self):
        from cache.smart_cache import get_video_details_smart, ChannelVideoCache
        channels = [{"channel_id": "UC001", "uploads_playlist_id": "UU001"}]
        on_api_call = MagicMock()
        fake_videos = [{"video_id": "v1"}]

        with patch.object(ChannelVideoCache, "get_channel_videos", return_value=(fake_videos, True)), \
             patch("cache.smart_cache.time.time", side_effect=[0.0, 0.3]):  # fresh but debug=False
            get_video_details_smart(
                MagicMock(), channels, max_videos=10,
                debug_mode=False, on_api_call=on_api_call,
            )

        on_api_call.assert_not_called()

    def test_handles_legacy_list_result_format(self):
        """Older callers may get a plain list instead of a (list, bool) tuple."""
        from cache.smart_cache import get_video_details_smart, ChannelVideoCache
        channels = [{"channel_id": "UC001", "uploads_playlist_id": "UU001"}]
        fake_videos = [{"video_id": "v1"}, {"video_id": "v2"}]

        with patch.object(ChannelVideoCache, "get_channel_videos", return_value=fake_videos):
            result = get_video_details_smart(MagicMock(), channels, max_videos=10)

        assert len(result) == 2

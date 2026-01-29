"""
smart_cache.py - Per-channel video caching for better hit rates
"""
import streamlit as st
import time
from typing import List, Dict, Any, Tuple, Optional, Callable


class ChannelVideoCache:
    """
    Cache individual channel videos instead of entire query results.

    Much higher hit rate because popular channels appear in multiple searches.
    """

    @staticmethod
    def _make_cache_key(channel_id: str, max_videos: int) -> str:
        """Generate cache key for a single channel"""
        return f"ch_vids_{channel_id}_{max_videos}"

    @staticmethod
    @st.cache_data(ttl=86400)  # 24 hours
    def get_channel_videos(
        channel_id: str,
        uploads_playlist_id: str,
        max_videos: int,
        _youtube_service  # Underscore prevents hashing
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Fetch and cache videos for a single channel.

        Returns:
            Tuple of (videos list, was_fresh) where was_fresh=True indicates
            this was an actual API call (not a cache hit).

        Note:
            The was_fresh flag allows callers to track API usage accurately.
            On cache hits, Streamlit returns the cached tuple directly,
            so was_fresh will still be True from the original call - but
            we detect cache hits by timing the function execution.
        """
        from googleapiclient.errors import HttpError

        try:
            # Fetch video IDs from uploads playlist
            playlist_response = _youtube_service.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=max_videos
            ).execute()

            video_ids = [
                item['snippet']['resourceId']['videoId']
                for item in playlist_response.get('items', [])
            ]

            if not video_ids:
                return [], True  # Fresh call, but no videos

            # Fetch video details
            videos_response = _youtube_service.videos().list(
                part="snippet,statistics",
                id=",".join(video_ids)
            ).execute()

            videos = []
            for item in videos_response.get('items', []):
                videos.append({
                    'channel_id': channel_id,
                    'video_id': item['id'],
                    'video_title': item['snippet']['title'],
                    'published_at': item['snippet']['publishedAt'],
                    'video_views': int(item['statistics'].get('viewCount', 0)),
                    'video_likes': int(item['statistics'].get('likeCount', 0)),
                    'video_comments': int(item['statistics'].get('commentCount', 0)),
                    'video_tags': item['snippet'].get('tags', []),
                })

            return videos, True  # True = this was a fresh API call

        except HttpError as e:
            return [], True  # Fresh call, but errored


def get_video_details_smart(
    youtube_service,
    channel_data: List[Dict],
    max_videos: int,
    debug_mode: bool = False,
    on_api_call: Optional[Callable[[str], None]] = None
) -> List[Dict]:
    """
    Fetch video details with per-channel caching.

    Args:
        youtube_service: YouTube API service instance
        channel_data: List of dicts with 'channel_id' and 'uploads_playlist_id'
        max_videos: Videos to fetch per channel
        debug_mode: Whether to track API calls for debugging
        on_api_call: Optional callback to track API calls (call_type: str) -> None

    Returns:
        List of all video dicts

    Notes:
        - Uses timing-based cache hit detection to avoid double-counting API quota
        - Cache hits (< 50ms per channel) are not counted toward quota
        - Fresh API calls (typically 200ms+) are tracked accurately
    """
    # Threshold for detecting cache hits (in seconds)
    # Fresh API calls typically take 200ms+, cache hits are < 10ms
    CACHE_HIT_THRESHOLD = 0.05  # 50ms

    all_videos = []

    for channel in channel_data:
        channel_id = channel['channel_id']
        uploads_id = channel['uploads_playlist_id']

        # Time the fetch to detect cache hits
        start_time = time.time()

        # Fetch with caching - returns (videos, was_fresh) tuple
        result = ChannelVideoCache.get_channel_videos(
            channel_id,
            uploads_id,
            max_videos,
            youtube_service
        )

        elapsed = time.time() - start_time

        # Handle both old format (list) and new format (tuple)
        if isinstance(result, tuple):
            videos, _ = result
        else:
            videos = result

        # Only track API calls if this was NOT a cache hit
        # Cache hits return almost instantly (< 50ms), fresh API calls take 200ms+
        is_cache_hit = elapsed < CACHE_HIT_THRESHOLD

        if debug_mode and videos and not is_cache_hit:
            # Each channel makes 2 API calls:
            # 1. playlistItems.list (get video IDs)
            # 2. videos.list (get video details)
            if on_api_call:
                on_api_call('youtube_playlist')
                on_api_call('youtube_video')

        if videos:
            all_videos.extend(videos)

    return all_videos

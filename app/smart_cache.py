"""
smart_cache.py - Per-channel video caching for better hit rates
"""
import streamlit as st
from typing import List, Dict, Any
from googleapiclient.errors import HttpError


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
    ) -> List[Dict[str, Any]]:
        """
        Fetch and cache videos for a single channel.
        
        NOTE: Tracking happens OUTSIDE this function in get_video_details_smart()
        
        Returns: List of video dicts (empty list on error)
        """
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
                return []
            
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
            
            return videos
            
        except HttpError as e:
            return []  # Graceful degradation


def get_video_details_smart(youtube_service, channel_data: List[Dict], max_videos: int) -> List[Dict]:
    """
    Fetch video details with per-channel caching.
    
    Args:
        channel_data: List of dicts with 'channel_id' and 'uploads_playlist_id'
        max_videos: Videos to fetch per channel
    
    Returns:
        List of all video dicts
    """
    try:
        from . import debug_tracker
    except ImportError:
        import debug_tracker
    
    all_videos = []
    
    for channel in channel_data:
        channel_id = channel['channel_id']
        uploads_id = channel['uploads_playlist_id']
        
        # Fetch with caching
        videos = ChannelVideoCache.get_channel_videos(
            channel_id,
            uploads_id,
            max_videos,
            youtube_service
        )
        
        # Track API calls AFTER fetching (only if we got results)
        if st.session_state.get('debug_mode', False) and videos:
            # Each channel makes 2 API calls:
            # 1. playlistItems.list (get video IDs)
            # 2. videos.list (get video details)
            debug_tracker.track_api_call('youtube_playlist')
            debug_tracker.track_api_call('youtube_video')
        
        if videos:
            all_videos.extend(videos)
    
    return all_videos
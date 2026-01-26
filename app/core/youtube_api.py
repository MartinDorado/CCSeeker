"""
youtube_api.py - YouTube Data API wrapper functions

Pure functions for interacting with the YouTube Data API.
These functions are Streamlit-agnostic and can be unit tested with mocked API clients.

Key design decisions:
- Functions accept youtube_service as a parameter (no global state)
- Callbacks are used for warnings/progress instead of direct st.* calls
- API tracking is optional via callback
- Results include both data and any warnings that occurred
"""

from datetime import datetime
from typing import Callable, Any
from dataclasses import dataclass, field
from googleapiclient.errors import HttpError

try:
    import dateutil.parser
except ImportError:
    dateutil = None


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class SearchResult:
    """Result from a channel search operation."""
    channels: list[dict]
    warnings: list[str] = field(default_factory=list)
    api_calls: int = 0


@dataclass
class ChannelStatsResult:
    """Result from fetching channel statistics."""
    stats: list[dict]
    api_calls: int = 0


@dataclass
class VideoDetailsResult:
    """Result from fetching video details."""
    videos: list[dict]
    warnings: list[str] = field(default_factory=list)
    api_calls: int = 0


# ============================================================================
# CHANNEL SEARCH
# ============================================================================

def search_channels_hybrid(
    youtube_service,
    query: str,
    region_code: str,
    max_videos: int = 100,
    max_channels: int = 50,
    on_api_call: Callable[[str], None] | None = None,
) -> SearchResult:
    """
    Hybrid search: Find channels by VIDEO content (primary) + channel names (secondary).

    Args:
        youtube_service: Authenticated YouTube Data API client
        query: Search query string
        region_code: ISO 3166-1 alpha-2 region code (e.g., "US") or empty string
        max_videos: Maximum videos to fetch from search (controls API quota)
        max_channels: Maximum channels to return
        on_api_call: Optional callback for tracking API calls (called with api_name string)

    Returns:
        SearchResult with:
        - channels: List[dict] with keys 'channel_id', 'channel_title', 'match_score'
        - warnings: List of warning messages (previously shown via st.warning)
        - api_calls: Number of API calls made

    Notes:
        - Scoring: video_matches * 10 (primary) + 5 for name_match (secondary)
        - On API errors, returns partial results with warnings
    """
    all_channels = {}  # {channel_id: {'title': str, 'video_matches': int, 'name_match': bool}}
    warnings = []
    api_calls = 0

    # === PART A: Search by video content (primary source) ===
    video_search_params = {
        'q': query,
        'part': 'snippet',
        'type': 'video',
        'maxResults': 50,
        'order': 'relevance',
    }
    if region_code:
        video_search_params['regionCode'] = region_code

    fetched_videos = 0
    next_page_token = None

    while fetched_videos < max_videos:
        if next_page_token:
            video_search_params['pageToken'] = next_page_token

        try:
            video_response = youtube_service.search().list(**video_search_params).execute()
            api_calls += 1
            if on_api_call:
                on_api_call('youtube_search')
        except HttpError as e:
            warnings.append(f"Video search error for '{query}': {e}")
            break

        items = video_response.get('items', [])
        if not items:
            break

        for item in items:
            channel_id = item['snippet'].get('channelId')
            channel_title = item['snippet'].get('channelTitle', 'Unknown')

            if channel_id:
                if channel_id not in all_channels:
                    all_channels[channel_id] = {
                        'title': channel_title,
                        'video_matches': 0,
                        'name_match': False
                    }
                all_channels[channel_id]['video_matches'] += 1

        fetched_videos += len(items)
        next_page_token = video_response.get('nextPageToken')
        if not next_page_token:
            break

    # === PART B: Search by channel name (secondary source) ===
    channel_search_params = {
        'q': query,
        'part': 'id,snippet',
        'type': 'channel',
        'maxResults': min(50, max_channels)
    }
    if region_code:
        channel_search_params['regionCode'] = region_code

    try:
        channel_response = youtube_service.search().list(**channel_search_params).execute()
        api_calls += 1
        if on_api_call:
            on_api_call('youtube_search')

        for item in channel_response.get('items', []):
            channel_id = item.get('id', {}).get('channelId')
            channel_title = item.get('snippet', {}).get('title', 'Unknown')

            if channel_id:
                if channel_id not in all_channels:
                    all_channels[channel_id] = {
                        'title': channel_title,
                        'video_matches': 0,
                        'name_match': True
                    }
                else:
                    all_channels[channel_id]['name_match'] = True

    except HttpError as e:
        warnings.append(f"Channel name search failed for '{query}': {e}. Showing video-based results only.")

    # === PART C: Calculate match scores and sort ===
    ranked_channels = []
    for channel_id, data in all_channels.items():
        match_score = data['video_matches'] * 10  # Video matches worth 10 points each
        if data['name_match']:
            match_score += 5  # Name match bonus

        ranked_channels.append({
            'channel_id': channel_id,
            'channel_title': data['title'],
            'match_score': match_score
        })

    ranked_channels.sort(key=lambda x: x['match_score'], reverse=True)

    return SearchResult(
        channels=ranked_channels,
        warnings=warnings,
        api_calls=api_calls
    )


def search_channels_multi_term(
    youtube_service,
    query: str,
    region_code: str,
    max_videos_per_term: int = 100,
    max_channels: int | None = None,
    on_api_call: Callable[[str], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> SearchResult:
    """
    Handle comma-separated queries as OR logic.

    Example: "manga, anime" -> search both terms, merge results

    Args:
        youtube_service: Authenticated YouTube Data API client
        query: Comma-separated search terms
        region_code: ISO 3166-1 alpha-2 region code
        max_videos_per_term: Max videos to fetch per search term
        max_channels: Max channels to return (None = unlimited)
        on_api_call: Optional callback for tracking API calls
        on_progress: Optional callback for progress updates (called with message string)

    Returns:
        SearchResult with merged and deduplicated channels sorted by total score
    """
    # Split by comma and clean
    terms = [t.strip() for t in query.split(',') if t.strip()]
    warnings = []
    total_api_calls = 0

    if len(terms) == 0:
        return SearchResult(channels=[], warnings=[], api_calls=0)

    # Enforce 2-term maximum (matching the old behavior)
    if len(terms) > 2:
        warnings.append(
            f"Search limited to 2 terms (received {len(terms)}). "
            f"Using: {', '.join(terms[:2])}"
        )
        terms = terms[:2]

    if len(terms) == 1:
        # Single term: use hybrid search directly
        result = search_channels_hybrid(
            youtube_service, terms[0], region_code, max_videos_per_term,
            on_api_call=on_api_call
        )
        channels = result.channels
        if max_channels is not None:
            channels = channels[:max_channels]
        return SearchResult(
            channels=channels,
            warnings=result.warnings,
            api_calls=result.api_calls
        )

    # Multiple terms: search each, then merge
    if on_progress:
        on_progress(f"Searching {len(terms)} topics: {', '.join(terms)}")

    all_channels = {}  # {channel_id: {'title': str, 'total_score': int}}

    for term in terms:
        if on_progress:
            on_progress(f"Searching for '{term}'...")

        result = search_channels_hybrid(
            youtube_service, term, region_code, max_videos_per_term,
            on_api_call=on_api_call
        )
        total_api_calls += result.api_calls
        warnings.extend(result.warnings)

        for channel in result.channels:
            channel_id = channel['channel_id']
            if channel_id not in all_channels:
                all_channels[channel_id] = {
                    'title': channel['channel_title'],
                    'total_score': 0
                }
            all_channels[channel_id]['total_score'] += channel['match_score']

    # Convert back to list format and sort
    merged = [
        {
            'channel_id': ch_id,
            'channel_title': data['title'],
            'match_score': data['total_score']
        }
        for ch_id, data in sorted(
            all_channels.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
    ]

    if max_channels is not None:
        merged = merged[:max_channels]

    return SearchResult(
        channels=merged,
        warnings=warnings,
        api_calls=total_api_calls
    )


# ============================================================================
# CHANNEL STATISTICS
# ============================================================================

def get_channel_stats(
    youtube_service,
    channel_ids: list[str],
    on_api_call: Callable[[str], None] | None = None,
) -> ChannelStatsResult:
    """
    Fetch detailed statistics for a list of channel IDs.

    Args:
        youtube_service: Authenticated YouTube Data API client
        channel_ids: List of YouTube channel IDs
        on_api_call: Optional callback for tracking API calls

    Returns:
        ChannelStatsResult with:
        - stats: List of dicts with channel metadata
        - api_calls: Number of API calls made

    Each stat dict contains:
        - channel_id, country, description, subscribers, views, videos
        - uploads_playlist_id (for fetching videos)
        - avg_views_per_video, channel_age_days
        - channel_keywords (list of keywords from brandingSettings)
        - default_language (from brandingSettings)
        - topic_categories (list of topic names from topicDetails)
    """
    stats_data = []
    api_calls = 0

    # Process in chunks of 50 (API limit)
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i + 50]
        request = youtube_service.channels().list(
            part="snippet,statistics,contentDetails,brandingSettings,topicDetails",
            id=",".join(chunk)
        )
        response = request.execute()
        api_calls += 1
        if on_api_call:
            on_api_call('youtube_channel')

        for item in response.get("items", []):
            content_details = item.get('contentDetails', {})
            related_playlists = content_details.get('relatedPlaylists', {})
            uploads_id = related_playlists.get('uploads')
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            branding = item.get("brandingSettings", {}).get("channel", {})
            topic_details = item.get("topicDetails", {})

            if uploads_id:
                # Calculate channel age in days
                published_at = snippet.get("publishedAt")
                channel_age_days = None
                if published_at and dateutil:
                    try:
                        created_date = dateutil.parser.parse(published_at)
                        channel_age_days = (datetime.now(created_date.tzinfo) - created_date).days
                    except Exception:
                        channel_age_days = None

                # Calculate derived metrics
                videos_count = int(statistics.get("videoCount", 0))
                total_views = int(statistics.get("viewCount", 0))
                avg_views_per_video = round(total_views / videos_count, 0) if videos_count > 0 else 0

                # Extract keywords from brandingSettings (space-separated string)
                keywords_str = branding.get("keywords", "")
                channel_keywords = keywords_str.split() if keywords_str else []

                # Extract topic names from Wikipedia URLs
                topic_urls = topic_details.get("topicCategories", [])
                topic_categories = [
                    url.split("/wiki/")[-1].replace("_", " ")
                    for url in topic_urls
                    if "/wiki/" in url
                ]

                stats_data.append({
                    "channel_id": item["id"],
                    "country": snippet.get("country", "N/A"),
                    "description": snippet.get("description", ""),
                    "subscribers": int(statistics.get("subscriberCount", 0)),
                    "views": total_views,
                    "videos": videos_count,
                    "uploads_playlist_id": uploads_id,
                    "avg_views_per_video": avg_views_per_video,
                    "channel_age_days": channel_age_days,
                    "channel_keywords": channel_keywords,
                    "default_language": branding.get("defaultLanguage", ""),
                    "topic_categories": topic_categories,
                })

    return ChannelStatsResult(stats=stats_data, api_calls=api_calls)


# ============================================================================
# VIDEO DETAILS
# ============================================================================

def get_video_details(
    youtube_service,
    channel_data: list[dict],
    max_videos_per_channel: int,
    on_api_call: Callable[[str], None] | None = None,
) -> VideoDetailsResult:
    """
    Fetch video details for multiple channels.

    Args:
        youtube_service: Authenticated YouTube Data API client
        channel_data: List of dicts with 'channel_id', 'uploads_playlist_id', optionally 'channel_title'
        max_videos_per_channel: Maximum videos to fetch per channel
        on_api_call: Optional callback for tracking API calls

    Returns:
        VideoDetailsResult with:
        - videos: List of video dicts
        - warnings: List of warning messages for failed channels
        - api_calls: Number of API calls made

    Each video dict contains:
        - channel_id, video_id, video_title, video_description, published_at
        - video_views, video_likes, video_comments, video_tags
    """
    all_video_details = []
    warnings = []
    api_calls = 0

    for channel in channel_data:
        playlist_id = channel.get("uploads_playlist_id")
        channel_id = channel.get("channel_id")
        channel_title = channel.get("channel_title", "(unknown)")

        if not playlist_id:
            continue

        try:
            # Paginate through playlist items up to the requested limit
            video_ids = []
            next_page_token = None
            fetched = 0

            while fetched < max_videos_per_channel:
                page_size = min(50, max_videos_per_channel - fetched)
                request = youtube_service.playlistItems().list(
                    part="snippet",
                    playlistId=playlist_id,
                    maxResults=page_size,
                    pageToken=next_page_token
                )
                response = request.execute()
                api_calls += 1
                if on_api_call:
                    on_api_call('youtube_playlist')

                items = response.get("items", [])
                for it in items:
                    vid = (it.get("snippet", {}).get("resourceId", {}) or {}).get("videoId")
                    if vid:
                        video_ids.append(vid)

                fetched += len(items)
                next_page_token = response.get("nextPageToken")
                if not next_page_token or not items:
                    break

        except HttpError:
            warnings.append(f"Could not fetch videos for '{channel_title}': Playlist not found or private.")
            continue

        if not video_ids:
            continue

        # Fetch video details
        video_request = youtube_service.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        )
        video_response = video_request.execute()
        api_calls += 1
        if on_api_call:
            on_api_call('youtube_video')

        for item in video_response.get("items", []):
            all_video_details.append({
                "channel_id": channel_id,
                "video_id": item["id"],
                "video_title": item["snippet"]["title"],
                "video_description": item["snippet"].get("description", ""),
                "published_at": item["snippet"]["publishedAt"],
                "video_views": int(item["statistics"].get("viewCount", 0)),
                "video_likes": int(item["statistics"].get("likeCount", 0)),
                "video_comments": int(item["statistics"].get("commentCount", 0)),
                "video_tags": item["snippet"].get("tags", []),
            })

    return VideoDetailsResult(
        videos=all_video_details,
        warnings=warnings,
        api_calls=api_calls
    )

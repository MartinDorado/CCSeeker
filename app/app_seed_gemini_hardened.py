import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import re
from googleapiclient.errors import HttpError
from collections import Counter
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv
try:
    # Try relative import (when run as module)
    from . import seed_topics_v2 as seedmod
    from . import similarity_engine
except ImportError:
    # Fallback for direct execution
    import seed_topics_v2 as seedmod
    import similarity_engine

try:
    import pycountry
    COUNTRY_OPTIONS_BASE = [f"{c.name} ({c.alpha_2})" for c in pycountry.countries]
except Exception:
    COUNTRY_OPTIONS_BASE = [
        "Argentina (AR)",
        "Australia (AU)",
        "Canada (CA)",
        "France (FR)",
        "Germany (DE)",
        "India (IN)",
        "Japan (JP)",
        "United Kingdom (GB)",
        "United States (US)",
    ]
try:
    from . import debug_tracker
except ImportError:
    import debug_tracker

import math
import time

# ============================================================================
# CACHING LAYER - Reduces API calls by 80%
# ============================================================================

@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_channel_stats_cached(channel_ids_tuple):
    """
    Cached wrapper for get_channel_stats
    
    Note: Uses tuple because lists aren't hashable for caching
    """
    youtube = get_youtube()
    return get_channel_stats(youtube, list(channel_ids_tuple))


@st.cache_data(ttl=7200)
def get_video_details_cached(channel_ids_tuple, max_videos=10):
    """
    Cached wrapper for get_video_details
    
    Fetches video details for a set of channels
    """
    youtube = get_youtube()
    
    # Reconstruct channel_data format
    channel_data = [
        {'channel_id': ch_id, 'uploads_playlist_id': None}
        for ch_id in channel_ids_tuple
    ]
    
    # We need to fetch uploads playlists first
    stats = get_channel_stats(youtube, list(channel_ids_tuple))
    channel_data_full = []
    
    for stat in stats:
        channel_data_full.append({
            'channel_id': stat['channel_id'],
            'uploads_playlist_id': stat['uploads_playlist_id']
        })
    
    return get_video_details(youtube, channel_data_full, max_videos)


@st.cache_data(ttl=3600)
def search_channels_multi_term_cached(query, region_code, max_videos=150):
    """Cached wrapper for search_channels_multi_term"""
    return search_channels_multi_term(query, region_code, max_videos)

# --- Securely Load API Keys ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Constants ---
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# --- Helper & API Functions ---

@st.cache_resource(show_spinner=False)
def get_youtube():
    """Create and cache a YouTube Data API client using the env API key."""
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is not configured")
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)

def get_gemini_model(temperature: float | None = None):
    """Configure and return a Gemini model for content generation."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")
    genai.configure(api_key=GEMINI_API_KEY)
    cfg = {}
    if temperature is not None:
        cfg["generation_config"] = {"temperature": float(temperature)}
    return genai.GenerativeModel('gemini-2.0-flash-lite', **cfg)

def extract_identifier_from_url(url):
    """Extract a channel identifier from common YouTube URL formats.

    Returns either a channel ID (starting with 'UC...') or a handle/username
    (e.g., 'SomeCreator'). Resolution to an actual channel ID is done elsewhere.
    """
    patterns = [
        r'(?:youtube\.com/channel/)([^/?&]+)',
        r'(?:youtube\.com/@)([^/?&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def resolve_channel_id(youtube_service, user_input: str):
    """Resolve a channel identifier from a URL, @handle, or UC... id.

    Returns channel_id or None.
    """
    if not user_input:
        return None
    s = user_input.strip()
    # Direct ID
    if s.startswith("UC") and len(s) >= 20:
        return s
    # Try parsing URL
    ident = extract_identifier_from_url(s) or s
    if ident.startswith("UC"):
        return ident
    # Treat as handle or name: remove leading '@'
    handle = ident[1:] if ident.startswith("@") else ident
    try:
        response = youtube_service.search().list(q=handle, part="id", type="channel", maxResults=1).execute()
        items = response.get("items", [])
        if items:
            return items[0]["id"]["channelId"]
        return None
    except HttpError:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def search_channels_hybrid(query: str, region_code: str, max_videos: int = 150, max_channels: int = 50):
    """
    Hybrid search: Find channels by their VIDEO content (primary) + channel names (secondary).
    
    Returns: List[dict] with keys 'channel_id', 'channel_title', 'match_score'
    """
    youtube = get_youtube()
    all_channels = {}  # {channel_id: {'title': str, 'video_matches': int, 'name_match': bool}}
    
    # === PART A: Search by video content (primary source) ===
    video_search_params = {
        'q': query,
        'part': 'snippet',
        'type': 'video',
        'maxResults': 50,
        'order': 'relevance',
        'relevanceLanguage': region_code if region_code else None,
        'regionCode': region_code if region_code else None
    }
    
    fetched_videos = 0
    next_page_token = None
    
    while fetched_videos < max_videos:
        if next_page_token:
            video_search_params['pageToken'] = next_page_token
        
        try:
            video_response = youtube.search().list(**video_search_params).execute()
        except HttpError as e:
            st.warning(f"Video search error for '{query}': {e}")
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
        channel_response = youtube.search().list(**channel_search_params).execute()
        
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
                    # Already found via videos, mark as also having name match
                    all_channels[channel_id]['name_match'] = True
    
    except HttpError as e:
        # Partial failure: return video results with warning
        st.warning(f"⚠️ Channel name search failed for '{query}': {e}. Showing video-based results only.")
    
    # === PART C: Calculate match scores and sort ===
    # Scoring: video_matches (primary) + name_match bonus (secondary)
    ranked_channels = []
    for channel_id, data in all_channels.items():
        match_score = data['video_matches'] * 10  # Video matches are worth 10 points each
        if data['name_match']:
            match_score += 5  # Name match bonus
        
        ranked_channels.append({
            'channel_id': channel_id,
            'channel_title': data['title'],
            'match_score': match_score
        })
    
    # Sort by match_score descending
    ranked_channels.sort(key=lambda x: x['match_score'], reverse=True)
    
    return ranked_channels

def search_channels_multi_term(query: str, region_code: str, max_videos_per_term: int = 150):
    """
    Handle comma-separated queries as OR logic.
    Example: "manga, anime" → search both terms, merge results
    
    Returns: List[dict] with deduplicated channels sorted by relevance
    """
    # Split by comma and clean
    terms = [t.strip() for t in query.split(',') if t.strip()]
    
    if len(terms) == 0:
        return []
    
    if len(terms) == 1:
        # Single term: use hybrid search directly
        return search_channels_hybrid(terms[0], region_code, max_videos_per_term)
    
    # Multiple terms: search each, then merge
    st.info(f"🔍 Searching {len(terms)} topics: {', '.join(terms)}")
    
    all_channels = {}  # {channel_id: {'title': str, 'total_score': int}}
    
    for term in terms:
        with st.spinner(f"Searching for '{term}'..."):
            results = search_channels_hybrid(term, region_code, max_videos_per_term)
        
        for channel in results:
            channel_id = channel['channel_id']
            if channel_id not in all_channels:
                all_channels[channel_id] = {
                    'title': channel['channel_title'],
                    'total_score': 0
                }
            # Accumulate scores across all terms
            all_channels[channel_id]['total_score'] += channel['match_score']
    
    # Convert back to list format and sort
    merged = [
        {
            'channel_id': ch_id,
            'channel_title': data['title']
        }
        for ch_id, data in sorted(
            all_channels.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
    ]
    
    return merged

# --- Boolean query helpers (minimal) ---
def _strip_outer_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
        return s[1:-1].strip()
    return s

def search_channels(youtube_service, query, region_code, total_results_to_fetch):
    """Channel search with proper parts so channelId is present."""
    channels = []
    next_page_token = None
    # IMPORTANT: include id so item["id"]["channelId"] exists
    search_params = {'q': query, 'part': 'id,snippet', 'type': 'channel'}
    if region_code:
        search_params['regionCode'] = region_code
    while len(channels) < total_results_to_fetch:
        search_params['maxResults'] = min(50, total_results_to_fetch - len(channels))
        if next_page_token:
            search_params['pageToken'] = next_page_token
        try:
            response = youtube_service.search().list(**search_params).execute()
        except HttpError as e:
            st.error(f"YouTube search API error: {e}")
            break
        for item in response.get("items", []):
            channel_id = item.get("id", {}).get("channelId")
            if not channel_id:
                continue
            channels.append({
                "channel_id": channel_id,
                "channel_title": item.get("snippet", {}).get("title", "Unknown")
            })
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return channels

def get_channel_stats(youtube_service, channel_ids):
    """
    Phase 1 enhancement: Add 3 high-value metrics with minimal breaking changes
    """
    stats_data = []
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i + 50]
        request = youtube_service.channels().list(
            part="snippet,statistics,contentDetails", 
            id=",".join(chunk)
        )
        response = request.execute()
        
        for item in response.get("items", []):
            content_details = item.get('contentDetails', {})
            related_playlists = content_details.get('relatedPlaylists', {})
            uploads_id = related_playlists.get('uploads')
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            
            if uploads_id:
                # Calculate channel age in days (for internal ranking, not displayed yet)
                published_at = snippet.get("publishedAt")
                channel_age_days = None
                if published_at:
                    try:
                        from datetime import datetime
                        import dateutil.parser
                        created_date = dateutil.parser.parse(published_at)
                        channel_age_days = (datetime.now(created_date.tzinfo) - created_date).days
                    except Exception:
                        channel_age_days = None  # Graceful fallback
                
                # Calculate derived metrics
                videos_count = int(statistics.get("videoCount", 0))
                total_views = int(statistics.get("viewCount", 0))
                avg_views_per_video = round(total_views / videos_count, 0) if videos_count > 0 else 0
                
                stats_data.append({
                    # Existing fields (unchanged)
                    "channel_id": item["id"],
                    "country": snippet.get("country", "N/A"),
                    "subscribers": int(statistics.get("subscriberCount", 0)),
                    "views": total_views,
                    "videos": videos_count,
                    "uploads_playlist_id": uploads_id,
                    
                    # NEW: Phase 1 additions (safe additions only)
                    "avg_views_per_video": avg_views_per_video,        # Main display metric
                    "channel_age_days": channel_age_days,              # For future ranking logic
                })
                    
    return stats_data


def get_channel_stats_cached_with_tracking(channel_ids_tuple):
    """Wrapper that tracks cache hits"""
    
    # Check if this exact call is cached (simplified check)
    cache_key = f"channel_stats_{hash(channel_ids_tuple)}"
    
    if st.session_state.get('debug_mode', False):
        # Simple heuristic: if function runs very fast, likely cache hit
        start = time.time()
        result = get_channel_stats_cached(channel_ids_tuple)
        elapsed = time.time() - start
        
        if elapsed < 0.01:  # Less than 10ms = probably cached
            debug_tracker.track_cache_hit()
        
        return result
    else:
        return get_channel_stats_cached(channel_ids_tuple)

def get_video_details(youtube_service, channel_data, max_videos_per_channel):
    # ... (This function is now also used by the seed analysis)
    all_video_details = []
    for channel in channel_data:
        playlist_id = channel["uploads_playlist_id"]
        try:
            # paginate through playlist items up to the requested limit
            video_ids = []
            next_page_token = None
            fetched = 0
            while fetched < max_videos_per_channel:
                page_size = min(50, max_videos_per_channel - fetched)
                request = youtube_service.playlistItems().list(
                    part="snippet", playlistId=playlist_id, maxResults=page_size,
                    pageToken=next_page_token
                )
                response = request.execute()
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
            st.warning(f"Could not fetch videos for '{channel.get('channel_title','(unknown)')}': Playlist not found or private.")
            continue
        if not video_ids: continue
        video_request = youtube_service.videos().list(part="snippet,statistics", id=",".join(video_ids))
        video_response = video_request.execute()
        for item in video_response.get("items", []):
            all_video_details.append({
                "channel_id": channel["channel_id"], "video_id": item["id"],
                "video_title": item["snippet"]["title"], "published_at": item["snippet"]["publishedAt"],
                "video_views": int(item["statistics"].get("viewCount", 0)),
                "video_likes": int(item["statistics"].get("likeCount", 0)),
                "video_comments": int(item["statistics"].get("commentCount", 0)),
                "video_tags": item["snippet"].get("tags", []),
            })
    return all_video_details

def calculate_keyword_relevance(df, query):
    """Compute per-channel relevance by matching query terms against title and tags.
    """
    if df.empty or not isinstance(query, str) or not query.strip():
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])

    if ',' in query:
        # Comma-separated format: "manga, anime, review"
        raw_terms = [t.strip() for t in query.split(',') if t.strip()]
    else:
        # Legacy format: "manga OR anime" or "manga AND review"
        # Split on OR/AND tokens, then normalize terms
        raw_terms = [word.strip() for word in re.split(r"OR|AND", query, flags=re.IGNORECASE)]
    
    cleaned = []
    for t in raw_terms:
        if not t:
            continue
        # Remove outer quotes if present
        t = _strip_outer_quotes(t)  # Use your existing helper
        if not t:
            continue
        cleaned.append(re.escape(t))

    if not cleaned:
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])

    # Match ANY term (OR logic)
    pattern = '(?:' + '|'.join(cleaned) + ')'

    # Build a combined text field: title + joined tags
    def _tags_to_text(x):
        if isinstance(x, list):
            return ' '.join([str(i) for i in x if i is not None])
        return str(x) if x is not None else ''

    df = df.copy()
    title_series = df['video_title'] if 'video_title' in df.columns else pd.Series([''] * len(df))
    tags_series = df['video_tags'] if 'video_tags' in df.columns else pd.Series([''] * len(df))
    df['combined_text'] = title_series.fillna('') + ' ' + tags_series.apply(_tags_to_text)
    df['is_relevant'] = df['combined_text'].str.contains(pattern, case=False, na=False)
    relevance = df.groupby('channel_id')['is_relevant'].mean().reset_index()
    relevance = relevance.rename(columns={'is_relevant': 'relevance_score'})
    return relevance

# --- Helpers for seed topic iteration ---
def _parse_topics_from_query(q: str) -> list[str]:
    """Split an OR-joined query into topic phrases, removing outer quotes."""
    if not q:
        return []
    parts = [p.strip() for p in q.split(" OR ") if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) >= 2 and p.startswith('"') and p.endswith('"'):
            out.append(p[1:-1])
        else:
            out.append(p)
    return out

def _build_query_from_topics(topics: list[str]) -> str:
    """Join topic phrases into an OR query, quoting multi-word phrases."""
    bits = []
    for t in topics:
        t = (t or "").strip()
        if not t:
            continue
        bits.append(t if " " not in t else f'"{t}"')
    return " OR ".join(bits)

def run_search(
    youtube,
    final_query: str,
    region_input: str,
    min_subs_input: int,
    months_ago_input: int,
    country_filter_input: str,
    boolean_fetch: bool = False,
):
    """Run the 4-step search + analysis pipeline and render results."""
    
    # Reset debug tracking for new search
    search_start_time = None
    if st.session_state.get('debug_mode', False):
        debug_tracker.reset_debug_tracking()
        search_start_time = time.time()
    
    try:
        
        # === STEP 1: Search for channels ===
        with st.spinner("Step 1/4: Searching for channels..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            # ✅ TRACK BEFORE calling cached function
            if st.session_state.get('debug_mode', False):
                debug_tracker.track_api_call('youtube_search')
            
            initial_channels = search_channels_multi_term_cached(final_query, region_input, max_videos=150)
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['search'] = time.time() - step_start
        

        if not initial_channels:
            st.error("Search did not return any channels.")
            return

        df_initial = pd.DataFrame(initial_channels)
        with st.expander("See raw channels found"):
            st.dataframe(df_initial)

        # === STEP 2: Fetch channel statistics ===
        with st.spinner("Step 2/4: Fetching channel statistics..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            channel_ids_tuple = tuple(df_initial['channel_id'].tolist())
            
            # ✅ TRACK BEFORE calling cached function
            if st.session_state.get('debug_mode', False):
                # We make 1 call per 50 channels (batching)
                num_calls = math.ceil(len(channel_ids_tuple) / 50)
                for _ in range(num_calls):
                    debug_tracker.track_api_call('youtube_channel')
            
            channel_statistics = get_channel_stats_cached(channel_ids_tuple)
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['channel_stats'] = time.time() - step_start

        if not channel_statistics:
            st.warning("Could not retrieve detailed stats for the found channels.")
            return

        df_stats = pd.DataFrame(channel_statistics)
        enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')

        # === STEP 3: Filter channels FIRST (before fetching videos!) ===
        with st.spinner("Step 3/5: Applying filters..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            # Filter by subscriber count
            filtered_channels = enriched_channel_data[
                enriched_channel_data['subscribers'] >= min_subs_input
            ].copy()
            
            # Filter by country (if specified)
            if country_filter_input:
                filtered_channels = filtered_channels[
                    filtered_channels['country'] == country_filter_input.upper()
                ]
            
            if filtered_channels.empty:
                st.error("No channels match your filtering criteria (subscribers, country).")
                return
            
            st.info(f"✅ {len(filtered_channels)} channels passed filters (min {min_subs_input:,} subs)")
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['filtering'] = time.time() - step_start

       # === STEP 4: Prepare channels for analysis (cap at 80 max) ===
        with st.spinner("Step 4/5: Preparing channels for analysis..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            # Sort by subscribers to prioritize established channels
            filtered_sorted = filtered_channels.sort_values('subscribers', ascending=False)
            
            # Cap at 80 channels to control quota usage
            channels_to_analyze = filtered_sorted.head(80).copy()
            
            channels_analyzed_count = len(channels_to_analyze)
            
            if channels_analyzed_count < len(filtered_channels):
                st.info(f"📊 Analyzing top {channels_analyzed_count} channels (from {len(filtered_channels)} total)")
            else:
                st.info(f"📊 Analyzing all {channels_analyzed_count} channels")
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['select_channels'] = time.time() - step_start

        # === STEP 5: Single-pass deep analysis (10 videos per channel) ===
        with st.spinner(f"Step 5/5: Deep analysis - fetching 10 videos from {channels_analyzed_count} channels..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            channel_ids_tuple = tuple(channels_to_analyze['channel_id'].tolist())
            
            # Track API calls for debug
            if st.session_state.get('debug_mode', False):
                for _ in channels_to_analyze.itertuples():
                    debug_tracker.track_api_call('youtube_video')
            
            # Fetch 10 videos for comprehensive relevance analysis
            video_data = get_video_details_cached(
                channel_ids_tuple, 
                max_videos=10  # 🔥 DEEP ANALYSIS: 10 videos per channel
            )
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['video_details'] = time.time() - step_start
        
        # Debug checkpoint
        if st.session_state.get('debug_mode', False):
            st.write(f"✓ Step 5: Retrieved {len(video_data)} videos from {channels_analyzed_count} channels")

        if not video_data:
            st.warning("Could not retrieve any video details from the channels.")
            st.dataframe(channels_to_analyze.sort_values(by="subscribers", ascending=False))
            return

        # === STEP 6: Calculate relevance and filter ===
        with st.spinner("Calculating relevance and engagement metrics..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            # Create DataFrame from video data
            df_videos = pd.DataFrame(video_data)
            
            # Calculate relevance scores (based on 10 videos per channel)
            relevance_scores = calculate_keyword_relevance(df_videos.copy(), final_query)
            
            # Calculate engagement rates
            df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
            df_videos['engagement_rate'] = (
                (df_videos['video_likes'] + df_videos['video_comments']) / 
                (df_videos['video_views'] + 1)
            )
            
            # Merge video data with channel data
            df_full = pd.merge(df_videos, channels_to_analyze, on='channel_id')
            
            # Filter by upload recency (if specified)
            if months_ago_input > 0:
                date_cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=months_ago_input)
                df_full = df_full[df_full['published_at'] >= date_cutoff]
            
            if df_full.empty:
                st.error("No channels have uploaded videos in the specified time range.")
                return
            
            # Calculate average engagement per channel
            avg_engagement = df_full.groupby('channel_id')['engagement_rate'].mean().reset_index()
            
            # Merge everything together
            final_channels = pd.merge(channels_to_analyze, avg_engagement, on='channel_id')
            final_channels = pd.merge(final_channels, relevance_scores, on='channel_id', how='left')
            
            # Fill NaN relevance scores with 0
            final_channels['relevance_score'] = final_channels['relevance_score'].fillna(0)
            
            # 🎯 CRITICAL FILTER: Keep only channels with relevance ≥ 5%
            relevant_channels = final_channels[final_channels['relevance_score'] >= 0.05].copy()
            
            if relevant_channels.empty:
                st.warning(f"⚠️ No channels found with relevance ≥ 5% for '{final_query}'. Try broadening your search terms or lowering minimum subscribers.")
                
                # Show top 10 channels by subscriber count as fallback
                st.info("Showing top 10 channels by size (may not be topic-focused):")
                fallback_channels = final_channels.nlargest(10, 'subscribers')
                top_channels = fallback_channels.copy()
            else:
                # Sort by relevance (primary) then subscribers (secondary)
                relevant_channels = relevant_channels.sort_values(
                    by=['relevance_score', 'subscribers'], 
                    ascending=False
                )
                
                # Add analysis badge
                relevant_channels['analysis_depth'] = '✓ 10 videos analyzed'
                
                top_channels = relevant_channels.copy()
                
                # Success message
                relevant_count = len(relevant_channels)
                st.success(f"✅ Found {relevant_count} relevant channel{'s' if relevant_count != 1 else ''} (relevance ≥ 5%)")
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['relevance_filtering'] = time.time() - step_start

        # Debug checkpoint
        if st.session_state.get('debug_mode', False):
            if not relevant_channels.empty:
                avg_relevance = relevant_channels['relevance_score'].mean()
                st.write(f"✓ Filtered to {len(relevant_channels)} channels (avg relevance: {avg_relevance:.1%})")
            else:
                st.write(f"✓ No channels passed 5% threshold (showing fallback)")

        # === SIMILARITY RANKING (if using seed) ===
        if 'seed_profile' in st.session_state:
            with st.spinner("🧠 Calculating similarity scores..."):
                seed_channel_id = st.session_state['seed_profile']['channel_id']
                before_exclusion = len(top_channels)
                
                top_channels = top_channels[top_channels['channel_id'] != seed_channel_id]
                
                excluded_count = before_exclusion - len(top_channels)
                if excluded_count > 0:
                    st.info(f"🚫 Excluded seed channel from results ({excluded_count} removed)")           
                
                if top_channels.empty:
                    st.error("No channels in the similar size range. Try disabling the size filter.")
                    return
                else:
                    st.info("📥 Fetching additional data for similarity analysis...")
                    
                    enriched_data = get_video_details(
                        youtube,
                        top_channels.to_dict('records'),
                        max_videos_per_channel=10
                    )
                    
                    if enriched_data:
                        df_enriched = pd.DataFrame(enriched_data)
                        
                        def flatten_tags(tag_series):
                            all_tags = []
                            for tags in tag_series:
                                if isinstance(tags, list):
                                    all_tags.extend(tags)
                                elif isinstance(tags, str):
                                    all_tags.append(tags)
                            unique_tags = list(set(t.lower().strip() for t in all_tags if t))
                            return unique_tags

                        channel_tags = (
                            df_enriched.groupby('channel_id')['video_tags']
                            .apply(flatten_tags)
                            .reset_index()
                            .rename(columns={'video_tags': 'tags'})
                        )

                        channel_keywords = (
                            df_enriched.groupby('channel_id')['video_title']
                            .apply(lambda x: list(x))
                            .reset_index()
                            .rename(columns={'video_title': 'recent_titles'})
                        )
                        
                        top_channels = top_channels.merge(channel_tags, on='channel_id', how='left')
                        top_channels = top_channels.merge(channel_keywords, on='channel_id', how='left')
                        
                        top_channels['tags'] = top_channels['tags'].apply(lambda x: x if isinstance(x, list) else [])
                        top_channels['recent_titles'] = top_channels['recent_titles'].apply(lambda x: x if isinstance(x, list) else [])
                        
                        def extract_keywords_from_titles(titles):
                            if not titles or not isinstance(titles, list):
                                return []
                            
                            text = " ".join(str(t) for t in titles if t)
                            words = re.findall(r'\b[a-záéíóúñü]{3,}\b', text.lower())
                            
                            from collections import Counter
                            word_counts = Counter(words)
                            
                            common_words = {'the', 'and', 'for', 'with', 'que', 'con', 'para', 'por', 'como'}
                            filtered_words = [w for w, _ in word_counts.most_common(30) if w not in common_words]
                            
                            return filtered_words[:20]
                        
                        top_channels['keywords'] = top_channels['recent_titles'].apply(extract_keywords_from_titles)
                        
                        st.info("🎯 Ranking channels by similarity...")
                        
                        candidates = top_channels.to_dict('records')
                        for candidate in candidates:
                            if 'channel_title' in candidate and 'channel_name' not in candidate:
                                candidate['channel_name'] = candidate['channel_title']
                        
                        ranked = similarity_engine.rank_channels_by_similarity(
                            candidates,
                            st.session_state['seed_profile'],
                            use_gemini=st.session_state.get('use_gemini_ranking', False),
                            gemini_api_key=GEMINI_API_KEY,
                            gemini_limit=10
                        )
                        
                        top_channels = pd.DataFrame(ranked)
                        
                        top_channels['similarity_score'] = top_channels['similarity'].apply(
                            lambda x: x.get('total_score', 0) if isinstance(x, dict) else 0
                        )
                        
                        top_channels['match_reasons'] = top_channels['similarity'].apply(
                            lambda x: ' • '.join(x.get('match_reasons', [])[:2]) if isinstance(x, dict) else ''
                        )
                        
                        top_channels = top_channels.sort_values('similarity_score', ascending=False)
                        
                        st.success(f"✅ Similarity ranking complete! Top match: {top_channels.iloc[0]['channel_title']} ({top_channels.iloc[0]['similarity_score']:.1f}/100)")
        
        st.info("💡 Results include channels whose content matches your search topics, not just their names.")

        # === AI SUMMARY ===
        if GEMINI_API_KEY:
            with st.spinner("✨ Generating AI Summary..."):
                step_start = time.time() if st.session_state.get('debug_mode', False) else None
                
                try:
                    summary_df = top_channels.copy()
                    summary_df['relevance_score'] = summary_df['relevance_score'].fillna(0).map('{:.0%}'.format)
                    summary_df['engagement_rate'] = summary_df['engagement_rate'].fillna(0).map('{:.2%}'.format)
                    summary_text = generate_summary(summary_df, final_query)
                    st.subheader("🤖 AI Generated Summary")
                    st.markdown(summary_text)
                except Exception as e:
                    st.error(f"Could not generate AI summary: {e}")
                
                if st.session_state.get('debug_mode', False) and step_start:
                    st.session_state.debug_data['timings']['ai_generation'] = time.time() - step_start
        else:
            st.warning("⚠️ GEMINI_API_KEY not configured. Skipping AI summary.")

        # === FORMAT AND DISPLAY RESULTS ===
        top_channels['relevance_score'] = top_channels['relevance_score'].fillna(0).map('{:.0%}'.format)
        top_channels['engagement_rate'] = top_channels['engagement_rate'].fillna(0).map('{:.2%}'.format)
        top_channels['avg_views_per_video'] = top_channels['avg_views_per_video'].fillna(0).map('{:,.0f}'.format)
        
        if 'similarity_score' in top_channels.columns:
            top_channels['similarity_score'] = top_channels['similarity_score'].fillna(0).map('{:.1f}'.format)
        
        # Store for outreach
        st.session_state['top_channels_for_outreach'] = top_channels[['channel_title']].reset_index(drop=True)
        st.session_state['final_query'] = final_query
        
        if 'similarity_score' in top_channels.columns:
            st.session_state['top_channels_full'] = top_channels.copy()

        # Choose display columns
        if 'similarity_score' in top_channels.columns:
            # Seed mode: show similarity, relevance, and avg views (remove match_reasons to save space)
            display_columns = ['channel_title', 'similarity_score', 'relevance_score', 
                            'avg_views_per_video', 'subscribers', 'country', 'engagement_rate']
        else:
            # Keyword mode: show relevance and avg views
            display_columns = ['channel_title', 'relevance_score', 'subscribers', 
                            'avg_views_per_video', 'country', 'engagement_rate']

        # Debug checkpoint
        if st.session_state.get('debug_mode', False):
            st.write(f"✓ Preparing to display {len(top_channels)} channels")
            st.write(f"Display columns: {display_columns}")

        # STORE IN SESSION STATE FOR DISPLAY
        st.session_state['display_df'] = top_channels[display_columns].copy()
        
        # Debug checkpoint
        if st.session_state.get('debug_mode', False):
            st.write(f"✓ display_df stored successfully with shape {st.session_state['display_df'].shape}")
        
        # === TRACK FINAL METRICS ===
        if st.session_state.get('debug_mode', False):
            if search_start_time:
                st.session_state.debug_data['timings']['total'] = time.time() - search_start_time
            
            if 'top_channels_full' in st.session_state:
                debug_tracker.track_similarity_scores(
                    st.session_state['top_channels_full'].to_dict('records')
                )
    
    except Exception as e:  # ← ERROR HANDLER
        st.error(f"❌ Error in run_search: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())
        
        # Show what step failed
        if st.session_state.get('debug_mode', False):
            st.write("### Debug Info at Failure:")
            st.write(f"- Session state keys: {list(st.session_state.keys())}")
            st.write(f"- Debug data: {st.session_state.get('debug_data', {})}")
      
        
# --- Function to resolve a channel handle (@username) to an ID ---
def get_channel_id_from_handle(youtube_service, handle):
    """Deprecated: use resolve_channel_id() instead."""
    return resolve_channel_id(youtube_service, handle)

# --- Gemini AI Integration ---
def generate_summary(df_results, query):
    """Formats the data and calls the Gemini API to generate a summary.
    
    Detects search mode (keywords vs channel-as-seed) and provides appropriate context.
    """
    try:
        # Track Gemini usage
        if st.session_state.get('debug_mode', False):
            try:
                import debug_tracker
                debug_tracker.track_api_call('gemini_summary')
                
                # Verify
                count = st.session_state.debug_data.get('gemini_summary_calls', 0)
                print(f"DEBUG: gemini_summary_calls now = {count}")
            except Exception as e:
                print(f"ERROR tracking summary call: {e}")               

        model = get_gemini_model()

        top_5_df = df_results.head(5)
        
        # Detect search mode by checking for similarity_score column
        is_seed_based = 'similarity_score' in df_results.columns
        
        data_string = ""
        for _, row in top_5_df.iterrows():
            data_string += f"- Channel: {row['channel_title']}\n"
            data_string += f"  - Subscribers: {row['subscribers']:,}\n"
            data_string += f"  - Country: {row['country']}\n"
            
            if is_seed_based:
                # Seed-based mode: show BOTH similarity and relevance
                similarity_score = row.get('similarity_score', 'N/A')
                relevance_score = row.get('relevance_score', 'N/A')
                
                # Extract match reasons from similarity dict if available
                if 'similarity' in row and isinstance(row['similarity'], dict):
                    reasons = row['similarity'].get('match_reasons', [])
                    reasons_text = ', '.join(reasons[:2]) if reasons else 'See detailed analysis'
                else:
                    reasons_text = 'N/A'
                
                data_string += f"  - Similarity Score: {similarity_score}/100\n"
                data_string += f"  - Why Similar: {reasons_text}\n"
                data_string += f"  - Topic Focus (Relevance): {relevance_score}\n"
            else:
                # Keyword mode: show relevance score
                data_string += f"  - Relevance Score: {row['relevance_score']}\n"
            
            data_string += f"  - Avg. Engagement Rate: {row['engagement_rate']}\n\n"

        # Adjust prompt based on search mode
        if is_seed_based:
            seed_name = st.session_state.get('seed_profile', {}).get('channel_name', 'the seed channel')
            prompt = f"""
You are an expert marketing analyst. Provide a concise summary of the top YouTube channels similar to "{seed_name}".

These channels were found using similarity analysis based on content topics, tags, audience size, and engagement patterns.

For each channel, consider:
- **Similarity Score (0-100)**: Overall match to the seed channel
- **Topic Focus (Relevance %)**: How focused they are on the auto-generated keywords from the seed
- **Engagement Rate**: How interactive their audience is

Base your analysis ONLY on the data below and highlight 2–3 standout matches and why they're good fits for collaboration.

Data:
{data_string}
"""
        else:
            prompt = f"""
You are an expert marketing analyst. Provide a concise summary of the top YouTube channels for the query "{query}".

Base your analysis ONLY on the data below and highlight 2–3 standout channels and why.

Data:
{data_string}
"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

def generate_outreach_drafts(
    top_channels_df: pd.DataFrame,
    original_query: str,
    limit: int = 3,
    temperature: float = 0.7,
    retries: int = 2,
    language: str = "en",   # <- NEW: "en" or "es"
) -> list[dict]:
    """
    Generate short, friendly outreach email drafts for the top N channels using Gemini.

    Parameters
    ----------
    top_channels_df : pd.DataFrame
        Must contain a 'channel_title' column (string-like).
    original_query : str
        The original search query to reference for authenticity.
    limit : int
        Number of channels to process (default 3).
    temperature : float
        Sampling temperature for Gemini (0.0-1.0 usually).
    retries : int
        How many times to retry a failed API call (default 2).
    language : str
        "en" for English or "es" for Spanish (default "en").

    Returns
    -------
    List[dict]: [{'channel_title': str, 'draft_text': str}, ...]
    """
    results: list[dict] = []

    if top_channels_df is None or top_channels_df.empty:
        return results
    if 'channel_title' not in top_channels_df.columns:
        return results

    model = get_gemini_model(temperature=temperature)

    df = (
        top_channels_df[['channel_title']]
        .dropna(subset=['channel_title'])
        .copy()
    )
    df['channel_title'] = df['channel_title'].astype(str).str.strip()
    df = df[df['channel_title'] != ""].drop_duplicates(subset=['channel_title'])
    df = df.head(max(0, int(limit)))

    oq = (original_query or "").strip() or "my audience’s interests"

    # Language instruction
    lang = (language or "en").lower()
    if lang.startswith("es"):
        lang_line = "Write the email in Spanish. Usa un español claro, neutro y profesional."
    else:
        lang_line = "Write the email in English in a clear, professional yet friendly tone."

    for _, row in df.iterrows():
        channel_name = row['channel_title']

         # Track each Gemini call
        if st.session_state.get('debug_mode', False):
            debug_tracker.track_api_call('gemini_outreach')

        prompt = f"""
Act as a marketing professional. Your task is to write a short, friendly, and professional outreach email to a YouTube creator.

**Instructions:**
- The tone should be respectful and concise.
- Mention the creator's channel name specifically.
- Reference the topic of my original search, which was "{oq}".
- The goal is to express interest in a potential collaboration.
- Do not use overly corporate language.
- {lang_line}

**Creator Channel Name:** {channel_name}

**Email Draft:**
"""

        draft_text = ""
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = model.generate_content(prompt)
                draft_text = (getattr(resp, "text", None) or str(resp)).strip()
                if draft_text.startswith("```"):
                    draft_text = draft_text.strip("`").strip()
                break
            except Exception as e:
                last_err = e
                continue

        if not draft_text and last_err:
            draft_text = f"(Error generating draft for '{channel_name}': {type(last_err).__name__}: {last_err})"

        results.append({
            "channel_title": channel_name,
            "draft_text": draft_text
        })

    return results


# --- Streamlit User Interface ---

st.set_page_config(
    page_title="CCSeeker - YouTube Creator Search",
    page_icon="appicons/app-icon-192x192.png",  # Relative path to your favicon
    layout="wide"
)

# Initialize debug tracking
debug_tracker.initialize_debug_tracking()

st.markdown("""
<style>
/* tighten the small toggle so it sits on the same baseline */
[data-testid="stToggle"] label { margin-top: 0 !important; }
/* ensure the right input matches the general field height */
[data-testid="stNumberInput"] input { padding-top: .55rem; padding-bottom: .55rem; }
</style>
""", unsafe_allow_html=True)


def inject_css(path: str):
    p = Path(path)
    if not p.exists():
        st.warning(f"CSS not found: {path}")
        return
    try:
        css = p.read_text(encoding="utf-8-sig")  # handles UTF-8 + BOM too
    except UnicodeDecodeError:
        css = p.read_text(encoding="latin-1", errors="replace")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_css("app/theme_ccseeker_dark.css")


# Header with logo and title
col_logo, col_title = st.columns([1, 10], vertical_alignment="center")

with col_logo:
    st.image("appicons/app-icon-192x192.png", width=100)  # Bigger logo

with col_title:
    st.title("CCSeeker")
    st.markdown("*Discover Niche YouTube Creators*")  # Slogan in italic

# === DEBUG MODE TOGGLE (in sidebar) ===
with st.sidebar:
    st.markdown("---")  # Visual separator
    
    # Toggle with explanation
    debug_enabled = st.toggle(
        "🔍 Debug Mode",
        value=st.session_state.get('debug_mode', False),
        help="Show API usage, performance metrics, and detailed similarity scores"
    )
    
    # Update session state
    st.session_state.debug_mode = debug_enabled

# The search method selector is outside the form to allow instant UI updates.
st.header("1. Search Method")
# Custom button-based selector for better layout control
st.markdown("#### Choose your search method:")

# Create columns for the method buttons
col1, col2, col3 = st.columns([2, 2, 3])

# Initialize session state if not exists
if 'search_method' not in st.session_state:
    st.session_state.search_method = "Keywords"

with col1:
    if st.button(
        "🔑 Keywords",
        key="btn_keywords",
        use_container_width=True,
        type="primary" if st.session_state.search_method == "Keywords" else "secondary"
    ):
        st.session_state.search_method = "Keywords"
        st.rerun()

with col2:
    if st.button(
        "📺 Channel-as-Seed",
        key="btn_seed",
        use_container_width=True,
        type="primary" if st.session_state.search_method == "Channel-as-Seed" else "secondary"
    ):
        st.session_state.search_method = "Channel-as-Seed"
        st.rerun()

# Use the session state value
search_method = st.session_state.search_method
with st.form("search_form"):
    # Inputs change depending on the selected method
    if search_method == "Keywords":
        query_input = st.text_input(
            "Search Keywords",
            "manga, anime",
            help=(
                "Enter topics separated by commas to find channels discussing any of them. "
                "Examples: 'manga, anime' or 'cooking, recipes, food'. "
                "Matches channel content (videos, descriptions) not just channel names."
            ),
        )
        seed_url_input = ""  # keep defined
    else:
        st.info("💡 Enter the full URL of a YouTube channel to find similar creators.")
        seed_url_input = st.text_input("YouTube Channel URL (the 'seed')", "https://www.youtube.com/@YourFavoriteChannel")
        query_input = ""  # keep defined
        with st.expander("How does Channel-as-Seed work?"):
            st.markdown("""
                This method discovers new channels based on a single example channel you provide.
                1.  **Analyze:** The agent fetches the latest videos from the URL you enter.
                2.  **Learn:** It extracts the most common topics and keywords from that channel's video titles and tags.
                3.  **Discover:** It then uses those learned keywords to launch a new, highly specific search to find other channels with similar content.
            """)

        # Output language (post-translation) selector
        output_lang_label = st.selectbox(
            "Translate topics into",
            [
                "Original (auto)",
                "English",
                "Español",
                "Português",
                "Français",
                "Deutsch",
            ],
            help="Extraction stays in the seed's original language; this only affects the final topic phrasing.")
        _lang_map = {
            "Original (auto)": "auto",
            "English": "en",
            "Español": "es",
            "Português": "pt",
            "Français": "fr",
            "Deutsch": "de",
        }
        target_language_code = _lang_map.get(output_lang_label, "auto")
        ignore_words_input = st.text_input(
            "Ignore words (comma-separated)",
            value="",
            help="Words to always ignore in topic extraction (e.g., brand names)."
        )

    country_options = COUNTRY_OPTIONS_BASE.copy()
    country_options.sort()
    country_options.insert(0, "Global")
    selected_country = st.selectbox(
    "Prioritize Region",
    country_options,
    index=0,  # Default to Global (no regional bias)
    help="YouTube will show channels popular in this region first, but results aren't limited to it.",
)
    region_input = "" if selected_country == "Global" else selected_country.split("(")[-1][:2]

    st.header("2. Filtering Criteria")
    c1, c2, c3 = st.columns(3)

    with c1:
        min_subs_input = st.number_input(
            "Minimum Subscribers",
            min_value=0, value=10000, step=1000, format="%d",
            help="Set to 0 to ignore."
        )

    with c2:
        # Create country dropdown options (same as Search Region)
        country_filter_options = country_options.copy()  # Uses same list as region selector
        selected_country_filter = st.selectbox(
            "Channel Country (strict filter)",
            country_filter_options,
            index=0,  # Default to "Global"
            help="Only show channels registered in this country. Leave as 'Global' to see all countries.",
            key="country_filter_select"
        )
        country_filter_input = "" if selected_country_filter == "Global" else selected_country_filter.split("(")[-1][:2]

    with c3:
        months_ago_input = st.number_input(
            "Published within last (months)",
            value=18, min_value=0, step=1,
            help="Only show channels with uploads in the last X months. Set to 0 to ignore upload recency."
        )

    submitted = st.form_submit_button("Find Creators")
    
# ============================================================================
# === HANDLE SEED-BASED SEARCH ===
# ============================================================================

if st.session_state.get('run_similarity_search'):
    # Clear the trigger
    st.session_state['run_similarity_search'] = False
    
    # Get the built query and run search
    if st.session_state.get('built_query') and st.session_state.get('seed_profile'):
        
        youtube = get_youtube()
        query = st.session_state['built_query']
        
        st.info(f"🔎 Searching for channels similar to: **{st.session_state['seed_profile']['channel_name']}**")
        
        # Run the search with the generated query
        run_search(
            youtube=youtube,
            final_query=query,
            region_input=region_input,
            min_subs_input=min_subs_input,
            months_ago_input=months_ago_input,
            country_filter_input=country_filter_input,
            boolean_fetch=False
        )

# --- Main Execution Logic ---
if submitted:
    # New search: clear previous outreach/session caches
    st.session_state.pop('top_channels_for_outreach', None)
    st.session_state.pop('final_query', None)
    st.session_state.pop('display_df', None)
    if not YOUTUBE_API_KEY:
        st.error("Please ensure your YOUTUBE_API_KEY is set in your .env file.")
    else:
        youtube = get_youtube()
        final_query = ""

        if search_method == "Channel-as-Seed":
            with st.spinner("Resolving channel..."):
                seed_channel_id = resolve_channel_id(youtube, seed_url_input)
            if not seed_channel_id:
                st.error("Could not resolve a channel ID from the provided input. Use a full channel URL, @handle, or a UC… ID.")
                final_query = None
            else:
                st.info(f"Using Channel ID: {seed_channel_id}")

            if seed_channel_id:
                # Apply user-provided penalties (banned words)
                penalties = set(
                    w.strip().lower() 
                    for w in (ignore_words_input or "").split(",") 
                    if w.strip()
                )

                # NEW: Analyze seed channel to extract complete profile
                with st.spinner("🔍 Analyzing seed channel..."):
                    seed_profile = seedmod.analyze_seed_channel_v2(
                        youtube,
                        seed_channel_id,
                        max_videos=30,
                        user_banned_words=penalties,
                        gemini_api_key=GEMINI_API_KEY
                    )
                
                if seed_profile:
                    # Store profile in session state for later use
                    st.session_state['seed_profile'] = seed_profile
                    st.session_state['seed_channel_id'] = seed_channel_id
                    
                    st.success("✅ Seed analysis complete! Review the profile below.")
                else:
                    st.error("Failed to analyze seed channel. Please check the URL and try again.")
                    seed_profile = None

               
        else:
            final_query = query_input

        # --- Proceed only if we have a valid query ---
        if final_query:
            run_search(
                youtube=youtube,
                final_query=final_query,
                region_input=region_input,
                min_subs_input=min_subs_input,
                months_ago_input=months_ago_input,
                country_filter_input=country_filter_input,
                boolean_fetch=bool(re.search(r"\b(AND|OR)\b", final_query, re.IGNORECASE)),
            )

# ============================================================================
# === SEED PROFILE REVIEW (NEW) ===
# ============================================================================

if st.session_state.get('seed_profile'):
    profile = st.session_state['seed_profile']
    
    st.header("3. 📊 Seed Channel Profile")
    
    # Display key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Subscribers", f"{profile['subscriber_count']:,}")
        st.metric("Tier", profile['subscriber_tier'].upper())
    
    with col2:
        st.metric("Language", profile['language'].upper())
        st.metric("Upload Frequency", f"{profile['upload_frequency']:.1f}/month")
    
    with col3:
        st.metric("Engagement Rate", f"{profile['avg_engagement_rate']:.2%}")
    
    # Show extracted topics
    st.subheader("📌 Primary Topics")
    if profile['primary_keywords']:
        st.write("**Multi-word phrases:**")
        st.write(", ".join(profile['primary_keywords']))
    else:
        st.info("No multi-word topics found")
    
    if profile['secondary_keywords']:
        st.write("**Single words:**")
        st.write(", ".join(profile['secondary_keywords'][:8]))
    
    # Show common tags
    st.subheader("🏷️ Most Common Tags")
    if profile['common_tags']:
        tag_display = profile['common_tags'][:12]
        st.write(", ".join(tag_display))
    else:
        st.info("No tags found in recent videos")
    
    # Show AI summary if available
    if profile.get('description_summary'):
        with st.expander("🤖 AI Channel Summary"):
            st.write(profile['description_summary'])
    
    # Search options
    st.subheader("🔍 Search Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_gemini_ranking = st.checkbox(
            "Use AI-enhanced ranking",
            value=True,
            help="Let Gemini analyze the top 10 matches for better accuracy (slower)"
        )
        st.session_state['use_gemini_ranking'] = use_gemini_ranking
    
    with col2:
        filter_by_size = st.checkbox(
            "Only similar-sized channels",
            value=True,
            help="Filter results to channels with ±50% subscriber count"
        )
        st.session_state['filter_by_size'] = filter_by_size
    
    # Build search query from profile
    search_terms = (
        profile['primary_keywords'][:3] +  # Top 3 phrases
        profile['common_tags'][:5]         # Top 5 tags
    )
    
    # Quote multi-word terms
    quoted_terms = [
        f'"{term}"' if ' ' in term else term 
        for term in search_terms
    ]
    
    built_query = ", ".join(quoted_terms[:6])  # Max 6 terms
    
    st.write("**Generated search query:**")
    st.code(built_query)
    
    # Main search button
    if st.button("🚀 Find Similar Channels", type="primary", key="btn_find_similar"):
        st.session_state['built_query'] = built_query
        st.session_state['run_similarity_search'] = True
        st.rerun()

# Keep the results table visible across reruns
if 'display_df' in st.session_state:   
    st.dataframe(
        st.session_state['display_df'],
        column_config={
            "similarity_score": st.column_config.Column(
                help="How similar this channel is to your seed channel (0-100). Based on shared topics, tags, audience size, engagement patterns, and upload frequency. Higher = better match."
            ),
            "relevance_score": st.column_config.Column(
                help="The percentage of a channel's recent videos that contain your search keywords in the title. A higher score means the channel is more focused on your topic."
            ),
            "avg_views_per_video": st.column_config.Column(
                help="Average views per video across the channel's recent uploads. Indicates reach and virality potential."
            ),
            "engagement_rate": st.column_config.Column(
                help="Calculated as (Likes + Comments) / Views, averaged across a channel's recent videos. This shows how interactive the audience is."
            ),
        }
    )
# ============================================================================
# === ENHANCED MATCH EXPLANATIONS ===
# ============================================================================
if 'similarity_score' in st.session_state.get('display_df', pd.DataFrame()).columns:
    
    # Only show if we have the full data
    if 'top_channels_full' not in st.session_state:
        st.info("Run a seed-based search to see detailed analysis")
    else:
        # Use a unique container to prevent duplicate keys
        with st.container():
            st.subheader("🔍 Detailed Match Analysis")
            
            top_channels_data = st.session_state['top_channels_full']
            channel_names = top_channels_data['channel_title'].tolist()[:10]
            
            # Create unique key based on session state
            analysis_key = f"channel_analysis_{id(st.session_state.get('seed_profile', {}))}"
            
            selected_channel = st.selectbox(
                "Select a channel to see detailed similarity analysis:",
                options=channel_names,
                key=analysis_key  # ← Unique key
            )
        
        if selected_channel:
            # Find the full row
            channel_row = top_channels_data[
                top_channels_data['channel_title'] == selected_channel
            ].iloc[0]
            
            # Generate detailed explanation
            explanation = similarity_engine.generate_match_explanation(
                channel_row.to_dict(),
                st.session_state['seed_profile'],
                detailed=True
            )
            
            # Display in an attractive format
            st.markdown(explanation)
            
            # Show tags comparison
            with st.expander("📊 Tag Comparison"):
                candidate_tags = set(channel_row.get('tags', []))
                seed_tags = set(st.session_state['seed_profile']['common_tags'])
                
                common = candidate_tags & seed_tags
                candidate_only = candidate_tags - seed_tags
                seed_only = seed_tags - candidate_tags
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Common Tags:**")
                    if common:
                        st.write(", ".join(list(common)[:15]))
                    else:
                        st.write("_(none)_")
                
                with col2:
                    st.write(f"**{selected_channel} Only:**")
                    if candidate_only:
                        st.write(", ".join(list(candidate_only)[:10]))
                    else:
                        st.write("_(none)_")
                
                with col3:
                    st.write(f"**Seed Only:**")
                    if seed_only:
                        st.write(", ".join(list(seed_only)[:10]))
                    else:
                        st.write("_(none)_")
    
    selected_channel = st.selectbox(
        "Select a channel to see detailed similarity analysis:",
        options=channel_names,
        key="selected_channel_analysis"
    )
    
    if selected_channel:
        # Find the full row
        channel_row = top_channels_data[top_channels_data['channel_title'] == selected_channel].iloc[0]
        
        # Generate detailed explanation
        explanation = similarity_engine.generate_match_explanation(
            channel_row.to_dict(),
            st.session_state['seed_profile'],
            detailed=True
        )
        
        # Display in an attractive format
        st.markdown(explanation)
        
        # Show tags comparison
        with st.expander("📊 Tag Comparison"):
            candidate_tags = set(channel_row.get('tags', []))
            seed_tags = set(st.session_state['seed_profile']['common_tags'])
            
            common = candidate_tags & seed_tags
            candidate_only = candidate_tags - seed_tags
            seed_only = seed_tags - candidate_tags
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Common Tags:**")
                if common:
                    st.write(", ".join(list(common)[:15]))
                else:
                    st.write("_(none)_")
            
            with col2:
                st.write(f"**{selected_channel} Only:**")
                if candidate_only:
                    st.write(", ".join(list(candidate_only)[:10]))
                else:
                    st.write("_(none)_")
            
            with col3:
                st.write(f"**Seed Only:**")
                if seed_only:
                    st.write(", ".join(list(seed_only)[:10]))
                else:
                    st.write("_(none)_")

# === Global Outreach Section (works across reruns) ===
if 'top_channels_for_outreach' in st.session_state and not st.session_state['top_channels_for_outreach'].empty:
    st.write("")  # spacer

    # Language switch
    lang_label = st.radio("Email language", ["English", "Español"], horizontal=True, key="outreach_lang")
    lang_code = "es" if lang_label == "Español" else "en"

    if st.button("Generate Outreach Drafts", key="btn_outreach"):
        if not GEMINI_API_KEY:
            st.error("Please ensure your GEMINI_API_KEY is set in your .env file to generate outreach drafts.")
        else:
            with st.spinner("Generating outreach drafts..."):
                try:
                    drafts = generate_outreach_drafts(
                        st.session_state['top_channels_for_outreach'],
                        st.session_state.get('final_query', ""),
                        limit=3,
                        language=lang_code,  # <-- pass the selected language
                    )
                    st.header("📧 AI Generated Outreach Drafts")
                    for i, d in enumerate(drafts, start=1):
                        st.subheader(f"{i}. {d['channel_title']}")
                        st.text_area(
                            label=f"Draft for {d['channel_title']}",
                            value=d['draft_text'],
                            height=220,
                            key=f"draft_text_{i}"
                        )
                except Exception as e:
                    st.error(f"Unexpected error while generating drafts: {type(e).__name__}: {e}")

# ============================================================================
# === RENDER DEBUG PANEL AT THE END (AFTER ALL TRACKING) ===
# ============================================================================

# This ensures the debug panel shows updated counters after search completes
if st.session_state.get('debug_mode', False):
    with st.sidebar:
        debug_tracker.render_debug_panel()
        # Optional: Add manual reset button (for testing)
        debug_tracker.render_quota_reset_button()

                            



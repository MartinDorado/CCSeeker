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
    from . import seed_topics_hardened as seedmod
except ImportError:
    # Fallback for direct execution
    import seed_topics_hardened as seedmod

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
    return genai.GenerativeModel('gemini-1.5-pro-latest', **cfg)

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

    - Strips outer quotes from multi-word phrases.
    - Escapes regex metacharacters in terms.
    - Matches in video title OR video tags.
    """
    if df.empty or not isinstance(query, str) or not query.strip():
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])

    # Split on OR/AND tokens, then normalize terms
    raw_terms = [word.strip() for word in re.split(r"OR|AND", query, flags=re.IGNORECASE)]
    cleaned = []
    for t in raw_terms:
        if not t:
            continue
        # remove outer quotes if present and escape for regex
        if (len(t) >= 2) and ((t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'"))):
            t = t[1:-1]
        t = t.strip()
        if not t:
            continue
        cleaned.append(re.escape(t))

    if not cleaned:
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])

    # Use non-capturing group to avoid pandas warning about match groups
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
    generate_summary_checkbox: bool,
    boolean_fetch: bool = False,
):
    """Run the 4-step search + analysis pipeline and render results."""
    with st.spinner("Step 1/4: Searching for channels..."):
        initial_channels = search_channels_multi_term(final_query, region_input, max_videos_per_term=150)

    if not initial_channels:
        st.error("Search did not return any channels.")
        return

    df_initial = pd.DataFrame(initial_channels)
    with st.expander("See raw channels found"):
        st.dataframe(df_initial)

    with st.spinner("Step 2/4: Fetching channel statistics..."):
        channel_statistics = get_channel_stats(youtube, df_initial['channel_id'].tolist())

    if not channel_statistics:
        st.warning("Could not retrieve detailed stats for the found channels.")
        return

    df_stats = pd.DataFrame(channel_statistics)
    enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')

    with st.spinner("Step 3/4: Fetching recent video details..."):
        video_data = get_video_details(youtube, enriched_channel_data.to_dict('records'), max_videos_per_channel=10)

    if not video_data:
        st.warning("Could not retrieve any video details from the channels.")
        st.dataframe(enriched_channel_data.sort_values(by="subscribers", ascending=False))
        return

    with st.spinner("Step 4/4: Analyzing and filtering results..."):
        df_videos = pd.DataFrame(video_data)
        relevance_scores = calculate_keyword_relevance(df_videos.copy(), final_query)
        df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
        df_videos['engagement_rate'] = (df_videos['video_likes'] + df_videos['video_comments']) / (df_videos['video_views'] + 1)
        df_full = pd.merge(df_videos, enriched_channel_data, on='channel_id')

        filtered_df = df_full[df_full['subscribers'] >= min_subs_input]

        # Conditionally apply the date filter
        if months_ago_input > 0:
            date_cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=months_ago_input)
            filtered_df = filtered_df[filtered_df['published_at'] >= date_cutoff]
        if country_filter_input:
            filtered_df = filtered_df[filtered_df['country'] == country_filter_input.upper()]

        if filtered_df.empty:
            st.error("No channels matched all your filtering criteria after full analysis.")
            return

        avg_engagement = filtered_df.groupby('channel_id')['engagement_rate'].mean().reset_index()
        final_channels = pd.merge(enriched_channel_data, avg_engagement, on='channel_id')
        final_channels = pd.merge(final_channels, relevance_scores, on='channel_id', how='left')

        top_channels = final_channels.sort_values(by=['relevance_score', 'subscribers'], ascending=False)

        st.success(f"Analysis Complete! Found {len(top_channels)} channels matching your criteria.")
        st.info("💡 Results include channels whose content matches your search topics, not just their names.")

        # --- AI Summary BEFORE formatting numbers for the table ---
        if generate_summary_checkbox:
            if not GEMINI_API_KEY:
                st.error("Please ensure your GEMINI_API_KEY is set in your .env file to generate a summary.")
            else:
                with st.spinner("Generating AI Summary..."):
                    summary_df = top_channels.copy()
                    summary_df['relevance_score'] = summary_df['relevance_score'].fillna(0).map('{:.0%}'.format)
                    summary_df['engagement_rate'] = summary_df['engagement_rate'].fillna(0).map('{:.2%}'.format)
                    summary_text = generate_summary(summary_df, final_query)
                    st.subheader("AI Generated Summary")
                    st.markdown(summary_text)

        # Format for display
        top_channels['relevance_score'] = top_channels['relevance_score'].fillna(0).map('{:.0%}'.format)
        top_channels['engagement_rate'] = top_channels['engagement_rate'].fillna(0).map('{:.2%}'.format)
        top_channels['avg_views_per_video'] = top_channels['avg_views_per_video'].fillna(0).map('{:,.0f}'.format)

        # Persist minimal data for outreach across reruns
        st.session_state['top_channels_for_outreach'] = top_channels[['channel_title']].reset_index(drop=True)
        st.session_state['final_query'] = final_query

        # (optional) also persist the displayed table so it stays visible after reruns
        st.session_state['display_df'] = top_channels[['channel_title', 'relevance_score', 'subscribers', 'avg_views_per_video', 'country', 'engagement_rate']].copy()

# --- Function to resolve a channel handle (@username) to an ID ---
def get_channel_id_from_handle(youtube_service, handle):
    """Deprecated: use resolve_channel_id() instead."""
    return resolve_channel_id(youtube_service, handle)

# --- Channel-as-Seed Function ---
def generate_summary(df_results, query):
    """Formats the data and calls the Gemini API to generate a summary."""
    try:
        model = get_gemini_model()

        top_5_df = df_results.head(5)
        data_string = ""
        for _, row in top_5_df.iterrows():
            data_string += f"- Channel: {row['channel_title']}\n"
            data_string += f"  - Subscribers: {row['subscribers']:,}\n"
            data_string += f"  - Country: {row['country']}\n"
            data_string += f"  - Relevance Score: {row['relevance_score']}\n"
            data_string += f"  - Avg. Engagement Rate: {row['engagement_rate']}\n\n"

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


c1, c2 = st.columns([1, 10], vertical_alignment="center")
c1.image("appicons/app-icon-192x192.png", width=64)
c2.title("YouTube Creator Search Agent")

# The search method selector is outside the form to allow instant UI updates.
st.header("1. Search Method")
search_method = st.radio("Choose your search method:", ("Keywords", "Channel-as-Seed"), horizontal=True)

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
        "Search Region (Bias)",
        country_options,
        index=0,  # Default to Global (no regional bias)
        help="This tells YouTube to prioritize results popular in this country. It is not a strict filter.",
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
        country_filter_input = st.text_input(
            "Channel Country (strict filter)", "",
            help="Leave blank to ignore."
        )

    with c3:
        months_ago_input = st.number_input(
            "Published within last (months)",
            value=18, min_value=0, step=1,
            help="Set to 0 to ignore this filter."
        )

    submitted = st.form_submit_button("Find Creators")



    # --- NEW: Checkbox for AI Summary (inside the form) ---
    generate_summary_checkbox = st.checkbox("Generate AI Summary of Top Results")

    

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
                # Apply user-provided penalties to the seed analyzer
                penalties = set(w.strip().lower() for w in (ignore_words_input or "").split(",") if w.strip())

                # Generate an initial topic query from the seed (but do not search yet)
                generated_query = seedmod.analyze_seed_channel(
                    youtube,
                    seed_channel_id,
                    max_seed_videos=30,
                    top_k=10,
                    use_gemini=True,
                    gemini_api_key=GEMINI_API_KEY,
                    language="auto",  # keep original-language topics for search
                    include_descriptions=False,
                    user_penalties=penalties,
                )

                if generated_query:
                    st.session_state['seed_candidates'] = _parse_topics_from_query(generated_query)
                    st.session_state['seed_channel_id'] = seed_channel_id
                    st.session_state['seed_ignore_bag'] = set(penalties)
                    # Keep original-language topics through the iteration loop
                    st.session_state['seed_target_language_code'] = "auto"
                    st.success("Seed topics generated. Review and refine them below, then run the search.")

                # Hold off running search until user reviews topics
                final_query = None
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
                generate_summary_checkbox=generate_summary_checkbox,
                boolean_fetch=bool(re.search(r"\b(AND|OR)\b", final_query, re.IGNORECASE)),
            )

# === Seed Topic Review (iterate + ignore bag) ===
if st.session_state.get('seed_candidates'):
    st.header("3. Review Seed Topics")
    candidates = st.session_state.get('seed_candidates', [])

    # Selection of topics to keep
    default_keep = st.session_state.get('seed_selected_topics', candidates)
    selected_topics = st.multiselect(
        "Select topics to keep",
        options=candidates,
        default=default_keep,
        key="seed_selected_topics",
        help="Keep the topics you like. Unselected ones will be added to the ignore bag when regenerating.")

    # Editable ignore bag
    ignore_set = set(st.session_state.get('seed_ignore_bag', set()))
    ignore_str_default = ", ".join(sorted(ignore_set))
    if 'seed_ignore_input' not in st.session_state:
        st.session_state['seed_ignore_input'] = ignore_str_default
    st.session_state['seed_ignore_input'] = st.text_input(
        "Ignore words (bag)",
        value=st.session_state['seed_ignore_input'],
        help="Words to always ignore in topic extraction (e.g., names/brands/places). You can edit this directly.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Regenerate topics", key="btn_regen_topics"):
            # Merge manual ignore entries with tokens from unselected topics
            manual_ignores = {w.strip().lower() for w in (st.session_state['seed_ignore_input'] or "").split(',') if w.strip()}
            dropped = set(candidates) - set(selected_topics)
            dropped_tokens = set()
            for t in dropped:
                for w in t.split():
                    w = w.strip().lower()
                    if w:
                        dropped_tokens.add(w)
            new_bag = set(ignore_set) | manual_ignores | dropped_tokens
            st.session_state['seed_ignore_bag'] = new_bag
            st.session_state['seed_ignore_input'] = ", ".join(sorted(new_bag))

            # Re-run seed analysis with updated penalties
            if not YOUTUBE_API_KEY:
                st.error("Please ensure your YOUTUBE_API_KEY is set in your .env file.")
            else:
                youtube2 = get_youtube()
                new_query = seedmod.analyze_seed_channel(
                    youtube2,
                    st.session_state.get('seed_channel_id', ''),
                    max_seed_videos=30,
                    top_k=10,
                    use_gemini=True,
                    gemini_api_key=GEMINI_API_KEY,
                    language="auto",
                    include_descriptions=False,
                    user_penalties=new_bag,
                )
                if new_query:
                    st.session_state['seed_candidates'] = _parse_topics_from_query(new_query)
                    st.success("Regenerated topics using updated ignore bag.")
                else:
                    st.warning("Could not regenerate topics with the current settings.")

    with c2:
        if st.button("Search with selected topics", key="btn_search_with_selected"):
            if not selected_topics:
                st.warning("Please select at least one topic to search.")
            else:
                built_query = _build_query_from_topics(selected_topics)
                if not YOUTUBE_API_KEY:
                    st.error("Please ensure your YOUTUBE_API_KEY is set in your .env file.")
                else:
                    youtube2 = get_youtube()
                    run_search(
                        youtube=youtube2,
                        final_query=built_query,
                        region_input=region_input,
                        min_subs_input=min_subs_input,
                        months_ago_input=months_ago_input,
                        country_filter_input=country_filter_input,
                        generate_summary_checkbox=generate_summary_checkbox,
                        boolean_fetch=False,
                    )

# Keep the results table visible across reruns
if 'display_df' in st.session_state:
    st.dataframe(
        st.session_state['display_df'],
        column_config={
            "relevance_score": st.column_config.Column(
                help="The percentage of a channel's recent videos that contain your search keywords in the title. A higher score means the channel is more focused on your topic."
            ),
            "engagement_rate": st.column_config.Column(
                help="Calculated as (Likes + Comments) / Views, averaged across a channel's recent videos. This shows how interactive the audience is."
            ),
        }
    )

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

                            



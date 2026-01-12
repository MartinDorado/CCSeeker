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
    from . import feedback_tracker
except ImportError:
    # Fallback for direct execution
    import seed_topics_v2 as seedmod
    import similarity_engine
    import feedback_tracker

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

# Core module imports (extracted pure functions)
try:
    from .core import (
        MAX_SEARCH_TERMS,
        validate_and_truncate_query,
        extract_identifier_from_url,
        resolve_channel_id,
        strip_outer_quotes,
        calculate_keyword_relevance,
        # YouTube API (Phase 2)
        SearchResult,
        ChannelStatsResult,
        VideoDetailsResult,
        search_channels_hybrid as _search_channels_hybrid_core,
        search_channels_multi_term as _search_channels_multi_term_core,
        get_channel_stats as _get_channel_stats_core,
        get_video_details as _get_video_details_core,
        # Gemini API (Phase 2)
        OutreachDraft,
        SummaryResult,
        generate_ai_relevance_score as _generate_ai_relevance_score_core,
        generate_summary as _generate_summary_core,
        generate_outreach_drafts as _generate_outreach_drafts_core,
    )
except ImportError:
    from core import (
        MAX_SEARCH_TERMS,
        validate_and_truncate_query,
        extract_identifier_from_url,
        resolve_channel_id,
        strip_outer_quotes,
        calculate_keyword_relevance,
        # YouTube API (Phase 2)
        SearchResult,
        ChannelStatsResult,
        VideoDetailsResult,
        search_channels_hybrid as _search_channels_hybrid_core,
        search_channels_multi_term as _search_channels_multi_term_core,
        get_channel_stats as _get_channel_stats_core,
        get_video_details as _get_video_details_core,
        # Gemini API (Phase 2)
        OutreachDraft,
        SummaryResult,
        generate_ai_relevance_score as _generate_ai_relevance_score_core,
        generate_summary as _generate_summary_core,
        generate_outreach_drafts as _generate_outreach_drafts_core,
    )

import math
import time

# ============================================================================
# --- TABLE OF CONTENTS ---
# ============================================================================
# LINES 1-154      | IMPORTS & CONFIGURATION
#                   - Module imports (seed_topics_v2, similarity_engine, smart_cache, debug_tracker)
#                   - Constants (MAX_SEARCH_TERMS=2, cache TTLs, thresholds)
#                   - Environment setup (API keys from .env)
                 

# LINES 155-276    | HELPER QUERY FUNCTIONS
#                  - validate_and_truncate_query() - Enforce 2-term limit
#                  - render_term_counter() - Visual query validation
#                  - extract_identifier_from_url() - Parse YouTube URLs
#                  - resolve_channel_id() - Convert handles/URLs to channel IDs
#                  - _strip_outer_quotes() - Clean user input

# LINES 277-342   | CACHING LAYER (Streamlit cache helpers)
#                  - get_channel_stats_cached() - 7-day TTL
#                  - get_video_details_cached() - delegates to per-channel cache (24h in smart_cache.py)
#                  - search_channels_multi_term_cached() - 3-day TTL

# LINES 343- 879   | CORE LOGIC - YOUTUBE API WRAPPERS + AI GENERATION (Gemini)
#                  - get_youtube() - Initialize API client
#                  - get_channel_stats() - Fetch channel metadata
#                  - search_channels_multi_term() - Multi-term OR search
#                  - search_channels_hybrid() - Two-phase search (video + name)
#                  - calculate_keyword_relevance() - Match scoring
#                  - generate_summary() - Channel overview
#                  - generate_outreach_emails() - Personalized drafts

# LINES 880 -1428  | MAIN PIPELINE: run_search()
#                  ~604-638: Docstring (pipeline overview)
#                  ~640-680: Step 0.5 - Query validation & truncation
#                  ~681-720: Step 1 - Search channels (hybrid strategy)
#                  ~721-760: Step 2 - Fetch channel stats
#                  ~761-800: Step 3 - Apply filters (subs, country)
#                  ~801-850: Step 4 - Quality selection (cap at 50)
#                  ~851-950: Step 5 - Deep analysis (fetch videos, calc relevance)
#                  ~951-1020: Step 6 - Similarity ranking (if seed mode)
#                  ~1021-1060: Step 7 - AI summary generation
#                  ~1061-1100: Step 8 - Format & store results

# LINES 1429-END | STREAMLIT UI SETUP
#                  - inject_css() - Load custom theme
#                  - Input form
#                  - Seed analysis handeler
#                  - Seed profile display
#                  - Result display
#                  - Global features


# === SESSION STATE KEYS ===
# This app uses the following st.session_state keys:
# - debug_mode: bool - Toggle for debug panel
# - search_method: str - "Keywords" or "Channel-as-Seed"
# - seed_profile: dict - Analyzed seed channel data
# - debug_data: dict - API call tracking and performance metrics
# - daily_quota: dict - Persistent quota tracking across sessions

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# API Configuration
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Search Parameters
# MAX_SEARCH_TERMS is imported from core.query_utils
MAX_SEARCH_RESULTS = 50           # Maximum channels to return per search
MAX_VIDEOS_PER_TERM = 100         # Videos to fetch from YouTube search per term
MAX_VIDEOS_PER_CHANNEL = 10       # Videos to analyze per channel (for relevance)
MAX_VIDEOS_PER_SEED = 50          # Videos to analyze for seed channel profile - to adjust it go to seed_topics_v2.py - def analyze_seed_channel_v2 -ln 175


# Cache TTLs (in seconds)
# These are documented here but must be hardcoded in decorators (Python limitation)
CACHE_TTL_CHANNEL_STATS = 604800   # 1 week - used in get_channel_stats_cached()
CACHE_TTL_VIDEO_DETAILS = 259200   # 3 days - used in get_video_details_cached()
CACHE_TTL_SEARCH_RESULTS = 259200  # 3 days - used in search_channels_multi_term_cached()

# Filtering Thresholds
MIN_MATCH_SCORE = 5         # 5 points on keyword match required. 5 points = 1 channel name match. 10 points = 1 video match.
MAX_CHANNELS_TO_ANALYZE = 50  # Cap channels for deep analysis to control quota
# Default values are defined in the STREAMLIT UI SETUP section.

# Similarity Scoring Weights (total: 100 points)
# These weights are hardcoded in similarity_engine.py on calculate_similarity_score(). Refactoring to make them configurable would require significant changes for minimal practical benefit.
# Changing these requires modifying multiple lines in that function.
SIMILARITY_WEIGHTS = {
    'tag_overlap': 30,        # Tag Jaccard similarity (most reliable signal)
    'keyword_match': 30,      # Topic presence in titles
    'subscriber_tier': 15,    # Subscriber count ratio
    'engagement_rate': 17,    # (Likes + comments) / views
    'upload_frequency': 8    # Videos per month comparison
}
# Total must equal 100

# For implementation details, see similarity_engine.py lines 137-260

# ============================================================================
# --- Securely Load API Keys ---
# ============================================================================


load_dotenv()  # Load .env for local dev

def _get_secret(name: str) -> str | None:
    """Try env var first, then Streamlit secrets (for Streamlit Cloud)."""
    # 1) Local env / .env
    value = os.getenv(name)
    if value:
        return value

    # 2) Streamlit Cloud secrets
    try:
        return st.secrets[name]  # raises KeyError if missing
    except Exception:
        return None

YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY")
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

# ============================================================================
# Helper query functions
# ============================================================================
# NOTE: validate_and_truncate_query, extract_identifier_from_url, resolve_channel_id,
# and strip_outer_quotes are now imported from core.query_utils


def render_term_counter(current_query: str) -> None:
    """
    Render a visual term counter showing X/2 terms used.
    
    Args:
        current_query: Current search query string
    """
    if not current_query:
        return
    
    terms = [t.strip() for t in current_query.split(',') if t.strip()]
    term_count = len(terms)
    
    # Color coding: green if OK, red if exceeding
    if term_count <= MAX_SEARCH_TERMS:
        color = "#10b981"  # Green
        icon = "✓"
        status = "OK"
    else:
        color = "#ef4444"  # Red
        icon = "⚠"
        status = "EXCEEDS LIMIT"
    
    st.markdown(
        f"""
        <div style='padding: 6px 12px; background-color: rgba(0,0,0,0.05); 
                    border-radius: 4px; border-left: 3px solid {color}; 
                    display: inline-block; margin-bottom: 8px;'>
            <span style='color: {color}; font-weight: 600;'>{icon} {term_count}/{MAX_SEARCH_TERMS} terms</span>
            <span style='color: #666; font-size: 0.85em; margin-left: 8px;'>({status})</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# CACHING LAYER - with accurate tracking
# ============================================================================

@st.cache_data(ttl=604800) # 7 days
def get_channel_stats_cached(channel_ids_tuple):
    # Normalize order to maximize cache hits and avoid duplicate fetches
    channel_ids = tuple(sorted(set(channel_ids_tuple)))
    if not channel_ids:
        return []
    youtube = get_youtube()
    return get_channel_stats(youtube, list(channel_ids))

def get_video_details_cached(channel_ids_tuple, max_videos= MAX_VIDEOS_PER_CHANNEL):
    """
    Wrapper using smart per-channel caching (see smart_cache.py).
    Tracking happens inside smart_cache.py.
    """
    from smart_cache import get_video_details_smart
    
    youtube = get_youtube()

    # Normalize order to improve cache hits downstream and deduplicate
    channel_ids_norm = tuple(sorted(set(channel_ids_tuple)))
    if not channel_ids_norm:
        return []
    
    # Get channel stats to get uploads playlist IDs
    stats = get_channel_stats_cached(channel_ids_norm)
    channel_data_full = []
    
    for stat in stats:
        channel_data_full.append({
            'channel_id': stat['channel_id'],
            'uploads_playlist_id': stat['uploads_playlist_id']
        })
    
    # Smart caching handles tracking internally
    return get_video_details_smart(youtube, channel_data_full, max_videos)


@st.cache_data(ttl=259200) # 3 days
def search_channels_multi_term_cached(query, region_code, max_videos= MAX_VIDEOS_PER_TERM, cache_bust: str = "v2-no-early-cap"):
    """Cached wrapper for multi-term search.

    cache_bust: change this string to invalidate prior cached results
    when search logic changes (e.g., removed early cap).
    """
    # Normalize query terms for a stable cache key (order-agnostic)
    terms = [t.strip() for t in (query or "").split(',') if t.strip()]
    if not terms:
        return []
    canonical_terms = sorted(set(terms), key=str.lower)
    canonical_query = ", ".join(canonical_terms)

    # cache_bust is unused in logic; only to affect cache key
    _ = cache_bust
    return search_channels_multi_term(canonical_query, region_code, max_videos)


# ============================================================================
# --- API Functions ---
# ============================================================================

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

# ============================================================================
#---- CORE LOGIC - YOUTUBE API WRAPPERS + AI GENERATION (Gemini)
# ============================================================================
# NOTE: Core YouTube API functions are now in core/youtube_api.py
# These wrappers add Streamlit-specific functionality (caching, warnings, spinners)


def _get_api_tracker():
    """Get API call tracker callback if debug mode is enabled."""
    if st.session_state.get('debug_mode', False):
        return debug_tracker.track_api_call
    return None


@st.cache_data(show_spinner=False, ttl=259200)  # 3 days
def search_channels_hybrid(query: str, region_code: str, max_videos: int = MAX_VIDEOS_PER_TERM, max_channels: int = MAX_SEARCH_RESULTS):
    """
    Hybrid search: Find channels by VIDEO content (primary) + channel names (secondary).

    Returns: List[dict] with keys 'channel_id', 'channel_title', 'match_score'

    Notes:
        - Delegates to core.youtube_api.search_channels_hybrid
        - Handles warnings via st.warning
        - Results cached for ~3 days via @st.cache_data(ttl=259200)
    """
    youtube = get_youtube()
    result = _search_channels_hybrid_core(
        youtube_service=youtube,
        query=query,
        region_code=region_code,
        max_videos=max_videos,
        max_channels=max_channels,
        on_api_call=_get_api_tracker(),
    )

    # Display any warnings that occurred
    for warning in result.warnings:
        st.warning(warning)

    return result.channels


def search_channels_multi_term(
    query: str,
    region_code: str,
    max_videos_per_term: int = MAX_VIDEOS_PER_TERM,
    max_channels: int | None = None,
):
    """
    Handle comma-separated queries as OR logic.
    Example: "manga, anime" -> search both terms, merge results

    Returns: List[dict] with deduplicated channels sorted by relevance

    Notes:
        - Adds Streamlit UI feedback (spinners, info messages)
        - Records pre_cap_channel_count in session state
    """
    # Split by comma and clean
    terms = [t.strip() for t in query.split(',') if t.strip()]

    if len(terms) == 0:
        return []

    # Enforce 2-term maximum with warning
    if len(terms) > 2:
        st.warning(
            f"Search limited to 2 terms (received {len(terms)}). "
            f"Using: **{', '.join(terms[:2])}**"
        )
        terms = terms[:2]

    if len(terms) == 1:
        # Single term: use hybrid search directly
        results_single = search_channels_hybrid(terms[0], region_code, max_videos_per_term)
        try:
            st.session_state['pre_cap_channel_count'] = len(results_single)
        except Exception:
            pass
        if max_channels is not None:
            results_single = results_single[:max_channels]
        return results_single

    # Multiple terms: search each, then merge
    st.info(f"Searching {len(terms)} topics: {', '.join(terms)}")

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

    # Record the true pre-cap count for logging
    try:
        st.session_state['pre_cap_channel_count'] = len(merged)
    except Exception:
        pass

    if max_channels is not None:
        merged = merged[:max_channels]

    return merged


def get_channel_stats(youtube_service, channel_ids):
    """
    Fetch detailed statistics for a list of channel IDs.

    Delegates to core.youtube_api.get_channel_stats with debug tracking.
    """
    result = _get_channel_stats_core(
        youtube_service=youtube_service,
        channel_ids=channel_ids,
        on_api_call=_get_api_tracker(),
    )
    return result.stats


def get_video_details(youtube_service, channel_data, max_videos_per_channel):
    """
    Fetch video details for multiple channels.

    Delegates to core.youtube_api.get_video_details with debug tracking.
    Displays warnings for failed channels via st.warning.
    """
    result = _get_video_details_core(
        youtube_service=youtube_service,
        channel_data=channel_data,
        max_videos_per_channel=max_videos_per_channel,
        on_api_call=_get_api_tracker(),
    )

    # Display any warnings that occurred
    for warning in result.warnings:
        st.warning(warning)

    return result.videos

# NOTE: calculate_keyword_relevance is now imported from core.relevance
# NOTE: Gemini AI functions are now in core/gemini_api.py
# These wrappers add Streamlit-specific functionality (debug tracking, session state)


def generate_ai_relevance_score(model, channel_data: dict, query: str) -> float:
    """
    Wrapper for core.gemini_api.generate_ai_relevance_score.

    Uses a Gemini model to score channel relevance based on video titles.
    Returns a relevance score between 0.0 and 1.0, or 0.0 on failure.
    """
    return _generate_ai_relevance_score_core(model, channel_data, query)


def generate_summary(df_results, query):
    """
    Generate a summary of the top YouTube channels using Gemini.

    Wrapper that adds Streamlit debug tracking and session state access.
    """
    model = get_gemini_model()

    # Get seed channel name if in seed-based mode
    seed_channel_name = None
    if 'similarity_score' in df_results.columns:
        seed_channel_name = st.session_state.get('seed_profile', {}).get('channel_name', 'the seed channel')

    result = _generate_summary_core(
        model=model,
        df_results=df_results,
        query=query,
        seed_channel_name=seed_channel_name,
        on_api_call=_get_api_tracker(),
    )

    if result.error:
        return result.error
    return result.text


def generate_outreach_drafts(
    top_channels_df: pd.DataFrame,
    original_query: str,
    limit: int = 3,
    temperature: float = 0.7,
    retries: int = 2,
    language: str = "en",
) -> list[dict]:
    """
    Generate short, friendly outreach email drafts for the top N channels using Gemini.

    Wrapper that adds Streamlit debug tracking.

    Returns:
        List[dict]: [{'channel_title': str, 'draft_text': str}, ...]
    """
    model = get_gemini_model(temperature=temperature)

    drafts = _generate_outreach_drafts_core(
        model=model,
        top_channels_df=top_channels_df,
        original_query=original_query,
        limit=limit,
        retries=retries,
        language=language,
        on_api_call=_get_api_tracker(),
    )

    # Convert OutreachDraft objects to dicts for backward compatibility
    return [{'channel_title': d.channel_title, 'draft_text': d.draft_text} for d in drafts]

# ============================================================================
#---MAIN PIPELINE: run_search()
# ============================================================================

def run_search(
    youtube,
    final_query: str,
    region_input: str,
    min_subs_input: int,
    months_ago_input: int,
    country_filter_input: str,
):

    """
Execute the end-to-end search pipeline: validate → search → stats → filter → analyze → rank → render.

Pipeline:

0.5 Validate and truncate query to 2 terms; if truncated, show a Streamlit warning and overwrite final_query with the validated value.
1 Search for channels (region-aware) via cached helper; track search calls in debug mode.
2 Fetch channel statistics in batches, then merge with initial search results.
3 Apply filters: subscribers ≥ min_subs_input; optional strict country match (uppercased ISO code).
4 Quality selection: require match_score ≥ 10, sort by match_score then subscribers, and cap to the top 50 channels for deep analysis.
5 Deep analysis: fetch up to 10 recent videos per channel;
6 Compute keyword relevance and engagement; optionally filter by recency (last months_ago_input months); sort by relevance_score then subscribers; annotate analysis_depth.

.Similarity (optional): if a seed_profile is present, exclude the seed, reuse fetched videos to derive tags/keywords, rank with similarity_engine (optionally using Gemini), compute similarity_score and match_reasons, and re-sort.
.AI Summary (optional): when GEMINI_API_KEY is configured, generate and store a textual summary of the top channels.
.Display and state: format metrics, choose display columns, and update session state; track timings in debug mode.

Args:
youtube: Authenticated YouTube Data API client.
final_query: Query string (comma-separated); auto-validated and truncated to 2 terms.
region_input: ISO 3166-1 alpha-2 region code (e.g., "US") for search.
min_subs_input: Minimum subscriber threshold.
months_ago_input: Only include videos from the last N months (0 = no recency filter).
country_filter_input: Optional strict country filter by ISO code (e.g., "US"); falsy disables.

Returns:
None. Renders UI and updates session state.

Session state:
- 'display_df', 'column_explanations', 'final_query'
- 'top_channels_for_outreach'
- 'top_channels_full' (when similarity ranking runs)
- 'ai_summary' or 'ai_summary_error'
- 'debug_data' timings and metrics (when debug mode is enabled)

Notes:
- Catches exceptions and surfaces them via Streamlit; does not raise.
- Tracks API/quota metrics when debug mode is enabled.
"""
    
    # Initialize search log
    search_log = []

    # Reset debug tracking for new search
    search_start_time = None
    if st.session_state.get('debug_mode', False):
        debug_tracker.reset_debug_tracking()
        search_start_time = time.time()
    
    try:
        # === STEP 0.5: Validate Query ===
        # Apply 2-term limit with auto-truncation
        final_query_validated, was_truncated = validate_and_truncate_query(final_query)

        # Show user-friendly warning if query was truncated
        if was_truncated:
            original_terms = [t.strip() for t in final_query.split(',') if t.strip()]
            kept_terms = original_terms[:2]
            removed_terms = original_terms[2:]
            
            st.warning(
                f"⚠ **Query automatically adjusted for optimal results**\n\n"
                f"You entered **{len(original_terms)} terms**, but searches are limited to **2 terms** "
                f"to optimize API usage and result quality.\n\n"
                f"✅ **Searching with**: {', '.join(kept_terms)}\n\n"
                f"❌ **Removed**: {', '.join(removed_terms)}\n\n"
                f"💡 *Tip: Use the most specific 2 terms for best results*"
            )
            
            # Update final_query for the rest of the function
            final_query = final_query_validated

        # === STEP 1: Search for channels ===
        with st.spinner("Step 1/4: Searching for channels..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None

            # Pass a cache_bust tag to ensure we don't reuse
            # older cached results from the pre-cap version
            initial_channels = search_channels_multi_term_cached(
                final_query,
                region_input,
                max_videos= MAX_VIDEOS_PER_TERM,
                cache_bust="mt-search-v2"
            )
            
            # Always set pre-cap count based on current Step 1 output
            try:
                st.session_state['pre_cap_channel_count'] = len(initial_channels)
            except Exception:
                pass

            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['search'] = time.time() - step_start
        

        if not initial_channels:
            st.error("Search did not return any channels.")
            return

        df_initial = pd.DataFrame(initial_channels)
        with st.expander("See raw channels found"):
            st.dataframe(df_initial)

        pre_cap_count = st.session_state.get('pre_cap_channel_count', len(initial_channels))
        if len(initial_channels) < pre_cap_count:
            # Early cap applied in the search helper
            log_msg = f"🔍 Ranked {pre_cap_count} channels before cap; showing {len(initial_channels)} after cap"
        else:
            # No early cap (cap happens later in Step 4)
            log_msg = f"🔍 Ranked {pre_cap_count} channels before cap (no early cap applied)"
        search_log.append(log_msg)

        # === STEP 2: Fetch channel statistics ===
        with st.spinner("Step 2/4: Fetching channel statistics..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            channel_ids_tuple = tuple(df_initial['channel_id'].tolist())

            channel_statistics = get_channel_stats_cached(channel_ids_tuple)
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['channel_stats'] = time.time() - step_start

        if not channel_statistics:
            st.warning("Could not retrieve detailed stats for the found channels.")
            return

        df_stats = pd.DataFrame(channel_statistics)
        enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')

        # === STEP 3: User seleted filters ===
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
            
            log_msg = f"✅ {len(filtered_channels)} channels passed filters (min {min_subs_input:,} subs)"
            search_log.append(log_msg)
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['filtering'] = time.time() - step_start

       # === STEP 4: Backend filtering ===
        with st.spinner("Step 4/5: Preparing channels for analysis..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            # Filter channels by minimum relevance threshold
            quality_channels = filtered_channels[
                filtered_channels['match_score'] >= MIN_MATCH_SCORE
            ].copy()
            
            # Fallback: If no channels meet threshold, show all results with warning
            if quality_channels.empty:
                st.info(
                    f"⚠️ No channels found with match_score ≥ {MIN_MATCH_SCORE}. "
                    f"Showing all {len(filtered_channels)} channels found."
                )
                quality_channels = filtered_channels.copy()
            else:
                # Show how many channels were filtered out
                filtered_out_count = len(filtered_channels) - len(quality_channels)
                if filtered_out_count > 0:
                    log_msg = (
                        f"✨ Quality filter applied: {filtered_out_count} low-relevance channels "
                        f"(match_score < {MIN_MATCH_SCORE}) excluded from deep analysis"
                    )
                    search_log.append(log_msg)

            # Sort by relevance FIRST (primary), then subscribers (secondary)
            # This ensures the most relevant channels are prioritized for deep analysis
            filtered_sorted = quality_channels.sort_values(
                by=['match_score', 'subscribers'],
                ascending=[False, False]  # Both descending
            )
            
            channels_to_analyze = filtered_sorted.head(MAX_CHANNELS_TO_ANALYZE).copy()
            
            channels_analyzed_count = len(channels_to_analyze)
            
            # Enhanced logging with relevance statistics
            if channels_analyzed_count < len(quality_channels):
                avg_match = channels_to_analyze['match_score'].mean()
                min_match = channels_to_analyze['match_score'].min()
                max_match = channels_to_analyze['match_score'].max()
                
                log_msg = (
                    f"📊 Analyzing top {channels_analyzed_count} channels "
                    f"(match scores: {min_match:.0f}-{max_match:.0f}, avg: {avg_match:.0f}) "
                    f"from {len(quality_channels)} quality matches"
                )
                search_log.append(log_msg)
            else:
                log_msg = (f"📊 Analyzing all {channels_analyzed_count} channels")
                search_log.append(log_msg)
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['select_channels'] = time.time() - step_start
                

        # === STEP 5: Deep analysis (10 videos per channel) ===
        with st.spinner(f"Deep analysis - fetching 10 videos from {channels_analyzed_count} channels..."):
            step_start = time.time() if st.session_state.get('debug_mode', False) else None
            
            channel_ids_tuple = tuple(channels_to_analyze['channel_id'].tolist())

            # Fetch 10 videos for comprehensive relevance analysis
            video_data = get_video_details_cached(
                channel_ids_tuple, 
                max_videos=10  # 🔥 DEEP ANALYSIS: 10 videos per channel
            )
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['video_details'] = time.time() - step_start

        if not video_data:
            st.warning("Could not retrieve any video details from the channels.")
            st.dataframe(channels_to_analyze.sort_values(by="subscribers", ascending=False))
            return
        
        log_msg = (f"🎬 Retrieved {len(video_data)} videos for deep analysis from {channels_analyzed_count} channels")
        search_log.append(log_msg)

        # === STEP 6: Calculate relevance and engagement ===
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
            

            # Calculate average engagement per channel
            avg_engagement = df_full.groupby('channel_id')['engagement_rate'].mean().reset_index()
            
            # Merge everything together
            final_channels = pd.merge(channels_to_analyze, avg_engagement, on='channel_id')
            final_channels = pd.merge(final_channels, relevance_scores, on='channel_id', how='left')
            
            # Fill NaN relevance scores with 0
            final_channels['relevance_score'] = final_channels['relevance_score'].fillna(0)
            
            # Sort by relevance (primary) then subscribers (secondary)
            # No hard filtering - let users see all results ranked by relevance
            final_channels_sorted = final_channels.sort_values(
                by=['relevance_score', 'engagement_rate'], 
                ascending=False
            ).copy()

            # Add analysis badge
            final_channels_sorted['analysis_depth'] = '✓ 10 videos analyzed'

            top_channels = final_channels_sorted.copy()

            # Success message with relevance stats
            total_count = len(top_channels)
            high_relevance_count = len(top_channels[top_channels['relevance_score'] >= 0.15])

            log_msg = f"✅ Found {total_count} channels ({high_relevance_count} with high topic focus ≥15%)"
            search_log.append(log_msg)
            
            if st.session_state.get('debug_mode', False) and step_start:
                st.session_state.debug_data['timings']['relevance_filtering'] = time.time() - step_start

        # === STEP 7: AI-Enhanced Relevance Scoring  ===
        # This step blends the keyword score with a semantic AI score.
        if GEMINI_API_KEY:
            with st.spinner("✨ Enhancing relevance with AI..."):
                step_start = time.time() if st.session_state.get('debug_mode', False) else None
                
                model = get_gemini_model()
                ai_scores = []

                # Group video titles by channel for AI analysis
                video_titles_by_channel = df_videos.groupby('channel_id')['video_title'].apply(list).to_dict()

                # Iterate through the channels we've already analyzed
                for _, row in top_channels.iterrows():
                    channel_id = row['channel_id']
                    
                    # Prepare data for the AI function
                    channel_data_for_ai = {
                        'channel_title': row['channel_title'],
                        'video_titles': video_titles_by_channel.get(channel_id, [])
                    }
                    
                    # Get the AI score
                    ai_score = generate_ai_relevance_score(model, channel_data_for_ai, final_query)
                    ai_scores.append({'channel_id': channel_id, 'ai_relevance_score': ai_score})

                if ai_scores:
                    df_ai_scores = pd.DataFrame(ai_scores)
                    # Merge AI scores back into the main dataframe
                    top_channels = pd.merge(top_channels, df_ai_scores, on='channel_id', how='left')
                    top_channels['ai_relevance_score'] = top_channels['ai_relevance_score'].fillna(0)

                    # Blend the scores: 80% algorithmic, 20% AI
                    top_channels['relevance_score'] = (0.8 * top_channels['relevance_score'] + 0.2 * top_channels['ai_relevance_score'])
                    
                    # Re-sort with the new blended score
                    top_channels = top_channels.sort_values(by=['relevance_score', 'engagement_rate'], ascending=False).copy()

                    log_msg = f"🧠 Blended keyword scores with AI relevance for higher accuracy"
                    search_log.append(log_msg)

        # === STEP 8: Similarity Raking Orchestation (if using seed)===            
        if 'seed_profile' in st.session_state:
            with st.spinner("🧠 Calculating similarity scores..."):
                seed_channel_id = st.session_state['seed_profile']['channel_id']
                before_exclusion = len(top_channels)
                
                top_channels = top_channels[top_channels['channel_id'] != seed_channel_id]
                
                excluded_count = before_exclusion - len(top_channels)
                if excluded_count > 0:
                    log_msg = (f"🚫 Excluded seed channel from results ({excluded_count} removed)") 
                    search_log.append(log_msg)          
                
                if top_channels.empty:
                    st.error("No channels in the similar size range. Try disabling the size filter.")
                    return
                else:
                       
                        # ✅ REUSE video data we already fetched!
                        # df_videos already has 10 videos from all channels (line 691)
                        
                        # Extract tags from existing df_videos
                        def flatten_tags(tag_series):
                            all_tags = []
                            for tags in tag_series:
                                if isinstance(tags, list):
                                    all_tags.extend(tags)
                                elif isinstance(tags, str):
                                    all_tags.append(tags)
                            unique_tags = list(set(t.lower().strip() for t in all_tags if t))
                            return unique_tags
                        
                        # Filter df_videos to only top_channels
                        top_channel_ids = set(top_channels['channel_id'])
                        df_videos_filtered = df_videos[df_videos['channel_id'].isin(top_channel_ids)]
                        
                        channel_tags = (
                            df_videos_filtered.groupby('channel_id')['video_tags']
                            .apply(flatten_tags)
                            .reset_index()
                            .rename(columns={'video_tags': 'tags'})
                        )
                        
                        channel_keywords = (
                            df_videos_filtered.groupby('channel_id')['video_title']
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
                        
                        log_msg = ("🎯 Ranking channels by similarity...")
                        search_log.append(log_msg)
                        
                        candidates = top_channels.to_dict('records')
                        for candidate in candidates:
                            if 'channel_title' in candidate and 'channel_name' not in candidate:
                                candidate['channel_name'] = candidate['channel_title']
                        
                        ranked = similarity_engine.rank_channels_by_similarity(
                            candidates,
                            st.session_state['seed_profile'],
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
                                             

        # === STEP 9: AI SUMMARY for Results (Generate but don't display yet - will show after results table) ===
        if GEMINI_API_KEY:
            with st.spinner("✨ Generating AI Summary..."):
                step_start = time.time() if st.session_state.get('debug_mode', False) else None
                
                try:
                    summary_df = top_channels.copy()
                    summary_df['relevance_score'] = summary_df['relevance_score'].fillna(0).map('{:.0%}'.format)
                    summary_df['engagement_rate'] = summary_df['engagement_rate'].fillna(0).map('{:.2%}'.format)
                    summary_text = generate_summary(summary_df, final_query)
                    # Store in session state for display after results table
                    st.session_state['ai_summary'] = summary_text
                except Exception as e:
                    st.session_state['ai_summary'] = None
                    st.session_state['ai_summary_error'] = str(e)
                
                if st.session_state.get('debug_mode', False) and step_start:
                    st.session_state.debug_data['timings']['ai_generation'] = time.time() - step_start
        else:
            st.session_state['ai_summary'] = None
            st.session_state['ai_summary_error'] = "GEMINI_API_KEY not configured"

        # === Step 10: Display results ranked by relevance (keywords mode) and similarity (for seed mode)===
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

        # Create YouTube channel URL from channel_id
        top_channels['channel_url'] = top_channels['channel_id'].apply(
            lambda x: f"https://www.youtube.com/channel/{x}" if x else ""
        )

        # Choose display columns
        if 'similarity_score' in top_channels.columns:
            # Seed mode: show similarity, relevance, and avg views (remove match_reasons to save space)
            display_columns = ['channel_title', 'channel_url', 'similarity_score', 'relevance_score',
                            'avg_views_per_video', 'subscribers', 'country', 'engagement_rate']
        else:
            # Keyword mode: show relevance and avg views
            display_columns = ['channel_title', 'channel_url', 'relevance_score', 'subscribers',
                            'avg_views_per_video', 'country', 'engagement_rate']

        # STORE IN SESSION STATE FOR DISPLAY
        st.session_state['display_df'] = top_channels[display_columns].copy()


        # Store column explanations for later display
        if 'similarity_score' in display_columns:
            st.session_state['column_explanations'] = {
                "channel_title": "Name of the YouTube channel",
                "similarity_score": "Overall similarity to your seed channel (0-100). Based on shared topics, tags, audience size, engagement patterns, and upload frequency. Higher = better match.",
                "relevance_score": "Percentage of recent videos containing your search keywords in titles/descriptions. Indicates topic focus.",
                "avg_views_per_video": "Average views per video across recent uploads. Shows reach and content virality.",
                "subscribers": "Total subscriber count. Indicates channel size and reach.",
                "country": "Country where the channel is registered (from YouTube's data).",
                "engagement_rate": "(Likes + Comments) / Views, averaged across recent videos. Shows audience interactivity."
            }
        else:
            st.session_state['column_explanations'] = {
                "channel_title": "Name of the YouTube channel",
                "relevance_score": "Percentage of recent videos containing your search keywords in titles/descriptions. Indicates topic focus.",
                "subscribers": "Total subscriber count. Indicates channel size and reach.",
                "avg_views_per_video": "Average views per video across recent uploads. Shows reach and content virality.",
                "country": "Country where the channel is registered (from YouTube's data).",
                "engagement_rate": "(Likes + Comments) / Views, averaged across recent videos. Shows audience interactivity."
            }
        
       # === TRACK FINAL METRICS ===
        if st.session_state.get('debug_mode', False):
            if search_start_time:
                st.session_state.debug_data['timings']['total'] = time.time() - search_start_time
            
            if 'top_channels_full' in st.session_state:
                debug_tracker.track_similarity_scores(
                    st.session_state['top_channels_full'].to_dict('records')
                )

        # Display search log in collapsible section
        if search_log:
            with st.expander("📋 Search Process Log", expanded=False):
                for msg in search_log:
                    st.markdown(f"- {msg}")

    except Exception as e:  # ← ERROR HANDLER
        st.error(f"❌ Error in run_search: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())
        
        # Show what step failed
        if st.session_state.get('debug_mode', False):
            st.write("### Debug Info at Failure:")
            st.write(f"- Session state keys: {list(st.session_state.keys())}")
            st.write(f"- Debug data: {st.session_state.get('debug_data', {})}")   
        

# ============================================================================
# --- Streamlit User Interface ---
# ============================================================================

st.set_page_config(
    page_title="CCSeeker - YouTube Creator Search",
    page_icon="docs/appicons/app-icon-192x192.png",  # Relative path to your favicon
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
    st.image("docs/appicons/app-icon-192x192.png", width=100)  # Bigger logo

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
    st.session_state.search_method = None

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
search_method = st.session_state.get('search_method')

# Initialize submitted to False to prevent NameError on first run
submitted = False
if search_method:
    with st.form("search_form"):
        # Inputs change depending on the selected method
        if search_method == "Keywords":
            query_input = st.text_input(
                "Search Keywords",
                "manga, anime",
                help=(
                    "⚠ Enter up to 2 topics separated by commas. "
                    "Examples: 'manga, anime' or 'cooking, recipes'. "
                    "If you enter more than 2 terms, only the first 2 will be used. "
                    "Matches channel content (videos, descriptions) not just channel names."
                ),
                key="keywords_input"
            )
            
            # Visual term counter
            if query_input:
                render_term_counter(query_input)
            with st.expander("How does Keyword Search work?"):
                st.markdown("""
                1. **Hybrid Search**: We find channels by matching your keywords in **video content** (primary signal) and **channel names** (secondary). Multi-term queries merge results, boosting channels that match all topics.
                2. **Filter & Select**: We apply your subscriber/country filters, remove weak matches, and cap the list at the top 50 strongest candidates.
                3. **Deep Analysis**: We fetch 10 recent videos per channel to calculate **engagement rates** and a **relevance score** (blending 80% keyword matching with 20% AI semantic analysis).
                4. **AI Summary**: The top 5 results are analyzed by Gemini to highlight the best collaboration opportunities.
                """)
            seed_url_input = ""  # keep defined
        else:
            st.info("💡 Enter the full URL of a YouTube channel to find similar creators or just the channel's name.")
            seed_url_input = st.text_input("YouTube Channel URL or name (the 'seed')", "https://www.youtube.com/@YourFavoriteChannel")
            query_input = ""  # keep defined

            # Set defaults for advanced options (will be shown in Search Options expander later)
            target_language_code = "auto"
            ignore_words_input = ""
            
            with st.expander("How does Channel-as-Seed work?"):
                st.markdown("""
                    This method discovers new channels based on a single example channel you provide.
                    1.  **Analyze:** It fetches the latest videos from the URL you enter.
                    2.  **Learn:** It extracts the most common topics and keywords from that channel's video titles and tags.
                    3.  **Discover:** It then uses those learned keywords to launch a new, highly specific search to find other channels with similar content.
                    4.  **Rank & Analyze:** Finally, it ranks the results by similarity to your seed channel based on a blend between an algorithmic score and an AI score. Giving 80% weight to the algorithmic score and 20% to the AI's score. The algorithmic score is based on similarity on topics, audience size, engagement patterns, and upload frequency.    
                """)
            
            # Initialize defaults for seed-specific options (will be shown in Search Options later)
            target_language_code = "auto"
            ignore_words_input = ""

        # For Keywords mode: show all filters in the form
        if search_method == "Keywords":
            country_options = COUNTRY_OPTIONS_BASE.copy()
            country_options.sort()
            country_options.insert(0, "Global")
            selected_country = st.selectbox(
            "Relevant in:",
            country_options,
            index=0,  # Default to Global (no regional bias)
            help="YouTube will show channels popular in this country first, but results aren't limited to it.",
        )
            region_input = "" if selected_country == "Global" else selected_country.split("(")[-1][:2]

            st.header("2. Filtering Criteria")
            c1, c2, c3 = st.columns(3)

            with c1:
                min_subs_input = st.number_input(
                    "Minimum Subscribers",
                    min_value=0, value=1000, step=1000, format="%d",
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
        else:
            # For Channel-as-Seed: initialize default values (filters will be shown after analysis)
            region_input = ""
            min_subs_input = 10000
            country_filter_input = ""
            months_ago_input = 18

        # Button label changes based on search method
        button_label = "Analyse Seed" if search_method == "Channel-as-Seed" else "Find Creators"
        submitted = st.form_submit_button(button_label)

# --- Main Execution Logic ---
if submitted:
    # New search: clear previous outreach/session caches
    st.session_state.pop('top_channels_for_outreach', None)
    st.session_state.pop('final_query', None)
    st.session_state.pop('display_df', None)
    # Reset feedback state for new search
    st.session_state['feedback_submitted'] = False
    st.session_state['show_reason_selector'] = False
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
                        max_videos=50,
                        user_banned_words=penalties,
                        gemini_api_key=GEMINI_API_KEY
                    )
                
                if seed_profile:
                    # Store profile in session state for later use
                    st.session_state['seed_profile'] = seed_profile
                    st.session_state['seed_channel_id'] = seed_channel_id
                    
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
            )

# ============================================================================
# === SEED PROFILE REVIEW (NEW) ===
# ============================================================================

if st.session_state.get('seed_profile'):
    profile = st.session_state['seed_profile']
    
    st.header("📊 Seed Channel Profile")
    
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
    
    # Show extracted topics with clear distinction
    st.subheader("📌 Topic Extraction")

    col_topics, col_tags = st.columns(2)

    with col_topics:
        st.markdown("**💬 Key Phrases** *(from video titles)*")
        if profile['primary_keywords']:
            phrases_display = ", ".join(f'"{phrase}"' for phrase in profile['primary_keywords'][:5])
            st.markdown(f"<div style='padding: 10px; background-color: rgba(59, 130, 246, 0.1); border-radius: 6px; border-left: 3px solid #3b82f6;'>{phrases_display}</div>", unsafe_allow_html=True)
        else:
            st.info("No multi-word phrases extracted")

        st.markdown("<div style='margin-top: 12px'></div>", unsafe_allow_html=True)
        
        # Show single keywords too
        st.markdown("**💬 Single Keywords** *(from video titles)*")
        if profile['secondary_keywords']:
            keywords_display ="Single keywords: " + ", ".join(profile['secondary_keywords'][:8])
            st.markdown(f"<div style='padding: 10px; background-color: rgba(174, 127, 57, 0.1); border-radius: 6px; border-left: 3px solid #AE7F39;'>{keywords_display}</div>", unsafe_allow_html=True)

    with col_tags:
        st.markdown("**🏷️ Creator Tags** *(used by the channel)*")
        if profile['common_tags']:
            tags_display = ", ".join(f"#{tag}" for tag in profile['common_tags'][:12])
            st.markdown(f"<div style='padding: 10px; background-color: rgba(16, 185, 129, 0.1); border-radius: 6px; border-left: 3px solid #10b981;'>{tags_display}</div>", unsafe_allow_html=True)
        else:
            st.info("No tags found in recent videos")
    
    st.markdown("<div style='margin-top: 12px'></div>", unsafe_allow_html=True)
    
    # Show AI summary if available
    if profile.get('description_summary'):
        with st.expander("🤖 AI Channel Summary"):
            st.write(profile['description_summary'])
    
    # Filtering Criteria (for Channel-as-Seed mode)
    st.header("2. Filtering Criteria")
    
    # Initialize country_options
    country_options = COUNTRY_OPTIONS_BASE.copy()
    country_options.sort()
    country_options.insert(0, "Global")
    
    # Relevant in region
    selected_country = st.selectbox(
        "Relevant in:",
        country_options,
        index=0,  # Default to Global
        help="YouTube will show channels popular in this country first, but results aren't limited to it.",
        key="seed_region_select"
    )
    region_input = "" if selected_country == "Global" else selected_country.split("(")[-1][:2]
    
    # Three-column layout for filters
    c1, c2, c3 = st.columns(3)
    
    with c1:
        min_subs_input = st.number_input(
            "Minimum Subscribers",
            min_value=0, value=10000, step=1000, format="%d",
            help="Set to 0 to ignore.",
            key="seed_min_subs"
        )
    
    with c2:
        country_filter_options = country_options.copy()
        selected_country_filter = st.selectbox(
            "Channel Country (strict filter)",
            country_filter_options,
            index=0,  # Default to "Global"
            help="Only show channels registered in this country. Leave as 'Global' to see all countries.",
            key="seed_country_filter"
        )
        country_filter_input = "" if selected_country_filter == "Global" else selected_country_filter.split("(")[-1][:2]
    
    with c3:
        months_ago_input = st.number_input(
            "Published within last (months)",
            value=18, min_value=0, step=1,
            help="Only show channels with uploads in the last X months. Set to 0 to ignore upload recency.",
            key="seed_months_ago"
        )
    
    # Build search query from profile (needed for the button)
    search_terms = profile['primary_keywords'][:2]  # Top 2 phrases
        
    # Fallback: If channel has fewer than 2 primary keywords, pad with common tags
    if len(search_terms) < 2:
        remaining_slots = 2 - len(search_terms)
        search_terms += profile['common_tags'][:remaining_slots]

    # Quote multi-word terms
    quoted_terms = [
        f'"{term}"' if ' ' in term else term 
        for term in search_terms
    ]
    
    default_query = ", ".join(quoted_terms[:2])

    # Initialize session state for editable query
    if 'editable_seed_query' not in st.session_state:
        st.session_state['editable_seed_query'] = default_query
    
    # Use the editable query (or default if not yet edited)
    built_query = st.session_state.get('editable_seed_query', default_query)

    # Advanced options (moved from search method section)
    with st.expander("⚙️ Advanced Analysis Options"):
        st.markdown("*Fine-tune how the seed channel is analyzed*")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
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
                help="The seed analysis stays in the original language, but final topic keywords can be translated for broader search results."
            )
            _lang_map = {
                "Original (auto)": "auto",
                "English": "en",
                "Español": "es",
                "Português": "pt",
                "Français": "fr",
                "Deutsch": "de",
            }
            target_language_code = _lang_map.get(output_lang_label, "auto")
        
        with col_adv2:
            ignore_words_input = st.text_input(
                "Ignore words (comma-separated)",
                value="",
                help="Words to exclude from topic extraction (e.g., brand names, common filler words). Useful for cleaning up noisy results."
            )
    
    # Query editor section
    col_query, col_reset = st.columns([4, 1])

    with col_query:
        built_query = st.text_area(
            "**Generated search query** (editable)",
            value=st.session_state['editable_seed_query'],
            height=80,
            help=(
                "Modify this query to refine your search. "
                "⚠ Maximum 2 terms allowed - if you add more, only the first 2 will be used. "
                "Separate terms with commas. Multi-word phrases are automatically quoted."
            ),
            key="query_editor"
        )
        st.session_state['editable_seed_query'] = built_query
    
    # Visual term counter below the text area
    if built_query:
        render_term_counter(built_query)

    with col_reset:
        st.write("")  # Spacer for alignment
        st.write("")  # Spacer for alignment
        # Re-calculate default_query for reset button
        search_terms = profile['primary_keywords'][:2]
        if len(search_terms) < 2:
            remaining_slots = 2 - len(search_terms)
            search_terms += profile['common_tags'][:remaining_slots]
        quoted_terms = [f'"{term}"' if ' ' in term else term for term in search_terms]
        default_query = ", ".join(quoted_terms[:2])
        
        if st.button("🔄 Reset", help="Reset to AI-generated default", key="reset_query"):
            st.session_state['editable_seed_query'] = default_query
            st.rerun()
    
    # Main search button - placed after all search options
    if st.button("🚀 Find Similar Channels", type="primary", key="btn_find_similar", use_container_width=True):
        st.session_state['built_query'] = built_query
        st.session_state['run_similarity_search'] = True
        st.rerun()

    # ============================================================================
    # === HANDLE SIMILARITY SEARCH ===
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
            )


# Keep the results table visible across reruns
if 'display_df' in st.session_state:
    st.subheader("📊 Search Results")
    
    st.dataframe(
        st.session_state['display_df'],
        column_config={
            "channel_url": st.column_config.LinkColumn(
                label="Link",
                display_text="Open",
                help="Click to open this YouTube channel in a new tab"
            ),
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
        },
        use_container_width=True
    )
    
    # Add explanations below table
    with st.expander("📖 Column Definitions", expanded=False):
        if 'column_explanations' in st.session_state:
            explanations = st.session_state['column_explanations']
            
            for col_key, col_name in [
                ("similarity_score", "**Similarity Score**"),
                ("relevance_score", "**Relevance Score**"),
                ("avg_views_per_video", "**Avg Views per Video**"),
                ("subscribers", "**Subscribers**"),
                ("country", "**Country**"),
                ("engagement_rate", "**Engagement Rate**")
            ]:
                if col_key in explanations:
                    st.markdown(f"{col_name}: {explanations[col_key]}")
    
    # === AI SUMMARY (Display after results table) ===
    if 'ai_summary' in st.session_state:
        if st.session_state['ai_summary']:
            st.subheader("🤖 AI Generated Summary")
            st.markdown(st.session_state['ai_summary'])
        elif 'ai_summary_error' in st.session_state:
            st.warning(f"⚠️ {st.session_state['ai_summary_error']}")

    # === USER FEEDBACK SECTION ===
    st.divider()
    st.markdown("#### 📝 How were these results?")

    # Initialize feedback state if needed
    if 'feedback_submitted' not in st.session_state:
        st.session_state['feedback_submitted'] = False
    if 'show_reason_selector' not in st.session_state:
        st.session_state['show_reason_selector'] = False

    if st.session_state['feedback_submitted']:
        st.success("Thanks for your feedback! It helps us improve.")
    else:
        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 4])

        with col_fb1:
            if st.button("👍 Good", use_container_width=True):
                # Gather search context
                display_df = st.session_state.get('display_df', pd.DataFrame())
                search_mode = "seed" if 'seed_profile' in st.session_state else "keyword"
                query = st.session_state.get('final_query', '')

                # Build top results list
                top_results = []
                if not display_df.empty:
                    for _, row in display_df.head(5).iterrows():
                        score = row.get('similarity_score') or row.get('relevance_score', 0)
                        top_results.append({
                            "channel_name": row.get('channel_title', ''),
                            "score": score
                        })

                # Get seed info if available
                seed_id = None
                seed_name = None
                if search_mode == "seed" and 'seed_profile' in st.session_state:
                    seed_profile = st.session_state['seed_profile']
                    seed_id = seed_profile.get('channel_id')
                    seed_name = seed_profile.get('channel_name')

                # Save positive feedback
                feedback_tracker.save_feedback(
                    feedback="up",
                    search_mode=search_mode,
                    query=query,
                    results_count=len(display_df),
                    top_results=top_results,
                    seed_channel_id=seed_id,
                    seed_channel_name=seed_name
                )
                st.session_state['feedback_submitted'] = True
                st.rerun()

        with col_fb2:
            if st.button("👎 Not helpful", use_container_width=True):
                st.session_state['show_reason_selector'] = True

        # Show reason selector if thumbs down was clicked
        if st.session_state.get('show_reason_selector', False):
            st.markdown("**What was the issue?**")
            reason = st.radio(
                "Select a reason:",
                options=[
                    ("wrong_topic", "Wrong topic/niche"),
                    ("size_mismatch", "Channel too big/small"),
                    ("inactive", "Outdated/inactive channels"),
                    ("other", "Other")
                ],
                format_func=lambda x: x[1],
                label_visibility="collapsed",
                horizontal=True,
                key="feedback_reason_radio"
            )

            if st.button("Submit Feedback", type="primary"):
                # Gather search context
                display_df = st.session_state.get('display_df', pd.DataFrame())
                search_mode = "seed" if 'seed_profile' in st.session_state else "keyword"
                query = st.session_state.get('final_query', '')

                # Build top results list
                top_results = []
                if not display_df.empty:
                    for _, row in display_df.head(5).iterrows():
                        score = row.get('similarity_score') or row.get('relevance_score', 0)
                        top_results.append({
                            "channel_name": row.get('channel_title', ''),
                            "score": score
                        })

                # Get seed info if available
                seed_id = None
                seed_name = None
                if search_mode == "seed" and 'seed_profile' in st.session_state:
                    seed_profile = st.session_state['seed_profile']
                    seed_id = seed_profile.get('channel_id')
                    seed_name = seed_profile.get('channel_name')

                # Save negative feedback with reason
                feedback_tracker.save_feedback(
                    feedback="down",
                    search_mode=search_mode,
                    query=query,
                    results_count=len(display_df),
                    top_results=top_results,
                    reason=reason[0],  # reason code
                    seed_channel_id=seed_id,
                    seed_channel_name=seed_name
                )
                st.session_state['feedback_submitted'] = True
                st.session_state['show_reason_selector'] = False
                st.rerun()

# ============================================================================
# === ENHANCED MATCH EXPLANATIONS ===
# ============================================================================
if 'similarity_score' in st.session_state.get('display_df', pd.DataFrame()).columns:
    
    # Only show if we have the full data
    if 'top_channels_full' not in st.session_state:
        st.info("Run a seed-based search to see detailed analysis")
    else:
        st.header("🔍 Detailed Match Analysis")
        st.markdown("*Deep dive into why channels match your seed*")
        
        top_channels_data = st.session_state['top_channels_full']
        channel_names = top_channels_data['channel_title'].tolist()[:10]
        
        # Comparison toggle
        enable_comparison = st.checkbox(
            "🆚 Compare two candidates side-by-side",
            value=False,
            help="Compare match quality between two discovered channels",
            key="enable_comparison"
        )
        
        if not enable_comparison:
            # SINGLE CHANNEL VIEW
            with st.container():
                analysis_key = f"channel_analysis_{id(st.session_state.get('seed_profile', {}))}"
                
                selected_channel = st.selectbox(
                    "Select a channel to analyze:",
                    options=channel_names,
                    key=analysis_key
                )
                
                if selected_channel:
                    channel_row = top_channels_data[
                        top_channels_data['channel_title'] == selected_channel
                    ].iloc[0]
                    
                    # Generate detailed explanation (includes header and score)
                    explanation = similarity_engine.generate_match_explanation(
                        channel_row.to_dict(),
                        st.session_state['seed_profile'],
                        detailed=True
                    )
                    
                    st.markdown(explanation)
                    
                    # Show tags comparison (moved right after explanation)
                    with st.expander("📊 Tag Comparison"):
                        candidate_tags = set(channel_row.get('tags', []))
                        seed_tags = set(st.session_state['seed_profile']['common_tags'])
                        
                        common = candidate_tags & seed_tags
                        candidate_only = candidate_tags - seed_tags
                        seed_only = seed_tags - candidate_tags
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**✅ Common Tags**")
                            if common:
                                st.write(", ".join(f"#{tag}" for tag in list(common)[:15]))
                            else:
                                st.caption("_(none)_")
                        
                        with col2:
                            st.markdown(f"**🔵 {selected_channel} Only**")
                            if candidate_only:
                                st.write(", ".join(f"#{tag}" for tag in list(candidate_only)[:10]))
                            else:
                                st.caption("_(none)_")
                        
                        with col3:
                            st.markdown(f"**🟢 Seed Only**")
                            if seed_only:
                                st.write(", ".join(f"#{tag}" for tag in list(seed_only)[:10]))
                            else:
                                st.caption("_(none)_")
        
        else:
            # COMPARISON VIEW
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                channel_a = st.selectbox(
                    "Channel A:",
                    options=channel_names,
                    key="comparison_channel_a"
                )
            
            with col_b:
                channel_b = st.selectbox(
                    "Channel B:",
                    options=[ch for ch in channel_names if ch != channel_a],
                    key="comparison_channel_b"
                )
            
            if channel_a and channel_b:
                row_a = top_channels_data[top_channels_data['channel_title'] == channel_a].iloc[0]
                row_b = top_channels_data[top_channels_data['channel_title'] == channel_b].iloc[0]
                
                sim_a = row_a.get('similarity', {})
                sim_b = row_b.get('similarity', {})
                
                # Side-by-side scores
                col_score_a, col_score_b = st.columns(2)
                
                with col_score_a:
                    st.markdown(f"### 📊 {channel_a}")
                    st.metric("Similarity Score", f"{sim_a.get('total_score', 0):.1f}/100")
                    
                    st.markdown("**Why it matches:**")
                    for reason in sim_a.get('match_reasons', [])[:3]:
                        st.markdown(f"- {reason}")
                
                with col_score_b:
                    st.markdown(f"### 📊 {channel_b}")
                    st.metric("Similarity Score", f"{sim_b.get('total_score', 0):.1f}/100")
                    
                    st.markdown("**Why it matches:**")
                    for reason in sim_b.get('match_reasons', [])[:3]:
                        st.markdown(f"- {reason}")
                
                # Comparative insights
                st.markdown("---")
                st.markdown("### 🆚 Comparison Insights")
                
                score_diff = abs(sim_a.get('total_score', 0) - sim_b.get('total_score', 0))
                
                if score_diff < 5:
                    st.info("💡 **Very close match!** Both channels are equally similar to your seed.")
                elif sim_a.get('total_score', 0) > sim_b.get('total_score', 0):
                    st.success(f"💡 **{channel_a}** is a stronger match (+{score_diff:.1f} points)")
                else:
                    st.success(f"💡 **{channel_b}** is a stronger match (+{score_diff:.1f} points)")
                
                # Tag overlap comparison
                with st.expander("🏷️ Tag Overlap Comparison"):
                    tags_a = set(row_a.get('tags', []))
                    tags_b = set(row_b.get('tags', []))
                    seed_tags = set(st.session_state['seed_profile']['common_tags'])
                    
                    common_both = tags_a & tags_b & seed_tags
                    only_a = tags_a & seed_tags - tags_b
                    only_b = tags_b & seed_tags - tags_a
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**✅ Both Share**")
                        if common_both:
                            st.write(", ".join(f"#{t}" for t in list(common_both)[:10]))
                        else:
                            st.caption("_(none)_")
                    
                    with col2:
                        st.markdown(f"**🔵 Only {channel_a}**")
                        if only_a:
                            st.write(", ".join(f"#{t}" for t in list(only_a)[:8]))
                        else:
                            st.caption("_(none)_")
                    
                    with col3:
                        st.markdown(f"**🟠 Only {channel_b}**")
                        if only_b:
                            st.write(", ".join(f"#{t}" for t in list(only_b)[:8]))
                        else:
                            st.caption("_(none)_")


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
        debug_tracker.display_debug_panel()

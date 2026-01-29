"""
debug_ui.py - Streamlit UI components for debug tracking

This module provides Streamlit-specific UI for:
- Initializing debug session state
- Displaying debug panels in the sidebar
- Tracking API calls via session state

Uses analytics.quota_tracker for pure business logic.
"""

import streamlit as st
from functools import wraps
import time

try:
    from .analytics.quota_tracker import (
        YOUTUBE_QUOTA_COSTS,
        calculate_youtube_quota_used,
        calculate_gemini_cost_estimate,
        get_total_youtube_calls,
        get_total_gemini_calls,
        get_current_date_pt,
        get_next_reset_time,
        load_daily_quota,
        save_daily_quota,
        accumulate_to_daily_quota,
        create_empty_debug_data,
    )
except ImportError:
    from analytics.quota_tracker import (
        YOUTUBE_QUOTA_COSTS,
        calculate_youtube_quota_used,
        calculate_gemini_cost_estimate,
        get_total_youtube_calls,
        get_total_gemini_calls,
        get_current_date_pt,
        get_next_reset_time,
        load_daily_quota,
        save_daily_quota,
        accumulate_to_daily_quota,
        create_empty_debug_data,
    )


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_debug_tracking():
    """
    Initialize debug tracking system.

    Includes:
    - Per-search tracking (resets each search)
    - Daily quota tracking (loaded from disk, persists across refreshes)
    """
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    # Daily quota tracking (persists across refreshes via file storage)
    if 'daily_quota' not in st.session_state:
        st.session_state.daily_quota = load_daily_quota()

    # Per-search tracking (resets each search)
    if 'debug_data' not in st.session_state:
        st.session_state.debug_data = create_empty_debug_data()


def reset_debug_tracking():
    """
    Reset per-search counters for a new search.

    IMPORTANT:
    - First accumulates current search's usage to daily total
    - Then saves daily total to disk
    - Then resets per-search counters
    - Daily quota persists and is NOT reset
    """
    if 'debug_data' in st.session_state and 'daily_quota' in st.session_state:
        # Add current search's usage to daily total and save
        st.session_state.daily_quota = accumulate_to_daily_quota(
            st.session_state.debug_data,
            st.session_state.daily_quota
        )

    # Reset per-search counters
    st.session_state.debug_data = create_empty_debug_data()


# ============================================================================
# API CALL TRACKING
# ============================================================================

def track_api_call(api_name: str):
    """
    Increment counter for specific API.

    Args:
        api_name: One of 'youtube_search', 'youtube_channel', 'youtube_video',
                  'youtube_playlist', 'gemini_summary', 'gemini_outreach',
                  'gemini_similarity', 'gemini_relevance'

    Example usage:
        track_api_call('youtube_search')
    """
    if 'debug_data' not in st.session_state:
        initialize_debug_tracking()

    key = f'{api_name}_calls'
    if key in st.session_state.debug_data:
        st.session_state.debug_data[key] += 1


# ============================================================================
# TIMING DECORATOR
# ============================================================================

def time_operation(operation_name: str):
    """
    Decorator to automatically time function execution.

    Example usage:
        @time_operation('search')
        def search_channels(...):
            # function code

    Args:
        operation_name: Key in debug_data['timings'] dict
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not st.session_state.get('debug_mode', False):
                return func(*args, **kwargs)

            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            if 'debug_data' in st.session_state:
                if operation_name in st.session_state.debug_data['timings']:
                    st.session_state.debug_data['timings'][operation_name] += elapsed

            return result
        return wrapper
    return decorator


# ============================================================================
# SIMILARITY SCORE TRACKING
# ============================================================================

def track_similarity_scores(channels_with_scores: list):
    """
    Store detailed similarity breakdowns for all channels.

    Args:
        channels_with_scores: List of dicts with 'channel_title', 'similarity' keys
    """
    if 'debug_data' not in st.session_state:
        initialize_debug_tracking()

    simplified = []
    for ch in channels_with_scores[:20]:
        similarity = ch.get('similarity', {})
        simplified.append({
            'channel': ch.get('channel_title', 'Unknown'),
            'total_score': similarity.get('total_score', 0),
            'breakdown': similarity.get('breakdown', {}),
            'reasons': similarity.get('match_reasons', [])
        })

    st.session_state.debug_data['similarity_details'] = simplified


# ============================================================================
# DEBUG PANEL DISPLAY
# ============================================================================

def display_debug_panel():
    """
    Display debug information in Streamlit sidebar.

    Only shows when debug_mode is enabled.
    """
    if not st.session_state.get('debug_mode', False):
        return

    if 'debug_data' not in st.session_state:
        initialize_debug_tracking()
        return

    data = st.session_state.debug_data
    daily = st.session_state.get('daily_quota', {})

    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Info")

    # API USAGE SECTION
    with st.sidebar.expander("API Call Summary", expanded=False):
        st.markdown("**This Search:**")

        # Calculate totals
        youtube_total = get_total_youtube_calls(data)
        gemini_total = get_total_gemini_calls(data)

        # Calculate quota units per category
        search_calls = data.get('youtube_search_calls', 0)
        search_units = search_calls * YOUTUBE_QUOTA_COSTS['search']
        channel_units = data.get('youtube_channel_calls', 0) * YOUTUBE_QUOTA_COSTS['channels']
        playlist_units = data.get('youtube_playlist_calls', 0) * YOUTUBE_QUOTA_COSTS['playlistItems']
        video_units = data.get('youtube_video_calls', 0) * YOUTUBE_QUOTA_COSTS['videos']

        st.caption("YouTube API Quota")
        st.text(f"  Total: {calculate_youtube_quota_used(data)} units")
        if search_calls > 0:
            st.text(f"  - Search: {search_units} ({search_calls}x100)")
        else:
            st.text(f"  - Search: {search_units}")
        st.text(f"  - Channels: {channel_units}")
        st.text(f"  - Playlists: {playlist_units}")
        st.text(f"  - Videos: {video_units}")

        st.caption("Gemini API Calls")
        st.text(f"  Total: {gemini_total}")
        st.text(f"  - Summary: {data.get('gemini_summary_calls', 0)}")
        st.text(f"  - Outreach: {data.get('gemini_outreach_calls', 0)}")
        st.text(f"  - Similarity: {data.get('gemini_similarity_calls', 0)}")
        st.text(f"  - Relevance: {data.get('gemini_relevance_calls', 0)}")

        this_search_youtube = calculate_youtube_quota_used(data)
        this_search_gemini = calculate_gemini_cost_estimate(data)

        st.text(f"YouTube: {this_search_youtube:,} units")
        st.text(f"Gemini: ~${this_search_gemini:.4f} USD")

        st.markdown("---")

        st.markdown("**Today's Total:**")

        total_youtube_units = daily.get('youtube_units', 0) + this_search_youtube
        total_gemini_cost = daily.get('gemini_cost_usd', 0.0) + this_search_gemini

        total_youtube_calls = daily.get('youtube_calls', 0) + youtube_total
        total_gemini_calls = daily.get('gemini_calls', 0) + gemini_total

        st.text(f"{total_youtube_calls} YouTube calls")
        st.text(f"{total_gemini_calls} Gemini calls")

        st.markdown("**Daily Quota:**")

        quota_pct = total_youtube_units / 100
        quota_remaining = 10000 - total_youtube_units

        st.progress(min(quota_pct / 100, 1.0))

        if quota_pct < 50:
            status = "OK"
        elif quota_pct < 80:
            status = "Warning"
        else:
            status = "Critical"

        st.text(f"{status}: {total_youtube_units:,}/10,000 units ({quota_pct:.1f}%)")
        st.caption(f"Remaining: {quota_remaining:,} units")

        if total_gemini_cost > 0:
            st.text(f"Gemini: ~${total_gemini_cost:.4f} USD")
        else:
            st.text(f"Gemini: ~$0.0000 USD")

        reset_time = get_next_reset_time()
        st.caption(f"Resets {reset_time} (midnight PT)")

    # TIMING DATA
    timings = data.get('timings', {})
    total_time = timings.get('total', 0)

    with st.sidebar.expander("Performance Timing", expanded=False):
        if total_time > 0:
            st.text(f"Search: {timings.get('search', 0):.2f}s")
            st.text(f"Channel stats: {timings.get('channel_stats', 0):.2f}s")
            st.text(f"Video details: {timings.get('video_details', 0):.2f}s")
            st.text(f"AI relevance: {timings.get('ai_relevance', 0):.2f}s")
            st.text(f"Similarity: {timings.get('similarity', 0):.2f}s")
            st.text(f"AI generation: {timings.get('ai_generation', 0):.2f}s")
            st.markdown("---")
            st.text(f"**Total: {total_time:.2f}s**")

            bottleneck = max(timings.items(), key=lambda x: x[1] if x[0] != 'total' else 0)
            if bottleneck[1] > 0 and bottleneck[0] != 'total':
                pct = (bottleneck[1] / total_time) * 100
                st.caption(f"Bottleneck: {bottleneck[0]} ({pct:.0f}%)")
        else:
            st.text("No timing data yet")
            st.caption("Times will appear after a search completes")

    # SIMILARITY SCORES
    similarity_details = data.get('similarity_details', [])

    with st.sidebar.expander("Detailed Similarity Scores", expanded=False):
        if similarity_details:
            for i, ch in enumerate(similarity_details[:10], 1):
                st.markdown(f"**{i}. {ch['channel']}**")
                st.text(f"Score: {ch['total_score']:.1f}")

                if ch.get('breakdown'):
                    st.caption("Breakdown:")
                    for metric, value in ch['breakdown'].items():
                        st.caption(f"  - {metric}: {value:.1f}")

                if ch.get('reasons'):
                    st.caption("Matches:")
                    for reason in ch['reasons'][:3]:
                        st.caption(f"  - {reason}")

                st.markdown("---")
        else:
            st.text("No similarity data yet")
            st.caption("Similarity scores appear after seed search")

"""
debug_tracker.py - API usage tracking and quota monitoring

Tracks:
- API calls per search (YouTube + Gemini)
- Daily quota usage (persists to disk)
- Performance timing
"""

import streamlit as st
import time
import json
import os
from functools import wraps
from datetime import datetime, timezone


# ============================================================================
# INITIALIZATION
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
    
    # ========================================================================
    # DAILY QUOTA TRACKING (persists across refreshes via file storage)
    # ========================================================================
    
    if 'daily_quota' not in st.session_state:
        # Load from disk (or create new if doesn't exist/expired)
        st.session_state.daily_quota = load_daily_quota()
    
    # ========================================================================
    # PER-SEARCH TRACKING (resets each search)
    # ========================================================================
    
    if 'debug_data' not in st.session_state:
        st.session_state.debug_data = {
            # API call counters (this search only)
            'youtube_search_calls': 0,
            'youtube_channel_calls': 0,
            'youtube_video_calls': 0,
            'youtube_playlist_calls': 0,
            'gemini_summary_calls': 0,
            'gemini_outreach_calls': 0,
            'gemini_similarity_calls': 0,
            
            # Timing data (in seconds)
            'timings': {
                'search': 0.0,
                'channel_stats': 0.0,
                'video_details': 0.0,
                'similarity_ranking': 0.0,
                'ai_generation': 0.0,
                'total': 0.0
            },
            
            # Detailed similarity scores (for all channels)
            'similarity_details': []
        }
    

def reset_debug_tracking():
    """
    Reset per-search counters for a new search.
    
    IMPORTANT: 
    - First accumulates current search's usage to daily total
    - Then saves daily total to disk
    - Then resets per-search counters
    - Daily quota persists and is NOT reset
    """
    # Add current search's usage to daily total BEFORE resetting
    _accumulate_to_daily_quota()
    
    # Save updated daily quota to disk
    save_daily_quota()
    
    # Reset per-search counters
    st.session_state.debug_data = {
        'youtube_search_calls': 0,
        'youtube_channel_calls': 0,
        'youtube_video_calls': 0,
        'youtube_playlist_calls': 0,
        'gemini_summary_calls': 0,
        'gemini_outreach_calls': 0,
        'gemini_similarity_calls': 0,
        'timings': {
            'search': 0.0,
            'channel_stats': 0.0,
            'video_details': 0.0,
            'similarity_ranking': 0.0,
            'ai_generation': 0.0,
            'total': 0.0
        },
        'similarity_details': []
    }


def _accumulate_to_daily_quota():
    """
    Add current search's API usage to the daily total.
    
    Called automatically by reset_debug_tracking() before resetting counters.
    This ensures we don't lose the current search's usage data.
    """
    if 'debug_data' not in st.session_state or 'daily_quota' not in st.session_state:
        return
    
    data = st.session_state.debug_data
    
    # Calculate this search's usage
    youtube_units = calculate_youtube_quota_used()
    gemini_cost = calculate_gemini_cost_estimate()
    
    total_youtube_calls = (
        data.get('youtube_search_calls', 0) +
        data.get('youtube_channel_calls', 0) +
        data.get('youtube_video_calls', 0) +
        data.get('youtube_playlist_calls', 0)
    )
    
    total_gemini_calls = (
        data.get('gemini_summary_calls', 0) +
        data.get('gemini_outreach_calls', 0) +
        data.get('gemini_similarity_calls', 0)
    )
    
    # Add to daily total
    st.session_state.daily_quota['youtube_calls'] += total_youtube_calls
    st.session_state.daily_quota['gemini_calls'] += total_gemini_calls
    st.session_state.daily_quota['youtube_units'] += youtube_units
    st.session_state.daily_quota['gemini_cost_usd'] += gemini_cost


# ============================================================================
# API CALL TRACKING - Simple counters
# ============================================================================

def track_api_call(api_name: str):
    """
    Increment counter for specific API.
    
    Args:
        api_name: One of 'youtube_search', 'youtube_channel', 'youtube_video', 'youtube_playlist',
                  'gemini_summary', 'gemini_outreach', 'gemini_similarity'
    
    Example usage:
        track_api_call('youtube_search')
    """
    if 'debug_data' not in st.session_state:
        initialize_debug_tracking()
    
    key = f'{api_name}_calls'
    if key in st.session_state.debug_data:
        st.session_state.debug_data[key] += 1


# ============================================================================
# TIMING DECORATOR - Measure function execution time
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
# QUOTA ESTIMATION
# ============================================================================

YOUTUBE_QUOTA_COSTS = {
    'search': 100,
    'channels': 1,
    'videos': 1,
    'playlistItems': 1
}

GEMINI_COSTS = {
    'flash': {'input': 0.0001, 'output': 0.0004},
}


def calculate_youtube_quota_used():
    """Estimate YouTube quota units consumed in current search."""
    if 'debug_data' not in st.session_state:
        return 0
    
    data = st.session_state.debug_data
    
    total = 0
    total += data.get('youtube_search_calls', 0) * YOUTUBE_QUOTA_COSTS['search']
    total += data.get('youtube_channel_calls', 0) * YOUTUBE_QUOTA_COSTS['channels']
    total += data.get('youtube_video_calls', 0) * YOUTUBE_QUOTA_COSTS['videos']
    total += data.get('youtube_playlist_calls', 0) * YOUTUBE_QUOTA_COSTS['playlistItems']
    
    return total


def calculate_gemini_cost_estimate():
    """Estimate Gemini API cost in USD for current search."""
    if 'debug_data' not in st.session_state:
        return 0.0
    
    data = st.session_state.debug_data
    
    # Rough estimates (input + output tokens)
    summary_calls = data.get('gemini_summary_calls', 0)
    outreach_calls = data.get('gemini_outreach_calls', 0)
    similarity_calls = data.get('gemini_similarity_calls', 0)
    
    # Estimate: ~500 tokens input + 200 tokens output per call
    cost_per_call = (500 * GEMINI_COSTS['flash']['input'] + 200 * GEMINI_COSTS['flash']['output']) / 1000
    
    total = (summary_calls + outreach_calls + similarity_calls) * cost_per_call
    
    return total


# ============================================================================
# DAILY QUOTA PERSISTENCE
# ============================================================================

QUOTA_CACHE_FILE = '.quota_cache.json'


def load_daily_quota():
    """Load daily quota from disk, or create new if expired/missing."""
    if os.path.exists(QUOTA_CACHE_FILE):
        try:
            with open(QUOTA_CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            # Check if it's from today (Pacific Time)
            saved_date = data.get('date', '')
            today = get_current_date_pt()
            
            if saved_date == today:
                return data
            
        except Exception:
            pass  # Fall through to create new
    
    # Create new daily quota
    return {
        'date': get_current_date_pt(),
        'youtube_calls': 0,
        'gemini_calls': 0,
        'youtube_units': 0,
        'gemini_cost_usd': 0.0
    }


def save_daily_quota():
    """Save daily quota to disk."""
    if 'daily_quota' not in st.session_state:
        return
    
    try:
        with open(QUOTA_CACHE_FILE, 'w') as f:
            json.dump(st.session_state.daily_quota, f, indent=2)
    except Exception as e:
        pass  # Silent fail - not critical


def get_current_date_pt():
    """Get current date in Pacific Time (YYYY-MM-DD format)."""
    from datetime import datetime, timezone, timedelta
    
    # Pacific Time is UTC-8 (or UTC-7 during DST)
    # For simplicity, using UTC-8
    pt_offset = timedelta(hours=-8)
    pt_time = datetime.now(timezone.utc) + pt_offset
    
    return pt_time.strftime('%Y-%m-%d')


def get_next_reset_time():
    """Get human-readable time until quota resets (midnight PT)."""
    from datetime import datetime, timezone, timedelta
    
    pt_offset = timedelta(hours=-8)
    pt_now = datetime.now(timezone.utc) + pt_offset
    
    # Calculate midnight PT
    midnight_pt = pt_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    time_until = midnight_pt - pt_now
    
    hours = int(time_until.total_seconds() // 3600)
    
    if hours < 1:
        return "in less than 1 hour"
    elif hours == 1:
        return "in 1 hour"
    else:
        return f"in {hours} hours"


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
    st.sidebar.subheader("🔍 Debug Info")
    
    # API USAGE SECTION
    with st.sidebar.expander("📊 API Call Summary", expanded=False):
        st.markdown("**This Search:**")

        # Calculate totals
        youtube_total = (
            data.get('youtube_search_calls', 0) +
            data.get('youtube_channel_calls', 0) +
            data.get('youtube_video_calls', 0) +
            data.get('youtube_playlist_calls', 0)
        )

        gemini_total = (
            data.get('gemini_summary_calls', 0) +
            data.get('gemini_outreach_calls', 0) +
            data.get('gemini_similarity_calls', 0)
        )

        st.caption("YouTube API Calls")
        st.text(f"  Total: {calculate_youtube_quota_used()}")
        st.text(f"  ├─ 🔎 Search: {data.get('youtube_search_calls', 0)}")
        st.text(f"  ├─ 📺 Channels: {data.get('youtube_channel_calls', 0)}")
        st.text(f"  ├─ 📃 Playlists: {data.get('youtube_playlist_calls', 0)}")
        st.text(f"  └─ 🎬 Videos: {data.get('youtube_video_calls', 0)}")

        st.caption("Gemini API Calls")
        st.text(f"  Total: {gemini_total}")
        st.text(f"  ├─ 📝 Summary: {data.get('gemini_summary_calls', 0)}")
        st.text(f"  ├─ ✉️  Outreach: {data.get('gemini_outreach_calls', 0)}")
        st.text(f"  └─ 🎯 Similarity: {data.get('gemini_similarity_calls', 0)}")
        
        this_search_youtube = calculate_youtube_quota_used()
        this_search_gemini = calculate_gemini_cost_estimate()
        
        st.text(f"YouTube: {this_search_youtube:,} units")
        st.text(f"Gemini: ~${this_search_gemini:.4f} USD")
        
        st.markdown("---")
        
        st.markdown("**Today's Total:**")
        
        total_youtube_units = daily.get('youtube_units', 0) + this_search_youtube
        total_gemini_cost = daily.get('gemini_cost_usd', 0.0) + this_search_gemini
        
        total_youtube_calls = daily.get('youtube_calls', 0) + (
            data.get('youtube_search_calls', 0) +
            data.get('youtube_channel_calls', 0) +
            data.get('youtube_video_calls', 0) +
            data.get('youtube_playlist_calls', 0)
        )
        
        total_gemini_calls = daily.get('gemini_calls', 0) + (
            data.get('gemini_summary_calls', 0) +
            data.get('gemini_outreach_calls', 0) +
            data.get('gemini_similarity_calls', 0)
        )
        
        st.text(f"📞 {total_youtube_calls} YouTube calls")
        st.text(f"🤖 {total_gemini_calls} Gemini calls")
        
        st.markdown("**Daily Quota:**")
        
        quota_pct = total_youtube_units/100
        quota_remaining = 10000 - total_youtube_units
        
        st.progress(min(quota_pct / 100, 1.0))
        
        if quota_pct < 50:
            status_emoji = "✅"
            status_color = "🟢"
        elif quota_pct < 80:
            status_emoji = "⚠️"
            status_color = "🟡"
        else:
            status_emoji = "🚨"
            status_color = "🔴"
        
        st.text(f"{status_emoji} {total_youtube_units:,}/10,000 units ({quota_pct:.1f}%)")
        st.caption(f"{status_color} Remaining: {quota_remaining:,} units")
        
        if total_gemini_cost > 0:
            st.text(f"💰 Gemini: ~${total_gemini_cost:.4f} USD")
        else:
            st.text(f"💰 Gemini: ~$0.0000 USD")
        
        reset_time = get_next_reset_time()
        st.caption(f"🔄 Resets {reset_time} (midnight PT)")
    
    # TIMING DATA
    timings = data.get('timings', {})
    total_time = timings.get('total', 0)
    
    with st.sidebar.expander("⏱️ Performance Timing", expanded=False):
        if total_time > 0:
            st.text(f"🔎 Search: {timings.get('search', 0):.2f}s")
            st.text(f"📊 Channel stats: {timings.get('channel_stats', 0):.2f}s")
            st.text(f"🎬 Video details: {timings.get('video_details', 0):.2f}s")
            st.text(f"🤖 AI relevance: {timings.get('ai_relevance', 0):.2f}s")
            st.text(f"🎯 Similarity: {timings.get('similarity', 0):.2f}s")
            st.text(f"✨ AI generation: {timings.get('ai_generation', 0):.2f}s")
            st.markdown("---")
            st.text(f"**Total: {total_time:.2f}s**")
            
            bottleneck = max(timings.items(), key=lambda x: x[1] if x[0] != 'total' else 0)
            if bottleneck[1] > 0 and bottleneck[0] != 'total':
                pct = (bottleneck[1] / total_time) * 100
                st.caption(f"🔥 Bottleneck: {bottleneck[0]} ({pct:.0f}%)")
        else:
            st.text("No timing data yet")
            st.caption("Times will appear after a search completes")
    
    # SIMILARITY SCORES
    similarity_details = data.get('similarity_details', [])
    
    with st.sidebar.expander("🎯 Detailed Similarity Scores", expanded=False):
        if similarity_details:
            for i, ch in enumerate(similarity_details[:10], 1):
                st.markdown(f"**{i}. {ch['channel']}**")
                st.text(f"Score: {ch['total_score']:.1f}")
                
                if ch.get('breakdown'):
                    st.caption("Breakdown:")
                    for metric, value in ch['breakdown'].items():
                        st.caption(f"  • {metric}: {value:.1f}")
                
                if ch.get('reasons'):
                    st.caption("Matches:")
                    for reason in ch['reasons'][:3]:
                        st.caption(f"  • {reason}")
                
                st.markdown("---")
        else:
            st.text("No similarity data yet")
            st.caption("Similarity scores appear after seed search")

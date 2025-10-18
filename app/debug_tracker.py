"""
debug_tracker.py - Performance and API usage monitoring with persistent daily quota tracking

This module tracks API calls, timing, and cache performance across the app.
Includes daily quota tracking that persists across page refreshes using local file storage.
"""

import time
import json
import streamlit as st
from typing import Optional, Any
from functools import wraps
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ============================================================================
# PERSISTENT STORAGE CONFIGURATION
# ============================================================================

# Where to store the persistent quota data
QUOTA_FILE = Path(".quota_cache.json")


def save_daily_quota():
    """
    Save daily quota to disk for persistence across page refreshes.
    
    Stores as JSON file in project root.
    """
    if 'daily_quota' not in st.session_state:
        return
    
    try:
        with open(QUOTA_FILE, 'w') as f:
            json.dump(st.session_state.daily_quota, f, indent=2)
    except Exception as e:
        # Silently fail - don't break the app if we can't save
        print(f"Warning: Could not save quota data: {e}")


def load_daily_quota() -> dict:
    """
    Load daily quota from disk.
    
    Returns:
        dict: Daily quota data, or new empty dict if file doesn't exist
    """
    if not QUOTA_FILE.exists():
        return {
            'date': get_current_quota_date(),
            'youtube_calls': 0,
            'gemini_calls': 0,
            'youtube_units': 0,
            'gemini_cost_usd': 0.0
        }
    
    try:
        with open(QUOTA_FILE, 'r') as f:
            data = json.load(f)
        
        # Check if it's from today - if not, reset
        current_date = get_current_quota_date()
        if data.get('date') != current_date:
            return {
                'date': current_date,
                'youtube_calls': 0,
                'gemini_calls': 0,
                'youtube_units': 0,
                'gemini_cost_usd': 0.0
            }
        
        return data
        
    except Exception as e:
        print(f"Warning: Could not load quota data: {e}")
        return {
            'date': get_current_quota_date(),
            'youtube_calls': 0,
            'gemini_calls': 0,
            'youtube_units': 0,
            'gemini_cost_usd': 0.0
        }


# ============================================================================
# TIMEZONE UTILITIES
# ============================================================================

def get_current_quota_date():
    """
    Get the current quota date in Pacific Time (UTC-7).
    
    Google's YouTube quota resets at midnight Pacific Time.
    Returns: String in format "YYYY-MM-DD"
    """
    pacific_offset = timedelta(hours=-7)
    pacific_tz = timezone(pacific_offset)
    
    now_pacific = datetime.now(pacific_tz)
    return now_pacific.strftime("%Y-%m-%d")


def get_next_reset_time():
    """
    Calculate when the quota will next reset (midnight PT).
    
    Returns: String like "in 14 hours" or "in 2 hours"
    """
    pacific_offset = timedelta(hours=-7)
    pacific_tz = timezone(pacific_offset)
    
    now_pacific = datetime.now(pacific_tz)
    
    # Calculate next midnight PT
    next_midnight = now_pacific.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    # Calculate time difference
    time_until_reset = next_midnight - now_pacific
    hours = int(time_until_reset.total_seconds() / 3600)
    
    if hours == 0:
        minutes = int(time_until_reset.total_seconds() / 60)
        return f"in {minutes} min"
    elif hours == 1:
        return "in 1 hour"
    else:
        return f"in {hours} hours"


# ============================================================================
# INITIALIZATION - Call this at app startup
# ============================================================================

def initialize_debug_tracking():
    """
    Initialize all debug tracking variables in session state.
    
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
            'gemini_summary_calls': 0,
            'gemini_outreach_calls': 0,
            'gemini_similarity_calls': 0,
            
            # Cache performance
            'cache_hits': 0,
            'cache_misses': 0,
            
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
        'gemini_summary_calls': 0,
        'gemini_outreach_calls': 0,
        'gemini_similarity_calls': 0,
        'cache_hits': 0,
        'cache_misses': 0,
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
        data.get('youtube_video_calls', 0)
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
        api_name: One of 'youtube_search', 'youtube_channel', 'youtube_video',
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
# CACHE TRACKING - Hit/miss ratio
# ============================================================================

def track_cache_hit():
    """Record when we successfully use cached data (no API call needed)"""
    if 'debug_data' not in st.session_state:
        initialize_debug_tracking()
    st.session_state.debug_data['cache_hits'] += 1


def track_cache_miss():
    """Record when we need to make a fresh API call"""
    if 'debug_data' not in st.session_state:
        initialize_debug_tracking()
    st.session_state.debug_data['cache_misses'] += 1


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
    
    return total


def calculate_gemini_cost_estimate():
    """Estimate Gemini API cost in USD for current search."""
    if 'debug_data' not in st.session_state:
        return 0.0
    
    data = st.session_state.debug_data
    
    summary_tokens = 500
    outreach_tokens = 300
    similarity_tokens = 400
    
    total_tokens = (
        data.get('gemini_summary_calls', 0) * summary_tokens +
        data.get('gemini_outreach_calls', 0) * outreach_tokens +
        data.get('gemini_similarity_calls', 0) * similarity_tokens
    )
    
    input_cost = (total_tokens * 0.6 / 1000) * GEMINI_COSTS['flash']['input']
    output_cost = (total_tokens * 0.4 / 1000) * GEMINI_COSTS['flash']['output']
    
    return input_cost + output_cost


# ============================================================================
# DEBUG PANEL UI
# ============================================================================

def render_debug_panel():
    """Render the complete debug information panel in the sidebar."""
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
    with st.sidebar.expander("📊 API Call Summary", expanded=True):
        st.markdown("**This Search:**")
        
        st.caption("YouTube API")
        st.text(f"  🔎 Search: {data.get('youtube_search_calls', 0)}")
        st.text(f"  📺 Channels: {data.get('youtube_channel_calls', 0)}")
        st.text(f"  🎬 Videos: {data.get('youtube_video_calls', 0)}")
        
        st.caption("Gemini API")
        st.text(f"  📝 Summary: {data.get('gemini_summary_calls', 0)}")
        st.text(f"  ✉️  Outreach: {data.get('gemini_outreach_calls', 0)}")
        st.text(f"  🎯 Similarity: {data.get('gemini_similarity_calls', 0)}")
        
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
            data.get('youtube_video_calls', 0)
        )
        
        total_gemini_calls = daily.get('gemini_calls', 0) + (
            data.get('gemini_summary_calls', 0) +
            data.get('gemini_outreach_calls', 0) +
            data.get('gemini_similarity_calls', 0)
        )
        
        st.text(f"📞 {total_youtube_calls} YouTube calls")
        st.text(f"🤖 {total_gemini_calls} Gemini calls")
        
        st.markdown("**Daily Quota:**")
        
        quota_pct = (total_youtube_units / 10000) * 100
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
    
    # CACHE PERFORMANCE
    total_cache_attempts = data.get('cache_hits', 0) + data.get('cache_misses', 0)
    
    with st.sidebar.expander("💾 Cache Performance", expanded=False):
        if total_cache_attempts > 0:
            cache_hit_rate = (data.get('cache_hits', 0) / total_cache_attempts) * 100
            st.metric("Hit Rate", f"{cache_hit_rate:.1f}%",
                     delta=f"{data.get('cache_hits', 0)} hits, {data.get('cache_misses', 0)} misses")
            st.caption("Higher is better - means fewer API calls")
        else:
            st.text("No cache data yet")
            st.caption("Hit Rate: 0.0%")
            st.caption("0 hits, 0 misses")
            st.caption("Higher is better - means fewer API calls")
    
    # TIMING DATA
    timings = data.get('timings', {})
    total_time = timings.get('total', 0)
    
    with st.sidebar.expander("⏱️ Performance Timing", expanded=False):
        if total_time > 0:
            st.text(f"🔎 Search: {timings.get('search', 0):.2f}s")
            st.text(f"📊 Channel stats: {timings.get('channel_stats', 0):.2f}s")
            st.text(f"🎬 Video details: {timings.get('video_details', 0):.2f}s")
            st.text(f"🎯 Similarity: {timings.get('similarity_ranking', 0):.2f}s")
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
                st.markdown(f"**{i}. {ch.get('channel', 'Unknown')}**")
                st.text(f"Score: {ch.get('total_score', 0):.1f}/100")
                
                breakdown = ch.get('breakdown', {})
                if breakdown:
                    st.caption(
                        f"Tags: {breakdown.get('tag_score', 0):.0f} | "
                        f"Keywords: {breakdown.get('keyword_score', 0):.0f} | "
                        f"Size: {breakdown.get('subscriber_score', 0):.0f}"
                    )
                
                reasons = ch.get('reasons', [])
                if reasons:
                    st.caption("• " + " • ".join(reasons[:2]))
                
                st.markdown("")
        else:
            st.text("No similarity data yet")
            st.caption("Only available for seed-based searches")


# ============================================================================
# MANUAL RESET
# ============================================================================

def render_quota_reset_button():
    """Render a button to manually reset daily quota."""
    if st.session_state.get('debug_mode', False):
        with st.sidebar:
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.caption("Manual quota reset")
            
            with col2:
                if st.button("🔄", key="reset_quota_btn", help="Reset daily quota counters"):
                    st.session_state.daily_quota = {
                        'date': get_current_quota_date(),
                        'youtube_calls': 0,
                        'gemini_calls': 0,
                        'youtube_units': 0,
                        'gemini_cost_usd': 0.0
                    }
                    save_daily_quota()  # Save to disk
                    st.success("✅ Daily quota reset!")
                    time.sleep(0.5)
                    st.rerun()


# ============================================================================
# SUMMARY STATS
# ============================================================================

def get_quota_summary() -> dict:
    """Get a quick summary of quota usage."""
    if 'daily_quota' not in st.session_state or 'debug_data' not in st.session_state:
        return {
            'youtube_units': 0,
            'youtube_remaining': 10000,
            'quota_pct': 0.0,
            'gemini_cost': 0.0,
            'status': 'ok'
        }
    
    daily = st.session_state.daily_quota
    
    this_search_youtube = calculate_youtube_quota_used()
    this_search_gemini = calculate_gemini_cost_estimate()
    
    total_youtube_units = daily.get('youtube_units', 0) + this_search_youtube
    total_gemini_cost = daily.get('gemini_cost_usd', 0.0) + this_search_gemini
    
    quota_pct = (total_youtube_units / 10000) * 100
    quota_remaining = 10000 - total_youtube_units
    
    if quota_pct < 50:
        status = 'ok'
    elif quota_pct < 80:
        status = 'warning'
    else:
        status = 'critical'
    
    return {
        'youtube_units': total_youtube_units,
        'youtube_remaining': quota_remaining,
        'quota_pct': quota_pct,
        'gemini_cost': total_gemini_cost,
        'status': status
    }
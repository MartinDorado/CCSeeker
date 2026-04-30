"""
pipeline.py - Search pipeline orchestration

Pure functions for executing the search pipeline.
These functions are Streamlit-agnostic and can be unit tested.

Key design decisions:
- Functions accept services (youtube, gemini) as parameters
- Callbacks are used for progress updates instead of direct st.* calls
- Results are returned as structured dataclasses, not stored in session_state
- All warnings/errors are collected and returned, not displayed directly
"""

import re
import time
import pandas as pd
from typing import Callable, Any, Protocol
from dataclasses import dataclass, field
from collections import Counter

# Import core functions
from .query_utils import validate_and_truncate_query
from .relevance import calculate_keyword_relevance
from .youtube_api import (
    search_channels_hybrid as _search_hybrid,
    search_channels_multi_term as _search_multi_term,
    get_channel_stats as _get_stats,
    get_video_details as _get_videos,
)
from .gemini_api import (
    generate_ai_relevance_score,
    generate_summary as _generate_summary,
    SummaryResult,
)
from .similarity import SimilarityCallbacks


# ============================================================================
# CONFIGURATION (mirrored from main.py for now)
# ============================================================================

MAX_VIDEOS_PER_TERM = 100
MAX_VIDEOS_PER_CHANNEL = 10
MIN_MATCH_SCORE = 5
MAX_CHANNELS_TO_ANALYZE = 50


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class PipelineResult:
    """
    Complete result from the search pipeline.

    Contains all data needed for UI rendering, stored separately from session_state.
    """
    # Core data
    channels_df: pd.DataFrame
    display_columns: list[str]
    column_explanations: dict[str, str]

    # For outreach feature
    top_channels_for_outreach: pd.DataFrame
    final_query: str

    # Optional seed mode data
    top_channels_full: pd.DataFrame | None = None

    # AI summary
    ai_summary: str | None = None
    ai_summary_error: str | None = None

    # Logging/debugging
    search_log: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    # Warnings that occurred during processing
    warnings: list[str] = field(default_factory=list)

    # Early exit info (non-None means pipeline stopped early)
    error: str | None = None

    # Intermediate data (for debugging/UI display)
    raw_channels_df: pd.DataFrame | None = None


@dataclass
class PipelineConfig:
    """Configuration parameters for the search pipeline."""
    # Search parameters
    max_videos_per_term: int = MAX_VIDEOS_PER_TERM
    max_videos_per_channel: int = MAX_VIDEOS_PER_CHANNEL
    max_channels_to_analyze: int = MAX_CHANNELS_TO_ANALYZE
    min_match_score: int = MIN_MATCH_SCORE

    # Filters
    min_subscribers: int = 0
    country_filter: str | None = None
    months_ago: int = 0

    # AI features
    enable_ai_relevance: bool = True
    enable_ai_summary: bool = True

    # Seed mode
    seed_profile: dict | None = None


class CacheFunctions(Protocol):
    """Protocol for cache functions passed to the pipeline."""
    def get_channel_stats_cached(self, channel_ids: tuple) -> list[dict]: ...
    def get_video_details_cached(self, channel_ids: tuple, max_videos: int) -> list[dict]: ...
    def search_channels_cached(self, query: str, region: str, max_videos: int) -> list[dict]: ...


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _flatten_tags(tag_series: pd.Series) -> list[str]:
    """Flatten a series of tag lists into unique lowercase tags."""
    all_tags = []
    for tags in tag_series:
        if isinstance(tags, list):
            all_tags.extend(tags)
        elif isinstance(tags, str):
            all_tags.append(tags)
    return list(set(t.lower().strip() for t in all_tags if t))


def _extract_keywords_from_titles(titles: list[str] | None) -> list[str]:
    """Extract common keywords from a list of video titles."""
    if not titles or not isinstance(titles, list):
        return []

    text = " ".join(str(t) for t in titles if t)
    words = re.findall(r'\b[a-záéíóúñü]{3,}\b', text.lower())

    word_counts = Counter(words)

    common_words = {'the', 'and', 'for', 'with', 'que', 'con', 'para', 'por', 'como'}
    filtered_words = [w for w, _ in word_counts.most_common(30) if w not in common_words]

    return filtered_words[:20]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_search_pipeline(
    youtube_service,
    query: str,
    region_code: str,
    config: PipelineConfig,
    gemini_model=None,
    similarity_engine=None,
    cache_functions: CacheFunctions | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    on_api_call: Callable[[str], None] | None = None,
) -> PipelineResult:
    """
    Execute the end-to-end search pipeline.

    This is a pure function that returns structured results instead of
    modifying session_state or rendering UI directly.

    Args:
        youtube_service: Authenticated YouTube Data API client
        query: Search query string (comma-separated terms)
        region_code: ISO 3166-1 alpha-2 region code (e.g., "US")
        config: Pipeline configuration parameters
        gemini_model: Optional Gemini model for AI features
        similarity_engine: Optional similarity engine module for seed mode
        cache_functions: Optional cache wrapper functions
        on_progress: Callback for progress updates (message, percentage)
        on_api_call: Callback for API call tracking

    Returns:
        PipelineResult with all data needed for UI rendering

    Notes:
        - Does not raise exceptions; errors are captured in result.error
        - Does not access st.session_state
        - All warnings are collected in result.warnings
    """
    # Initialize result containers
    search_log: list[str] = []
    warnings: list[str] = []
    timings: dict[str, float] = {}
    pipeline_start = time.time()

    try:
        # === STEP 0.5: Validate Query ===
        if on_progress:
            on_progress("Validating query...", 0.05)

        final_query, was_truncated = validate_and_truncate_query(query)

        if was_truncated:
            original_terms = [t.strip() for t in query.split(',') if t.strip()]
            kept_terms = original_terms[:2]
            removed_terms = original_terms[2:]
            warnings.append(
                f"Query automatically adjusted: Using '{', '.join(kept_terms)}'. "
                f"Removed: {', '.join(removed_terms)}"
            )

        if not final_query:
            return PipelineResult(
                channels_df=pd.DataFrame(),
                display_columns=[],
                column_explanations={},
                top_channels_for_outreach=pd.DataFrame(),
                final_query="",
                error="Empty query after validation",
                warnings=warnings,
            )

        # === STEP 1: Search for channels ===
        if on_progress:
            on_progress("Step 1/4: Searching for channels...", 0.1)

        step_start = time.time()

        if cache_functions:
            initial_channels = cache_functions.search_channels_cached(
                final_query, region_code, config.max_videos_per_term
            )
        else:
            result = _search_multi_term(
                youtube_service=youtube_service,
                query=final_query,
                region_code=region_code,
                max_videos_per_term=config.max_videos_per_term,
                on_api_call=on_api_call,
            )
            initial_channels = result.channels
            warnings.extend(result.warnings)

        timings['search'] = time.time() - step_start

        if not initial_channels:
            return PipelineResult(
                channels_df=pd.DataFrame(),
                display_columns=[],
                column_explanations={},
                top_channels_for_outreach=pd.DataFrame(),
                final_query=final_query,
                error="Search did not return any channels",
                search_log=search_log,
                timings=timings,
                warnings=warnings,
            )

        df_initial = pd.DataFrame(initial_channels)
        pre_cap_count = len(initial_channels)
        search_log.append(f"🔍 Found {pre_cap_count} channels from search")

        # === STEP 2: Fetch channel statistics ===
        if on_progress:
            on_progress("Step 2/4: Fetching channel statistics...", 0.25)

        step_start = time.time()
        channel_ids_tuple = tuple(df_initial['channel_id'].tolist())

        if cache_functions:
            channel_statistics = cache_functions.get_channel_stats_cached(channel_ids_tuple)
        else:
            stats_result = _get_stats(
                youtube_service=youtube_service,
                channel_ids=list(channel_ids_tuple),
                on_api_call=on_api_call,
            )
            channel_statistics = stats_result.stats

        timings['channel_stats'] = time.time() - step_start

        if not channel_statistics:
            return PipelineResult(
                channels_df=pd.DataFrame(),
                display_columns=[],
                column_explanations={},
                top_channels_for_outreach=pd.DataFrame(),
                final_query=final_query,
                error="Could not retrieve channel statistics",
                search_log=search_log,
                timings=timings,
                warnings=warnings,
                raw_channels_df=df_initial,
            )

        df_stats = pd.DataFrame(channel_statistics)
        enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')

        # === STEP 3: Apply user filters ===
        if on_progress:
            on_progress("Step 3/5: Applying filters...", 0.35)

        step_start = time.time()

        # Filter by subscriber count
        filtered_channels = enriched_channel_data[
            enriched_channel_data['subscribers'] >= config.min_subscribers
        ].copy()

        # Filter by country (if specified); channels with no country set pass through
        if config.country_filter:
            target = config.country_filter.upper()
            mask = (filtered_channels['country'] == target) | filtered_channels['country'].isna()
            filtered_channels = filtered_channels[mask]

        timings['filtering'] = time.time() - step_start

        if filtered_channels.empty:
            return PipelineResult(
                channels_df=pd.DataFrame(),
                display_columns=[],
                column_explanations={},
                top_channels_for_outreach=pd.DataFrame(),
                final_query=final_query,
                error="No channels match your filtering criteria (subscribers, country)",
                search_log=search_log,
                timings=timings,
                warnings=warnings,
                raw_channels_df=df_initial,
            )

        search_log.append(
            f"✅ {len(filtered_channels)} channels passed filters "
            f"(min {config.min_subscribers:,} subs)"
        )

        # === STEP 4: Quality selection ===
        if on_progress:
            on_progress("Step 4/5: Preparing channels for analysis...", 0.45)

        step_start = time.time()

        quality_channels = filtered_channels[
            filtered_channels['match_score'] >= config.min_match_score
        ].copy()

        if quality_channels.empty:
            warnings.append(
                f"No channels found with match_score >= {config.min_match_score}. "
                f"Showing all {len(filtered_channels)} channels found."
            )
            quality_channels = filtered_channels.copy()
        else:
            filtered_out_count = len(filtered_channels) - len(quality_channels)
            if filtered_out_count > 0:
                search_log.append(
                    f"✨ Quality filter: {filtered_out_count} low-relevance channels excluded"
                )

        # Sort by relevance then subscribers
        filtered_sorted = quality_channels.sort_values(
            by=['match_score', 'subscribers'],
            ascending=[False, False]
        )

        channels_to_analyze = filtered_sorted.head(config.max_channels_to_analyze).copy()
        channels_analyzed_count = len(channels_to_analyze)

        if channels_analyzed_count < len(quality_channels):
            avg_match = channels_to_analyze['match_score'].mean()
            search_log.append(
                f"📊 Analyzing top {channels_analyzed_count} channels "
                f"(avg match: {avg_match:.0f}) from {len(quality_channels)} quality matches"
            )
        else:
            search_log.append(f"📊 Analyzing all {channels_analyzed_count} channels")

        timings['select_channels'] = time.time() - step_start

        # === STEP 5: Deep analysis (fetch videos) ===
        if on_progress:
            on_progress(
                f"Deep analysis - fetching videos from {channels_analyzed_count} channels...",
                0.55
            )

        step_start = time.time()
        channel_ids_tuple = tuple(channels_to_analyze['channel_id'].tolist())

        if cache_functions:
            video_data = cache_functions.get_video_details_cached(
                channel_ids_tuple, config.max_videos_per_channel
            )
        else:
            # Need to get uploads playlist IDs first
            channel_data = [
                {
                    'channel_id': row['channel_id'],
                    'uploads_playlist_id': row.get('uploads_playlist_id'),
                    'channel_title': row.get('channel_title', '(unknown)'),
                }
                for _, row in channels_to_analyze.iterrows()
            ]
            videos_result = _get_videos(
                youtube_service=youtube_service,
                channel_data=channel_data,
                max_videos_per_channel=config.max_videos_per_channel,
                on_api_call=on_api_call,
            )
            video_data = videos_result.videos
            warnings.extend(videos_result.warnings)

        timings['video_details'] = time.time() - step_start

        if not video_data:
            # Return partial results with warning
            warnings.append("Could not retrieve video details from channels")
            return PipelineResult(
                channels_df=channels_to_analyze.sort_values(by="subscribers", ascending=False),
                display_columns=['channel_title', 'subscribers', 'country'],
                column_explanations={"channel_title": "Name", "subscribers": "Subs", "country": "Country"},
                top_channels_for_outreach=channels_to_analyze[['channel_title']],
                final_query=final_query,
                search_log=search_log,
                timings=timings,
                warnings=warnings,
            )

        search_log.append(
            f"🎬 Retrieved {len(video_data)} videos for deep analysis"
        )

        # === STEP 6: Calculate relevance and engagement ===
        if on_progress:
            on_progress("Calculating relevance and engagement metrics...", 0.65)

        step_start = time.time()

        df_videos = pd.DataFrame(video_data)

        # Calculate relevance scores
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
        # Filters out channels whose most recent video is older than the cutoff
        if config.months_ago > 0:
            date_cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=config.months_ago)
            most_recent_per_channel = df_full.groupby('channel_id')['published_at'].max()
            active_channel_ids = most_recent_per_channel[most_recent_per_channel >= date_cutoff].index
            channels_before = df_full['channel_id'].nunique()
            df_full = df_full[df_full['channel_id'].isin(active_channel_ids)]
            channels_after = df_full['channel_id'].nunique()
            inactive_count = channels_before - channels_after
            if inactive_count > 0:
                search_log.append(
                    f"📅 Recency filter: {inactive_count} channels excluded (no uploads in {config.months_ago} months)"
                )
            else:
                search_log.append(
                    f"📅 Recency filter applied ({config.months_ago} months): all {channels_after} channels have recent uploads"
                )

        # Calculate average engagement per channel
        avg_engagement = df_full.groupby('channel_id')['engagement_rate'].mean().reset_index()

        # Merge everything together
        final_channels = pd.merge(channels_to_analyze, avg_engagement, on='channel_id')
        final_channels = pd.merge(final_channels, relevance_scores, on='channel_id', how='left')

        # Fill NaN relevance scores with 0
        final_channels['relevance_score'] = final_channels['relevance_score'].fillna(0)

        # Drop channels that scored exactly 0 in keyword-only mode — they matched no criteria.
        # In seed mode the similarity step re-scores channels, so we keep them all here.
        if not config.seed_profile:
            final_channels = final_channels[final_channels['relevance_score'] > 0]

        # Sort by relevance then engagement
        final_channels_sorted = final_channels.sort_values(
            by=['relevance_score', 'engagement_rate'],
            ascending=False
        ).copy()

        # Add analysis badge
        final_channels_sorted['analysis_depth'] = '✓ videos analyzed'

        top_channels = final_channels_sorted.copy()

        # Success message with relevance stats
        total_count = len(top_channels)
        high_relevance_count = len(top_channels[top_channels['relevance_score'] >= 0.15])
        search_log.append(
            f"✅ Found {total_count} channels ({high_relevance_count} with high topic focus ≥15%)"
        )

        timings['relevance_filtering'] = time.time() - step_start

        # === STEP 7: AI-Enhanced Relevance Scoring ===
        if gemini_model and config.enable_ai_relevance:
            if on_progress:
                on_progress("✨ Enhancing relevance with AI...", 0.75)

            step_start = time.time()

            ai_scores = []
            video_titles_by_channel = df_videos.groupby('channel_id')['video_title'].apply(list).to_dict()

            for _, row in top_channels.iterrows():
                channel_id = row['channel_id']
                channel_data_for_ai = {
                    'channel_title': row['channel_title'],
                    'video_titles': video_titles_by_channel.get(channel_id, [])
                }
                ai_score = generate_ai_relevance_score(
                    gemini_model,
                    channel_data_for_ai,
                    final_query,
                    on_api_call=on_api_call,
                )
                ai_scores.append({'channel_id': channel_id, 'ai_relevance_score': ai_score})

            if ai_scores:
                df_ai_scores = pd.DataFrame(ai_scores)
                top_channels = pd.merge(top_channels, df_ai_scores, on='channel_id', how='left')
                top_channels['ai_relevance_score'] = top_channels['ai_relevance_score'].fillna(0)

                # Blend: 80% algorithmic, 20% AI
                top_channels['relevance_score'] = (
                    0.8 * top_channels['relevance_score'] +
                    0.2 * top_channels['ai_relevance_score']
                )

                # Re-sort with blended score
                top_channels = top_channels.sort_values(
                    by=['relevance_score', 'engagement_rate'],
                    ascending=False
                ).copy()

                search_log.append("🧠 Blended keyword scores with AI relevance")

            timings['ai_relevance'] = time.time() - step_start

        # === STEP 8: Similarity Ranking (if using seed) ===
        if config.seed_profile and similarity_engine:
            if on_progress:
                on_progress("🧠 Calculating similarity scores...", 0.85)

            step_start = time.time()
            seed_channel_id = config.seed_profile['channel_id']
            before_exclusion = len(top_channels)

            top_channels = top_channels[top_channels['channel_id'] != seed_channel_id]

            excluded_count = before_exclusion - len(top_channels)
            if excluded_count > 0:
                search_log.append(f"🚫 Excluded seed channel from results")

            if top_channels.empty:
                return PipelineResult(
                    channels_df=pd.DataFrame(),
                    display_columns=[],
                    column_explanations={},
                    top_channels_for_outreach=pd.DataFrame(),
                    final_query=final_query,
                    error="No channels in the similar size range. Try disabling the size filter.",
                    search_log=search_log,
                    timings=timings,
                    warnings=warnings,
                )

            # Extract tags from existing df_videos
            top_channel_ids = set(top_channels['channel_id'])
            df_videos_filtered = df_videos[df_videos['channel_id'].isin(top_channel_ids)]

            channel_tags = (
                df_videos_filtered.groupby('channel_id')['video_tags']
                .apply(_flatten_tags)
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

            top_channels['tags'] = top_channels['tags'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            top_channels['recent_titles'] = top_channels['recent_titles'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            top_channels['keywords'] = top_channels['recent_titles'].apply(_extract_keywords_from_titles)

            search_log.append("🎯 Ranking channels by similarity...")

            candidates = top_channels.to_dict('records')
            for candidate in candidates:
                if 'channel_title' in candidate and 'channel_name' not in candidate:
                    candidate['channel_name'] = candidate['channel_title']

            # Get GEMINI_API_KEY for similarity engine (passed via config)
            gemini_api_key = config.seed_profile.get('gemini_api_key')

            # Create callbacks for similarity engine
            # Captures warnings in the warnings list, tracks API calls via callback
            similarity_callbacks = SimilarityCallbacks(
                on_info=lambda msg: search_log.append(msg),
                on_warning=lambda msg: warnings.append(msg),
                on_success=lambda msg: search_log.append(msg),
                on_api_call=on_api_call,
                debug_mode=True
            )

            ranked = similarity_engine.rank_channels_by_similarity(
                candidates,
                config.seed_profile,
                gemini_api_key=gemini_api_key,
                gemini_limit=10,
                debug=True,  # Enable breakdown for feedback analytics
                callbacks=similarity_callbacks
            )

            top_channels = pd.DataFrame(ranked)

            top_channels['similarity_score'] = top_channels['similarity'].apply(
                lambda x: x.get('total_score', 0) if isinstance(x, dict) else 0
            )

            top_channels['match_reasons'] = top_channels['similarity'].apply(
                lambda x: ' • '.join(x.get('match_reasons', [])[:2]) if isinstance(x, dict) else ''
            )

            top_channels = top_channels.sort_values('similarity_score', ascending=False)

            # Drop channels that scored exactly 0 — no meaningful similarity detected
            top_channels = top_channels[top_channels['similarity_score'] > 0]

            timings['similarity'] = time.time() - step_start

        # === STEP 9: AI Summary ===
        ai_summary = None
        ai_summary_error = None

        if gemini_model and config.enable_ai_summary:
            if on_progress:
                on_progress("✨ Generating AI Summary...", 0.9)

            step_start = time.time()

            try:
                summary_df = top_channels.copy()
                summary_df['relevance_score'] = summary_df['relevance_score'].fillna(0).map('{:.0%}'.format)
                summary_df['engagement_rate'] = summary_df['engagement_rate'].fillna(0).map('{:.2%}'.format)

                seed_channel_name = None
                if config.seed_profile:
                    seed_channel_name = config.seed_profile.get('channel_name', 'the seed channel')

                result = _generate_summary(
                    model=gemini_model,
                    df_results=summary_df,
                    query=final_query,
                    seed_channel_name=seed_channel_name,
                    on_api_call=on_api_call,
                )

                if result.error:
                    ai_summary_error = result.error
                else:
                    ai_summary = result.text
            except Exception as e:
                ai_summary_error = str(e)

            timings['ai_generation'] = time.time() - step_start
        else:
            ai_summary_error = "Gemini not configured"

        # === STEP 10: Format results ===
        if on_progress:
            on_progress("Formatting results...", 0.95)

        # Create YouTube channel URL (before saving full data)
        top_channels['channel_url'] = top_channels['channel_id'].apply(
            lambda x: f"https://www.youtube.com/channel/{x}" if x else ""
        )

        # Save full data BEFORE formatting (for feedback analytics with raw scores)
        top_channels_full = top_channels.copy()

        # Format display values
        top_channels['relevance_score'] = top_channels['relevance_score'].fillna(0).map('{:.0%}'.format)
        top_channels['engagement_rate'] = top_channels['engagement_rate'].fillna(0).map('{:.2%}'.format)
        top_channels['avg_views_per_video'] = top_channels['avg_views_per_video'].fillna(0).map('{:,.0f}'.format)

        if 'similarity_score' in top_channels.columns:
            top_channels['similarity_score'] = top_channels['similarity_score'].fillna(0).map('{:.1f}'.format)

        # Choose display columns
        # Note: channel_id is included for feedback tracking but not displayed in UI
        if 'similarity_score' in top_channels.columns:
            display_columns = [
                'channel_id', 'channel_title', 'channel_url', 'similarity_score', 'relevance_score',
                'avg_views_per_video', 'subscribers', 'country', 'engagement_rate'
            ]
            column_explanations = {
                "channel_title": "Name of the YouTube channel",
                "similarity_score": "Overall similarity to your seed channel (0-100).",
                "relevance_score": "Percentage of recent videos containing your search keywords.",
                "avg_views_per_video": "Average views per video across recent uploads.",
                "subscribers": "Total subscriber count.",
                "country": "Country where the channel is registered.",
                "engagement_rate": "(Likes + Comments) / Views, averaged across recent videos."
            }
        else:
            display_columns = [
                'channel_id', 'channel_title', 'channel_url', 'relevance_score', 'subscribers',
                'avg_views_per_video', 'country', 'engagement_rate'
            ]
            column_explanations = {
                "channel_title": "Name of the YouTube channel",
                "relevance_score": "Percentage of recent videos containing your search keywords.",
                "subscribers": "Total subscriber count.",
                "avg_views_per_video": "Average views per video across recent uploads.",
                "country": "Country where the channel is registered.",
                "engagement_rate": "(Likes + Comments) / Views, averaged across recent videos."
            }

        # Prepare outreach data
        top_channels_for_outreach = top_channels[['channel_title']].reset_index(drop=True)

        timings['total'] = time.time() - pipeline_start

        if on_progress:
            on_progress("Complete!", 1.0)

        return PipelineResult(
            channels_df=top_channels[display_columns].copy(),
            display_columns=display_columns,
            column_explanations=column_explanations,
            top_channels_for_outreach=top_channels_for_outreach,
            final_query=final_query,
            top_channels_full=top_channels_full,
            ai_summary=ai_summary,
            ai_summary_error=ai_summary_error,
            search_log=search_log,
            timings=timings,
            warnings=warnings,
            raw_channels_df=df_initial,
        )

    except Exception as e:
        import traceback
        return PipelineResult(
            channels_df=pd.DataFrame(),
            display_columns=[],
            column_explanations={},
            top_channels_for_outreach=pd.DataFrame(),
            final_query=query,
            error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            search_log=search_log,
            timings=timings,
            warnings=warnings,
        )

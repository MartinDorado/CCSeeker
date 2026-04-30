"""
similarity.py - Multi-signal channel similarity scoring

Pure business logic for ranking candidate channels by similarity to a seed channel.
This module is Streamlit-agnostic and uses callbacks for UI notifications.

Scoring Breakdown (100 points total):
- Tag Overlap (30 pts): Jaccard similarity on video tags
- Keyword Overlap (30 pts): Jaccard similarity on keywords from titles
- Subscriber Similarity (15 pts): Ratio-based scoring
- Engagement Rate Similarity (17 pts): Absolute difference penalty
- Upload Frequency Similarity (8 pts): Ratio-based scoring

When Gemini API key is available:
- Final Score = 80% algorithmic + 20% Gemini "vibe" analysis
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from .scoring_version import SEED_WEIGHTS
except ImportError:
    from scoring_version import SEED_WEIGHTS


# ============================================================================
# CALLBACK INTERFACE
# ============================================================================

@dataclass
class SimilarityCallbacks:
    """
    Callbacks for UI notifications during similarity analysis.

    This allows the similarity module to remain Streamlit-agnostic while
    still providing progress updates to the UI layer.

    Example usage:
        callbacks = SimilarityCallbacks(
            on_info=lambda msg: st.info(msg),
            on_warning=lambda msg: st.warning(msg),
            on_success=lambda msg: st.success(msg),
            on_api_call=lambda call_type: track_api_call(call_type),
            debug_mode=True
        )
    """
    on_info: Optional[Callable[[str], None]] = None
    on_warning: Optional[Callable[[str], None]] = None
    on_success: Optional[Callable[[str], None]] = None
    on_api_call: Optional[Callable[[str], None]] = None
    debug_mode: bool = False


def _default_callbacks() -> SimilarityCallbacks:
    """Return a SimilarityCallbacks instance with no-op callbacks."""
    return SimilarityCallbacks()


# ============================================================================
# SIMILARITY METRICS
# ============================================================================

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity coefficient.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set1: First set of items
        set2: Second set of items

    Returns:
        float: 0.0 (no overlap) to 1.0 (identical)

    Examples:
        >>> jaccard_similarity({'a', 'b', 'c'}, {'b', 'c', 'd'})
        0.5
        >>> jaccard_similarity({'a', 'b'}, {'a', 'b'})
        1.0
        >>> jaccard_similarity({'a'}, {'b'})
        0.0
    """
    set1 = set(set1)
    set2 = set(set2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def overlap_count(set1: set, set2: set) -> int:
    """
    Count items in common between two sets.

    Args:
        set1: First set of items
        set2: Second set of items

    Returns:
        int: Number of common items

    Examples:
        >>> overlap_count({'a', 'b', 'c'}, {'b', 'c', 'd'})
        2
        >>> overlap_count({'a'}, {'b'})
        0
    """
    return len(set(set1) & set(set2))


# ============================================================================
# SUBSCRIBER TIER UTILITIES
# ============================================================================

def get_subscriber_similarity(candidate_subs: int, seed_subs: int) -> float:
    """
    Calculate subscriber count similarity using ratio-based scoring.

    Uses min(a/b, b/a) to get a symmetric similarity measure:
    - Same size (1:1 ratio) = 1.0
    - 2x difference = 0.5
    - 10x difference = 0.1

    Args:
        candidate_subs: Candidate channel's subscriber count
        seed_subs: Seed channel's subscriber count

    Returns:
        float: 0.0 to 1.0 similarity score

    Examples:
        >>> get_subscriber_similarity(100000, 100000)
        1.0
        >>> get_subscriber_similarity(50000, 100000)
        0.5
        >>> get_subscriber_similarity(10000, 100000)
        0.1
    """
    if seed_subs == 0 or candidate_subs == 0:
        return 0.0

    ratio = min(candidate_subs / seed_subs, seed_subs / candidate_subs)
    return ratio


def is_within_tier_range(candidate_subs: int, seed_subs: int, tolerance: float = 0.5) -> bool:
    """
    Check if candidate is within ±tolerance of seed's subscriber count.

    Args:
        candidate_subs: Candidate channel's subscriber count
        seed_subs: Seed channel's subscriber count
        tolerance: Fractional tolerance (0.5 = ±50%)

    Returns:
        bool: True if candidate is within range

    Examples:
        >>> is_within_tier_range(75000, 100000, 0.5)  # 50K-150K range
        True
        >>> is_within_tier_range(200000, 100000, 0.5)
        False
    """
    lower_bound = int(seed_subs * (1 - tolerance))
    upper_bound = int(seed_subs * (1 + tolerance))

    return lower_bound <= candidate_subs <= upper_bound


# ============================================================================
# MAIN SIMILARITY SCORING
# ============================================================================

def calculate_similarity_score(candidate: dict, seed_profile: dict, debug: bool = False) -> dict:
    """
    Calculate multi-factor similarity score between candidate and seed channel.

    Scoring Breakdown (100 points total):
    - Tag Overlap (30 pts): Jaccard similarity on video tags
    - Keyword Overlap (30 pts): Jaccard similarity on keywords from titles
    - Subscriber Similarity (15 pts): Ratio-based scoring
    - Engagement Rate Similarity (17 pts): Absolute difference penalty
    - Upload Frequency Similarity (8 pts): Ratio-based scoring

    Args:
        candidate: Dict with keys: 'keywords', 'tags', 'subscribers',
                   'engagement_rate', 'upload_frequency'
        seed_profile: Dict from analyze_seed_channel() with keys: 'common_tags',
                      'primary_keywords', 'secondary_keywords', 'subscriber_count',
                      'avg_engagement_rate', 'upload_frequency'
        debug: If True, include detailed breakdown in result

    Returns:
        dict: {
            'total_score': float (0-100),
            'match_reasons': list[str],
            'breakdown': dict (if debug=True)
        }

    Example:
        >>> candidate = {
        ...     'tags': ['manga', 'anime', 'review'],
        ...     'subscribers': 50000,
        ...     'engagement_rate': 0.05
        ... }
        >>> seed = {
        ...     'common_tags': ['manga', 'anime', 'shonen'],
        ...     'subscriber_count': 100000,
        ...     'avg_engagement_rate': 0.04,
        ...     'upload_frequency': 2.0,
        ...     'primary_keywords': ['manga'],
        ...     'secondary_keywords': ['review']
        ... }
        >>> score = calculate_similarity_score(candidate, seed)
        >>> print(score['total_score'])  # Will be between 0-100
    """

    score = 0.0
    reasons = []
    breakdown = {}

    # ========================================================================
    # FACTOR 1: Tag Overlap (30 points)
    # ========================================================================

    # Safety check: ensure tags are lists
    candidate_tags = candidate.get('tags', [])
    if not isinstance(candidate_tags, list):
        candidate_tags = []

    seed_tags = seed_profile.get('common_tags', [])
    if not isinstance(seed_tags, list):
        seed_tags = []

    # When the seed has no tags, tag points are redistributed to keyword overlap
    # so channels can still reach a meaningful total score.
    seed_has_tags = bool(seed_tags)

    # If either has no tags, this factor scores 0
    if not candidate_tags or not seed_tags:
        tag_score = 0.0
        tag_overlap = 0.0
        common_tag_count = 0
    else:
        candidate_tags_set = set(candidate_tags)
        seed_tags_set = set(seed_tags)

        tag_overlap = jaccard_similarity(candidate_tags_set, seed_tags_set)
        common_tag_count = overlap_count(candidate_tags_set, seed_tags_set)

        tag_score = tag_overlap * SEED_WEIGHTS.tag_overlap

    score += tag_score

    if common_tag_count >= 5:
        reasons.append(f"Shares {common_tag_count} common tags")
    elif common_tag_count >= 3:
        reasons.append(f"Shares {common_tag_count} tags")

    breakdown['tag_score'] = round(tag_score, 1)
    breakdown['tag_overlap'] = round(tag_overlap, 2)
    breakdown['common_tags'] = common_tag_count

    # ========================================================================
    # FACTOR 2: Keyword Overlap (30 points, or 60 when seed has no tags)
    # When the seed channel has no tags the 30-point tag budget is folded into
    # this factor so the total can still reach 100.
    # ========================================================================

    # Safety check
    candidate_keywords = candidate.get('keywords', [])
    if not isinstance(candidate_keywords, list):
        candidate_keywords = []

    seed_keywords = (
        seed_profile.get('primary_keywords', []) +
        seed_profile.get('secondary_keywords', [])
    )

    keyword_max = (
        SEED_WEIGHTS.keyword_overlap + SEED_WEIGHTS.tag_overlap
        if not seed_has_tags
        else SEED_WEIGHTS.keyword_overlap
    )

    if not candidate_keywords or not seed_keywords:
        keyword_score = 0.0
        keyword_overlap = 0.0
        common_keyword_count = 0
    else:
        candidate_keywords_set = set(candidate_keywords)
        seed_keywords_set = set(seed_keywords)

        keyword_overlap = jaccard_similarity(candidate_keywords_set, seed_keywords_set)
        common_keyword_count = overlap_count(candidate_keywords_set, seed_keywords_set)

        keyword_score = keyword_overlap * keyword_max

    score += keyword_score

    if not seed_has_tags:
        reasons.append("Tags unavailable — keyword overlap weighted higher")

    if keyword_overlap >= 0.5:
        reasons.append(f"High keyword overlap ({keyword_overlap:.0%})")
    elif common_keyword_count >= 3:
        reasons.append(f"{common_keyword_count} matching keywords")

    breakdown['keyword_score'] = round(keyword_score, 1)
    breakdown['keyword_overlap'] = round(keyword_overlap, 2)
    breakdown['common_keywords'] = common_keyword_count

    # ========================================================================
    # FACTOR 3: Subscriber Similarity (15 points)
    # ========================================================================

    candidate_subs = candidate.get('subscribers', 0)
    seed_subs = seed_profile.get('subscriber_count', 0)

    sub_similarity = get_subscriber_similarity(candidate_subs, seed_subs)
    sub_score = sub_similarity * SEED_WEIGHTS.subscriber_similarity
    score += sub_score

    # Human-readable size comparison
    if is_within_tier_range(candidate_subs, seed_subs, tolerance=0.3):
        reasons.append(f"Very similar audience size ({candidate_subs:,} vs {seed_subs:,})")
    elif is_within_tier_range(candidate_subs, seed_subs, tolerance=0.6):
        reasons.append(f"Similar audience size ({candidate_subs:,} vs {seed_subs:,})")

    breakdown['subscriber_score'] = round(sub_score, 1)
    breakdown['subscriber_similarity'] = round(sub_similarity, 2)

    # ========================================================================
    # FACTOR 4: Engagement Rate Similarity (17 points)
    # ========================================================================

    candidate_engagement = candidate.get('engagement_rate', 0.0)
    seed_engagement = seed_profile.get('avg_engagement_rate', 0.0)

    # Calculate absolute difference
    engagement_diff = abs(candidate_engagement - seed_engagement)

    # Score inversely proportional to difference
    # diff=0.0 → score=max_points
    # diff=0.05 → score=half
    # diff=0.10+ → score=0
    max_engagement_pts = SEED_WEIGHTS.engagement_rate
    engagement_score = max(0, max_engagement_pts - (engagement_diff * max_engagement_pts * 10))
    score += engagement_score

    if engagement_score >= 12:
        reasons.append("Similar audience engagement")

    breakdown['engagement_score'] = round(engagement_score, 1)
    breakdown['engagement_diff'] = round(engagement_diff, 4)

    # ========================================================================
    # FACTOR 5: Upload Frequency Similarity (8 points)
    # ========================================================================

    candidate_freq = candidate.get('upload_frequency', 0.0)
    seed_freq = seed_profile.get('upload_frequency', 0.0)

    if seed_freq > 0 and candidate_freq > 0:
        freq_ratio = min(candidate_freq / seed_freq, seed_freq / candidate_freq)
        freq_score = freq_ratio * SEED_WEIGHTS.upload_frequency
    else:
        freq_ratio = 0.0
        freq_score = 0.0

    score += freq_score

    if freq_score >= 6:
        reasons.append("Similar upload schedule")

    breakdown['frequency_score'] = round(freq_score, 1)
    breakdown['frequency_ratio'] = round(freq_ratio, 2)

    # ========================================================================
    # RETURN
    # ========================================================================

    result = {
        'total_score': round(score, 1),
        'match_reasons': reasons
    }

    if debug:
        result['breakdown'] = breakdown

    return result


# ============================================================================
# GEMINI-ENHANCED SIMILARITY (OPTIONAL)
# ============================================================================

def gemini_similarity_analysis(
    candidate: dict,
    seed_profile: dict,
    gemini_api_key: str,
    callbacks: Optional[SimilarityCallbacks] = None
) -> dict:
    """
    Use Gemini to analyze "vibe" similarity between channels.

    Args:
        candidate: Candidate channel data
        seed_profile: Seed channel profile from analyze_seed_channel()
        gemini_api_key: Gemini API key
        callbacks: Optional callbacks for UI notifications and API tracking

    Returns:
        dict: {
            'gemini_score': int (0-10),
            'gemini_reason': str
        }
    """
    if callbacks is None:
        callbacks = _default_callbacks()

    if not gemini_api_key or not genai:
        return {'gemini_score': 0, 'gemini_reason': 'Gemini not configured'}

    try:
        # Track Gemini usage via callback
        if callbacks.debug_mode and callbacks.on_api_call:
            callbacks.on_api_call('gemini_similarity')

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        # Build comparison prompt
        prompt = f"""
You are a YouTube content strategist. Compare these two channels and rate their similarity.

SEED CHANNEL:
- Name: {seed_profile.get('channel_name', 'Unknown')}
- Subscribers: {seed_profile.get('subscriber_count', 0):,}
- Summary: {seed_profile.get('description_summary', 'N/A')}
- Recent titles: {', '.join(seed_profile.get('recent_titles', [])[:5])}
- Main topics: {', '.join(seed_profile.get('primary_keywords', []))}

CANDIDATE CHANNEL:
- Name: {candidate.get('channel_name') or candidate.get('channel_title', 'Unknown')}
- Subscribers: {candidate.get('subscribers', 0):,}
- Recent titles: {', '.join(candidate.get('recent_titles', [])[:5])}
- Topics: {', '.join(candidate.get('keywords', [])[:5])}

Rate their content similarity 0-10 considering:
1. Topic overlap (are they in the same niche?)
2. Content style (educational vs entertainment vs review etc.)
3. Target audience (age, interests, expertise level)
4. Production approach (vlogs vs tutorials vs commentary)

Respond in this exact format:
Score: X/10
Reason: [One clear sentence explaining the rating]
"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse response
        score_match = re.search(r'Score:\s*(\d+)/10', text)
        reason_match = re.search(r'Reason:\s*(.+)', text, re.IGNORECASE)

        gemini_score = int(score_match.group(1)) if score_match else 5
        gemini_reason = reason_match.group(1).strip() if reason_match else "Similar content focus"

        return {
            'gemini_score': gemini_score,
            'gemini_reason': gemini_reason
        }

    except Exception as e:
        # Notify via callback instead of st.warning
        if callbacks.on_warning:
            callbacks.on_warning(f"Gemini analysis failed: {e}")
        return {
            'gemini_score': 0,
            'gemini_reason': f'Analysis error: {str(e)}'
        }


# ============================================================================
# COMBINED SCORING (Algorithmic + AI)
# ============================================================================

def calculate_final_score(
    candidate: dict,
    seed_profile: dict,
    gemini_api_key: Optional[str] = None,
    debug: bool = False,
    callbacks: Optional[SimilarityCallbacks] = None
) -> dict:
    """
    Calculate final similarity score combining algorithmic analysis with AI.

    Uses blended scoring (80% algorithmic, 20% AI) when Gemini API key is available.
    Falls back to 100% algorithmic scoring silently if no API key.

    Args:
        candidate: Candidate channel data
        seed_profile: Seed channel profile from analyze_seed_channel()
        gemini_api_key: Optional Gemini API key for AI enhancement
        debug: If True, include detailed breakdown
        callbacks: Optional callbacks for UI notifications

    Returns:
        dict: {
            'total_score': float (0-100),
            'algorithmic_score': float (0-100),
            'gemini_score': int (0-10),
            'match_reasons': list[str],
            'gemini_reason': str,
            'breakdown': dict (if debug=True)
        }
    """
    if callbacks is None:
        callbacks = _default_callbacks()

    # Normalize field names (handle both channel_name and channel_title)
    if 'channel_title' in candidate and 'channel_name' not in candidate:
        candidate['channel_name'] = candidate['channel_title']

    # Calculate algorithmic score
    algo_result = calculate_similarity_score(candidate, seed_profile, debug=debug)

    result = {
        'algorithmic_score': algo_result['total_score'],
        'match_reasons': algo_result['match_reasons']
    }

    if debug:
        result['breakdown'] = algo_result.get('breakdown', {})

    # Apply blended scoring if Gemini API key is available
    if gemini_api_key:
        gemini_result = gemini_similarity_analysis(
            candidate, seed_profile, gemini_api_key, callbacks
        )

        result['gemini_score'] = gemini_result['gemini_score']
        result['gemini_reason'] = gemini_result['gemini_reason']

        # Only blend if Gemini analysis succeeded (score > 0)
        if gemini_result['gemini_score'] > 0:
            # Combine scores using configured blend ratio
            # Normalize Gemini score to 0-100 scale
            gemini_normalized = gemini_result['gemini_score'] * 10
            ai_blend = SEED_WEIGHTS.ai_blend_ratio

            combined_score = (
                algo_result['total_score'] * (1 - ai_blend) +
                gemini_normalized * ai_blend
            )

            result['total_score'] = round(combined_score, 1)

            # Add Gemini insight to reasons
            if gemini_result['gemini_score'] >= 8:
                result['match_reasons'].insert(0, f"AI: {gemini_result['gemini_reason']}")
        else:
            # Gemini failed - use algorithmic score only
            result['total_score'] = algo_result['total_score']
    else:
        # No Gemini API key - use algorithmic score only (silent fallback)
        result['total_score'] = algo_result['total_score']
        result['gemini_score'] = 0
        result['gemini_reason'] = 'Not configured'

    return result


# ============================================================================
# BATCH RANKING
# ============================================================================

def rank_channels_by_similarity(
    candidates: List[dict],
    seed_profile: dict,
    gemini_api_key: Optional[str] = None,
    gemini_limit: int = 10,
    debug: bool = False,
    callbacks: Optional[SimilarityCallbacks] = None
) -> List[dict]:
    """
    Rank all candidate channels by similarity to seed.

    Uses blended scoring (80% algorithmic, 20% AI) for the top N channels when
    Gemini API key is available. Falls back to 100% algorithmic scoring silently
    if no API key is provided.

    Args:
        candidates: List of dicts, each with: channel_id, channel_name,
                    keywords, tags, subscribers, etc.
        seed_profile: Output from analyze_seed_channel()
        gemini_api_key: Optional API key for AI enhancement
        gemini_limit: Only analyze top N channels with Gemini (default: 10)
        debug: If True, include detailed breakdown in similarity results
        callbacks: Optional callbacks for UI notifications

    Returns:
        List of candidates with added 'similarity' field, sorted by
        total_score descending
    """
    if callbacks is None:
        callbacks = _default_callbacks()

    # Notify start of ranking
    if callbacks.on_info:
        callbacks.on_info(f"🔢 Ranking {len(candidates)} candidates by similarity...")

    # First pass: Calculate algorithmic scores for ALL candidates
    for candidate in candidates:
        similarity = calculate_similarity_score(candidate, seed_profile, debug=debug)

        # Store algorithmic score
        similarity['algorithmic_score'] = similarity['total_score']
        similarity['gemini_score'] = 0
        similarity['gemini_reason'] = 'Not analyzed'

        candidate['similarity'] = similarity

    # Sort by algorithmic score
    candidates.sort(key=lambda x: x['similarity']['total_score'], reverse=True)

    # Second pass: Enhance top N with Gemini blended scoring (if API key available)
    if gemini_api_key:
        if callbacks.on_info:
            callbacks.on_info(f"✨ Enhancing top {gemini_limit} channels with AI analysis...")

        for candidate in candidates[:gemini_limit]:
            # Use calculate_final_score for blended scoring
            final_result = calculate_final_score(
                candidate,
                seed_profile,
                gemini_api_key=gemini_api_key,
                debug=debug,
                callbacks=callbacks
            )

            # Update candidate's similarity with blended results
            candidate['similarity']['total_score'] = final_result['total_score']
            candidate['similarity']['gemini_score'] = final_result.get('gemini_score', 0)
            candidate['similarity']['gemini_reason'] = final_result.get('gemini_reason', 'Not analyzed')
            candidate['similarity']['match_reasons'] = final_result['match_reasons']

            if debug and 'breakdown' in final_result:
                candidate['similarity']['breakdown'] = final_result['breakdown']

        # Re-sort after Gemini enhancement (rankings may have changed)
        candidates.sort(key=lambda x: x['similarity']['total_score'], reverse=True)

    # Log completion
    if candidates:
        top_channel_name = candidates[0].get('channel_name') or candidates[0].get('channel_title', 'Unknown')
        top_score = candidates[0]['similarity']['total_score']
        if callbacks.on_success:
            callbacks.on_success(f"✅ Ranking complete! Top match: {top_channel_name} ({top_score:.1f}/100)")
    else:
        if callbacks.on_warning:
            callbacks.on_warning("No candidates to rank")

    return candidates


# ============================================================================
# FILTER BY SUBSCRIBER TIER
# ============================================================================

def filter_by_subscriber_range(
    candidates: List[dict],
    seed_subs: int,
    tolerance: float = 0.5,
    callbacks: Optional[SimilarityCallbacks] = None
) -> List[dict]:
    """
    Filter candidates to only those within subscriber range.

    Args:
        candidates: List of candidate channel dicts
        seed_subs: Seed channel's subscriber count
        tolerance: Fractional tolerance (0.5 = ±50%)
        callbacks: Optional callbacks for UI notifications

    Returns:
        List of candidates within the subscriber range

    Example:
        >>> candidates = [{'subscribers': 75000}, {'subscribers': 200000}]
        >>> filtered = filter_by_subscriber_range(candidates, 100000, 0.5)
        >>> len(filtered)  # Only the 75K channel is in 50K-150K range
        1
    """
    if callbacks is None:
        callbacks = _default_callbacks()

    lower = int(seed_subs * (1 - tolerance))
    upper = int(seed_subs * (1 + tolerance))

    filtered = [
        c for c in candidates
        if lower <= c.get('subscribers', 0) <= upper
    ]

    if callbacks.on_info:
        callbacks.on_info(f"🎯 Filtered to {len(filtered)} channels in range {lower:,}-{upper:,} subs")

    return filtered


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def generate_match_explanation(candidate: dict, seed_profile: dict, detailed: bool = True) -> str:
    """
    Generate human-readable explanation of why a channel matches.

    Args:
        candidate: Candidate channel dict with 'similarity' field
        seed_profile: Seed channel profile (unused but kept for API consistency)
        detailed: If True, include full score breakdown

    Returns:
        str: Formatted markdown string

    Example:
        >>> candidate = {'similarity': {'total_score': 75.5, 'match_reasons': ['High overlap']}}
        >>> explanation = generate_match_explanation(candidate, {})
        >>> '75.5/100' in explanation
        True
    """

    similarity = candidate.get('similarity', {})
    total_score = similarity.get('total_score', 0)
    reasons = similarity.get('match_reasons', [])
    breakdown = similarity.get('breakdown', {})

    # Header
    explanation = f"## Similarity Score: {total_score:.1f}/100\n\n"

    # Main reasons
    if reasons:
        explanation += "**Why this channel matches:**\n"
        for reason in reasons:
            explanation += f"- {reason}\n"
        explanation += "\n"

    # Detailed breakdown (if requested)
    if detailed and breakdown:
        explanation += "**Score Breakdown:**\n"

        tag_score = breakdown.get('tag_score', 0)
        common_tags = breakdown.get('common_tags', 0)
        explanation += f"- **Tags** ({tag_score:.1f}/25): {common_tags} tags in common\n"

        keyword_score = breakdown.get('keyword_score', 0)
        common_keywords = breakdown.get('common_keywords', 0)
        explanation += f"- **Keywords** ({keyword_score:.1f}/30): {common_keywords} keywords match\n"

        sub_score = breakdown.get('subscriber_score', 0)
        explanation += f"- **Audience Size** ({sub_score:.1f}/20): Similar subscriber count\n"

        eng_score = breakdown.get('engagement_score', 0)
        explanation += f"- **Engagement** ({eng_score:.1f}/15): Similar interaction rates\n"

        freq_score = breakdown.get('frequency_score', 0)
        explanation += f"- **Upload Schedule** ({freq_score:.1f}/10): Similar posting frequency\n"

        explanation += "\n"

    # Gemini insight
    gemini_reason = similarity.get('gemini_reason')
    gemini_score = similarity.get('gemini_score', 0)

    if gemini_reason and gemini_reason != 'Not analyzed':
        explanation += f"**AI Analysis ({gemini_score}/10):** {gemini_reason}\n"

    return explanation

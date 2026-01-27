"""
similarity_engine.py - Multi-signal channel similarity scoring

Ranks candidate channels by similarity to seed channel
"""

import re
from typing import Optional
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from core.scoring_version import SEED_WEIGHTS
except ImportError:
    from .core.scoring_version import SEED_WEIGHTS


# ============================================================================
# SIMILARITY METRICS
# ============================================================================

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity coefficient
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Returns: 0.0 (no overlap) to 1.0 (identical)
    """
    set1 = set(set1)
    set2 = set(set2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def overlap_count(set1: set, set2: set) -> int:
    """Count items in common between two sets"""
    return len(set(set1) & set(set2))


# ============================================================================
# SUBSCRIBER TIER UTILITIES
# ============================================================================

def get_subscriber_similarity(candidate_subs: int, seed_subs: int) -> float:
    """
    Calculate subscriber count similarity (0.0 to 1.0)
    
    Uses ratio-based scoring:
    - Same size (1:1 ratio) = 1.0
    - 2x difference = 0.5
    - 10x difference = 0.1
    """
    if seed_subs == 0 or candidate_subs == 0:
        return 0.0
    
    ratio = min(candidate_subs / seed_subs, seed_subs / candidate_subs)
    
    # Convert ratio to score (logarithmic decay)
    # ratio=1.0 → score=1.0
    # ratio=0.5 → score=0.5
    # ratio=0.1 → score=0.1
    
    return ratio


def is_within_tier_range(candidate_subs: int, seed_subs: int, tolerance: float = 0.5) -> bool:
    """
    Check if candidate is within ±tolerance of seed's subscriber count
    
    tolerance=0.5 means 50K-150K range for a 100K seed
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

    Scoring Breakdown (100 points total - Proposed):
    - Tag Overlap (30 pts): Jaccard similarity on video tags. Strongest signal for topic alignment.
    - Keyword Overlap (30 pts): Jaccard similarity on keywords from titles. Equally important as tags.
    - Subscriber Similarity (15 pts): Ratio-based scoring. Important for finding peers.
    - Engagement Rate Similarity (17 pts): How audience interaction compares.
    - Upload Frequency Similarity (8 pts): How content velocity compares.
    
    Parameters:
    -----------
    candidate : dict
        Must have: 'keywords', 'tags', 'subscribers', 'engagement_rate', 'upload_frequency'
    
    seed_profile : dict
        Output from analyze_seed_channel_v2()
    
    debug : bool
        If True, include detailed breakdown
    
    Returns:
    --------
    {
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
        ...     'subscribers': 100000,
        ...     'engagement_rate': 0.04
        ... }
        >>> score = calculate_similarity_score(candidate, seed)
        >>> print(score['total_score'])
        67.5
        >>> print(score['match_reasons'])
        ['Strong tag overlap (67% similar)', 'Similar subscriber tier', ...]
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
    # FACTOR 2: Keyword Overlap (30 points)
    # ========================================================================

    # Safety check
    candidate_keywords = candidate.get('keywords', [])
    if not isinstance(candidate_keywords, list):
        candidate_keywords = []

    seed_keywords = (
        seed_profile.get('primary_keywords', []) +
        seed_profile.get('secondary_keywords', [])
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

        keyword_score = keyword_overlap * SEED_WEIGHTS.keyword_overlap

    score += keyword_score

    if keyword_overlap >= 0.5:
        reasons.append(f"High keyword overlap ({keyword_overlap:.0%})")
    elif common_keyword_count >= 3:
        reasons.append(f"{common_keyword_count} matching keywords")

    breakdown['keyword_score'] = round(keyword_score, 1)
    breakdown['keyword_overlap'] = round(keyword_overlap, 2)
    breakdown['common_keywords'] = common_keyword_count
    
    # ========================================================================
    # FACTOR 3: Subscriber Similarity (15 points) - 
    # ========================================================================
    
    candidate_subs = candidate.get('subscribers', 0)
    seed_subs = seed_profile['subscriber_count']
    
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
    # FACTOR 4: Engagement Rate Similarity (17 points) - 
    # ========================================================================
    
    candidate_engagement = candidate.get('engagement_rate', 0.0)
    seed_engagement = seed_profile['avg_engagement_rate']
    
    # Calculate absolute difference
    engagement_diff = abs(candidate_engagement - seed_engagement)
    
    # Score inversely proportional to difference
    # diff=0.0 → score=max_points
    # diff=0.05 → score=half
    # diff=0.10+ → score=0
    max_engagement_pts = SEED_WEIGHTS.engagement_rate
    engagement_score = max(0, max_engagement_pts - (engagement_diff * max_engagement_pts * 10))
    score += engagement_score
    
    if engagement_score >= 12: # Adjusted threshold
        reasons.append("Similar audience engagement")
    
    breakdown['engagement_score'] = round(engagement_score, 1)
    breakdown['engagement_diff'] = round(engagement_diff, 4)
    
    # ========================================================================
    # FACTOR 5: Upload Frequency Similarity (8 points) 
    # ======================================================================== 
    
    candidate_freq = candidate.get('upload_frequency', 0.0)
    seed_freq = seed_profile['upload_frequency']
    
    if seed_freq > 0 and candidate_freq > 0:
        freq_ratio = min(candidate_freq / seed_freq, seed_freq / candidate_freq)
        freq_score = freq_ratio * SEED_WEIGHTS.upload_frequency
    else:
        freq_score = 0.0
    
    score += freq_score
    
    if freq_score >= 6:  # Adjusted threshold
        reasons.append("Similar upload schedule")
    
    breakdown['frequency_score'] = round(freq_score, 1)
    breakdown['frequency_ratio'] = round(freq_ratio, 2) if seed_freq > 0 and candidate_freq > 0 else 0.0
    
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
    gemini_api_key: str
) -> dict:
    """
    Use Gemini to analyze "vibe" similarity between channels
    
    Returns:
    {
        'gemini_score': int (0-10),
        'gemini_reason': str
    }
    """
    
    if not gemini_api_key or not genai:
        return {'gemini_score': 0, 'gemini_reason': 'Gemini not configured'}
    
    try:
        # Track Gemini usage
        import streamlit as st
        if st.session_state.get('debug_mode', False):
            # Import here to avoid circular dependency
            try:
                from . import debug_tracker
            except ImportError:
                import debug_tracker
            debug_tracker.track_api_call('gemini_similarity')
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Build comparison prompt
        prompt = f"""
You are a YouTube content strategist. Compare these two channels and rate their similarity.

SEED CHANNEL:
- Name: {seed_profile['channel_name']}
- Subscribers: {seed_profile['subscriber_count']:,}
- Summary: {seed_profile.get('description_summary', 'N/A')}
- Recent titles: {', '.join(seed_profile['recent_titles'][:5])}
- Main topics: {', '.join(seed_profile['primary_keywords'])}

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
        st.warning(f"Gemini analysis failed: {e}")
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
    debug: bool = False
) -> dict:
    """
    Calculate final similarity score combining algorithmic analysis with AI.

    Uses blended scoring (80% algorithmic, 20% AI) when Gemini API key is available.
    Falls back to 100% algorithmic scoring silently if no API key.

    Returns:
    {
        'total_score': float (0-100),
        'algorithmic_score': float (0-100),
        'gemini_score': int (0-10),
        'match_reasons': list[str],
        'gemini_reason': str,
        'breakdown': dict (if debug=True)
    }
    """
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
        gemini_result = gemini_similarity_analysis(candidate, seed_profile, gemini_api_key)

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
    candidates: list[dict],
    seed_profile: dict,
    gemini_api_key: Optional[str] = None,
    gemini_limit: int = 10,
    debug: bool = False
) -> list[dict]:
    """
    Rank all candidate channels by similarity to seed.

    Uses blended scoring (80% algorithmic, 20% AI) for the top N channels when
    Gemini API key is available. Falls back to 100% algorithmic scoring silently
    if no API key is provided.

    Parameters:
    -----------
    candidates : list[dict]
        Each must have: channel_id, channel_name, keywords, tags, subscribers, etc.

    seed_profile : dict
        Output from analyze_seed_channel_v2()

    gemini_api_key : str, optional
        If provided, enables blended scoring for top candidates

    gemini_limit : int
        Only analyze top N channels with Gemini (to save API calls). Default: 10

    debug : bool
        If True, include detailed breakdown in similarity results

    Returns:
    --------
    List of candidates with added 'similarity' field, sorted by total_score descending
    """

    st.info(f"🔢 Ranking {len(candidates)} candidates by similarity...")

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
        st.info(f"✨ Enhancing top {gemini_limit} channels with AI analysis...")

        for candidate in candidates[:gemini_limit]:
            # Use calculate_final_score for blended scoring
            final_result = calculate_final_score(
                candidate,
                seed_profile,
                gemini_api_key=gemini_api_key,
                debug=debug
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
        st.success(f"✅ Ranking complete! Top match: {top_channel_name} ({top_score:.1f}/100)")
    else:
        st.warning("No candidates to rank")

    return candidates


# ============================================================================
# FILTER BY SUBSCRIBER TIER
# ============================================================================

def filter_by_subscriber_range(
    candidates: list[dict],
    seed_subs: int,
    tolerance: float = 0.5
) -> list[dict]:
    """
    Filter candidates to only those within subscriber range
    
    tolerance=0.5 means ±50% of seed's subscriber count
    """
    
    lower = int(seed_subs * (1 - tolerance))
    upper = int(seed_subs * (1 + tolerance))
    
    filtered = [
        c for c in candidates
        if lower <= c.get('subscribers', 0) <= upper
    ]
    
    st.info(f"🎯 Filtered to {len(filtered)} channels in range {lower:,}-{upper:,} subs")
    
    return filtered


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def generate_match_explanation(candidate: dict, seed_profile: dict, detailed: bool = True) -> str:
    """
    Generate human-readable explanation of why a channel matches
    
    Returns formatted markdown string
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

"""
synthetic_data_generator.py - Generate realistic feedback data for ML training

This module creates synthetic feedback data that models realistic user behavior
patterns for demonstration and testing of the ML pipeline.

IMPORTANT: This is synthetic data for portfolio demonstration.
Production would use real user feedback collected over time.

Documented Assumptions:
-----------------------
1. Tag overlap and keyword overlap are strongest predictors of satisfaction
   - tag_score > 20 -> 75% relevant
   - tag_score < 10 -> 30% relevant

2. Presented rank affects relevance (position bias)
   - rank 1: 70% base relevance
   - rank 5: 50% base relevance

3. AI enhancement provides ~5-10% lift in relevance rate

4. Negative feedback reasons distribution:
   - wrong_topic: 45%
   - low_quality: 25%
   - poor_fit: 20%
   - other: 10%

5. Score distributions (seed mode):
   - tag_score: Normal(mu=18, sigma=8), clipped to [0, 30]
   - keyword_score: Normal(mu=16, sigma=9), clipped to [0, 30]
   - subscriber_score: Normal(mu=10, sigma=4), clipped to [0, 15]
   - engagement_score: Normal(mu=11, sigma=5), clipped to [0, 17]
   - frequency_score: Normal(mu=5, sigma=2), clipped to [0, 8]

6. Edge case: ~15% of channels don't use tags (tag_score = 0 or very low)
   - These channels may still be relevant if keyword_score is high
   - Models real-world behavior where some creators skip video tags
"""

import random
from datetime import datetime
from typing import Optional
import sys
import os

# NumPy import with availability check
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Add parent directory for imports when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..core.scoring_version import (
        SEED_WEIGHTS,
        CHANNEL_FEEDBACK_REASONS,
        VALID_RATINGS,
        get_scoring_version,
    )
except ImportError:
    from core.scoring_version import (
        SEED_WEIGHTS,
        CHANNEL_FEEDBACK_REASONS,
        VALID_RATINGS,
        get_scoring_version,
    )

try:
    from .feedback_tracker import save_channel_feedback, build_channel_feedback_entry
except ImportError:
    from feedback_tracker import save_channel_feedback, build_channel_feedback_entry


def _clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def _normal(mean: float, std: float) -> float:
    """Generate a normally distributed random value."""
    if HAS_NUMPY:
        return np.random.normal(mean, std)
    else:
        # Fallback using Box-Muller transform
        import math
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mean + std * z


def _generate_component_scores(
    target_relevance: float,
    ai_enabled: bool = True,
    no_tags: bool = False,
) -> dict:
    """
    Generate realistic component scores for seed mode.

    Higher target_relevance produces higher scores on average.

    Args:
        target_relevance: Probability that this should be relevant (0-1)
        ai_enabled: Whether AI scoring is available
        no_tags: If True, simulate a channel that doesn't use tags

    Returns:
        dict with all component scores
    """
    # Base means shift based on target relevance
    # If target_relevance is high, scores tend to be higher
    relevance_boost = (target_relevance - 0.5) * 10

    # Generate tag score - handle no-tags edge case
    if no_tags:
        # Channel doesn't use tags: score is 0 or very low (0-3)
        tag_score = _clip(_normal(1, 1.5), 0, 3)
    else:
        tag_score = _clip(
            _normal(18 + relevance_boost, 8), 0, 30
        )

    # For no-tags channels, keyword score might compensate
    # (good title keywords can still make channel relevant)
    keyword_boost = relevance_boost
    if no_tags:
        # Slight boost to keyword score for no-tags channels that are still relevant
        keyword_boost = relevance_boost * 1.2

    keyword_score = _clip(
        _normal(16 + keyword_boost * 0.8, 9), 0, 30
    )
    subscriber_score = _clip(
        _normal(10 + relevance_boost * 0.3, 4), 0, 15
    )
    engagement_score = _clip(
        _normal(11 + relevance_boost * 0.4, 5), 0, 17
    )
    frequency_score = _clip(
        _normal(5 + relevance_boost * 0.2, 2), 0, 8
    )

    # Calculate algorithmic total
    algorithmic_score = (
        tag_score + keyword_score + subscriber_score +
        engagement_score + frequency_score
    )

    # Gemini score (if AI enabled)
    if ai_enabled:
        # Gemini generally agrees with algorithmic but with noise
        gemini_base = (algorithmic_score / 100) * 10  # Scale to 0-10
        gemini_score = _clip(
            _normal(gemini_base, 1.5), 0, 10
        )
    else:
        gemini_score = None

    return {
        "tag_score": round(tag_score, 2),
        "keyword_score": round(keyword_score, 2),
        "subscriber_score": round(subscriber_score, 2),
        "engagement_score": round(engagement_score, 2),
        "frequency_score": round(frequency_score, 2),
        "algorithmic_score": round(algorithmic_score, 2),
        "gemini_score": round(gemini_score, 2) if gemini_score is not None else None,
    }


def _determine_rating(
    component_scores: dict,
    presented_rank: int,
    ai_enabled: bool,
    no_tags: bool = False,
) -> tuple[str, Optional[str]]:
    """
    Determine rating based on scores with realistic noise.

    Args:
        component_scores: Dict with scoring components
        presented_rank: Position in results (1-5)
        ai_enabled: Whether AI was enabled
        no_tags: Whether this is a no-tags channel

    Returns:
        (rating, reason) tuple
    """
    tag_score = component_scores["tag_score"]
    keyword_score = component_scores["keyword_score"]
    algorithmic_score = component_scores["algorithmic_score"]

    # Base probability from scores
    # Tag and keyword are strongest predictors
    if no_tags:
        # For no-tags channels, rely more heavily on keyword score
        score_factor = (
            keyword_score / 30 * 0.5 +
            algorithmic_score / 100 * 0.5
        )
    else:
        score_factor = (
            tag_score / 30 * 0.4 +
            keyword_score / 30 * 0.3 +
            algorithmic_score / 100 * 0.3
        )

    # Position bias (rank 1 more likely rated relevant)
    position_factor = 1 - (presented_rank - 1) * 0.05

    # AI boost
    ai_factor = 1.05 if ai_enabled else 1.0

    # Final probability
    relevance_prob = _clip(score_factor * position_factor * ai_factor, 0.1, 0.95)

    # Determine rating
    roll = random.random()

    if roll < relevance_prob:
        return ("relevant", None)
    elif roll < relevance_prob + 0.05:  # 5% skip rate
        return ("skip", None)
    else:
        # Not relevant - pick a reason
        reason_roll = random.random()
        if reason_roll < 0.45:
            reason = "wrong_topic"
        elif reason_roll < 0.70:
            reason = "low_quality"
        elif reason_roll < 0.90:
            reason = "poor_fit"
        else:
            reason = "other"
        return ("not_relevant", reason)


def _generate_fake_channel(rank: int) -> dict:
    """Generate a fake channel with plausible metadata."""
    channel_id = f"UC{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=22))}"

    adjectives = ["Gaming", "Tech", "Creative", "Daily", "Pro", "Casual", "Epic", "Retro"]
    nouns = ["Reviews", "Plays", "Studio", "Channel", "Hub", "Zone", "Central", "World"]

    channel_name = f"{random.choice(adjectives)} {random.choice(nouns)} {random.randint(1, 999)}"
    channel_url = f"https://youtube.com/channel/{channel_id}"

    return {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_url": channel_url,
    }


def generate_search_feedback(
    search_mode: str = "seed",
    ai_enabled: bool = True,
    quality_level: str = "mixed",  # "good", "bad", or "mixed"
    no_tags_probability: float = 0.15,  # 15% of channels don't use tags
) -> dict:
    """
    Generate a single search feedback entry with 5 channel ratings.

    Args:
        search_mode: "seed" or "keyword"
        ai_enabled: Whether AI enhancement was used
        quality_level: Controls overall quality of results
            - "good": Mostly relevant results
            - "bad": Mostly irrelevant results
            - "mixed": Realistic distribution
        no_tags_probability: Probability that a channel doesn't use tags (0-1)

    Returns:
        dict ready for save_channel_feedback()
    """
    # Generate query
    topics = [
        "gaming commentary", "tech reviews", "cooking tutorials",
        "fitness motivation", "travel vlog", "music production",
        "DIY crafts", "financial advice", "language learning",
        "car restoration", "book reviews", "photography tips"
    ]
    query = random.choice(topics)

    # Results count
    results_count = random.randint(8, 50)

    # Quality level affects target relevance
    if quality_level == "good":
        base_relevance = 0.75
    elif quality_level == "bad":
        base_relevance = 0.25
    else:
        base_relevance = 0.55

    # Generate 5 channel feedback entries
    channel_feedback = []
    for rank in range(1, 6):
        channel = _generate_fake_channel(rank)

        # Higher ranks get slightly better scores on average
        rank_adjustment = (5 - rank) * 0.03
        target_relevance = base_relevance + rank_adjustment

        # Determine if this channel uses tags (edge case)
        no_tags = random.random() < no_tags_probability

        component_scores = _generate_component_scores(
            target_relevance, ai_enabled, no_tags=no_tags
        )

        # Calculate presented score
        algo_score = component_scores["algorithmic_score"]
        gemini_score = component_scores["gemini_score"]

        if ai_enabled and gemini_score is not None:
            presented_score = algo_score * 0.8 + gemini_score * 10 * 0.2
        else:
            presented_score = algo_score

        rating, reason = _determine_rating(
            component_scores, rank, ai_enabled, no_tags=no_tags
        )

        entry = build_channel_feedback_entry(
            channel_id=channel["channel_id"],
            channel_name=channel["channel_name"],
            channel_url=channel["channel_url"],
            presented_rank=rank,
            presented_score=round(presented_score, 2),
            rating=rating,
            component_scores=component_scores,
            reason=reason,
        )
        channel_feedback.append(entry)

    return {
        "search_mode": search_mode,
        "query": query,
        "results_count": results_count,
        "channel_feedback": channel_feedback,
        "seed_channel_id": f"UC{''.join(random.choices('0123456789', k=22))}" if search_mode == "seed" else None,
        "seed_channel_name": f"Seed Channel {random.randint(1, 100)}" if search_mode == "seed" else None,
        "filters": {
            "min_subscribers": random.choice([1000, 5000, 10000, 50000]),
            "country_filter": random.choice([None, "US", "GB", "CA"]),
            "months_ago": random.choice([3, 6, 12, None]),
        },
        "ai_enabled": ai_enabled,
    }


def generate_synthetic_feedback(
    n_searches: int = 60,
    seed_mode_ratio: float = 0.7,
    ai_enabled_ratio: float = 0.8,
    no_tags_probability: float = 0.15,
    save_to_file: bool = True,
) -> list[dict]:
    """
    Generate synthetic feedback dataset for ML training.

    This creates a realistic distribution of feedback with:
    - Mix of seed and keyword mode searches
    - Mix of AI-enabled and non-AI searches
    - Mix of good, bad, and mixed quality results
    - Realistic correlations between scores and ratings
    - Edge cases: ~15% of channels don't use tags

    Args:
        n_searches: Number of search feedback entries to generate
        seed_mode_ratio: Fraction of searches in seed mode (0-1)
        ai_enabled_ratio: Fraction with AI enabled (0-1)
        no_tags_probability: Probability that a channel doesn't use tags (0-1)
        save_to_file: If True, saves via feedback_tracker

    Returns:
        List of generated feedback entries

    Example:
        >>> entries = generate_synthetic_feedback(n_searches=100, save_to_file=True)
        >>> print(f"Generated {len(entries)} search entries")
        >>> print(f"Total channel ratings: {sum(len(e['channel_feedback']) for e in entries)}")
    """
    entries = []

    # Quality distribution: 20% good, 20% bad, 60% mixed
    quality_weights = ["good"] * 20 + ["bad"] * 20 + ["mixed"] * 60

    for i in range(n_searches):
        # Determine search parameters
        search_mode = "seed" if random.random() < seed_mode_ratio else "keyword"
        ai_enabled = random.random() < ai_enabled_ratio
        quality_level = random.choice(quality_weights)

        # Generate feedback
        feedback_data = generate_search_feedback(
            search_mode=search_mode,
            ai_enabled=ai_enabled,
            quality_level=quality_level,
            no_tags_probability=no_tags_probability,
        )

        entries.append(feedback_data)

        # Save to file if requested
        if save_to_file:
            save_channel_feedback(
                search_mode=feedback_data["search_mode"],
                query=feedback_data["query"],
                results_count=feedback_data["results_count"],
                channel_feedback=feedback_data["channel_feedback"],
                seed_channel_id=feedback_data.get("seed_channel_id"),
                seed_channel_name=feedback_data.get("seed_channel_name"),
                filters=feedback_data.get("filters"),
                ai_enabled=feedback_data.get("ai_enabled"),
            )

    return entries


def get_synthetic_data_summary(entries: list[dict]) -> dict:
    """
    Get summary statistics of generated synthetic data.

    Args:
        entries: List of feedback entries from generate_synthetic_feedback()

    Returns:
        dict with summary statistics
    """
    total_channels = sum(len(e["channel_feedback"]) for e in entries)

    ratings = {"relevant": 0, "not_relevant": 0, "skip": 0}
    reasons = {"wrong_topic": 0, "low_quality": 0, "poor_fit": 0, "other": 0}

    # Track no-tags edge cases
    no_tags_count = 0
    no_tags_relevant = 0

    for entry in entries:
        for cf in entry["channel_feedback"]:
            ratings[cf["rating"]] += 1
            if cf["reason"]:
                reasons[cf["reason"]] += 1

            # Check for no-tags edge case (tag_score <= 3)
            component_scores = cf.get("component_scores", {})
            tag_score = component_scores.get("tag_score", 0)
            if tag_score <= 3:
                no_tags_count += 1
                if cf["rating"] == "relevant":
                    no_tags_relevant += 1

    seed_count = sum(1 for e in entries if e["search_mode"] == "seed")
    ai_count = sum(1 for e in entries if e.get("ai_enabled"))

    return {
        "total_searches": len(entries),
        "total_channel_ratings": total_channels,
        "seed_mode_searches": seed_count,
        "keyword_mode_searches": len(entries) - seed_count,
        "ai_enabled_searches": ai_count,
        "rating_distribution": ratings,
        "reason_distribution": reasons,
        "relevance_rate": ratings["relevant"] / total_channels if total_channels > 0 else 0,
        "no_tags_channels": no_tags_count,
        "no_tags_relevance_rate": no_tags_relevant / no_tags_count if no_tags_count > 0 else 0,
    }


if __name__ == "__main__":
    # Demo: Generate synthetic data and print summary
    print("Generating synthetic feedback data...")
    print("(Including ~15% channels without tags as edge case)")
    entries = generate_synthetic_feedback(n_searches=60, save_to_file=False)

    summary = get_synthetic_data_summary(entries)
    print(f"\n=== Synthetic Data Summary ===")
    print(f"Total searches: {summary['total_searches']}")
    print(f"Total channel ratings: {summary['total_channel_ratings']}")
    print(f"Seed mode: {summary['seed_mode_searches']}")
    print(f"Keyword mode: {summary['keyword_mode_searches']}")
    print(f"AI enabled: {summary['ai_enabled_searches']}")
    print(f"Relevance rate: {summary['relevance_rate']:.1%}")
    print(f"\nRating distribution: {summary['rating_distribution']}")
    print(f"Reason distribution: {summary['reason_distribution']}")
    print(f"\n=== Edge Case: No-Tags Channels ===")
    print(f"Channels without tags: {summary['no_tags_channels']}")
    print(f"No-tags relevance rate: {summary['no_tags_relevance_rate']:.1%}")

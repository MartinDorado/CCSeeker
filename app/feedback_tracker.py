"""
feedback_tracker.py - User feedback collection for search results

Collects and stores user feedback on search quality to help improve
similarity and relevance scoring algorithms.
"""

import json
import os
from datetime import datetime
from typing import Optional

# Feedback storage file path (in app directory)
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), ".feedback_data.json")

# Reason codes for negative feedback
FEEDBACK_REASONS = {
    "few_results": "Few results",
    "low_quality": "Low quality content",
    "wrong_topic": "Wrong topic/niche",
    "other": "Other"
}


def _load_feedback_data() -> dict:
    """Load existing feedback data from JSON file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Corrupted or unreadable file - start fresh
            return {"feedback_entries": []}
    return {"feedback_entries": []}


def _save_feedback_data(data: dict) -> bool:
    """Save feedback data to JSON file."""
    try:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


def save_feedback(
    feedback: str,
    search_mode: str,
    query: str,
    results_count: int,
    top_results: list[dict],
    reason: Optional[str] = None,
    seed_channel_id: Optional[str] = None,
    seed_channel_name: Optional[str] = None,
    filters: Optional[dict] = None,
    ai_enabled: Optional[bool] = None,
    scoring_context: Optional[dict] = None
) -> bool:
    """
    Save user feedback for a search.

    Parameters:
    -----------
    feedback : str
        "up" for positive, "down" for negative

    search_mode : str
        "seed" or "keyword"

    query : str
        The search query used

    results_count : int
        Total number of results returned

    top_results : list[dict]
        Top 5 results with channel_id, channel_name, channel_url, and score

    reason : str, optional
        Reason code for negative feedback (few_results, low_quality, wrong_topic, other)

    seed_channel_id : str, optional
        Channel ID of seed (if seed mode)

    seed_channel_name : str, optional
        Channel name of seed (if seed mode)

    filters : dict, optional
        Search filter settings used (min_subscribers, country_filter, months_ago, region)

    ai_enabled : bool, optional
        Whether AI enhancement was enabled for this search

    scoring_context : dict, optional
        For seed mode: similarity scoring details for top result
        {
            'top_result_total_score': float,
            'top_result_algorithmic_score': float,
            'top_result_gemini_score': float,
            'score_distribution': {'max': float, 'min': float, 'avg': float}
        }

    Returns:
    --------
    bool: True if saved successfully
    """
    data = _load_feedback_data()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "search_mode": search_mode,
        "query": query,
        "results_count": results_count,
        "top_results": top_results[:5],  # Store top 5 for analysis
        "feedback": feedback,
        "reason": reason
    }

    # Add seed info if applicable
    if search_mode == "seed":
        entry["seed_channel_id"] = seed_channel_id
        entry["seed_channel_name"] = seed_channel_name

    # Add filter settings if provided
    if filters:
        entry["filters"] = filters

    # Add AI enabled flag if provided
    if ai_enabled is not None:
        entry["ai_enabled"] = ai_enabled

    # Add scoring context if provided (seed mode)
    if scoring_context:
        entry["scoring_context"] = scoring_context

    data["feedback_entries"].append(entry)

    return _save_feedback_data(data)


def get_feedback_stats() -> dict:
    """
    Get summary statistics of collected feedback.

    Returns:
    --------
    dict with keys:
        - total_entries: int
        - positive_count: int
        - negative_count: int
        - reason_breakdown: dict[str, int]
        - by_search_mode: dict[str, dict]
    """
    data = _load_feedback_data()
    entries = data.get("feedback_entries", [])

    stats = {
        "total_entries": len(entries),
        "positive_count": 0,
        "negative_count": 0,
        "reason_breakdown": {
            "few_results": 0,
            "low_quality": 0,
            "wrong_topic": 0,
            "other": 0
        },
        "by_search_mode": {
            "seed": {"positive": 0, "negative": 0},
            "keyword": {"positive": 0, "negative": 0}
        }
    }

    for entry in entries:
        feedback = entry.get("feedback")
        mode = entry.get("search_mode", "keyword")
        reason = entry.get("reason")

        if feedback == "up":
            stats["positive_count"] += 1
            if mode in stats["by_search_mode"]:
                stats["by_search_mode"][mode]["positive"] += 1
        elif feedback == "down":
            stats["negative_count"] += 1
            if mode in stats["by_search_mode"]:
                stats["by_search_mode"][mode]["negative"] += 1
            if reason and reason in stats["reason_breakdown"]:
                stats["reason_breakdown"][reason] += 1

    return stats


def get_negative_feedback_entries(limit: int = 50) -> list[dict]:
    """
    Get recent negative feedback entries for analysis.

    Parameters:
    -----------
    limit : int
        Maximum number of entries to return

    Returns:
    --------
    List of negative feedback entries, most recent first
    """
    data = _load_feedback_data()
    entries = data.get("feedback_entries", [])

    negative = [e for e in entries if e.get("feedback") == "down"]

    # Sort by timestamp descending (most recent first)
    negative.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return negative[:limit]


def export_feedback_csv(filepath: str) -> bool:
    """
    Export all feedback to CSV for analysis.

    Parameters:
    -----------
    filepath : str
        Path to save the CSV file

    Returns:
    --------
    bool: True if exported successfully
    """
    import csv

    data = _load_feedback_data()
    entries = data.get("feedback_entries", [])

    if not entries:
        return False

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "timestamp", "search_mode", "query", "results_count",
                "feedback", "reason", "seed_channel_id", "seed_channel_name",
                "ai_enabled", "min_subscribers", "country_filter", "months_ago", "region",
                "top_result_total_score", "top_result_algorithmic_score", "top_result_gemini_score",
                "score_dist_max", "score_dist_min", "score_dist_avg",
                "top_result_1_name", "top_result_1_id", "top_result_1_url", "top_result_1_score",
                "top_result_2_name", "top_result_2_id", "top_result_2_url", "top_result_2_score",
                "top_result_3_name", "top_result_3_id", "top_result_3_url", "top_result_3_score"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                filters = entry.get("filters", {})
                scoring_context = entry.get("scoring_context", {})
                score_dist = scoring_context.get("score_distribution", {})
                row = {
                    "timestamp": entry.get("timestamp"),
                    "search_mode": entry.get("search_mode"),
                    "query": entry.get("query"),
                    "results_count": entry.get("results_count"),
                    "feedback": entry.get("feedback"),
                    "reason": entry.get("reason"),
                    "seed_channel_id": entry.get("seed_channel_id"),
                    "seed_channel_name": entry.get("seed_channel_name"),
                    "ai_enabled": entry.get("ai_enabled"),
                    "min_subscribers": filters.get("min_subscribers"),
                    "country_filter": filters.get("country_filter"),
                    "months_ago": filters.get("months_ago"),
                    "region": filters.get("region"),
                    "top_result_total_score": scoring_context.get("top_result_total_score"),
                    "top_result_algorithmic_score": scoring_context.get("top_result_algorithmic_score"),
                    "top_result_gemini_score": scoring_context.get("top_result_gemini_score"),
                    "score_dist_max": score_dist.get("max"),
                    "score_dist_min": score_dist.get("min"),
                    "score_dist_avg": score_dist.get("avg")
                }

                # Add top 3 results as separate columns for name, id, url, and score
                top_results = entry.get("top_results", [])
                for i in range(3):
                    if i < len(top_results):
                        r = top_results[i]
                        row[f"top_result_{i+1}_name"] = r.get('channel_name', '')
                        row[f"top_result_{i+1}_id"] = r.get('channel_id', '')
                        row[f"top_result_{i+1}_url"] = r.get('channel_url', '')
                        row[f"top_result_{i+1}_score"] = r.get('score', '')
                    else:
                        row[f"top_result_{i+1}_name"] = ""
                        row[f"top_result_{i+1}_id"] = ""
                        row[f"top_result_{i+1}_url"] = ""
                        row[f"top_result_{i+1}_score"] = ""

                writer.writerow(row)

        return True
    except IOError:
        return False

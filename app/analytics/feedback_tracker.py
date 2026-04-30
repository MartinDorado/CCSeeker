"""
feedback_tracker.py - Per-channel user feedback collection for search results

Collects and stores user feedback at the individual channel level (top 5 results)
to enable ML-based improvements to similarity and relevance scoring algorithms.

Schema version: 2.0.0 - Per-channel feedback with version signatures
"""

import json
import os
from datetime import datetime
from typing import Optional, Literal, Protocol, runtime_checkable

try:
    from ..core.scoring_version import (
        CHANNEL_FEEDBACK_REASONS,
        VALID_RATINGS,
        get_scoring_version,
        is_version_compatible,
    )
except ImportError:
    from core.scoring_version import (
        CHANNEL_FEEDBACK_REASONS,
        VALID_RATINGS,
        get_scoring_version,
        is_version_compatible,
    )

# Feedback storage file path (in app directory)
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".feedback_data.json")

# Schema version for this feedback format
FEEDBACK_SCHEMA_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# FeedbackStore Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FeedbackStore(Protocol):
    def save_entry(self, entry: dict) -> bool: ...
    def load_entries(self, mode: str | None = None) -> list[dict]: ...
    def clear_all(self) -> bool: ...
    def clear_incompatible(self) -> int: ...


# ---------------------------------------------------------------------------
# JSONFeedbackStore — file-based implementation
# ---------------------------------------------------------------------------

class JSONFeedbackStore:
    """File-based feedback store backed by a local JSON file."""

    def __init__(self, filepath: str = None):
        self._filepath = filepath if filepath is not None else FEEDBACK_FILE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load raw JSON data from file."""
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"schema_version": FEEDBACK_SCHEMA_VERSION, "feedback_entries": []}
        return {"schema_version": FEEDBACK_SCHEMA_VERSION, "feedback_entries": []}

    def _dump(self, data: dict) -> bool:
        """Write raw JSON data to file."""
        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def save_entry(self, entry: dict) -> bool:
        """Append an entry to the JSON file."""
        data = self._load()
        data["feedback_entries"].append(entry)
        return self._dump(data)

    def load_entries(self, mode: str | None = None) -> list[dict]:
        """Load all entries, optionally filtered by search_mode."""
        data = self._load()
        entries = data.get("feedback_entries", [])
        if mode is not None:
            entries = [e for e in entries if e.get("search_mode") == mode]
        return entries

    def clear_all(self) -> bool:
        """Reset the store to an empty state."""
        data = {"schema_version": FEEDBACK_SCHEMA_VERSION, "feedback_entries": []}
        return self._dump(data)

    def clear_incompatible(self) -> int:
        """Remove entries incompatible with the current scoring version. Returns count removed."""
        data = self._load()
        entries = data.get("feedback_entries", [])
        original_count = len(entries)
        compatible = [
            e for e in entries
            if is_version_compatible(
                e.get("scoring_version", {}),
                e.get("search_mode", "keyword"),
            )
        ]
        data["feedback_entries"] = compatible
        self._dump(data)
        return original_count - len(compatible)


# ---------------------------------------------------------------------------
# SupabaseFeedbackStore — Supabase/Postgres backend
# ---------------------------------------------------------------------------

class SupabaseFeedbackStore:
    """
    Supabase (Postgres) backend for feedback.

    Requires two env vars: SUPABASE_URL and SUPABASE_SERVICE_KEY.
    If the supabase package is not installed, raises ImportError.

    Two-table schema:
    - feedback_entries  (parent — one row per search submission)
    - channel_feedback  (child  — one row per channel rating, FK entry_id)

    The 'channel_feedback' column in the joined result is the list of
    per-channel dicts, identical in shape to the JSON format.
    """

    def __init__(self, url: str, key: str):
        from supabase import create_client  # noqa: PLC0415 — ImportError intentional
        self._db = create_client(url, key)

    def save_entry(self, entry: dict) -> bool:
        """Insert a feedback entry (and its per-channel rows) into Supabase."""
        try:
            parent_row = {
                "timestamp": entry.get("timestamp"),
                "search_mode": entry.get("search_mode"),
                "query": entry.get("query"),
                "results_count": entry.get("results_count"),
                "scoring_version": entry.get("scoring_version"),
                "seed_channel_id": entry.get("seed_channel_id"),
                "seed_channel_name": entry.get("seed_channel_name"),
                "filters": entry.get("filters"),
                "ai_enabled": entry.get("ai_enabled"),
            }
            resp = self._db.table("feedback_entries").insert(parent_row).execute()
            entry_id = resp.data[0]["id"]

            cf_list = entry.get("channel_feedback", [])
            if cf_list:
                cf_rows = [
                    {
                        "entry_id": entry_id,
                        "channel_id": cf.get("channel_id"),
                        "channel_name": cf.get("channel_name"),
                        "channel_url": cf.get("channel_url"),
                        "presented_rank": cf.get("presented_rank"),
                        "presented_score": cf.get("presented_score"),
                        "rating": cf.get("rating"),
                        "reason": cf.get("reason"),
                        "component_scores": cf.get("component_scores"),
                    }
                    for cf in cf_list
                ]
                self._db.table("channel_feedback").insert(cf_rows).execute()

            return True
        except Exception:
            return False

    def load_entries(self, mode: str | None = None) -> list[dict]:
        """Load all entries from Supabase, optionally filtered by search_mode."""
        try:
            if mode is not None:
                resp = (
                    self._db
                    .table("feedback_entries")
                    .select("*, channel_feedback(*)")
                    .eq("search_mode", mode)
                    .execute()
                )
            else:
                resp = (
                    self._db
                    .table("feedback_entries")
                    .select("*, channel_feedback(*)")
                    .execute()
                )

            results = []
            for row in resp.data:
                entry = {
                    "timestamp": row.get("timestamp"),
                    "search_mode": row.get("search_mode"),
                    "query": row.get("query"),
                    "results_count": row.get("results_count"),
                    "scoring_version": row.get("scoring_version"),
                    "channel_feedback": row.get("channel_feedback", []),
                }
                if row.get("seed_channel_id") is not None:
                    entry["seed_channel_id"] = row["seed_channel_id"]
                if row.get("seed_channel_name") is not None:
                    entry["seed_channel_name"] = row["seed_channel_name"]
                if row.get("filters") is not None:
                    entry["filters"] = row["filters"]
                if row.get("ai_enabled") is not None:
                    entry["ai_enabled"] = row["ai_enabled"]
                results.append(entry)
            return results
        except Exception:
            return []

    def clear_all(self) -> bool:
        """Delete all rows from both tables."""
        try:
            self._db.table("channel_feedback").delete().neq("id", 0).execute()
            self._db.table("feedback_entries").delete().neq("id", 0).execute()
            return True
        except Exception:
            return False

    def clear_incompatible(self) -> int:
        """Remove entries incompatible with the current scoring version. Returns count removed."""
        try:
            entries = self.load_entries()
            incompatible_ids = []
            for row in entries:
                if not is_version_compatible(
                    row.get("scoring_version", {}),
                    row.get("search_mode", "keyword"),
                ):
                    incompatible_ids.append(row.get("_id"))

            if not incompatible_ids:
                return 0

            for row_id in incompatible_ids:
                if row_id is not None:
                    self._db.table("channel_feedback").delete().eq("entry_id", row_id).execute()
                    self._db.table("feedback_entries").delete().eq("id", row_id).execute()

            return len(incompatible_ids)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Module-level store selector
# ---------------------------------------------------------------------------

def _get_supabase_secret(name: str) -> str | None:
    """Return a Supabase secret from environment."""
    return os.getenv(name)


def _get_store() -> FeedbackStore:
    """Return the active FeedbackStore based on environment."""
    url = _get_supabase_secret("SUPABASE_URL")
    key = _get_supabase_secret("SUPABASE_SERVICE_KEY")
    if url and key:
        return SupabaseFeedbackStore(url, key)
    return JSONFeedbackStore()


# ---------------------------------------------------------------------------
# Thin wrappers kept for backward-compatibility (tests import these)
# ---------------------------------------------------------------------------

def _load_feedback_data() -> dict:
    """Load existing feedback data from JSON file."""
    store = JSONFeedbackStore(FEEDBACK_FILE)
    return store._load()


def _save_feedback_data(data: dict) -> bool:
    """Save feedback data to JSON file."""
    store = JSONFeedbackStore(FEEDBACK_FILE)
    return store._dump(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_channel_feedback(
    search_mode: Literal["seed", "keyword"],
    query: str,
    results_count: int,
    channel_feedback: list[dict],
    seed_channel_id: Optional[str] = None,
    seed_channel_name: Optional[str] = None,
    filters: Optional[dict] = None,
    ai_enabled: Optional[bool] = None,
) -> bool:
    """
    Save per-channel user feedback for a search.

    Parameters:
    -----------
    search_mode : Literal["seed", "keyword"]
        The search mode used

    query : str
        The search query used

    results_count : int
        Total number of results returned

    channel_feedback : list[dict]
        Per-channel feedback, each entry should have:
        - channel_id: str
        - channel_name: str
        - channel_url: str
        - presented_rank: int (1-5)
        - presented_score: float
        - rating: "relevant" | "not_relevant" | "skip"
        - reason: str | None (required if rating is "not_relevant")
        - component_scores: dict (scoring breakdown for ML training)

    seed_channel_id : str, optional
        Channel ID of seed (if seed mode)

    seed_channel_name : str, optional
        Channel name of seed (if seed mode)

    filters : dict, optional
        Search filter settings used (min_subscribers, country_filter, months_ago, region)

    ai_enabled : bool, optional
        Whether AI enhancement was enabled for this search

    Returns:
    --------
    bool: True if saved successfully
    """
    scoring_version = get_scoring_version(search_mode)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "search_mode": search_mode,
        "query": query,
        "results_count": results_count,
        "scoring_version": scoring_version.to_dict(),
        "channel_feedback": channel_feedback,
    }

    if search_mode == "seed":
        entry["seed_channel_id"] = seed_channel_id
        entry["seed_channel_name"] = seed_channel_name

    if filters:
        entry["filters"] = filters

    if ai_enabled is not None:
        entry["ai_enabled"] = ai_enabled

    return _get_store().save_entry(entry)


def build_channel_feedback_entry(
    channel_id: str,
    channel_name: str,
    channel_url: str,
    presented_rank: int,
    presented_score: float,
    rating: Literal["relevant", "not_relevant", "skip"],
    component_scores: dict,
    reason: Optional[str] = None,
) -> dict:
    """
    Build a properly structured channel feedback entry.

    Parameters:
    -----------
    channel_id : str
        YouTube channel ID

    channel_name : str
        Channel display name

    channel_url : str
        Channel URL

    presented_rank : int
        Position in results (1-5)

    presented_score : float
        The score shown to user (relevance or similarity)

    rating : Literal["relevant", "not_relevant", "skip"]
        User's rating for this channel

    component_scores : dict
        Breakdown of scoring components for ML training.
        For seed mode: tag_score, keyword_score, subscriber_score,
                      engagement_score, frequency_score, algorithmic_score, gemini_score
        For keyword mode: title_match_score, tags_match_score,
                         algorithmic_relevance, ai_relevance

    reason : str, optional
        Reason code if rating is "not_relevant"
        Valid values: "wrong_topic", "low_quality", "poor_fit", "other"

    Returns:
    --------
    dict: Structured channel feedback entry
    """
    if rating not in VALID_RATINGS:
        raise ValueError(f"Invalid rating: {rating}. Must be one of {VALID_RATINGS}")

    if rating == "not_relevant" and reason and reason not in CHANNEL_FEEDBACK_REASONS:
        raise ValueError(
            f"Invalid reason: {reason}. Must be one of {list(CHANNEL_FEEDBACK_REASONS.keys())}"
        )

    return {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_url": channel_url,
        "presented_rank": presented_rank,
        "presented_score": presented_score,
        "rating": rating,
        "reason": reason if rating == "not_relevant" else None,
        "component_scores": component_scores,
    }


def get_feedback_stats() -> dict:
    """
    Get summary statistics of collected feedback.

    Returns:
    --------
    dict with keys:
        - total_entries: int (number of feedback submissions)
        - total_channel_ratings: int (total individual channel ratings)
        - rating_breakdown: dict[str, int] (relevant/not_relevant/skip counts)
        - reason_breakdown: dict[str, int] (for not_relevant ratings)
        - by_search_mode: dict[str, dict]
        - compatible_entries: int (entries compatible with current scoring version)
    """
    entries = _get_store().load_entries()

    stats = {
        "total_entries": len(entries),
        "total_channel_ratings": 0,
        "rating_breakdown": {
            "relevant": 0,
            "not_relevant": 0,
            "skip": 0,
        },
        "reason_breakdown": {
            "wrong_topic": 0,
            "low_quality": 0,
            "poor_fit": 0,
            "other": 0,
        },
        "by_search_mode": {
            "seed": {"entries": 0, "relevant": 0, "not_relevant": 0},
            "keyword": {"entries": 0, "relevant": 0, "not_relevant": 0},
        },
        "compatible_entries": {
            "seed": 0,
            "keyword": 0,
        },
    }

    for entry in entries:
        mode = entry.get("search_mode", "keyword")
        channel_feedback = entry.get("channel_feedback", [])
        scoring_version = entry.get("scoring_version", {})

        if mode in stats["by_search_mode"]:
            stats["by_search_mode"][mode]["entries"] += 1

        if is_version_compatible(scoring_version, mode):
            stats["compatible_entries"][mode] += 1

        for cf in channel_feedback:
            rating = cf.get("rating")
            reason = cf.get("reason")

            stats["total_channel_ratings"] += 1

            if rating in stats["rating_breakdown"]:
                stats["rating_breakdown"][rating] += 1

            if mode in stats["by_search_mode"] and rating in ("relevant", "not_relevant"):
                stats["by_search_mode"][mode][rating] += 1

            if rating == "not_relevant" and reason in stats["reason_breakdown"]:
                stats["reason_breakdown"][reason] += 1

    return stats


def get_training_data(
    mode: Literal["seed", "keyword"],
    only_compatible: bool = True,
) -> list[dict]:
    """
    Extract training data for ML models from feedback.

    Returns flattened list of channel ratings with their component scores,
    suitable for training logistic regression (weight learning) or
    linear regression (score calibration).

    Parameters:
    -----------
    mode : Literal["seed", "keyword"]
        Filter by search mode

    only_compatible : bool
        If True, only include feedback collected under compatible scoring version

    Returns:
    --------
    List of dicts, each containing:
        - All component_scores fields (features)
        - presented_score: float
        - rating: str (label)
        - is_relevant: bool (binary label for logistic regression)
        - reason: str | None
        - query: str
        - timestamp: str
    """
    entries = _get_store().load_entries(mode=mode)
    training_data = []

    for entry in entries:
        scoring_version = entry.get("scoring_version", {})
        if only_compatible and not is_version_compatible(scoring_version, mode):
            continue

        query = entry.get("query", "")
        timestamp = entry.get("timestamp", "")

        for cf in entry.get("channel_feedback", []):
            rating = cf.get("rating")

            if rating == "skip":
                continue

            record = {
                "query": query,
                "timestamp": timestamp,
                "channel_id": cf.get("channel_id"),
                "presented_rank": cf.get("presented_rank"),
                "presented_score": cf.get("presented_score"),
                "rating": rating,
                "is_relevant": rating == "relevant",
                "reason": cf.get("reason"),
            }

            component_scores = cf.get("component_scores", {})
            for key, value in component_scores.items():
                record[f"component_{key}"] = value

            training_data.append(record)

    return training_data


def get_negative_feedback_entries(limit: int = 50) -> list[dict]:
    """
    Get recent feedback entries containing negative channel ratings for analysis.

    Parameters:
    -----------
    limit : int
        Maximum number of entries to return

    Returns:
    --------
    List of feedback entries that contain at least one "not_relevant" rating,
    most recent first
    """
    entries = _get_store().load_entries()

    negative = [
        e for e in entries
        if any(
            cf.get("rating") == "not_relevant"
            for cf in e.get("channel_feedback", [])
        )
    ]

    negative.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return negative[:limit]


def export_feedback_csv(filepath: str) -> bool:
    """
    Export all feedback to CSV for analysis.

    The CSV is flattened with one row per channel rating, suitable for
    importing into BI tools or ML pipelines.

    Parameters:
    -----------
    filepath : str
        Path to save the CSV file

    Returns:
    --------
    bool: True if exported successfully
    """
    import csv

    entries = _get_store().load_entries()

    if not entries:
        return False

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                # Entry-level fields
                "timestamp",
                "search_mode",
                "query",
                "results_count",
                "seed_channel_id",
                "seed_channel_name",
                "ai_enabled",
                # Filters
                "min_subscribers",
                "country_filter",
                "months_ago",
                "region",
                # Version signature
                "scoring_version",
                "scoring_weights",
                "ai_blend_ratio",
                "pipeline_hash",
                # Per-channel fields
                "channel_id",
                "channel_name",
                "channel_url",
                "presented_rank",
                "presented_score",
                "rating",
                "reason",
                # Component scores (seed mode)
                "component_tag_score",
                "component_keyword_score",
                "component_subscriber_score",
                "component_engagement_score",
                "component_frequency_score",
                "component_algorithmic_score",
                "component_gemini_score",
                # Component scores (keyword mode)
                "component_title_match_score",
                "component_tags_match_score",
                "component_algorithmic_relevance",
                "component_ai_relevance",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                filters = entry.get("filters", {})
                scoring_version = entry.get("scoring_version", {})

                base_row = {
                    "timestamp": entry.get("timestamp"),
                    "search_mode": entry.get("search_mode"),
                    "query": entry.get("query"),
                    "results_count": entry.get("results_count"),
                    "seed_channel_id": entry.get("seed_channel_id"),
                    "seed_channel_name": entry.get("seed_channel_name"),
                    "ai_enabled": entry.get("ai_enabled"),
                    "min_subscribers": filters.get("min_subscribers"),
                    "country_filter": filters.get("country_filter"),
                    "months_ago": filters.get("months_ago"),
                    "region": filters.get("region"),
                    "scoring_version": scoring_version.get("version"),
                    "scoring_weights": json.dumps(scoring_version.get("weights", {})),
                    "ai_blend_ratio": scoring_version.get("ai_blend_ratio"),
                    "pipeline_hash": scoring_version.get("pipeline_hash"),
                }

                for cf in entry.get("channel_feedback", []):
                    component_scores = cf.get("component_scores", {})

                    row = {
                        **base_row,
                        "channel_id": cf.get("channel_id"),
                        "channel_name": cf.get("channel_name"),
                        "channel_url": cf.get("channel_url"),
                        "presented_rank": cf.get("presented_rank"),
                        "presented_score": cf.get("presented_score"),
                        "rating": cf.get("rating"),
                        "reason": cf.get("reason"),
                        # Seed mode components
                        "component_tag_score": component_scores.get("tag_score"),
                        "component_keyword_score": component_scores.get("keyword_score"),
                        "component_subscriber_score": component_scores.get("subscriber_score"),
                        "component_engagement_score": component_scores.get("engagement_score"),
                        "component_frequency_score": component_scores.get("frequency_score"),
                        "component_algorithmic_score": component_scores.get("algorithmic_score"),
                        "component_gemini_score": component_scores.get("gemini_score"),
                        # Keyword mode components
                        "component_title_match_score": component_scores.get("title_match_score"),
                        "component_tags_match_score": component_scores.get("tags_match_score"),
                        "component_algorithmic_relevance": component_scores.get("algorithmic_relevance"),
                        "component_ai_relevance": component_scores.get("ai_relevance"),
                    }

                    writer.writerow(row)

        return True
    except IOError:
        return False


def clear_incompatible_feedback() -> int:
    """
    Remove feedback entries that are incompatible with current scoring version.

    This should be called when scoring logic changes significantly and old
    feedback would pollute ML training data.

    Returns:
    --------
    int: Number of entries removed
    """
    return _get_store().clear_incompatible()


def clear_all_feedback() -> bool:
    """
    Remove all feedback entries.

    Returns:
    --------
    bool: True if cleared successfully
    """
    return _get_store().clear_all()

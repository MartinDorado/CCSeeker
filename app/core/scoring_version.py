"""
scoring_version.py - Centralized scoring weights and version signature

This module is the single source of truth for all scoring weights used in
CCSeeker's relevance and similarity calculations. It provides:

1. Weight constants for keyword mode (relevance scoring)
2. Weight constants for seed mode (similarity scoring)
3. Version signature generation for ML training data compatibility

All scoring modules (relevance.py, similarity.py) should import
weights from here rather than defining them locally.

This module is Streamlit-agnostic and can be unit tested independently.
"""

import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import Literal

# ============================================================================
# VERSION IDENTIFIER
# ============================================================================

# Increment this when scoring logic changes in a way that affects results
# Follow semantic versioning: MAJOR.MINOR.PATCH
# - MAJOR: Breaking changes to scoring (old feedback incompatible)
# - MINOR: New features that don't break existing scoring
# - PATCH: Bug fixes that don't change scoring behavior
SCORING_VERSION = "2.1.0"


# ============================================================================
# KEYWORD MODE WEIGHTS (used by relevance.py)
# ============================================================================

@dataclass(frozen=True)
class KeywordWeights:
    """Weights for keyword mode relevance scoring."""
    title_weight: float = 2.0
    tags_weight: float = 1.0

    # AI blend ratio (algorithmic vs AI score)
    # Final score = (1 - ai_blend) * algorithmic + ai_blend * ai_score
    ai_blend_ratio: float = 0.20


# Default instance for import
KEYWORD_WEIGHTS = KeywordWeights()


# ============================================================================
# SEED MODE WEIGHTS (used by similarity.py)
# ============================================================================

@dataclass(frozen=True)
class SeedWeights:
    """Weights for seed mode similarity scoring (100 points total)."""
    tag_overlap: int = 30
    keyword_overlap: int = 30
    subscriber_similarity: int = 15
    engagement_rate: int = 17
    upload_frequency: int = 8

    # AI blend ratio (algorithmic vs AI score)
    ai_blend_ratio: float = 0.20

    @property
    def total_points(self) -> int:
        """Verify weights sum to 100."""
        return (
            self.tag_overlap +
            self.keyword_overlap +
            self.subscriber_similarity +
            self.engagement_rate +
            self.upload_frequency
        )


# Default instance for import
SEED_WEIGHTS = SeedWeights()


# ============================================================================
# PER-CHANNEL FEEDBACK REASONS
# ============================================================================

# Reason codes for negative per-channel feedback
CHANNEL_FEEDBACK_REASONS = {
    "wrong_topic": "Wrong topic/niche",
    "low_quality": "Low quality content",
    "poor_fit": "Not a good fit (size/style)",
    "other": "Other"
}

# Valid rating values for per-channel feedback
VALID_RATINGS = ("relevant", "not_relevant", "skip")


# ============================================================================
# VERSION SIGNATURE GENERATION
# ============================================================================

def _get_file_hash(filepath: str) -> str:
    """Generate MD5 hash of a file's contents."""
    if not os.path.exists(filepath):
        return "file_not_found"

    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except IOError:
        return "read_error"


def generate_pipeline_hash() -> str:
    """
    Generate a hash representing the current state of scoring logic files.

    This hash changes when any of the core scoring files are modified,
    helping identify when feedback was collected under different logic.

    Returns:
        12-character hash string
    """
    # Get the directory where this file lives
    core_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(core_dir)

    # Files that affect scoring
    scoring_files = [
        os.path.join(core_dir, "scoring_version.py"),
        os.path.join(core_dir, "relevance.py"),
        os.path.join(core_dir, "similarity.py"),
    ]

    # Combine hashes
    combined = ""
    for filepath in scoring_files:
        combined += _get_file_hash(filepath)

    # Return hash of combined hashes
    return hashlib.md5(combined.encode()).hexdigest()[:12]


@dataclass
class ScoringVersionSignature:
    """
    Complete version signature for feedback entries.

    This captures all information needed to determine if feedback
    is compatible with current scoring logic for ML training.
    """
    version: str
    mode: Literal["seed", "keyword"]
    weights: dict = field(default_factory=dict)
    ai_blend_ratio: float = 0.20
    pipeline_hash: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_scoring_version(mode: Literal["seed", "keyword"]) -> ScoringVersionSignature:
    """
    Get the current scoring version signature for a given mode.

    Args:
        mode: Either "seed" or "keyword"

    Returns:
        ScoringVersionSignature with current weights and version info

    Example:
        >>> sig = get_scoring_version("seed")
        >>> sig.version
        '2.0.0'
        >>> sig.weights['tag_overlap']
        30
    """
    if mode == "seed":
        weights = {
            "tag_overlap": SEED_WEIGHTS.tag_overlap,
            "keyword_overlap": SEED_WEIGHTS.keyword_overlap,
            "subscriber_similarity": SEED_WEIGHTS.subscriber_similarity,
            "engagement_rate": SEED_WEIGHTS.engagement_rate,
            "upload_frequency": SEED_WEIGHTS.upload_frequency,
        }
        ai_blend = SEED_WEIGHTS.ai_blend_ratio
    elif mode == "keyword":
        weights = {
            "title_weight": KEYWORD_WEIGHTS.title_weight,
            "tags_weight": KEYWORD_WEIGHTS.tags_weight,
        }
        ai_blend = KEYWORD_WEIGHTS.ai_blend_ratio
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'seed' or 'keyword'.")

    return ScoringVersionSignature(
        version=SCORING_VERSION,
        mode=mode,
        weights=weights,
        ai_blend_ratio=ai_blend,
        pipeline_hash=generate_pipeline_hash(),
    )


def is_version_compatible(
    feedback_version: dict,
    current_mode: Literal["seed", "keyword"]
) -> bool:
    """
    Check if feedback was collected under fully compatible scoring logic.

    Feedback is fully compatible if:
    1. Same major.minor version (patch differences are fine)
    2. Same mode (seed/keyword)
    3. Same weights (exact match)

    Cross-minor-version feedback (e.g. 2.0.x collected under 2.1.x) is
    "soft-incompatible": the algorithmic weights are unchanged but the AI
    blend sees richer input, so direct ranking comparisons are unreliable.
    Use is_soft_compatible() to load that feedback for trend analysis only.

    Args:
        feedback_version: The scoring_version dict from a feedback entry
        current_mode: The mode to check compatibility with

    Returns:
        True if feedback is fully compatible for ML training
    """
    if not feedback_version:
        return False

    if feedback_version.get("mode") != current_mode:
        return False

    fv = feedback_version.get("version", "0.0.0").split(".")
    cv = SCORING_VERSION.split(".")

    # Major version must match
    if fv[0] != cv[0]:
        return False

    # Minor version must also match (cross-minor = soft-incompatible)
    if len(fv) > 1 and len(cv) > 1 and fv[1] != cv[1]:
        return False

    current_sig = get_scoring_version(current_mode)
    return feedback_version.get("weights", {}) == current_sig.weights


def is_soft_compatible(
    feedback_version: dict,
    current_mode: Literal["seed", "keyword"]
) -> bool:
    """
    Check if feedback is usable for trend analysis (lenient compatibility).

    Returns True when the major version and weights match, even if minor
    versions differ.  Cross-minor-version feedback (e.g. 2.0.x vs 2.1.x)
    is soft-incompatible: acceptable for observability/trend analysis but
    NOT for direct ranking comparison or ML training.

    Args:
        feedback_version: The scoring_version dict from a feedback entry
        current_mode: The mode to check compatibility with

    Returns:
        True if feedback can be used for trend analysis
    """
    if not feedback_version:
        return False

    if feedback_version.get("mode") != current_mode:
        return False

    feedback_major = feedback_version.get("version", "0.0.0").split(".")[0]
    current_major = SCORING_VERSION.split(".")[0]
    if feedback_major != current_major:
        return False

    current_sig = get_scoring_version(current_mode)
    return feedback_version.get("weights", {}) == current_sig.weights

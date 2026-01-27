"""
Tests for core.scoring_version module

Tests cover:
- Weight constants validity
- Version signature generation
- Version compatibility checking
- Pipeline hash generation
"""

import pytest
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.scoring_version import (
    SCORING_VERSION,
    KEYWORD_WEIGHTS,
    SEED_WEIGHTS,
    CHANNEL_FEEDBACK_REASONS,
    VALID_RATINGS,
    get_scoring_version,
    is_version_compatible,
    generate_pipeline_hash,
    ScoringVersionSignature,
)


class TestScoringConstants:
    """Tests for scoring weight constants."""

    def test_scoring_version_format(self):
        """Version string follows semantic versioning format."""
        parts = SCORING_VERSION.split(".")
        assert len(parts) == 3, "Version should be MAJOR.MINOR.PATCH"
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' should be numeric"

    def test_keyword_weights_positive(self):
        """Keyword mode weights are positive numbers."""
        assert KEYWORD_WEIGHTS.title_weight > 0
        assert KEYWORD_WEIGHTS.tags_weight > 0
        assert 0 < KEYWORD_WEIGHTS.ai_blend_ratio < 1

    def test_seed_weights_sum_to_100(self):
        """Seed mode weights sum to 100 points."""
        assert SEED_WEIGHTS.total_points == 100

    def test_seed_weights_positive(self):
        """All seed mode weights are positive."""
        assert SEED_WEIGHTS.tag_overlap > 0
        assert SEED_WEIGHTS.keyword_overlap > 0
        assert SEED_WEIGHTS.subscriber_similarity > 0
        assert SEED_WEIGHTS.engagement_rate > 0
        assert SEED_WEIGHTS.upload_frequency > 0

    def test_seed_ai_blend_ratio_valid(self):
        """AI blend ratio is between 0 and 1."""
        assert 0 < SEED_WEIGHTS.ai_blend_ratio < 1

    def test_channel_feedback_reasons_not_empty(self):
        """Channel feedback reasons dictionary has entries."""
        assert len(CHANNEL_FEEDBACK_REASONS) > 0
        assert "wrong_topic" in CHANNEL_FEEDBACK_REASONS
        assert "low_quality" in CHANNEL_FEEDBACK_REASONS
        assert "poor_fit" in CHANNEL_FEEDBACK_REASONS
        assert "other" in CHANNEL_FEEDBACK_REASONS

    def test_valid_ratings_tuple(self):
        """Valid ratings contains expected values."""
        assert "relevant" in VALID_RATINGS
        assert "not_relevant" in VALID_RATINGS
        assert "skip" in VALID_RATINGS
        assert len(VALID_RATINGS) == 3


class TestGetScoringVersion:
    """Tests for get_scoring_version function."""

    def test_seed_mode_returns_signature(self):
        """Seed mode returns valid signature."""
        sig = get_scoring_version("seed")
        assert isinstance(sig, ScoringVersionSignature)
        assert sig.version == SCORING_VERSION
        assert sig.mode == "seed"

    def test_keyword_mode_returns_signature(self):
        """Keyword mode returns valid signature."""
        sig = get_scoring_version("keyword")
        assert isinstance(sig, ScoringVersionSignature)
        assert sig.version == SCORING_VERSION
        assert sig.mode == "keyword"

    def test_seed_mode_has_correct_weights(self):
        """Seed mode signature contains seed weights."""
        sig = get_scoring_version("seed")
        assert "tag_overlap" in sig.weights
        assert "keyword_overlap" in sig.weights
        assert "subscriber_similarity" in sig.weights
        assert "engagement_rate" in sig.weights
        assert "upload_frequency" in sig.weights

    def test_keyword_mode_has_correct_weights(self):
        """Keyword mode signature contains keyword weights."""
        sig = get_scoring_version("keyword")
        assert "title_weight" in sig.weights
        assert "tags_weight" in sig.weights

    def test_invalid_mode_raises_error(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            get_scoring_version("invalid")

    def test_signature_to_dict(self):
        """Signature can be converted to dict for JSON serialization."""
        sig = get_scoring_version("seed")
        d = sig.to_dict()
        assert isinstance(d, dict)
        assert d["version"] == SCORING_VERSION
        assert d["mode"] == "seed"
        assert "weights" in d
        assert "ai_blend_ratio" in d
        assert "pipeline_hash" in d

    def test_pipeline_hash_included(self):
        """Signature includes pipeline hash."""
        sig = get_scoring_version("seed")
        assert len(sig.pipeline_hash) == 12  # MD5 hash truncated to 12 chars


class TestPipelineHash:
    """Tests for generate_pipeline_hash function."""

    def test_hash_is_deterministic(self):
        """Same files produce same hash."""
        hash1 = generate_pipeline_hash()
        hash2 = generate_pipeline_hash()
        assert hash1 == hash2

    def test_hash_is_12_characters(self):
        """Hash is truncated to 12 characters."""
        h = generate_pipeline_hash()
        assert len(h) == 12

    def test_hash_is_hexadecimal(self):
        """Hash contains only hex characters."""
        h = generate_pipeline_hash()
        assert all(c in "0123456789abcdef" for c in h)


class TestVersionCompatibility:
    """Tests for is_version_compatible function."""

    def test_current_version_is_compatible(self):
        """Current version signature is compatible with itself."""
        sig = get_scoring_version("seed")
        assert is_version_compatible(sig.to_dict(), "seed")

    def test_wrong_mode_not_compatible(self):
        """Seed signature not compatible with keyword mode."""
        sig = get_scoring_version("seed")
        assert not is_version_compatible(sig.to_dict(), "keyword")

    def test_empty_dict_not_compatible(self):
        """Empty dict is not compatible."""
        assert not is_version_compatible({}, "seed")
        assert not is_version_compatible({}, "keyword")

    def test_none_not_compatible(self):
        """None is not compatible."""
        assert not is_version_compatible(None, "seed")

    def test_different_major_version_not_compatible(self):
        """Different major version is not compatible."""
        sig = get_scoring_version("seed").to_dict()
        sig["version"] = "99.0.0"  # Different major version
        assert not is_version_compatible(sig, "seed")

    def test_different_weights_not_compatible(self):
        """Different weights make signature incompatible."""
        sig = get_scoring_version("seed").to_dict()
        sig["weights"]["tag_overlap"] = 999  # Changed weight
        assert not is_version_compatible(sig, "seed")

    def test_same_major_version_compatible(self):
        """Same major version with same weights is compatible."""
        sig = get_scoring_version("seed").to_dict()
        # Change minor/patch but keep major and weights the same
        major = SCORING_VERSION.split(".")[0]
        sig["version"] = f"{major}.99.99"
        assert is_version_compatible(sig, "seed")


class TestDataclassesFrozen:
    """Test that weight dataclasses are immutable."""

    def test_keyword_weights_frozen(self):
        """KeywordWeights is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            KEYWORD_WEIGHTS.title_weight = 5.0

    def test_seed_weights_frozen(self):
        """SeedWeights is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            SEED_WEIGHTS.tag_overlap = 50

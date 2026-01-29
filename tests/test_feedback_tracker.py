"""
Tests for feedback_tracker module

Tests cover:
- Per-channel feedback saving
- Feedback entry building with validation
- Statistics calculation
- Training data extraction
- CSV export format
- Version compatibility filtering
"""

import pytest
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.analytics.feedback_tracker import (
    FEEDBACK_SCHEMA_VERSION,
    save_channel_feedback,
    build_channel_feedback_entry,
    get_feedback_stats,
    get_training_data,
    get_negative_feedback_entries,
    export_feedback_csv,
    clear_all_feedback,
    clear_incompatible_feedback,
    _load_feedback_data,
    _save_feedback_data,
)
from app.core.scoring_version import VALID_RATINGS, CHANNEL_FEEDBACK_REASONS


@pytest.fixture
def temp_feedback_file(tmp_path):
    """Create a temporary feedback file for testing."""
    feedback_file = tmp_path / ".feedback_data.json"
    # Patch the FEEDBACK_FILE constant
    with patch("app.analytics.feedback_tracker.FEEDBACK_FILE", str(feedback_file)):
        yield feedback_file


@pytest.fixture
def sample_channel_feedback():
    """Sample channel feedback data for testing."""
    return [
        build_channel_feedback_entry(
            channel_id="UC123",
            channel_name="Test Channel 1",
            channel_url="https://youtube.com/@testchannel1",
            presented_rank=1,
            presented_score=85.5,
            rating="relevant",
            component_scores={
                "tag_score": 25.0,
                "keyword_score": 28.0,
                "subscriber_score": 12.0,
                "engagement_score": 15.0,
                "frequency_score": 5.5,
                "algorithmic_score": 85.5,
                "gemini_score": 8,
            },
        ),
        build_channel_feedback_entry(
            channel_id="UC456",
            channel_name="Test Channel 2",
            channel_url="https://youtube.com/@testchannel2",
            presented_rank=2,
            presented_score=72.3,
            rating="not_relevant",
            reason="wrong_topic",
            component_scores={
                "tag_score": 20.0,
                "keyword_score": 22.0,
                "subscriber_score": 10.0,
                "engagement_score": 12.0,
                "frequency_score": 8.3,
                "algorithmic_score": 72.3,
                "gemini_score": 6,
            },
        ),
        build_channel_feedback_entry(
            channel_id="UC789",
            channel_name="Test Channel 3",
            channel_url="https://youtube.com/@testchannel3",
            presented_rank=3,
            presented_score=65.0,
            rating="skip",
            component_scores={
                "tag_score": 18.0,
                "keyword_score": 20.0,
                "subscriber_score": 8.0,
                "engagement_score": 12.0,
                "frequency_score": 7.0,
                "algorithmic_score": 65.0,
                "gemini_score": 5,
            },
        ),
    ]


class TestBuildChannelFeedbackEntry:
    """Tests for build_channel_feedback_entry function."""

    def test_valid_relevant_entry(self):
        """Valid relevant rating creates proper entry."""
        entry = build_channel_feedback_entry(
            channel_id="UC123",
            channel_name="Test Channel",
            channel_url="https://youtube.com/@test",
            presented_rank=1,
            presented_score=80.0,
            rating="relevant",
            component_scores={"tag_score": 25.0},
        )
        assert entry["channel_id"] == "UC123"
        assert entry["rating"] == "relevant"
        assert entry["reason"] is None

    def test_valid_not_relevant_entry_with_reason(self):
        """Not relevant rating with reason creates proper entry."""
        entry = build_channel_feedback_entry(
            channel_id="UC123",
            channel_name="Test Channel",
            channel_url="https://youtube.com/@test",
            presented_rank=1,
            presented_score=80.0,
            rating="not_relevant",
            reason="wrong_topic",
            component_scores={"tag_score": 25.0},
        )
        assert entry["rating"] == "not_relevant"
        assert entry["reason"] == "wrong_topic"

    def test_skip_rating_clears_reason(self):
        """Skip rating sets reason to None even if provided."""
        entry = build_channel_feedback_entry(
            channel_id="UC123",
            channel_name="Test Channel",
            channel_url="https://youtube.com/@test",
            presented_rank=1,
            presented_score=80.0,
            rating="skip",
            reason="wrong_topic",  # Should be ignored
            component_scores={"tag_score": 25.0},
        )
        assert entry["rating"] == "skip"
        assert entry["reason"] is None

    def test_invalid_rating_raises_error(self):
        """Invalid rating raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rating"):
            build_channel_feedback_entry(
                channel_id="UC123",
                channel_name="Test Channel",
                channel_url="https://youtube.com/@test",
                presented_rank=1,
                presented_score=80.0,
                rating="invalid_rating",
                component_scores={"tag_score": 25.0},
            )

    def test_invalid_reason_raises_error(self):
        """Invalid reason raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reason"):
            build_channel_feedback_entry(
                channel_id="UC123",
                channel_name="Test Channel",
                channel_url="https://youtube.com/@test",
                presented_rank=1,
                presented_score=80.0,
                rating="not_relevant",
                reason="invalid_reason",
                component_scores={"tag_score": 25.0},
            )

    def test_entry_contains_component_scores(self):
        """Entry includes component scores."""
        scores = {"tag_score": 25.0, "keyword_score": 28.0}
        entry = build_channel_feedback_entry(
            channel_id="UC123",
            channel_name="Test Channel",
            channel_url="https://youtube.com/@test",
            presented_rank=1,
            presented_score=80.0,
            rating="relevant",
            component_scores=scores,
        )
        assert entry["component_scores"] == scores


class TestSaveChannelFeedback:
    """Tests for save_channel_feedback function."""

    def test_save_creates_file(self, temp_feedback_file, sample_channel_feedback):
        """Saving feedback creates the file."""
        save_channel_feedback(
            search_mode="seed",
            query="manga, anime",
            results_count=25,
            channel_feedback=sample_channel_feedback,
        )
        assert temp_feedback_file.exists()

    def test_save_includes_scoring_version(self, temp_feedback_file, sample_channel_feedback):
        """Saved feedback includes scoring version signature."""
        save_channel_feedback(
            search_mode="seed",
            query="manga, anime",
            results_count=25,
            channel_feedback=sample_channel_feedback,
        )
        data = json.loads(temp_feedback_file.read_text())
        entry = data["feedback_entries"][0]
        assert "scoring_version" in entry
        assert "version" in entry["scoring_version"]
        assert "weights" in entry["scoring_version"]
        assert "pipeline_hash" in entry["scoring_version"]

    def test_save_includes_seed_info(self, temp_feedback_file, sample_channel_feedback):
        """Seed mode includes seed channel info."""
        save_channel_feedback(
            search_mode="seed",
            query="manga, anime",
            results_count=25,
            channel_feedback=sample_channel_feedback,
            seed_channel_id="UC_SEED",
            seed_channel_name="Seed Channel",
        )
        data = json.loads(temp_feedback_file.read_text())
        entry = data["feedback_entries"][0]
        assert entry["seed_channel_id"] == "UC_SEED"
        assert entry["seed_channel_name"] == "Seed Channel"

    def test_save_includes_filters(self, temp_feedback_file, sample_channel_feedback):
        """Filters are saved when provided."""
        filters = {"min_subscribers": 1000, "country_filter": "US"}
        save_channel_feedback(
            search_mode="keyword",
            query="cooking",
            results_count=10,
            channel_feedback=sample_channel_feedback,
            filters=filters,
        )
        data = json.loads(temp_feedback_file.read_text())
        entry = data["feedback_entries"][0]
        assert entry["filters"] == filters


class TestGetFeedbackStats:
    """Tests for get_feedback_stats function."""

    def test_empty_stats(self, temp_feedback_file):
        """Empty feedback returns zero stats."""
        stats = get_feedback_stats()
        assert stats["total_entries"] == 0
        assert stats["total_channel_ratings"] == 0

    def test_stats_count_ratings(self, temp_feedback_file, sample_channel_feedback):
        """Stats correctly count ratings."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        stats = get_feedback_stats()
        assert stats["total_entries"] == 1
        assert stats["total_channel_ratings"] == 3
        assert stats["rating_breakdown"]["relevant"] == 1
        assert stats["rating_breakdown"]["not_relevant"] == 1
        assert stats["rating_breakdown"]["skip"] == 1

    def test_stats_count_reasons(self, temp_feedback_file, sample_channel_feedback):
        """Stats correctly count reasons."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        stats = get_feedback_stats()
        assert stats["reason_breakdown"]["wrong_topic"] == 1

    def test_stats_by_search_mode(self, temp_feedback_file, sample_channel_feedback):
        """Stats are grouped by search mode."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        stats = get_feedback_stats()
        assert stats["by_search_mode"]["seed"]["entries"] == 1
        assert stats["by_search_mode"]["seed"]["relevant"] == 1
        assert stats["by_search_mode"]["seed"]["not_relevant"] == 1


class TestGetTrainingData:
    """Tests for get_training_data function."""

    def test_training_data_excludes_skip(self, temp_feedback_file, sample_channel_feedback):
        """Training data excludes skip ratings."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        training = get_training_data("seed")
        # sample_channel_feedback has 1 relevant, 1 not_relevant, 1 skip
        assert len(training) == 2
        ratings = [t["rating"] for t in training]
        assert "skip" not in ratings

    def test_training_data_has_is_relevant_label(self, temp_feedback_file, sample_channel_feedback):
        """Training data includes binary is_relevant label."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        training = get_training_data("seed")
        relevant_record = next(t for t in training if t["rating"] == "relevant")
        not_relevant_record = next(t for t in training if t["rating"] == "not_relevant")
        assert relevant_record["is_relevant"] is True
        assert not_relevant_record["is_relevant"] is False

    def test_training_data_flattens_components(self, temp_feedback_file, sample_channel_feedback):
        """Training data flattens component scores with prefix."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        training = get_training_data("seed")
        record = training[0]
        assert "component_tag_score" in record
        assert "component_keyword_score" in record

    def test_training_data_filters_by_mode(self, temp_feedback_file, sample_channel_feedback):
        """Training data filters by search mode."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        seed_training = get_training_data("seed")
        keyword_training = get_training_data("keyword")
        assert len(seed_training) == 2
        assert len(keyword_training) == 0


class TestGetNegativeFeedbackEntries:
    """Tests for get_negative_feedback_entries function."""

    def test_returns_entries_with_negative_ratings(self, temp_feedback_file, sample_channel_feedback):
        """Returns entries containing not_relevant ratings."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        negative = get_negative_feedback_entries()
        assert len(negative) == 1

    def test_respects_limit(self, temp_feedback_file, sample_channel_feedback):
        """Respects limit parameter."""
        # Save multiple entries
        for i in range(5):
            save_channel_feedback(
                search_mode="seed",
                query=f"test_{i}",
                results_count=10,
                channel_feedback=sample_channel_feedback,
            )
        negative = get_negative_feedback_entries(limit=3)
        assert len(negative) == 3


class TestExportFeedbackCsv:
    """Tests for export_feedback_csv function."""

    def test_export_creates_file(self, temp_feedback_file, sample_channel_feedback, tmp_path):
        """Export creates CSV file."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        csv_path = tmp_path / "export.csv"
        result = export_feedback_csv(str(csv_path))
        assert result is True
        assert csv_path.exists()

    def test_export_one_row_per_channel(self, temp_feedback_file, sample_channel_feedback, tmp_path):
        """Export creates one row per channel rating."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        csv_path = tmp_path / "export.csv"
        export_feedback_csv(str(csv_path))

        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # 3 channels in sample_channel_feedback
        assert len(rows) == 3

    def test_export_empty_returns_false(self, temp_feedback_file, tmp_path):
        """Export returns False when no data."""
        csv_path = tmp_path / "export.csv"
        result = export_feedback_csv(str(csv_path))
        assert result is False


class TestClearFunctions:
    """Tests for clear_all_feedback and clear_incompatible_feedback."""

    def test_clear_all_removes_entries(self, temp_feedback_file, sample_channel_feedback):
        """clear_all_feedback removes all entries."""
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        stats_before = get_feedback_stats()
        assert stats_before["total_entries"] == 1

        clear_all_feedback()

        stats_after = get_feedback_stats()
        assert stats_after["total_entries"] == 0

    def test_clear_incompatible_returns_count(self, temp_feedback_file, sample_channel_feedback):
        """clear_incompatible_feedback returns count of removed entries."""
        # Save valid entry
        save_channel_feedback(
            search_mode="seed",
            query="test",
            results_count=10,
            channel_feedback=sample_channel_feedback,
        )
        # All entries should be compatible with current version
        removed = clear_incompatible_feedback()
        assert removed == 0


class TestValidRatingsAndReasons:
    """Verify that valid ratings and reasons match expected values."""

    def test_all_valid_ratings_accepted(self):
        """All VALID_RATINGS create valid entries."""
        for rating in VALID_RATINGS:
            reason = "wrong_topic" if rating == "not_relevant" else None
            entry = build_channel_feedback_entry(
                channel_id="UC123",
                channel_name="Test",
                channel_url="https://youtube.com/@test",
                presented_rank=1,
                presented_score=50.0,
                rating=rating,
                reason=reason,
                component_scores={},
            )
            assert entry["rating"] == rating

    def test_all_valid_reasons_accepted(self):
        """All CHANNEL_FEEDBACK_REASONS create valid entries."""
        for reason_code in CHANNEL_FEEDBACK_REASONS.keys():
            entry = build_channel_feedback_entry(
                channel_id="UC123",
                channel_name="Test",
                channel_url="https://youtube.com/@test",
                presented_rank=1,
                presented_score=50.0,
                rating="not_relevant",
                reason=reason_code,
                component_scores={},
            )
            assert entry["reason"] == reason_code

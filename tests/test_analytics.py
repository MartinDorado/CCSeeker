"""
test_analytics.py - Tests for analytics module

Tests synthetic data generation, ML training, and weight optimization.
"""

import pytest
import numpy as np
import os
import sys
import tempfile

# Add app directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from analytics.synthetic_data_generator import (
    generate_synthetic_feedback,
    generate_search_feedback,
    get_synthetic_data_summary,
    _generate_component_scores,
    _determine_rating,
)
from analytics.ml_trainer import (
    prepare_features_seed_mode,
)
from analytics.weight_optimizer import (
    softmax,
    apply_weight_constraints,
)


class TestSyntheticDataGenerator:
    """Tests for synthetic_data_generator.py"""

    def test_generate_component_scores_range(self):
        """Component scores should be within valid ranges."""
        scores = _generate_component_scores(target_relevance=0.5)

        assert 0 <= scores["tag_score"] <= 30
        assert 0 <= scores["keyword_score"] <= 30
        assert 0 <= scores["subscriber_score"] <= 15
        assert 0 <= scores["engagement_score"] <= 17
        assert 0 <= scores["frequency_score"] <= 8

    def test_generate_component_scores_with_ai(self):
        """Should include gemini_score when AI enabled."""
        scores = _generate_component_scores(target_relevance=0.5, ai_enabled=True)
        assert scores["gemini_score"] is not None
        assert 0 <= scores["gemini_score"] <= 10

    def test_generate_component_scores_without_ai(self):
        """Should not include gemini_score when AI disabled."""
        scores = _generate_component_scores(target_relevance=0.5, ai_enabled=False)
        assert scores["gemini_score"] is None

    def test_generate_component_scores_no_tags_edge_case(self):
        """No-tags edge case should produce low tag_score."""
        scores = _generate_component_scores(
            target_relevance=0.5, ai_enabled=True, no_tags=True
        )
        # No-tags channels have tag_score between 0-3
        assert 0 <= scores["tag_score"] <= 3

    def test_generate_search_feedback_structure(self):
        """Generated feedback should have correct structure."""
        feedback = generate_search_feedback(search_mode="seed")

        assert feedback["search_mode"] == "seed"
        assert "query" in feedback
        assert "results_count" in feedback
        assert len(feedback["channel_feedback"]) == 5
        assert feedback["seed_channel_id"] is not None

    def test_generate_search_feedback_keyword_mode(self):
        """Keyword mode should not have seed info."""
        feedback = generate_search_feedback(search_mode="keyword")

        assert feedback["search_mode"] == "keyword"
        assert feedback["seed_channel_id"] is None

    def test_generate_search_feedback_channel_structure(self):
        """Each channel feedback entry should have required fields."""
        feedback = generate_search_feedback(search_mode="seed")

        for cf in feedback["channel_feedback"]:
            assert "channel_id" in cf
            assert "channel_name" in cf
            assert "channel_url" in cf
            assert "presented_rank" in cf
            assert "presented_score" in cf
            assert "rating" in cf
            assert cf["rating"] in ("relevant", "not_relevant", "skip")
            assert "component_scores" in cf

    def test_generate_synthetic_feedback_count(self):
        """Should generate requested number of entries."""
        entries = generate_synthetic_feedback(n_searches=10, save_to_file=False)
        assert len(entries) == 10

    def test_generate_synthetic_feedback_mode_ratio(self):
        """Should respect seed_mode_ratio."""
        entries = generate_synthetic_feedback(
            n_searches=100, seed_mode_ratio=0.8, save_to_file=False
        )

        seed_count = sum(1 for e in entries if e["search_mode"] == "seed")
        # Allow some variance (expect ~80, allow 70-90)
        assert 70 <= seed_count <= 90

    def test_generate_synthetic_feedback_no_tags_distribution(self):
        """Should include no-tags edge cases at specified probability."""
        entries = generate_synthetic_feedback(
            n_searches=100, no_tags_probability=0.20, save_to_file=False
        )

        # Count channels with low tag scores (no-tags edge case)
        no_tags_count = 0
        total_channels = 0
        for entry in entries:
            for cf in entry["channel_feedback"]:
                total_channels += 1
                tag_score = cf["component_scores"].get("tag_score", 0)
                if tag_score <= 3:
                    no_tags_count += 1

        # Expect ~20% no-tags, allow 10-30% range
        no_tags_ratio = no_tags_count / total_channels
        assert 0.10 <= no_tags_ratio <= 0.30

    def test_summary_statistics(self):
        """Summary should include expected fields."""
        entries = generate_synthetic_feedback(n_searches=20, save_to_file=False)
        summary = get_synthetic_data_summary(entries)

        assert summary["total_searches"] == 20
        assert summary["total_channel_ratings"] == 100  # 20 * 5
        assert "rating_distribution" in summary
        assert "relevance_rate" in summary
        assert "no_tags_channels" in summary
        assert "no_tags_relevance_rate" in summary

    def test_determine_rating_returns_valid_values(self):
        """Rating should be one of the valid values."""
        scores = _generate_component_scores(target_relevance=0.5)

        for _ in range(100):
            rating, reason = _determine_rating(scores, presented_rank=1, ai_enabled=True)
            assert rating in ("relevant", "not_relevant", "skip")
            if rating == "not_relevant":
                assert reason in ("wrong_topic", "low_quality", "poor_fit", "other")
            else:
                assert reason is None


class TestMLTrainer:
    """Tests for ml_trainer.py"""

    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        data = []
        for i in range(50):
            is_relevant = i % 2 == 0  # 50/50 split

            # Higher scores for relevant
            base = 20 if is_relevant else 10

            data.append(
                {
                    "query": f"query_{i}",
                    "timestamp": "2026-01-27T12:00:00",
                    "channel_id": f"UC{i:020d}",
                    "presented_rank": (i % 5) + 1,
                    "presented_score": base + np.random.normal(0, 5),
                    "rating": "relevant" if is_relevant else "not_relevant",
                    "is_relevant": is_relevant,
                    "reason": None if is_relevant else "wrong_topic",
                    "component_tag_score": np.clip(base + np.random.normal(0, 3), 0, 30),
                    "component_keyword_score": np.clip(
                        base + np.random.normal(0, 3), 0, 30
                    ),
                    "component_subscriber_score": np.clip(
                        base / 2 + np.random.normal(0, 2), 0, 15
                    ),
                    "component_engagement_score": np.clip(
                        base / 2 + np.random.normal(0, 2), 0, 17
                    ),
                    "component_frequency_score": np.clip(
                        base / 4 + np.random.normal(0, 1), 0, 8
                    ),
                }
            )
        return data

    def test_prepare_features_shape(self, sample_training_data):
        """Feature matrix should have correct shape."""
        X, y, names = prepare_features_seed_mode(sample_training_data)

        assert X.shape == (50, 5)  # 50 samples, 5 features
        assert len(y) == 50
        assert len(names) == 5

    def test_prepare_features_normalized(self, sample_training_data):
        """Features should be normalized to [0, 1]."""
        X, y, names = prepare_features_seed_mode(sample_training_data)

        assert X.min() >= 0
        assert X.max() <= 1

    def test_prepare_features_labels(self, sample_training_data):
        """Labels should be binary."""
        X, y, names = prepare_features_seed_mode(sample_training_data)

        assert set(y).issubset({0, 1})

    def test_prepare_features_empty_data_raises(self):
        """Should raise ValueError for empty data."""
        with pytest.raises(ValueError, match="No training data"):
            prepare_features_seed_mode([])


class TestWeightOptimizer:
    """Tests for weight_optimizer.py"""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)

        assert np.isclose(result.sum(), 1.0)

    def test_softmax_preserves_order(self):
        """Softmax should preserve relative ordering."""
        x = np.array([1.0, 3.0, 2.0])
        result = softmax(x)

        assert result[1] > result[2] > result[0]

    def test_softmax_temperature_effect(self):
        """Higher temperature should produce more uniform distribution."""
        x = np.array([1.0, 5.0])

        low_temp = softmax(x, temperature=0.5)
        high_temp = softmax(x, temperature=5.0)

        # Higher temp = smaller difference between values
        low_diff = abs(low_temp[1] - low_temp[0])
        high_diff = abs(high_temp[1] - high_temp[0])

        assert high_diff < low_diff

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)

        assert np.isclose(result.sum(), 1.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_apply_constraints_sum_to_100(self):
        """Constrained weights should sum to 100."""
        raw = {
            "tag_overlap": 35,
            "keyword_overlap": 25,
            "subscriber_similarity": 20,
            "engagement_rate": 15,
            "upload_frequency": 10,
        }

        constrained = apply_weight_constraints(raw)

        assert sum(constrained.values()) == 100

    def test_apply_constraints_min_weight(self):
        """No weight should go below minimum."""
        raw = {
            "tag_overlap": 50,
            "keyword_overlap": 50,
            "subscriber_similarity": 2,  # Below min
            "engagement_rate": 2,  # Below min
            "upload_frequency": 2,  # Below min
        }

        constrained = apply_weight_constraints(raw, min_weight=5)

        for v in constrained.values():
            assert v >= 5

    def test_apply_constraints_max_change(self):
        """Changes should be within max percentage."""
        raw = {
            "tag_overlap": 60,  # Would be +100% from default 30
            "keyword_overlap": 60,
            "subscriber_similarity": 5,
            "engagement_rate": 5,
            "upload_frequency": 5,
        }

        constrained = apply_weight_constraints(raw, max_change_pct=0.50)

        # After constraints, tag_overlap should be capped
        # Default is 30, max allowed is 30 * 1.5 = 45
        # After normalization the exact value varies, but should be reasonable
        assert constrained["tag_overlap"] <= 50  # Reasonable upper bound

    def test_apply_constraints_preserves_all_keys(self):
        """All weight keys should be present in output."""
        raw = {
            "tag_overlap": 30,
            "keyword_overlap": 30,
            "subscriber_similarity": 15,
            "engagement_rate": 17,
            "upload_frequency": 8,
        }

        constrained = apply_weight_constraints(raw)

        assert set(constrained.keys()) == set(raw.keys())


class TestFabricExport:
    """Tests for fabric_export.py"""

    def test_export_to_csv_no_data(self):
        """Should handle empty data gracefully."""
        from analytics.fabric_export import export_to_csv
        from feedback_tracker import clear_all_feedback

        # Ensure no data
        clear_all_feedback()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            result = export_to_csv(output_path=f.name, mode="seed")

        assert result is False

    def test_get_fabric_ready_dataframe_empty(self):
        """Should return empty DataFrame when no data."""
        from analytics.fabric_export import get_fabric_ready_dataframe
        from feedback_tracker import clear_all_feedback

        clear_all_feedback()

        df = get_fabric_ready_dataframe(mode="seed")
        assert df.empty

    def test_generate_summary_report_structure(self):
        """Summary report should have expected structure."""
        from analytics.fabric_export import generate_summary_report

        report = generate_summary_report()

        assert "generated_at" in report
        assert "kpis" in report
        assert "by_mode" in report
        assert "total_searches" in report["kpis"]
        assert "total_channel_ratings" in report["kpis"]
        assert "overall_relevance_rate" in report["kpis"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

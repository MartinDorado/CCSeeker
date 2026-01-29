"""
weight_optimizer.py - Convert ML coefficients to scoring weights

This module takes the coefficients from a trained logistic regression
model and converts them into adjusted scoring weights that can be
used by similarity.py (in app/core/).

The key insight: softmax(coefficients) gives relative importance,
which we can scale to sum to the target total (100 points).
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

# Add parent directory for imports when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from ..core.scoring_version import SEED_WEIGHTS, SCORING_VERSION
except ImportError:
    from core.scoring_version import SEED_WEIGHTS, SCORING_VERSION

try:
    from .ml_trainer import TrainedModel
except ImportError:
    from ml_trainer import TrainedModel


@dataclass
class OptimizedWeights:
    """Container for optimized weight configuration."""
    version: str
    updated_at: str
    model_accuracy: float
    sample_count: int

    # Similarity weights (seed mode)
    tag_overlap: int
    keyword_overlap: int
    subscriber_similarity: int
    engagement_rate: int
    upload_frequency: int

    # Metadata
    changes_from_default: dict
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> dict:
        return asdict(self)

    def total_points(self) -> int:
        return (
            self.tag_overlap + self.keyword_overlap +
            self.subscriber_similarity + self.engagement_rate +
            self.upload_frequency
        )


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax function with temperature scaling.

    Higher temperature = more uniform distribution
    Lower temperature = more extreme distribution

    Args:
        x: Input array (coefficients)
        temperature: Scaling factor (default 1.0)

    Returns:
        Probability distribution summing to 1
    """
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def apply_weight_constraints(
    raw_weights: dict[str, float],
    min_weight: int = 5,
    max_change_pct: float = 0.50,
) -> dict[str, int]:
    """
    Apply constraints to prevent extreme weight changes.

    Constraints:
    1. Minimum weight of 5 points per component
    2. Maximum +/-50% change from default weights

    Args:
        raw_weights: Unconstrained weights (may not sum to 100)
        min_weight: Minimum allowed weight per component
        max_change_pct: Maximum percentage change from default (0-1)

    Returns:
        Constrained weights summing to 100
    """
    defaults = {
        "tag_overlap": SEED_WEIGHTS.tag_overlap,
        "keyword_overlap": SEED_WEIGHTS.keyword_overlap,
        "subscriber_similarity": SEED_WEIGHTS.subscriber_similarity,
        "engagement_rate": SEED_WEIGHTS.engagement_rate,
        "upload_frequency": SEED_WEIGHTS.upload_frequency,
    }

    constrained = {}

    for name, raw in raw_weights.items():
        default = defaults[name]

        # Apply min weight
        weight = max(raw, min_weight)

        # Apply max change constraint
        min_allowed = default * (1 - max_change_pct)
        max_allowed = default * (1 + max_change_pct)
        weight = np.clip(weight, min_allowed, max_allowed)

        constrained[name] = weight

    # Normalize to sum to 100
    total = sum(constrained.values())
    normalized = {k: round(v / total * 100) for k, v in constrained.items()}

    # Adjust for rounding errors (ensure sum is exactly 100)
    diff = 100 - sum(normalized.values())
    if diff != 0:
        # Add/subtract from largest weight
        largest = max(normalized, key=normalized.get)
        normalized[largest] += diff

    return normalized


def optimize_weights(
    trained_model: TrainedModel,
    temperature: float = 2.0,
    min_accuracy: float = 0.65,
) -> Optional[OptimizedWeights]:
    """
    Convert trained model coefficients to optimized scoring weights.

    Process:
    1. Extract coefficients from trained model
    2. Apply softmax to get relative importance
    3. Scale to target total (100 points)
    4. Apply constraints (min weight, max change)
    5. Package with metadata

    Args:
        trained_model: Output from ml_trainer.train_weight_model()
        temperature: Softmax temperature (higher = more conservative changes)
        min_accuracy: Minimum CV accuracy required to trust results

    Returns:
        OptimizedWeights ready for use
        None if model doesn't meet confidence threshold
    """
    # Check confidence threshold
    cv_accuracy = trained_model.metrics.cv_accuracy_mean
    cv_std = trained_model.metrics.cv_accuracy_std

    if cv_accuracy < min_accuracy:
        print(f"Model accuracy ({cv_accuracy:.1%}) below threshold ({min_accuracy:.1%})")
        print("Optimization rejected - continue collecting feedback.")
        return None

    # Map coefficient names to weight names
    coef_to_weight = {
        "component_tag_score": "tag_overlap",
        "component_keyword_score": "keyword_overlap",
        "component_subscriber_score": "subscriber_similarity",
        "component_engagement_score": "engagement_rate",
        "component_frequency_score": "upload_frequency",
    }

    # Extract coefficients (only for scoring components)
    coefficients = []
    weight_names = []

    for coef_name, weight_name in coef_to_weight.items():
        if coef_name in trained_model.coefficients:
            coefficients.append(trained_model.coefficients[coef_name])
            weight_names.append(weight_name)

    if len(coefficients) != 5:
        print(f"Expected 5 coefficients, got {len(coefficients)}")
        return None

    coefficients = np.array(coefficients)

    # Convert coefficients to importance via softmax
    # Add small positive shift to handle negative coefficients meaningfully
    shifted_coefs = coefficients - coefficients.min() + 0.1
    importance = softmax(shifted_coefs, temperature=temperature)

    # Scale to 100 points
    raw_weights = {name: imp * 100 for name, imp in zip(weight_names, importance)}

    # Apply constraints
    constrained_weights = apply_weight_constraints(raw_weights)

    # Calculate changes from default
    defaults = {
        "tag_overlap": SEED_WEIGHTS.tag_overlap,
        "keyword_overlap": SEED_WEIGHTS.keyword_overlap,
        "subscriber_similarity": SEED_WEIGHTS.subscriber_similarity,
        "engagement_rate": SEED_WEIGHTS.engagement_rate,
        "upload_frequency": SEED_WEIGHTS.upload_frequency,
    }

    changes = {
        name: constrained_weights[name] - defaults[name]
        for name in constrained_weights
    }

    # Determine confidence level
    if cv_accuracy >= 0.75 and cv_std < 0.10:
        confidence = "high"
    elif cv_accuracy >= 0.65 and cv_std < 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    return OptimizedWeights(
        version=f"learned-{SCORING_VERSION}",
        updated_at=datetime.now().isoformat(),
        model_accuracy=round(cv_accuracy, 4),
        sample_count=trained_model.metrics.n_samples,
        tag_overlap=constrained_weights["tag_overlap"],
        keyword_overlap=constrained_weights["keyword_overlap"],
        subscriber_similarity=constrained_weights["subscriber_similarity"],
        engagement_rate=constrained_weights["engagement_rate"],
        upload_frequency=constrained_weights["upload_frequency"],
        changes_from_default=changes,
        confidence=confidence,
    )


def save_optimized_weights(
    weights: OptimizedWeights,
    filepath: str = None,
) -> bool:
    """
    Save optimized weights to JSON file.

    Args:
        weights: OptimizedWeights to save
        filepath: Output path (default: app/config/learned_weights.json)

    Returns:
        True if saved successfully
    """
    if filepath is None:
        # Default to config directory
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(app_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        filepath = os.path.join(config_dir, "learned_weights.json")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(weights.to_dict(), f, indent=2)
        print(f"Saved optimized weights to: {filepath}")
        return True
    except IOError as e:
        print(f"Error saving weights: {e}")
        return False


def print_optimization_report(weights: OptimizedWeights) -> None:
    """Print a formatted optimization report."""
    defaults = {
        "tag_overlap": SEED_WEIGHTS.tag_overlap,
        "keyword_overlap": SEED_WEIGHTS.keyword_overlap,
        "subscriber_similarity": SEED_WEIGHTS.subscriber_similarity,
        "engagement_rate": SEED_WEIGHTS.engagement_rate,
        "upload_frequency": SEED_WEIGHTS.upload_frequency,
    }

    print("\n" + "=" * 60)
    print("WEIGHT OPTIMIZATION REPORT")
    print("=" * 60)

    print(f"\nMODEL INFO")
    print(f"  Version: {weights.version}")
    print(f"  Accuracy: {weights.model_accuracy:.1%}")
    print(f"  Samples: {weights.sample_count}")
    print(f"  Confidence: {weights.confidence.upper()}")

    print(f"\nOPTIMIZED WEIGHTS (vs defaults)")

    optimized = {
        "tag_overlap": weights.tag_overlap,
        "keyword_overlap": weights.keyword_overlap,
        "subscriber_similarity": weights.subscriber_similarity,
        "engagement_rate": weights.engagement_rate,
        "upload_frequency": weights.upload_frequency,
    }

    for name, value in optimized.items():
        default = defaults[name]
        change = value - default
        arrow = "^" if change > 0 else "v" if change < 0 else "="
        print(f"  {name}: {value} ({arrow} {abs(change)} from {default})")

    print(f"\n  Total: {weights.total_points()} points")

    print(f"\nCHANGES SUMMARY")
    if all(c == 0 for c in weights.changes_from_default.values()):
        print("  No significant changes from defaults.")
    else:
        sorted_changes = sorted(
            weights.changes_from_default.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for name, change in sorted_changes:
            if change != 0:
                direction = "increased" if change > 0 else "decreased"
                print(f"  {name}: {direction} by {abs(change)} points")

    print("=" * 60)


if __name__ == "__main__":
    from analytics.ml_trainer import train_weight_model

    print("Training model...")
    model = train_weight_model(mode="seed")

    if model:
        print("\nOptimizing weights...")
        weights = optimize_weights(model)

        if weights:
            print_optimization_report(weights)

            # Optionally save
            print("\nTo save optimized weights, run:")
            print("  save_optimized_weights(weights)")
    else:
        print("No model trained - cannot optimize weights.")
        print("\nTo generate synthetic data first, run:")
        print("  python -c \"from app.analytics.synthetic_data_generator import generate_synthetic_feedback; generate_synthetic_feedback(60)\"")

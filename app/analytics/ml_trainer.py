"""
ml_trainer.py - Train ML models on feedback data for weight optimization

This module implements logistic regression training to learn which scoring
components best predict user satisfaction (relevant vs not_relevant).

The key insight: logistic regression coefficients directly indicate
feature importance, which we can convert to scoring weights.
"""

import sys
import os
from typing import Optional
from dataclasses import dataclass

# Add parent directory for imports when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NumPy and pandas
import numpy as np
import pandas as pd

# scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from ..core.scoring_version import SEED_WEIGHTS
except ImportError:
    from core.scoring_version import SEED_WEIGHTS

try:
    from .feedback_tracker import get_training_data
except ImportError:
    from feedback_tracker import get_training_data


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    n_samples: int
    n_positive: int
    n_negative: int


@dataclass
class TrainedModel:
    """Container for trained model and associated data."""
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    metrics: ModelMetrics
    coefficients: dict[str, float]


def prepare_features_seed_mode(
    training_data: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare feature matrix and labels for seed mode training.

    Features are normalized by their maximum possible values to put
    them on comparable scales for coefficient interpretation.

    Args:
        training_data: Output from feedback_tracker.get_training_data("seed")

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - 1 for relevant, 0 for not_relevant
        feature_names: List of feature names in order
    """
    if not training_data:
        raise ValueError("No training data provided")

    df = pd.DataFrame(training_data)

    # Define features and their normalization factors
    # These are the max possible scores from SEED_WEIGHTS
    features_config = {
        "component_tag_score": SEED_WEIGHTS.tag_overlap,  # 30
        "component_keyword_score": SEED_WEIGHTS.keyword_overlap,  # 30
        "component_subscriber_score": SEED_WEIGHTS.subscriber_similarity,  # 15
        "component_engagement_score": SEED_WEIGHTS.engagement_rate,  # 17
        "component_frequency_score": SEED_WEIGHTS.upload_frequency,  # 8
    }

    # Build feature matrix with normalized values
    feature_names = list(features_config.keys())
    X = np.zeros((len(df), len(feature_names)))

    for i, feat in enumerate(feature_names):
        max_val = features_config[feat]
        X[:, i] = df[feat].fillna(0).values / max_val

    # Labels
    y = df["is_relevant"].astype(int).values

    return X, y, feature_names


def train_weight_model(
    mode: str = "seed",
    only_compatible: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Optional[TrainedModel]:
    """
    Train logistic regression model on feedback data.

    The model learns which scoring components best predict whether
    a user will rate a channel as "relevant".

    Args:
        mode: "seed" or "keyword"
        only_compatible: Only use feedback compatible with current scoring version
        test_size: Fraction of data to hold out for testing
        random_state: Random seed for reproducibility

    Returns:
        TrainedModel with model, metrics, and coefficients
        None if insufficient data

    Raises:
        ValueError: If mode is not supported or data is insufficient
    """
    # Get training data
    training_data = get_training_data(mode=mode, only_compatible=only_compatible)

    if len(training_data) < 30:
        print(f"Warning: Only {len(training_data)} samples available. Need at least 30.")
        return None

    # Prepare features based on mode
    if mode == "seed":
        X, y, feature_names = prepare_features_seed_mode(training_data)
    else:
        raise ValueError(f"Mode '{mode}' not yet implemented. Use 'seed'.")

    # Check class balance
    n_positive = int(sum(y))
    n_negative = len(y) - n_positive

    if n_positive < 10 or n_negative < 10:
        print(f"Warning: Class imbalance - {n_positive} positive, {n_negative} negative")
        print("Need at least 10 samples per class for reliable training.")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features (important for coefficient interpretation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight="balanced",  # Handle any remaining imbalance
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Cross-validation on full dataset for robust accuracy estimate
    X_scaled_full = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled_full, y, cv=5)

    metrics = ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cv_accuracy_mean=float(cv_scores.mean()),
        cv_accuracy_std=float(cv_scores.std()),
        n_samples=len(y),
        n_positive=n_positive,
        n_negative=n_negative,
    )

    # Extract coefficients
    coefficients = {
        name: float(coef) for name, coef in zip(feature_names, model.coef_[0])
    }

    return TrainedModel(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        metrics=metrics,
        coefficients=coefficients,
    )


def evaluate_model(trained_model: TrainedModel) -> dict:
    """
    Generate a human-readable evaluation report.

    Args:
        trained_model: Output from train_weight_model()

    Returns:
        dict with formatted evaluation results
    """
    m = trained_model.metrics

    report = {
        "summary": {
            "samples": m.n_samples,
            "positive_samples": m.n_positive,
            "negative_samples": m.n_negative,
            "class_balance": f"{m.n_positive / m.n_samples:.1%} positive",
        },
        "test_metrics": {
            "accuracy": f"{m.accuracy:.1%}",
            "precision": f"{m.precision:.1%}",
            "recall": f"{m.recall:.1%}",
            "f1_score": f"{m.f1:.1%}",
        },
        "cross_validation": {
            "mean_accuracy": f"{m.cv_accuracy_mean:.1%}",
            "std_accuracy": f"+/-{m.cv_accuracy_std:.1%}",
            "reliable": m.cv_accuracy_std < 0.15,  # Low variance = reliable
        },
        "coefficients": {
            name.replace("component_", ""): {
                "value": round(coef, 4),
                "direction": "positive" if coef > 0 else "negative",
                "importance": (
                    "high" if abs(coef) > 0.5
                    else "medium" if abs(coef) > 0.2
                    else "low"
                ),
            }
            for name, coef in trained_model.coefficients.items()
        },
        "recommendation": _get_recommendation(trained_model),
    }

    return report


def _get_recommendation(trained_model: TrainedModel) -> str:
    """Generate actionable recommendation based on model results."""
    m = trained_model.metrics

    if m.cv_accuracy_std > 0.15:
        return "Model variance is high. Collect more feedback data before updating weights."

    if m.cv_accuracy_mean < 0.60:
        return "Model accuracy is low. Current features may not predict satisfaction well."

    if m.cv_accuracy_mean >= 0.70:
        return "Model is reliable. Consider applying learned weights to scoring."

    return "Model shows promise. Continue collecting feedback to improve reliability."


def print_training_report(trained_model: TrainedModel) -> None:
    """Print a formatted training report to console."""
    report = evaluate_model(trained_model)

    print("\n" + "=" * 60)
    print("ML TRAINING REPORT - Weight Learning")
    print("=" * 60)

    print("\nDATA SUMMARY")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")

    print("\nTEST METRICS")
    for k, v in report["test_metrics"].items():
        print(f"  {k}: {v}")

    print("\nCROSS-VALIDATION")
    for k, v in report["cross_validation"].items():
        print(f"  {k}: {v}")

    print("\nFEATURE COEFFICIENTS (higher = more predictive of relevance)")
    sorted_coefs = sorted(
        report["coefficients"].items(),
        key=lambda x: abs(x[1]["value"]),
        reverse=True,
    )
    for name, data in sorted_coefs:
        arrow = "+" if data["direction"] == "positive" else "-"
        print(f"  {arrow} {name}: {data['value']:+.4f} ({data['importance']})")

    print(f"\nRECOMMENDATION: {report['recommendation']}")
    print("=" * 60)


if __name__ == "__main__":
    print("Training ML model on feedback data...")

    result = train_weight_model(mode="seed")

    if result:
        print_training_report(result)
    else:
        print("Insufficient data for training. Generate synthetic data first.")
        print("\nTo generate synthetic data, run:")
        print("  python -c \"from app.analytics.synthetic_data_generator import generate_synthetic_feedback; generate_synthetic_feedback(60)\"")

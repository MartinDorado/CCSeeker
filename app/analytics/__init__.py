"""
Analytics module for CCSeeker ML training and data export.

This module provides tools for:
- Generating synthetic feedback data for ML training demos
- Training logistic regression models to learn optimal scoring weights
- Optimizing weights based on learned coefficients
- Exporting data for Microsoft Fabric / Power BI analytics
"""

from .synthetic_data_generator import (
    generate_synthetic_feedback,
    generate_search_feedback,
    get_synthetic_data_summary,
)
from .ml_trainer import (
    train_weight_model,
    evaluate_model,
    TrainedModel,
    ModelMetrics,
)
from .weight_optimizer import (
    optimize_weights,
    apply_weight_constraints,
    save_optimized_weights,
    OptimizedWeights,
)
from .fabric_export import (
    export_to_parquet,
    export_to_csv,
    get_fabric_ready_dataframe,
    generate_summary_report,
)

__all__ = [
    # Synthetic data generation
    "generate_synthetic_feedback",
    "generate_search_feedback",
    "get_synthetic_data_summary",
    # ML training
    "train_weight_model",
    "evaluate_model",
    "TrainedModel",
    "ModelMetrics",
    # Weight optimization
    "optimize_weights",
    "apply_weight_constraints",
    "save_optimized_weights",
    "OptimizedWeights",
    # Fabric export
    "export_to_parquet",
    "export_to_csv",
    "get_fabric_ready_dataframe",
    "generate_summary_report",
]

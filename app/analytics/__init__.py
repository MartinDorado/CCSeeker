"""
Analytics module for CCSeeker ML training and data export.

This module provides tools for:
- Collecting and managing user feedback data
- Tracking API quota usage and costs
- Generating synthetic feedback data for ML training demos
- Training logistic regression models to learn optimal scoring weights
- Optimizing weights based on learned coefficients
- Exporting data for Microsoft Fabric / Power BI analytics
"""

from .quota_tracker import (
    YOUTUBE_QUOTA_COSTS,
    GEMINI_COSTS,
    DebugData,
    DailyQuota,
    calculate_youtube_quota_used,
    calculate_gemini_cost_estimate,
    get_total_youtube_calls,
    get_total_gemini_calls,
    get_current_date_pt,
    get_next_reset_time,
    load_daily_quota,
    save_daily_quota,
    track_api_call,
    track_timing,
    track_similarity_scores,
    accumulate_to_daily_quota,
    create_empty_debug_data,
)
from .feedback_tracker import (
    FEEDBACK_SCHEMA_VERSION,
    save_channel_feedback,
    build_channel_feedback_entry,
    get_feedback_stats,
    get_training_data,
    get_negative_feedback_entries,
    export_feedback_csv,
    clear_incompatible_feedback,
    clear_all_feedback,
)
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
    # Quota tracking
    "YOUTUBE_QUOTA_COSTS",
    "GEMINI_COSTS",
    "DebugData",
    "DailyQuota",
    "calculate_youtube_quota_used",
    "calculate_gemini_cost_estimate",
    "get_total_youtube_calls",
    "get_total_gemini_calls",
    "get_current_date_pt",
    "get_next_reset_time",
    "load_daily_quota",
    "save_daily_quota",
    "track_api_call",
    "track_timing",
    "track_similarity_scores",
    "accumulate_to_daily_quota",
    "create_empty_debug_data",
    # Feedback tracking
    "FEEDBACK_SCHEMA_VERSION",
    "save_channel_feedback",
    "build_channel_feedback_entry",
    "get_feedback_stats",
    "get_training_data",
    "get_negative_feedback_entries",
    "export_feedback_csv",
    "clear_incompatible_feedback",
    "clear_all_feedback",
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

"""
fabric_export.py - Export feedback data for Microsoft Fabric analytics

This module exports feedback data in formats suitable for import into
Microsoft Fabric Lakehouse (Parquet) or Power BI (CSV).

The exported data is flattened and denormalized for easy analytics
and dashboard creation.
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory for imports when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from feedback_tracker import get_training_data, get_feedback_stats, _load_feedback_data


def export_to_parquet(output_path: str = None, mode: str = "all") -> bool:
    """
    Export feedback data to Parquet format for Fabric Lakehouse.

    Parquet is the preferred format for Fabric because:
    - Columnar storage = fast analytics queries
    - Efficient compression
    - Native support in Fabric notebooks

    Args:
        output_path: Path for output file (default: feedback_export.parquet)
        mode: "seed", "keyword", or "all"

    Returns:
        True if exported successfully
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"feedback_export_{timestamp}.parquet"

    # Get data for each mode
    records = []

    if mode in ("seed", "all"):
        seed_data = get_training_data(mode="seed", only_compatible=False)
        for r in seed_data:
            r["search_mode"] = "seed"
        records.extend(seed_data)

    if mode in ("keyword", "all"):
        keyword_data = get_training_data(mode="keyword", only_compatible=False)
        for r in keyword_data:
            r["search_mode"] = "keyword"
        records.extend(keyword_data)

    if not records:
        print("No data to export.")
        return False

    df = pd.DataFrame(records)

    try:
        df.to_parquet(output_path, index=False)
        print(f"Exported {len(df)} records to: {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def export_to_csv(output_path: str = None, mode: str = "all") -> bool:
    """
    Export feedback data to CSV format.

    CSV is simpler for:
    - Manual inspection
    - Excel import
    - Quick Power BI connection

    Args:
        output_path: Path for output file
        mode: "seed", "keyword", or "all"

    Returns:
        True if exported successfully
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"feedback_export_{timestamp}.csv"

    records = []

    if mode in ("seed", "all"):
        seed_data = get_training_data(mode="seed", only_compatible=False)
        for r in seed_data:
            r["search_mode"] = "seed"
        records.extend(seed_data)

    if mode in ("keyword", "all"):
        keyword_data = get_training_data(mode="keyword", only_compatible=False)
        for r in keyword_data:
            r["search_mode"] = "keyword"
        records.extend(keyword_data)

    if not records:
        print("No data to export.")
        return False

    df = pd.DataFrame(records)

    try:
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to: {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def get_fabric_ready_dataframe(mode: str = "seed") -> pd.DataFrame:
    """
    Get a DataFrame ready for Fabric analytics.

    This adds computed columns useful for dashboards:
    - is_relevant (boolean)
    - total_component_score
    - relevance_rate calculation base

    Args:
        mode: "seed" or "keyword"

    Returns:
        pd.DataFrame ready for analysis
    """
    data = get_training_data(mode=mode, only_compatible=False)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Add computed columns for seed mode
    if mode == "seed":
        component_cols = [
            "component_tag_score",
            "component_keyword_score",
            "component_subscriber_score",
            "component_engagement_score",
            "component_frequency_score",
        ]

        # Total component score
        existing_cols = [c for c in component_cols if c in df.columns]
        if existing_cols:
            df["total_component_score"] = df[existing_cols].fillna(0).sum(axis=1)

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour

    return df


def generate_summary_report() -> dict:
    """
    Generate a summary report for Fabric dashboard.

    Returns:
        dict with KPIs and aggregated metrics
    """
    stats = get_feedback_stats()

    # Get DataFrames for detailed analysis
    seed_df = get_fabric_ready_dataframe("seed")

    total_ratings = stats["total_channel_ratings"]
    relevant_count = stats["rating_breakdown"]["relevant"]

    report = {
        "generated_at": datetime.now().isoformat(),
        "kpis": {
            "total_searches": stats["total_entries"],
            "total_channel_ratings": total_ratings,
            "overall_relevance_rate": (
                relevant_count / total_ratings if total_ratings > 0 else 0
            ),
        },
        "by_mode": {},
        "by_reason": stats.get("reason_breakdown", {}),
    }

    # Mode breakdown
    for mode in ["seed", "keyword"]:
        mode_stats = stats["by_search_mode"].get(mode, {})
        relevant = mode_stats.get("relevant", 0)
        not_relevant = mode_stats.get("not_relevant", 0)
        total = relevant + not_relevant

        report["by_mode"][mode] = {
            "searches": mode_stats.get("entries", 0),
            "relevant": relevant,
            "not_relevant": not_relevant,
            "relevance_rate": relevant / total if total > 0 else 0,
        }

    # Score distributions for seed mode
    if not seed_df.empty and "total_component_score" in seed_df.columns:
        report["seed_score_stats"] = {
            "mean": float(seed_df["total_component_score"].mean()),
            "median": float(seed_df["total_component_score"].median()),
            "std": float(seed_df["total_component_score"].std()),
            "min": float(seed_df["total_component_score"].min()),
            "max": float(seed_df["total_component_score"].max()),
        }

    return report


if __name__ == "__main__":
    print("Exporting feedback data for Fabric...")

    # Check if there's data to export
    stats = get_feedback_stats()
    if stats["total_entries"] == 0:
        print("No feedback data found. Generate synthetic data first:")
        print("  python -c \"from app.analytics.synthetic_data_generator import generate_synthetic_feedback; generate_synthetic_feedback(60)\"")
    else:
        # Export Parquet
        export_to_parquet()

        # Generate summary
        report = generate_summary_report()
        print("\n=== SUMMARY REPORT ===")
        print(json.dumps(report, indent=2, default=str))

"""
Shared fixtures and cache-clearing for AppTest UI tests.

All tests in this directory run AppTest.from_file("app/main.py") against the
real app script with external calls mocked at the module boundary.
"""
import os
import sys

import pandas as pd
import pytest
import streamlit as st

# Ensure app/ is on the path so core imports resolve when building fixtures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

APP_PATH = "app/main.py"


@pytest.fixture(autouse=True)
def clear_st_caches():
    """Prevent @st.cache_data / @st.cache_resource bleed between test runs."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def fake_channels_df():
    return pd.DataFrame({
        "channel_id":        ["UC001", "UC002", "UC003"],
        "channel_title":     ["Manga World", "Anime Hub", "Otaku Central"],
        "channel_url":       [
            "https://youtube.com/c/mangaworld",
            "https://youtube.com/c/animehub",
            "https://youtube.com/c/otakucentral",
        ],
        "subscribers":       [50_000, 120_000, 35_000],
        "relevance_score":   [0.92, 0.85, 0.78],
        "avg_views":         [10_000, 25_000, 8_000],
        "engagement_rate":   [0.05, 0.04, 0.06],
        "upload_frequency":  [4.2, 3.1, 2.8],
    })


@pytest.fixture
def fake_pipeline_result(fake_channels_df):
    from core.pipeline import PipelineResult
    return PipelineResult(
        channels_df=fake_channels_df,
        display_columns=list(fake_channels_df.columns),
        column_explanations={"relevance_score": "How relevant this channel is to your query"},
        top_channels_for_outreach=fake_channels_df.head(3),
        final_query="manga, anime",
    )


@pytest.fixture
def fake_pipeline_error():
    from core.pipeline import PipelineResult
    return PipelineResult(
        channels_df=pd.DataFrame(),
        display_columns=[],
        column_explanations={},
        top_channels_for_outreach=pd.DataFrame(),
        final_query="manga, anime",
        error="Search did not return any channels matching your query.",
    )


@pytest.fixture
def fake_seed_profile():
    return {
        "channel_name":       "MangaWorld",
        "channel_id":         "UC001",
        "subscriber_count":   150_000,
        "subscriber_tier":    "medium",
        "language":           "en",
        "upload_frequency":   4.5,
        "avg_engagement_rate": 0.045,
        "primary_keywords":   ["manga review", "anime analysis"],
        "secondary_keywords": ["manga", "anime", "review"],
        "common_tags":        ["manga", "anime", "comics", "review"],
        "description_summary": None,
        "gemini_api_key":     None,
    }

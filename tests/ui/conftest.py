"""
Shared fixtures and cache-clearing for AppTest UI tests.

All tests in this directory run AppTest.from_file("app/main.py") against the
real app script with external calls mocked at the module boundary.
"""
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
import streamlit as st

# Ensure app/ is on the path so core imports resolve when building fixtures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))

APP_PATH = "app/main.py"
# AppTest re-executes module-level code (pycountry, CSS read, image load) on every
# run() call.  3s (the default) is too tight on a cold start; 10s gives headroom.
DEFAULT_TIMEOUT = 10


def make_at(**initial_session_state) -> "AppTest":
    """Create an AppTest instance with a safe timeout and optional pre-set session state."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP_PATH, default_timeout=DEFAULT_TIMEOUT)
    for key, value in initial_session_state.items():
        at.session_state[key] = value
    return at


@pytest.fixture(scope="session", autouse=True)
def warmup_apptest():
    """Absorb the cold-start cost (Python bytecode compilation + first imports of
    supabase, scikit-learn, pycountry, google-api-python-client, etc.) once per
    test session so individual tests see a warm sys.modules and don't time out."""
    from streamlit.testing.v1 import AppTest
    try:
        AppTest.from_file(APP_PATH, default_timeout=60).run()
    except Exception:
        pass  # individual tests will surface the real error if boot is broken


@pytest.fixture
def no_api_keys():
    """Shadow API keys with empty strings so the app takes the 'not configured' path.

    load_dotenv() uses override=False by default, so it will not overwrite an env var
    that is already set — even to an empty string.  This ensures the 'no key' error
    branch in main.py is reached regardless of what is in the developer's .env file.
    """
    with patch.dict(os.environ, {"YOUTUBE_API_KEY": "", "GEMINI_API_KEY": ""}):
        yield


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

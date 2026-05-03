"""
AppTest tests for error states and edge cases.

Verifies the app surfaces errors gracefully rather than crashing,
and that contextual suggestions appear for specific error types.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.ui.conftest import make_at

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))


def _error_result(error_message: str):
    from core.pipeline import PipelineResult
    return PipelineResult(
        channels_df=pd.DataFrame(),
        display_columns=[],
        column_explanations={},
        top_channels_for_outreach=pd.DataFrame(),
        final_query="manga, anime",
        error=error_message,
    )


def test_pipeline_error_no_channels_shows_suggestions():
    """'did not return any channels' error should surface actionable suggestions."""
    with patch("core.run_search_pipeline",
               return_value=_error_result("Search did not return any channels matching your query.")), \
         patch("googleapiclient.discovery.build", return_value=MagicMock()):

        at = make_at(search_method="Keywords", user_youtube_key="AIza_fake_key_for_testing")
        at.run()
        next(b for b in at.button if b.label == "Find Creators").click().run()

    assert not at.exception
    assert len(at.error) > 0
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "broader" in all_markdown or "Suggestions" in all_markdown


def test_pipeline_error_filtering_shows_suggestions():
    """'filtering criteria' error should surface filter-adjustment suggestions."""
    with patch("core.run_search_pipeline",
               return_value=_error_result("All channels were removed by filtering criteria.")), \
         patch("googleapiclient.discovery.build", return_value=MagicMock()):

        at = make_at(search_method="Keywords", user_youtube_key="AIza_fake_key_for_testing")
        at.run()
        next(b for b in at.button if b.label == "Find Creators").click().run()

    assert not at.exception
    assert len(at.error) > 0
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "subscriber" in all_markdown.lower() or "Suggestions" in all_markdown


def test_ai_summary_error_does_not_crash(fake_channels_df):
    """An ai_summary_error in session state must not raise an exception."""
    at = make_at(
        display_df=fake_channels_df,
        ai_summary=None,
        ai_summary_error="Gemini quota exceeded.",
    )
    at.run()
    assert not at.exception


def test_missing_youtube_key_shows_setup_instructions(no_api_keys):
    """Submitting with no key should include a hint about where to get one."""
    at = make_at(search_method="Keywords")
    at.run()

    next(b for b in at.button if b.label == "Find Creators").click().run()

    assert not at.exception
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "API key" in all_markdown or "console.cloud.google" in all_markdown

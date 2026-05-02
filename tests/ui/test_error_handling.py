"""
AppTest tests for error states and edge cases.

These verify that the app surfaces errors gracefully rather than crashing,
and that contextual suggestions appear for specific error types.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import APP_PATH


def test_pipeline_error_no_channels_shows_suggestions():
    """'did not return any channels' error should surface actionable suggestions."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))
    from core.pipeline import PipelineResult

    error_result = PipelineResult(
        channels_df=pd.DataFrame(),
        display_columns=[],
        column_explanations={},
        top_channels_for_outreach=pd.DataFrame(),
        final_query="manga, anime",
        error="Search did not return any channels matching your query.",
    )

    with patch("app.main.run_search_pipeline", return_value=error_result), \
         patch("app.main.build", return_value=MagicMock()):

        at = AppTest.from_file(APP_PATH)
        at.session_state["search_method"] = "Keywords"
        at.session_state["user_youtube_key"] = "AIza_fake_key_for_testing"
        at.run()

        submit = next(b for b in at.button if b.label == "Find Creators")
        submit.click().run()

    assert not at.exception
    assert len(at.error) > 0
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "broader" in all_markdown or "Suggestions" in all_markdown


def test_pipeline_error_filtering_shows_suggestions():
    """'filtering criteria' error should surface filter-adjustment suggestions."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))
    from core.pipeline import PipelineResult

    error_result = PipelineResult(
        channels_df=pd.DataFrame(),
        display_columns=[],
        column_explanations={},
        top_channels_for_outreach=pd.DataFrame(),
        final_query="manga, anime",
        error="All channels were removed by filtering criteria.",
    )

    with patch("app.main.run_search_pipeline", return_value=error_result), \
         patch("app.main.build", return_value=MagicMock()):

        at = AppTest.from_file(APP_PATH)
        at.session_state["search_method"] = "Keywords"
        at.session_state["user_youtube_key"] = "AIza_fake_key_for_testing"
        at.run()

        submit = next(b for b in at.button if b.label == "Find Creators")
        submit.click().run()

    assert not at.exception
    assert len(at.error) > 0
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "subscriber" in all_markdown.lower() or "Suggestions" in all_markdown


def test_ai_summary_error_does_not_crash(fake_channels_df):
    """An ai_summary_error in session state must not raise an exception."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["display_df"] = fake_channels_df
    at.session_state["ai_summary"] = None
    at.session_state["ai_summary_error"] = "Gemini quota exceeded."
    at.run()
    assert not at.exception


def test_missing_youtube_key_shows_setup_instructions():
    """Submitting with no key should include a hint about where to get one."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["search_method"] = "Keywords"
    at.run()

    submit = next(b for b in at.button if b.label == "Find Creators")
    submit.click().run()

    assert not at.exception
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "API key" in all_markdown or "console.cloud.google" in all_markdown

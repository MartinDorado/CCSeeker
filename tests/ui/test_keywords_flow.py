"""
AppTest tests for the Keywords search flow.

Strategy:
- Tests that need the search form → pre-set session_state['search_method']
- Tests that need results rendered → pre-set session_state['display_df'] directly
  (this bypasses the search entirely and purely tests the UI wiring)
- Tests that need the pipeline to run → patch app.main.run_search_pipeline +
  app.main.build and simulate form submission
"""
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import APP_PATH


def test_keywords_form_renders_when_method_selected():
    """Setting search_method before run should show the Keywords form."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["search_method"] = "Keywords"
    at.run()
    assert not at.exception
    # The search keywords text input should be present
    assert at.text_input(key="keywords_input") is not None


def test_keywords_submit_without_api_key_shows_error():
    """Submitting the form with no API key must show the missing-key error."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["search_method"] = "Keywords"
    at.run()

    # Click the form submit button (label "Find Creators", no key)
    submit = next(b for b in at.button if b.label == "Find Creators")
    submit.click().run()

    assert not at.exception
    assert len(at.error) > 0
    assert any("API key" in e.value or "not configured" in e.value for e in at.error)


def test_keywords_results_render_from_session_state(fake_channels_df):
    """Pre-populating display_df should render the Search Results subheader."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["display_df"] = fake_channels_df
    at.run()
    assert not at.exception
    assert any("Search Results" in s.value for s in at.subheader)


def test_keywords_results_show_channel_data(fake_channels_df):
    """Results dataframe should be present when display_df is in session state."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["display_df"] = fake_channels_df
    at.run()
    assert not at.exception
    assert len(at.dataframe) > 0


def test_keywords_ai_summary_renders_when_available(fake_channels_df):
    """AI summary block should appear when ai_summary is stored in session state."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["display_df"] = fake_channels_df
    at.session_state["ai_summary"] = "Top channels for manga & anime content."
    at.run()
    assert not at.exception
    all_text = " ".join(m.value for m in at.markdown)
    assert "Top channels" in all_text


def test_keywords_search_pipeline_called_on_submit(fake_pipeline_result):
    """Submitting the Keywords form with a fake key should invoke run_search_pipeline."""
    with patch("app.main.run_search_pipeline", return_value=fake_pipeline_result) as mock_pipeline, \
         patch("app.main.build", return_value=MagicMock()):

        at = AppTest.from_file(APP_PATH)
        at.session_state["search_method"] = "Keywords"
        at.session_state["user_youtube_key"] = "AIza_fake_key_for_testing"
        at.run()

        submit = next(b for b in at.button if b.label == "Find Creators")
        submit.click().run()

    assert mock_pipeline.called
    assert "display_df" in at.session_state


def test_keywords_results_absent_without_display_df():
    """No display_df in session state means no results section, no dataframe."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    assert "display_df" not in at.session_state
    assert len(at.dataframe) == 0

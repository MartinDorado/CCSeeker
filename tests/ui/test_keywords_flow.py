"""
AppTest tests for the Keywords search flow.

Strategy:
- Tests that need results rendered → pre-set session_state['display_df'] directly.
  This tests the UI wiring at line 1288 without any API calls.
- Tests that need the pipeline to run → patch at SOURCE modules so re-imports
  during AppTest script re-execution pick up the mocks:
    core.pipeline.run_search_pipeline  (not app.main.run_search_pipeline)
    googleapiclient.discovery.build    (not app.main.build)
"""
from unittest.mock import MagicMock, patch

import pytest

from tests.ui.conftest import make_at


def test_keywords_form_renders_when_method_selected():
    """Setting search_method before run should show the Keywords form."""
    at = make_at(search_method="Keywords")
    at.run()
    assert not at.exception
    assert at.text_input(key="keywords_input") is not None


def test_keywords_submit_without_api_key_shows_error(no_api_keys):
    """Submitting the form with no API key must show the missing-key error."""
    at = make_at(search_method="Keywords")
    at.run()

    submit = next(b for b in at.button if b.label == "Find Creators")
    submit.click().run()

    assert not at.exception
    assert len(at.error) > 0
    assert any("API key" in e.value or "not configured" in e.value for e in at.error)


def test_keywords_results_render_from_session_state(fake_channels_df):
    """Pre-populating display_df should render the Search Results subheader."""
    at = make_at(display_df=fake_channels_df)
    at.run()
    assert not at.exception
    assert any("Search Results" in s.value for s in at.subheader)


def test_keywords_results_show_channel_data(fake_channels_df):
    """Results dataframe should be present when display_df is in session state."""
    at = make_at(display_df=fake_channels_df)
    at.run()
    assert not at.exception
    assert len(at.dataframe) > 0


def test_keywords_ai_summary_renders_when_available(fake_channels_df):
    """AI summary block should appear when ai_summary is stored in session state."""
    at = make_at(
        display_df=fake_channels_df,
        ai_summary="Top channels for manga & anime content.",
    )
    at.run()
    assert not at.exception
    all_text = " ".join(m.value for m in at.markdown)
    assert "Top channels" in all_text


def test_keywords_search_pipeline_called_on_submit(fake_pipeline_result):
    """Submitting the Keywords form with a fake key should invoke run_search_pipeline."""
    # Patch at source modules — AppTest re-executes the script on every run(),
    # which re-imports these names. Patching the source ensures the re-import
    # gets the mock instead of the real function.
    # Patch core.__init__ namespace: main.py does `from core import run_search_pipeline`
    # which looks up sys.modules['core'].run_search_pipeline on each re-execution.
    with patch("core.run_search_pipeline", return_value=fake_pipeline_result) as mock_pipeline, \
         patch("googleapiclient.discovery.build", return_value=MagicMock()):

        at = make_at(search_method="Keywords", user_youtube_key="AIza_fake_key_for_testing")
        at.run()

        submit = next(b for b in at.button if b.label == "Find Creators")
        submit.click().run()

    assert not at.exception
    assert mock_pipeline.called
    assert "display_df" in at.session_state


def test_keywords_results_absent_without_display_df():
    """No display_df in session state means no results section, no dataframe."""
    at = make_at()
    at.run()
    assert not at.exception
    assert "display_df" not in at.session_state
    assert len(at.dataframe) == 0

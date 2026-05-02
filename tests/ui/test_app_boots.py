"""
AppTest boot tests — verifies app/main.py loads cleanly in all baseline states.
These are the cheapest safety net: a top-level exception would fail all four.
"""
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import APP_PATH


def test_app_loads_without_exception():
    """App should boot cleanly with no API keys and no session state set."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception


def test_app_shows_ccseeker_title():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    assert any("CCSeeker" in t.value for t in at.title)


def test_app_shows_both_search_method_buttons():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Keywords" in lbl for lbl in labels)
    assert any("Seed" in lbl for lbl in labels)


def test_no_results_section_on_fresh_load():
    """Results subheader must not appear before any search has run."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    assert "display_df" not in at.session_state
    assert not any("Search Results" in s.value for s in at.subheader)

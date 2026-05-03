"""
AppTest tests for the Channel-as-Seed flow.

Same patching strategy as test_keywords_flow.py: patch at source modules
(core.seed_topics, core.query_utils, googleapiclient.discovery) so that
AppTest's script re-execution picks up the mocks during re-import.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.ui.conftest import make_at

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app"))


def test_seed_form_renders_when_method_selected():
    """Setting search_method to Channel-as-Seed should show the seed URL input."""
    at = make_at(search_method="Channel-as-Seed")
    at.run()
    assert not at.exception
    assert len(at.text_input) > 0


def test_seed_profile_section_renders_when_in_session_state(fake_seed_profile):
    """Pre-populating seed_profile should show the Seed Channel Profile header."""
    at = make_at(seed_profile=fake_seed_profile)
    at.run()
    assert not at.exception
    assert any("Seed Channel Profile" in h.value for h in at.header)


def test_seed_profile_shows_subscriber_metric(fake_seed_profile):
    """Subscriber count metric should render from the seed profile data."""
    at = make_at(seed_profile=fake_seed_profile)
    at.run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert any("Subscribers" in lbl for lbl in metric_labels)


def test_seed_profile_shows_extracted_keywords(fake_seed_profile):
    """Primary keywords from the profile should appear in the rendered page."""
    at = make_at(seed_profile=fake_seed_profile)
    at.run()
    assert not at.exception
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "manga review" in all_markdown


def test_seed_submit_without_api_key_shows_error(no_api_keys):
    """Submitting Analyse Seed with no API key must show the missing-key error."""
    at = make_at(search_method="Channel-as-Seed")
    at.run()

    submit = next(b for b in at.button if b.label == "Analyse Seed")
    submit.click().run()

    assert not at.exception
    assert len(at.error) > 0
    assert any("API key" in e.value or "not configured" in e.value for e in at.error)


def test_seed_analysis_success_stores_profile_in_session_state(fake_seed_profile):
    """A successful analyze_seed_channel call must store the profile in session state."""
    from core.seed_topics import SeedAnalysisResult, SeedProfile

    fake_profile_obj = SeedProfile(
        channel_name=fake_seed_profile["channel_name"],
        channel_id=fake_seed_profile["channel_id"],
        subscriber_count=fake_seed_profile["subscriber_count"],
        subscriber_tier=fake_seed_profile["subscriber_tier"],
        language=fake_seed_profile["language"],
        upload_frequency=fake_seed_profile["upload_frequency"],
        avg_engagement_rate=fake_seed_profile["avg_engagement_rate"],
        category="entertainment",
        primary_keywords=fake_seed_profile["primary_keywords"],
        secondary_keywords=fake_seed_profile["secondary_keywords"],
        common_tags=fake_seed_profile["common_tags"],
    )
    fake_result = SeedAnalysisResult(profile=fake_profile_obj)

    # Patch core.__init__ namespace so re-imports during AppTest re-execution
    # pick up the mocks (main.py uses `from core import analyze_seed_channel`).
    with patch("core.analyze_seed_channel", return_value=fake_result), \
         patch("core.resolve_channel_id", return_value="UC001"), \
         patch("googleapiclient.discovery.build", return_value=MagicMock()):

        at = make_at(
            search_method="Channel-as-Seed",
            user_youtube_key="AIza_fake_key_for_testing",
        )
        at.run()

        submit = next(b for b in at.button if b.label == "Analyse Seed")
        submit.click().run()

    assert not at.exception
    assert "seed_profile" in at.session_state
    assert at.session_state["seed_profile"]["channel_name"] == "MangaWorld"

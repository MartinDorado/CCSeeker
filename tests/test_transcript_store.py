"""
Tests for app/cache/transcript_store.py

Covers:
- JSONTranscriptStore: save, get, negative-result caching
- SupabaseTranscriptStore: mocked client (mirrors test_feedback_tracker.py pattern)
- get_transcript_cached: L2 cache hit (no fetcher call), miss (fetcher called + saved)
"""

import sys
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.transcription import TranscriptResult, FakeTranscriptFetcher
from app.cache.transcript_store import (
    JSONTranscriptStore,
    SupabaseTranscriptStore,
    get_transcript_cached,
    TRANSCRIPT_STORE_FILE,
)
from tests.fixtures.transcripts import (
    TRANSCRIPT_EN_EDUCATIONAL,
    TRANSCRIPT_DISABLED,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_result(video_id: str = "v1", status: str = "ok", text: str = "hello") -> TranscriptResult:
    return TranscriptResult(
        video_id=video_id,
        channel_id="UC_test",
        language="en",
        text=text,
        status=status,
    )


# ============================================================================
# JSONTranscriptStore
# ============================================================================

@pytest.fixture
def json_store(tmp_path):
    filepath = str(tmp_path / ".test_transcripts.json")
    return JSONTranscriptStore(filepath=filepath)


class TestJSONTranscriptStore:
    def test_get_returns_none_when_empty(self, json_store):
        assert json_store.get_transcript("nonexistent") is None

    def test_save_and_get_roundtrip(self, json_store):
        result = _make_result(video_id="v1", text="some content")
        json_store.save_transcript(result)
        retrieved = json_store.get_transcript("v1")
        assert retrieved is not None
        assert retrieved.video_id == "v1"
        assert retrieved.text == "some content"
        assert retrieved.status == "ok"

    def test_negative_result_cached(self, json_store):
        disabled = _make_result(video_id="v_disabled", status="disabled", text="")
        json_store.save_transcript(disabled)
        retrieved = json_store.get_transcript("v_disabled")
        assert retrieved is not None
        assert retrieved.status == "disabled"

    def test_no_captions_result_cached(self, json_store):
        no_cap = _make_result(video_id="v_nocap", status="no_captions", text="")
        json_store.save_transcript(no_cap)
        retrieved = json_store.get_transcript("v_nocap")
        assert retrieved is not None
        assert retrieved.status == "no_captions"

    def test_overwrite_existing_entry(self, json_store):
        v1 = _make_result(video_id="v1", text="old text")
        v1_new = _make_result(video_id="v1", text="new text")
        json_store.save_transcript(v1)
        json_store.save_transcript(v1_new)
        retrieved = json_store.get_transcript("v1")
        assert retrieved.text == "new text"

    def test_multiple_entries_independent(self, json_store):
        for i in range(5):
            json_store.save_transcript(_make_result(video_id=f"v{i}", text=f"text{i}"))
        for i in range(5):
            r = json_store.get_transcript(f"v{i}")
            assert r is not None
            assert r.text == f"text{i}"

    def test_corrupted_file_returns_none(self, tmp_path):
        filepath = str(tmp_path / "corrupt.json")
        with open(filepath, "w") as f:
            f.write("{not valid json")
        store = JSONTranscriptStore(filepath=filepath)
        assert store.get_transcript("v1") is None

    def test_save_returns_true_on_success(self, json_store):
        result = _make_result()
        assert json_store.save_transcript(result) is True

    def test_save_persists_to_file(self, json_store):
        result = _make_result(video_id="persist_test", text="persisted content")
        json_store.save_transcript(result)
        # Re-open the file and verify the data
        with open(json_store._filepath, "r") as f:
            raw = json.load(f)
        assert "persist_test" in raw
        assert raw["persist_test"]["transcript_text"] == "persisted content"

    def test_fetched_at_is_stored(self, json_store):
        result = _make_result(video_id="v_ts")
        json_store.save_transcript(result)
        with open(json_store._filepath, "r") as f:
            raw = json.load(f)
        assert "fetched_at" in raw["v_ts"]


# ============================================================================
# SupabaseTranscriptStore (mocked)
# ============================================================================

def _make_supabase_mock():
    """Build a mock Supabase client."""
    mock_client = MagicMock()
    # Chain: .table().select().eq().limit().execute()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    # Default: empty data
    mock_execute = MagicMock()
    mock_execute.data = []
    mock_table.execute.return_value = mock_execute
    return mock_client, mock_table, mock_execute


class TestSupabaseTranscriptStore:
    def test_get_returns_none_when_no_data(self):
        mock_client, mock_table, mock_execute = _make_supabase_mock()
        with patch("app.cache.transcript_store.SupabaseTranscriptStore.__init__",
                   lambda self, url, key: setattr(self, '_db', mock_client)):
            store = SupabaseTranscriptStore.__new__(SupabaseTranscriptStore)
            store._db = mock_client
        mock_execute.data = []
        result = store.get_transcript("v1")
        assert result is None

    def test_get_returns_result_when_found(self):
        mock_client, mock_table, mock_execute = _make_supabase_mock()
        mock_execute.data = [{
            "video_id": "v1",
            "channel_id": "UC1",
            "language": "en",
            "transcript_text": "hello world",
            "fetch_status": "ok",
            "error_message": None,
        }]
        store = SupabaseTranscriptStore.__new__(SupabaseTranscriptStore)
        store._db = mock_client
        result = store.get_transcript("v1")
        assert result is not None
        assert result.video_id == "v1"
        assert result.text == "hello world"
        assert result.status == "ok"

    def test_save_calls_upsert(self):
        mock_client, mock_table, mock_execute = _make_supabase_mock()
        store = SupabaseTranscriptStore.__new__(SupabaseTranscriptStore)
        store._db = mock_client
        result = _make_result(video_id="v2", text="transcript text")
        ok = store.save_transcript(result)
        assert ok is True
        mock_client.table.assert_called_with("transcripts")
        mock_table.upsert.assert_called_once()

    def test_save_negative_result_upserts_status(self):
        mock_client, mock_table, mock_execute = _make_supabase_mock()
        store = SupabaseTranscriptStore.__new__(SupabaseTranscriptStore)
        store._db = mock_client
        disabled = _make_result(video_id="v_dis", status="disabled", text="")
        ok = store.save_transcript(disabled)
        assert ok is True
        upsert_call_args = mock_table.upsert.call_args[0][0]
        assert upsert_call_args["fetch_status"] == "disabled"

    def test_get_returns_none_on_exception(self):
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("connection refused")
        store = SupabaseTranscriptStore.__new__(SupabaseTranscriptStore)
        store._db = mock_client
        result = store.get_transcript("v1")
        assert result is None

    def test_save_returns_false_on_exception(self):
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("connection refused")
        store = SupabaseTranscriptStore.__new__(SupabaseTranscriptStore)
        store._db = mock_client
        ok = store.save_transcript(_make_result())
        assert ok is False


# ============================================================================
# get_transcript_cached (L2 integration)
# ============================================================================

class TestGetTranscriptCached:
    def test_cache_hit_does_not_call_fetcher(self, tmp_path):
        store = JSONTranscriptStore(filepath=str(tmp_path / "cache.json"))
        existing = _make_result(video_id="v_hit", text="cached content")
        store.save_transcript(existing)

        fetcher = MagicMock()
        result = get_transcript_cached(
            video_id="v_hit",
            channel_id="UC1",
            fetcher=fetcher,
            store=store,
        )
        assert result.text == "cached content"
        fetcher.fetch.assert_not_called()

    def test_cache_miss_calls_fetcher_and_saves(self, tmp_path):
        store = JSONTranscriptStore(filepath=str(tmp_path / "cache.json"))
        fetch_result = _make_result(video_id="v_miss", text="fresh content")
        fetcher = FakeTranscriptFetcher({"v_miss": fetch_result})

        result = get_transcript_cached(
            video_id="v_miss",
            channel_id="UC1",
            fetcher=fetcher,
            store=store,
        )
        assert result.text == "fresh content"
        # Verify it's now in the store (negative caching works too)
        stored = store.get_transcript("v_miss")
        assert stored is not None
        assert stored.text == "fresh content"

    def test_negative_result_saved_to_store(self, tmp_path):
        store = JSONTranscriptStore(filepath=str(tmp_path / "cache.json"))
        disabled = TranscriptResult("v_dis", "UC1", None, "", "disabled")
        fetcher = FakeTranscriptFetcher({"v_dis": disabled})
        result = get_transcript_cached(
            video_id="v_dis",
            channel_id="UC1",
            fetcher=fetcher,
            store=store,
        )
        assert result.status == "disabled"
        stored = store.get_transcript("v_dis")
        assert stored is not None
        assert stored.status == "disabled"

    def test_channel_id_filled_when_empty(self, tmp_path):
        store = JSONTranscriptStore(filepath=str(tmp_path / "cache.json"))
        bare = TranscriptResult("v_bare", "", "en", "text", "ok")
        fetcher = FakeTranscriptFetcher({"v_bare": bare})
        result = get_transcript_cached(
            video_id="v_bare",
            channel_id="UC_filled",
            fetcher=fetcher,
            store=store,
        )
        assert result.channel_id == "UC_filled"

"""
Tests for app/core/transcription.py

Covers:
- FakeTranscriptFetcher behaviour
- Status mapping (disabled, rate_limited, no_captions, error)
- fetch_transcripts_parallel: parallelism, ordering, circuit breaker
- extract_niche_summary: happy path, low-confidence flag (<3 transcripts)
- _parse_niche_json: markdown-fenced JSON, missing required key, malformed JSON
- NicheExtractionResult when gemini_model is None
- Zero-transcripts → no Gemini call
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.transcription import (
    TranscriptionConfig,
    TranscriptResult,
    NicheExtractionResult,
    FakeTranscriptFetcher,
    YouTubeTranscriptFetcher,
    fetch_transcripts_parallel,
    extract_niche_summary,
    _parse_niche_json,
)
from tests.fixtures.transcripts import (
    TRANSCRIPT_EN_EDUCATIONAL,
    TRANSCRIPT_ES_VLOG,
    TRANSCRIPT_DISABLED,
    TRANSCRIPT_EMPTY,
    TRANSCRIPT_SHORTS_FRAGMENT,
    FAKE_RESPONSES,
)


# ============================================================================
# FakeTranscriptFetcher
# ============================================================================

class TestFakeTranscriptFetcher:
    def test_returns_mapped_result(self):
        fetcher = FakeTranscriptFetcher(FAKE_RESPONSES)
        result = fetcher.fetch("vid_en_edu_1", "en")
        assert result.status == "ok"
        assert "gradient" in result.text

    def test_unmapped_id_returns_no_captions(self):
        fetcher = FakeTranscriptFetcher({})
        result = fetcher.fetch("nonexistent", None)
        assert result.status == "no_captions"
        assert result.video_id == "nonexistent"

    def test_string_shorthand_status(self):
        fetcher = FakeTranscriptFetcher({"abc": "rate_limited"})
        result = fetcher.fetch("abc", None)
        assert result.status == "rate_limited"

    def test_disabled_status(self):
        fetcher = FakeTranscriptFetcher({"vid_disabled_1": TRANSCRIPT_DISABLED})
        result = fetcher.fetch("vid_disabled_1", None)
        assert result.status == "disabled"


# ============================================================================
# fetch_transcripts_parallel
# ============================================================================

class TestFetchTranscriptsParallel:
    def _make_config(self, **kwargs) -> TranscriptionConfig:
        defaults = dict(enabled=True, max_videos=8, per_video_chars=1500,
                        corpus_chars=12000, per_video_timeout_s=5.0,
                        skip_shorts=True)
        defaults.update(kwargs)
        return TranscriptionConfig(**defaults)

    def test_returns_results_for_all_ids(self):
        fetcher = FakeTranscriptFetcher(FAKE_RESPONSES)
        ids = ["vid_en_edu_1", "vid_es_vlog_1", "vid_disabled_1"]
        config = self._make_config()
        results = fetch_transcripts_parallel(ids, fetcher, config)
        assert len(results) == 3
        assert [r.video_id for r in results] == ids

    def test_preserves_order(self):
        fetcher = FakeTranscriptFetcher(FAKE_RESPONSES)
        ids = ["vid_disabled_1", "vid_en_edu_1", "vid_es_vlog_1"]
        config = self._make_config()
        results = fetch_transcripts_parallel(ids, fetcher, config)
        assert [r.video_id for r in results] == ids

    def test_truncates_text_to_per_video_chars(self):
        long_text = "x" * 3000
        fetcher = FakeTranscriptFetcher({
            "vid1": TranscriptResult("vid1", "", "en", long_text, "ok")
        })
        config = self._make_config(per_video_chars=500)
        results = fetch_transcripts_parallel(["vid1"], fetcher, config)
        assert len(results[0].text) <= 500

    def test_circuit_breaker_after_3_rate_limited(self):
        responses = {f"vid{i}": "rate_limited" for i in range(10)}
        fetcher = FakeTranscriptFetcher(responses)
        config = self._make_config()
        ids = [f"vid{i}" for i in range(10)]
        results = fetch_transcripts_parallel(ids, fetcher, config)
        # At least 3 results; remaining may be cancelled/error
        assert len(results) == len(ids)
        rate_limited = [r for r in results if r.status == "rate_limited"]
        assert len(rate_limited) >= 3

    def test_progress_callback_called(self):
        fetcher = FakeTranscriptFetcher(FAKE_RESPONSES)
        config = self._make_config()
        progress_calls = []
        fetch_transcripts_parallel(
            ["vid_en_edu_1", "vid_es_vlog_1"],
            fetcher,
            config,
            on_progress=lambda msg, pct: progress_calls.append((msg, pct)),
        )
        assert len(progress_calls) > 0

    def test_api_call_callback_only_for_ok(self):
        responses = {
            "vid_ok": TranscriptResult("vid_ok", "", "en", "text", "ok"),
            "vid_fail": TRANSCRIPT_DISABLED,
        }
        fetcher = FakeTranscriptFetcher(responses)
        config = self._make_config()
        api_calls = []
        fetch_transcripts_parallel(
            ["vid_ok", "vid_fail"],
            fetcher,
            config,
            on_api_call=lambda name: api_calls.append(name),
        )
        assert api_calls.count("transcript_fetch") == 1

    def test_empty_id_list_returns_empty(self):
        fetcher = FakeTranscriptFetcher({})
        config = self._make_config()
        results = fetch_transcripts_parallel([], fetcher, config)
        assert results == []


# ============================================================================
# _parse_niche_json
# ============================================================================

class TestParseNicheJson:
    def _valid_niche_dict(self, **overrides) -> dict:
        base = {
            "niche": "Educational Python programming",
            "audience": "beginner developers",
            "style": "tutorial",
            "topic_emphasis": ["python basics", "data structures"],
            "tone": "friendly, clear",
            "confidence": "high",
        }
        base.update(overrides)
        return base

    def test_valid_json_parsed(self):
        data = self._valid_niche_dict()
        result = _parse_niche_json(json.dumps(data))
        assert result["niche"] == "Educational Python programming"
        assert result["style"] == "tutorial"

    def test_markdown_fenced_json(self):
        data = self._valid_niche_dict()
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_niche_json(raw)
        assert result["niche"] == "Educational Python programming"

    def test_markdown_fenced_without_language_tag(self):
        data = self._valid_niche_dict()
        raw = f"```\n{json.dumps(data)}\n```"
        result = _parse_niche_json(raw)
        assert "niche" in result

    def test_missing_required_key_returns_empty(self):
        data = self._valid_niche_dict()
        del data["niche"]
        result = _parse_niche_json(json.dumps(data))
        assert result == {}

    def test_malformed_json_returns_empty(self):
        result = _parse_niche_json("{not valid json")
        assert result == {}

    def test_invalid_style_replaced_with_hybrid(self):
        data = self._valid_niche_dict(style="unknown_style")
        result = _parse_niche_json(json.dumps(data))
        assert result["style"] == "hybrid"

    def test_invalid_confidence_replaced_with_low(self):
        data = self._valid_niche_dict(confidence="maybe")
        result = _parse_niche_json(json.dumps(data))
        assert result["confidence"] == "low"

    def test_topic_emphasis_coerced_to_list_when_string(self):
        data = self._valid_niche_dict(topic_emphasis="single phrase")
        result = _parse_niche_json(json.dumps(data))
        assert isinstance(result["topic_emphasis"], list)

    def test_non_dict_response_returns_empty(self):
        result = _parse_niche_json('["not", "a", "dict"]')
        assert result == {}


# ============================================================================
# extract_niche_summary
# ============================================================================

def _make_gemini_model(response_text: str):
    """Build a minimal mock Gemini model that returns a fixed text."""
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate_content.return_value = response
    return model


class TestExtractNicheSummary:
    def _ok_transcript(self, vid: str, text: str = "some transcript text") -> TranscriptResult:
        return TranscriptResult(vid, "UC1", "en", text, "ok")

    def test_happy_path_returns_summary(self):
        valid_niche = json.dumps({
            "niche": "Python tutorials",
            "audience": "developers",
            "style": "tutorial",
            "topic_emphasis": ["python", "scripting"],
            "tone": "concise",
            "confidence": "high",
        })
        model = _make_gemini_model(valid_niche)
        transcripts = [self._ok_transcript(f"v{i}") for i in range(5)]
        result = extract_niche_summary(transcripts, model)
        assert result.summary != {}
        assert result.transcripts_used == 5
        assert result.confidence == "high"
        assert result.api_calls == 1

    def test_fewer_than_3_ok_forces_low_confidence(self):
        valid_niche = json.dumps({
            "niche": "Gaming channel",
            "audience": "gamers",
            "style": "entertainment",
            "topic_emphasis": ["gaming"],
            "tone": "energetic",
            "confidence": "high",
        })
        model = _make_gemini_model(valid_niche)
        transcripts = [
            self._ok_transcript("v1"),
            self._ok_transcript("v2"),
            TRANSCRIPT_DISABLED,
            TRANSCRIPT_DISABLED,
        ]
        result = extract_niche_summary(transcripts, model)
        assert result.confidence == "low"
        assert result.summary.get("confidence") == "low"

    def test_no_ok_transcripts_returns_unavailable_no_call(self):
        model = _make_gemini_model("{}")
        transcripts = [TRANSCRIPT_DISABLED, TRANSCRIPT_DISABLED]
        result = extract_niche_summary(transcripts, model)
        assert result.summary == {}
        assert result.confidence == "unavailable"
        assert result.api_calls == 0
        model.generate_content.assert_not_called()

    def test_empty_transcripts_list_returns_unavailable(self):
        model = _make_gemini_model("{}")
        result = extract_niche_summary([], model)
        assert result.summary == {}
        assert result.confidence == "unavailable"
        assert result.api_calls == 0

    def test_gemini_model_none_returns_unavailable(self):
        transcripts = [self._ok_transcript(f"v{i}") for i in range(4)]
        result = extract_niche_summary(transcripts, None)
        assert result.summary == {}
        assert result.confidence == "unavailable"
        assert result.api_calls == 0

    def test_malformed_gemini_response_returns_unavailable(self):
        model = _make_gemini_model("{broken json}")
        transcripts = [self._ok_transcript(f"v{i}") for i in range(4)]
        result = extract_niche_summary(transcripts, model)
        assert result.summary == {}
        assert result.confidence == "unavailable"
        assert result.api_calls == 1

    def test_gemini_missing_required_field_returns_unavailable(self):
        incomplete = json.dumps({
            "audience": "developers",
            "style": "tutorial",
            "topic_emphasis": ["python"],
            "tone": "concise",
            "confidence": "high",
            # "niche" key is absent
        })
        model = _make_gemini_model(incomplete)
        transcripts = [self._ok_transcript(f"v{i}") for i in range(4)]
        result = extract_niche_summary(transcripts, model)
        assert result.summary == {}
        assert result.confidence == "unavailable"

    def test_api_call_callback_fired(self):
        valid_niche = json.dumps({
            "niche": "Cooking channel",
            "audience": "home cooks",
            "style": "tutorial",
            "topic_emphasis": ["recipes"],
            "tone": "warm",
            "confidence": "medium",
        })
        model = _make_gemini_model(valid_niche)
        transcripts = [self._ok_transcript(f"v{i}") for i in range(4)]
        api_calls = []
        extract_niche_summary(
            transcripts, model, on_api_call=lambda name: api_calls.append(name)
        )
        assert "gemini_niche" in api_calls

    def test_corpus_is_capped_at_corpus_chars(self):
        long_text = "word " * 5000
        transcripts = [self._ok_transcript(f"v{i}", long_text) for i in range(5)]
        model = _make_gemini_model(json.dumps({
            "niche": "test",
            "audience": "test",
            "style": "tutorial",
            "topic_emphasis": ["test"],
            "tone": "neutral",
            "confidence": "high",
        }))
        result = extract_niche_summary(transcripts, model, corpus_chars=5000)
        assert result.api_calls == 1  # call was made despite large input
        # Verify the prompt text is under corpus_chars + overhead
        call_args = model.generate_content.call_args
        prompt_arg = call_args[0][0]
        assert len(prompt_arg) < 5000 + 1000  # corpus + prompt template overhead

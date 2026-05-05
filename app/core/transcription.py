"""
transcription.py - Transcript fetching and niche extraction for seed channels

Streamlit-agnostic: uses callbacks for progress/API tracking, mirrors the
youtube_api.py pattern. Production fetcher wraps youtube-transcript-api;
a FakeTranscriptFetcher is provided for tests.

Scope: seed mode only. Keyword-mode Deep Analysis is a deferred follow-up.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable


# ============================================================================
# CONFIG & RESULT TYPES
# ============================================================================

@dataclass
class TranscriptionConfig:
    """Runtime settings for the transcript-fetching pipeline."""
    enabled: bool = True
    max_videos: int = 8
    per_video_chars: int = 1500
    corpus_chars: int = 12000
    per_video_timeout_s: float = 5.0
    skip_shorts: bool = True
    proxy_url: str | None = None


@dataclass
class TranscriptResult:
    """Result for a single video transcript fetch."""
    video_id: str
    channel_id: str
    language: str | None
    text: str
    status: Literal["ok", "no_captions", "disabled", "error", "rate_limited"]
    error_message: str | None = None


@dataclass
class NicheExtractionResult:
    """Structured niche profile derived from transcript corpus."""
    summary: dict = field(default_factory=dict)
    transcripts_used: int = 0
    transcripts_failed: int = 0
    confidence: Literal["high", "medium", "low", "unavailable"] = "unavailable"
    api_calls: int = 0


# ============================================================================
# FETCHER PROTOCOL + IMPLEMENTATIONS
# ============================================================================

@runtime_checkable
class TranscriptFetcher(Protocol):
    def fetch(self, video_id: str, language_pref: str | None) -> TranscriptResult: ...


class YouTubeTranscriptFetcher:
    """Production fetcher — wraps youtube-transcript-api."""

    def __init__(self, channel_id: str = "", proxy_url: str | None = None):
        self._channel_id = channel_id
        self._proxy_url = proxy_url

    def fetch(self, video_id: str, language_pref: str | None) -> TranscriptResult:
        try:
            from youtube_transcript_api import (
                YouTubeTranscriptApi,
                TranscriptsDisabled,
                NoTranscriptFound,
            )
            # Import error classes that may exist in different versions
            try:
                from youtube_transcript_api._errors import TooManyRequests
            except ImportError:
                TooManyRequests = None

            proxies = {"https": self._proxy_url} if self._proxy_url else None

            # Build language priority list.
            # Always include the auto-generated variant for the preferred language
            # so channels without manual captions (e.g. "a.it" for Italian) are covered.
            languages = []
            if language_pref:
                languages.append(language_pref)
                if language_pref not in ("en", "es"):
                    languages.append(f"a.{language_pref}")
            languages += ["en", "es", "a.en", "a.es"]  # English/Spanish fallbacks

            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=languages if languages else None,
                proxies=proxies,
            )

            if not transcript_list:
                return TranscriptResult(
                    video_id=video_id,
                    channel_id=self._channel_id,
                    language=None,
                    text="",
                    status="no_captions",
                )

            text = " ".join(entry.get("text", "") for entry in transcript_list)
            lang = transcript_list[0].get("lang") if transcript_list else language_pref

            return TranscriptResult(
                video_id=video_id,
                channel_id=self._channel_id,
                language=lang,
                text=text,
                status="ok",
            )

        except Exception as exc:
            exc_name = type(exc).__name__
            exc_str = str(exc).lower()

            if "TranscriptsDisabled" in exc_name or "disabled" in exc_str:
                return TranscriptResult(
                    video_id=video_id,
                    channel_id=self._channel_id,
                    language=None,
                    text="",
                    status="disabled",
                    error_message=str(exc),
                )
            if (
                "TooManyRequests" in exc_name
                or "IpBlocked" in exc_name
                or "too many requests" in exc_str
                or "ip" in exc_str and "block" in exc_str
            ):
                return TranscriptResult(
                    video_id=video_id,
                    channel_id=self._channel_id,
                    language=None,
                    text="",
                    status="rate_limited",
                    error_message=str(exc),
                )
            if "NoTranscriptFound" in exc_name or "no transcript" in exc_str:
                return TranscriptResult(
                    video_id=video_id,
                    channel_id=self._channel_id,
                    language=None,
                    text="",
                    status="no_captions",
                    error_message=str(exc),
                )

            return TranscriptResult(
                video_id=video_id,
                channel_id=self._channel_id,
                language=None,
                text="",
                status="error",
                error_message=str(exc),
            )


class FakeTranscriptFetcher:
    """
    Test double for TranscriptFetcher.

    Pass a dict mapping video_id → TranscriptResult (or status string shorthand).
    Unmapped IDs get status="no_captions".
    """

    def __init__(self, responses: dict[str, "TranscriptResult | str"] | None = None):
        self._responses = responses or {}

    def fetch(self, video_id: str, language_pref: str | None) -> TranscriptResult:
        result = self._responses.get(video_id)
        if result is None:
            return TranscriptResult(
                video_id=video_id,
                channel_id="",
                language=None,
                text="",
                status="no_captions",
            )
        if isinstance(result, str):
            return TranscriptResult(
                video_id=video_id,
                channel_id="",
                language=None,
                text="",
                status=result,  # type: ignore[arg-type]
            )
        return result


# ============================================================================
# PARALLEL FETCH
# ============================================================================

_CIRCUIT_BREAKER_THRESHOLD = 3


def fetch_transcripts_parallel(
    video_ids: list[str],
    fetcher: TranscriptFetcher,
    config: TranscriptionConfig,
    language_pref: str | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    on_api_call: Callable[[str], None] | None = None,
) -> list[TranscriptResult]:
    """
    Fetch transcripts for multiple videos in parallel.

    Applies a circuit breaker: if 3+ consecutive rate_limited results appear
    across all in-flight futures, remaining fetches are aborted.

    Args:
        video_ids: Video IDs to fetch (already filtered for Shorts).
        fetcher: TranscriptFetcher implementation.
        config: TranscriptionConfig controlling concurrency and limits.
        language_pref: Preferred language code (e.g. "en").
        on_progress: Optional progress callback.
        on_api_call: Optional API call tracker callback.

    Returns:
        List of TranscriptResult in the same order as video_ids.
    """
    results: dict[str, TranscriptResult] = {}
    consecutive_rate_limited = 0
    total = len(video_ids)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_vid = {
            executor.submit(
                fetcher.fetch, vid, language_pref
            ): vid
            for vid in video_ids
        }

        completed = 0
        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                result = future.result(timeout=config.per_video_timeout_s)
            except Exception as exc:
                result = TranscriptResult(
                    video_id=vid,
                    channel_id="",
                    language=None,
                    text="",
                    status="error",
                    error_message=str(exc),
                )

            # Truncate text to per_video_chars
            if result.status == "ok" and len(result.text) > config.per_video_chars:
                result.text = result.text[: config.per_video_chars]

            results[vid] = result

            if result.status == "rate_limited":
                consecutive_rate_limited += 1
            else:
                consecutive_rate_limited = 0

            if on_api_call and result.status == "ok":
                on_api_call("transcript_fetch")

            completed += 1
            if on_progress:
                on_progress(
                    f"Transcripts: {completed}/{total}",
                    completed / total,
                )

            if consecutive_rate_limited >= _CIRCUIT_BREAKER_THRESHOLD:
                for remaining_future in future_to_vid:
                    remaining_future.cancel()
                break

    # Fill any video_ids that were cancelled / never started
    for vid in video_ids:
        if vid not in results:
            results[vid] = TranscriptResult(
                video_id=vid,
                channel_id="",
                language=None,
                text="",
                status="error",
                error_message="cancelled by circuit breaker",
            )

    return [results[vid] for vid in video_ids]


# ============================================================================
# NICHE EXTRACTION VIA GEMINI
# ============================================================================

_NICHE_SCHEMA = {
    "niche": str,
    "audience": str,
    "style": str,
    "topic_emphasis": list,
    "tone": str,
    "confidence": str,
}

_VALID_STYLES = {
    "educational", "entertainment", "tutorial",
    "commentary", "vlog", "review", "hybrid",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def _build_niche_prompt(corpus: str) -> str:
    return f"""You are a YouTube content analyst. Read the following transcript excerpts from a YouTube channel's recent videos and extract a structured niche profile.

TRANSCRIPT EXCERPTS:
{corpus}

Return ONLY valid JSON with exactly these keys:
{{
  "niche": "<1-sentence specific niche statement>",
  "audience": "<who watches this>",
  "style": "<educational|entertainment|tutorial|commentary|vlog|review|hybrid>",
  "topic_emphasis": ["<3-5 short topical phrases>"],
  "tone": "<1-2 adjectives>",
  "confidence": "<high|medium|low>"
}}"""


def _parse_niche_json(text: str) -> dict:
    """
    Parse Gemini's response into a niche dict.

    Handles markdown-fenced JSON, extra whitespace, and missing/extra keys.
    Returns {} on any parse or validation failure.
    """
    # Strip markdown fences (```json ... ```)
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    raw = fenced.group(1) if fenced else text.strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    # Require all mandatory keys
    required = {"niche", "audience", "style", "topic_emphasis", "tone", "confidence"}
    if not required.issubset(data.keys()):
        return {}

    # Coerce types and clamp enum values
    if not isinstance(data.get("topic_emphasis"), list):
        data["topic_emphasis"] = []

    style = str(data.get("style", "")).lower()
    data["style"] = style if style in _VALID_STYLES else "hybrid"

    conf = str(data.get("confidence", "")).lower()
    data["confidence"] = conf if conf in _VALID_CONFIDENCE else "low"

    return {k: data[k] for k in required}


def extract_niche_summary(
    transcripts: list[TranscriptResult],
    gemini_model,
    on_api_call: Callable[[str], None] | None = None,
    corpus_chars: int = 12000,
) -> NicheExtractionResult:
    """
    Call Gemini once to extract a structured niche profile from transcripts.

    Args:
        transcripts: List of TranscriptResult (only "ok" ones are used).
        gemini_model: Configured Gemini GenerativeModel instance.
        on_api_call: Optional tracking callback.
        corpus_chars: Maximum total characters in the combined corpus.

    Returns:
        NicheExtractionResult with summary dict, counts, and confidence.
    """
    successful = [t for t in transcripts if t.status == "ok" and t.text]
    failed = len(transcripts) - len(successful)

    if not successful or gemini_model is None:
        return NicheExtractionResult(
            summary={},
            transcripts_used=0,
            transcripts_failed=failed,
            confidence="unavailable",
            api_calls=0,
        )

    # Build corpus with per-video separator
    corpus_parts = []
    total_chars = 0
    for tr in successful:
        part = f"[video {tr.video_id}]\n{tr.text}"
        if total_chars + len(part) > corpus_chars:
            part = part[: corpus_chars - total_chars]
            corpus_parts.append(part)
            break
        corpus_parts.append(part)
        total_chars += len(part)

    corpus = "\n\n".join(corpus_parts)
    prompt = _build_niche_prompt(corpus)

    # Force low confidence if fewer than 3 transcripts succeeded
    forced_low = len(successful) < 3

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        raw_text = response.text.strip()
    except Exception:
        # Fallback: try without mime_type (older SDK versions)
        try:
            response = gemini_model.generate_content(prompt)
            raw_text = response.text.strip()
        except Exception as exc:
            return NicheExtractionResult(
                summary={},
                transcripts_used=len(successful),
                transcripts_failed=failed,
                confidence="unavailable",
                api_calls=1,
            )

    if on_api_call:
        on_api_call("gemini_niche")

    summary = _parse_niche_json(raw_text)

    if not summary:
        return NicheExtractionResult(
            summary={},
            transcripts_used=len(successful),
            transcripts_failed=failed,
            confidence="unavailable",
            api_calls=1,
        )

    if forced_low:
        summary["confidence"] = "low"

    confidence = summary.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return NicheExtractionResult(
        summary=summary,
        transcripts_used=len(successful),
        transcripts_failed=failed,
        confidence=confidence,  # type: ignore[arg-type]
        api_calls=1,
    )

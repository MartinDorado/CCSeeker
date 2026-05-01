"""
transcript_store.py - Two-tier transcript cache (JSON local + Supabase)

Mirrors the Protocol + JSONStore + SupabaseStore pattern from feedback_tracker.py.
Negative results (no_captions, disabled) are cached to avoid re-fetching the
~30% of videos that will never have captions.

Supabase schema:
    transcripts(
        video_id         text primary key,
        channel_id       text not null,
        language         text,
        transcript_text  text,
        fetched_at       timestamptz default now(),
        fetch_status     text,
        error_message    text
    )
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

try:
    from ..core.transcription import TranscriptResult
except ImportError:
    from core.transcription import TranscriptResult

# Default file path for JSON store (in app directory)
TRANSCRIPT_STORE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".transcript_cache.json"
)


# ============================================================================
# PROTOCOL
# ============================================================================

@runtime_checkable
class TranscriptStore(Protocol):
    def get_transcript(self, video_id: str) -> Optional[TranscriptResult]: ...
    def save_transcript(self, result: TranscriptResult) -> bool: ...


# ============================================================================
# JSON IMPLEMENTATION (local dev)
# ============================================================================

class JSONTranscriptStore:
    """File-backed transcript cache using a local JSON file."""

    def __init__(self, filepath: str | None = None):
        self._filepath = filepath if filepath is not None else TRANSCRIPT_STORE_FILE

    def _load(self) -> dict:
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _dump(self, data: dict) -> bool:
        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    def get_transcript(self, video_id: str) -> Optional[TranscriptResult]:
        data = self._load()
        entry = data.get(video_id)
        if entry is None:
            return None
        return TranscriptResult(
            video_id=entry["video_id"],
            channel_id=entry.get("channel_id", ""),
            language=entry.get("language"),
            text=entry.get("transcript_text", ""),
            status=entry.get("fetch_status", "error"),
            error_message=entry.get("error_message"),
        )

    def save_transcript(self, result: TranscriptResult) -> bool:
        data = self._load()
        data[result.video_id] = {
            "video_id": result.video_id,
            "channel_id": result.channel_id,
            "language": result.language,
            "transcript_text": result.text,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "fetch_status": result.status,
            "error_message": result.error_message,
        }
        return self._dump(data)


# ============================================================================
# SUPABASE IMPLEMENTATION (production)
# ============================================================================

class SupabaseTranscriptStore:
    """Supabase-backed transcript cache."""

    def __init__(self, url: str, key: str):
        from supabase import create_client  # noqa: PLC0415 — ImportError intentional
        self._db = create_client(url, key)

    def get_transcript(self, video_id: str) -> Optional[TranscriptResult]:
        try:
            resp = (
                self._db.table("transcripts")
                .select("*")
                .eq("video_id", video_id)
                .limit(1)
                .execute()
            )
            if not resp.data:
                return None
            row = resp.data[0]
            return TranscriptResult(
                video_id=row["video_id"],
                channel_id=row.get("channel_id", ""),
                language=row.get("language"),
                text=row.get("transcript_text", ""),
                status=row.get("fetch_status", "error"),
                error_message=row.get("error_message"),
            )
        except Exception:
            return None

    def save_transcript(self, result: TranscriptResult) -> bool:
        try:
            row = {
                "video_id": result.video_id,
                "channel_id": result.channel_id,
                "language": result.language,
                "transcript_text": result.text,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "fetch_status": result.status,
                "error_message": result.error_message,
            }
            self._db.table("transcripts").upsert(row).execute()
            return True
        except Exception:
            return False


# ============================================================================
# SELECTOR
# ============================================================================

def _get_supabase_secret(name: str) -> str | None:
    return os.getenv(name)


def _get_transcript_store() -> TranscriptStore:
    """Return the active TranscriptStore based on environment."""
    url = _get_supabase_secret("SUPABASE_URL")
    key = _get_supabase_secret("SUPABASE_SERVICE_KEY")
    if url and key:
        return SupabaseTranscriptStore(url, key)
    return JSONTranscriptStore()


# ============================================================================
# HELPERS WITH STORE INTEGRATION
# ============================================================================

def get_transcript_cached(
    video_id: str,
    channel_id: str,
    fetcher,
    language_pref: str | None = None,
    store: TranscriptStore | None = None,
) -> TranscriptResult:
    """
    Fetch a transcript with L2 store lookup + write-back.

    Cache-misses call fetcher.fetch(); the result (including negatives) is
    persisted so no-captions videos are never re-fetched.

    Args:
        video_id: YouTube video ID.
        channel_id: Parent channel ID (for store metadata).
        fetcher: TranscriptFetcher implementation.
        language_pref: Preferred language code.
        store: Optional pre-constructed store; defaults to _get_transcript_store().
    """
    if store is None:
        store = _get_transcript_store()

    cached = store.get_transcript(video_id)
    if cached is not None:
        return cached

    result = fetcher.fetch(video_id, language_pref)
    if result.channel_id == "":
        result.channel_id = channel_id
    store.save_transcript(result)
    return result

"""
seed_topics.py - Seed channel topic extraction and profiling

This module analyzes a seed YouTube channel to extract comprehensive profile data
(topics, keywords, tags, metrics) for similarity-based channel discovery.

Streamlit-agnostic: Uses callbacks for progress/API tracking (like youtube_api.py).
"""

import re
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

from .youtube_api import get_channel_stats, get_video_details
from .transcription import (
    TranscriptionConfig,
    YouTubeTranscriptFetcher,
    fetch_transcripts_parallel,
    extract_niche_summary,
)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Topic extraction thresholds
MIN_TOKEN_LENGTH = 3
MIN_DOC_FREQUENCY_RATIO = 0.20  # Term must appear in 20% of videos
MIN_DESC_DOC_FREQUENCY_RATIO = 0.15

# Scoring weights for term types
WEIGHT_TAGS = 2.0       # Tags are strongest signal
WEIGHT_BIGRAMS = 1.6    # Title phrases
WEIGHT_UNIGRAMS = 1.0   # Single words
WEIGHT_DESCRIPTION = 0.5  # Descriptions are noisy

# Output limits
MAX_PRIMARY_KEYWORDS = 5
MAX_SECONDARY_KEYWORDS = 10
MAX_COMMON_TAGS = 15
MAX_RECENT_TITLES = 20
MAX_DESCRIPTION_CHARS = 200
MAX_VIDEOS_FOR_DESC = 10

# Penalty thresholds
PENALTY_YEAR = 0.5
PENALTY_NUMBER = 0.3
PENALTY_MONTH = 0.4
PENALTY_PROMO = 0.3
PENALTY_EVENT = 0.5

# Subscriber tier boundaries
TIER_NANO_MAX = 10_000
TIER_MICRO_MAX = 100_000
TIER_MID_MAX = 1_000_000
TIER_MACRO_MAX = 10_000_000


# ============================================================================
# STOPWORDS AND NOISE PATTERNS
# ============================================================================

STOPWORDS_EN = {
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "about", "into", "over", "under", "very", "more", "without", "only",
    "here", "why", "also", "any", "some", "all", "every", "new", "best",
    "how", "your", "you", "our", "we", "but", "not", "what", "when", "where"
}

STOPWORDS_ES = {
    "que", "con", "para", "por", "las", "los", "una", "unos", "unas", "del",
    "sus", "este", "esta", "estas", "estos", "sobre", "entre", "como", "cuando",
    "donde", "desde", "hasta", "muy", "más", "mas", "sin", "solo", "sólo",
    "aqui", "aquí", "porque", "tambien", "también", "todo", "toda", "todos",
    "todas", "algo", "algun", "algún", "alguna", "algunas", "algunos", "pero"
}

STOPWORDS_COMMON = {
    "oficial", "official", "channel", "canal", "clips", "clip", "podcast",
    "tv", "shorts", "live", "directo", "video", "videos", "en", "de", "y"
}

# Noise patterns (light penalties, not hard blocks)
PROMO_WORDS = {"trailer", "estreno", "like", "share", "suscribete", "suscríbete"}
EVENT_WORDS = {"webinar", "congreso", "seminario", "conferencia", "tour", "festival"}
MONTHS_EN = {"january", "february", "march", "april", "may", "june", "july",
             "august", "september", "october", "november", "december"}
MONTHS_ES = {"enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SeedProfile:
    """Complete profile of a seed channel for similarity matching."""
    # Identity
    channel_id: str
    channel_name: str

    # Metrics
    subscriber_count: int
    subscriber_tier: str  # "nano", "micro", "mid", "macro", "mega"
    upload_frequency: float  # videos per month
    avg_engagement_rate: float

    # Content classification
    category: str
    language: str  # "en" or "es"

    # Topic extraction results
    primary_keywords: list[str] = field(default_factory=list)   # Multi-word phrases (max 5)
    secondary_keywords: list[str] = field(default_factory=list)  # Single words (max 10)
    common_tags: list[str] = field(default_factory=list)        # Video tags (max 15)

    # Context data
    recent_titles: list[str] = field(default_factory=list)      # Last 20 video titles
    description_summary: str = ""                               # AI-generated summary

    # Enhanced data from YouTube API (unused before, now captured)
    topic_categories: list[str] = field(default_factory=list)   # YouTube's topic classification
    channel_keywords: list[str] = field(default_factory=list)   # From brandingSettings

    # Transcript-derived niche profile (seed mode only, {} when unavailable)
    transcript_niche_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility with similarity.py."""
        return asdict(self)


@dataclass
class SeedAnalysisResult:
    """Result from analyzing a seed channel."""
    profile: Optional[SeedProfile] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    api_calls: int = 0


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(texts: list[str]) -> str:
    """
    Detect language from video titles using stopword analysis.

    Args:
        texts: List of video titles to analyze

    Returns:
        'es' for Spanish, 'en' for English (default)
    """
    en_hits = 0
    es_hits = 0

    for text in texts:
        words = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())
        for word in words:
            if word in STOPWORDS_EN:
                en_hits += 1
            elif word in STOPWORDS_ES:  # FIX: elif instead of if to avoid double-counting
                es_hits += 1

    return 'es' if es_hits > en_hits else 'en'


def get_stopwords(language: str) -> set:
    """Get combined stopwords for detected language."""
    if language == 'es':
        return STOPWORDS_ES | STOPWORDS_COMMON
    return STOPWORDS_EN | STOPWORDS_COMMON


# ============================================================================
# TOKENIZATION & TEXT PROCESSING
# ============================================================================

def tokenize(text: str, stopwords: set, min_length: int = MIN_TOKEN_LENGTH) -> list[str]:
    """
    Extract clean tokens from text.

    Rules:
    - Only alphabetic characters (keeps accents)
    - Minimum length
    - Not in stopwords
    - No numbers
    """
    words = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())

    return [
        word for word in words
        if len(word) >= min_length
        and word not in stopwords
        and not any(char.isdigit() for char in word)
    ]


def extract_bigrams(tokens: list[str], stopwords: set) -> list[str]:
    """
    Create meaningful two-word phrases from token list.

    Example: ['healthy', 'vegan', 'recipes'] -> ['healthy vegan', 'vegan recipes']
    """
    bigrams = []

    for i in range(len(tokens) - 1):
        word1, word2 = tokens[i], tokens[i + 1]

        # Skip if either word is a stopword
        if word1 in stopwords or word2 in stopwords:
            continue

        bigrams.append(f"{word1} {word2}")

    return bigrams


# ============================================================================
# SCORING & PENALTIES
# ============================================================================

def calculate_term_penalty(term: str) -> float:
    """
    Calculate penalty score for a term (0.0 = perfect, 1.0 = remove).

    Applies soft penalties for:
    - Years/dates (2024): 0.5 penalty
    - Numbers (ep5): 0.3 penalty
    - Month names: 0.4 penalty
    - Promotional words (like, share, subscribe): 0.3 penalty
    - Event-specific terms (webinar, festival): 0.5 penalty
    """
    tokens = set(term.lower().split())
    penalty = 0.0

    # Numbers or years (medium penalty)
    if any(t.isdigit() and len(t) == 4 for t in tokens):  # "2024"
        penalty += PENALTY_YEAR
    elif any(char.isdigit() for t in tokens for char in t):  # "ep5"
        penalty += PENALTY_NUMBER

    # Months (medium penalty)
    if tokens & MONTHS_EN or tokens & MONTHS_ES:
        penalty += PENALTY_MONTH

    # Promotional language (light penalty)
    if tokens & PROMO_WORDS:
        penalty += PENALTY_PROMO

    # Event-specific terms (medium penalty)
    if tokens & EVENT_WORDS:
        penalty += PENALTY_EVENT

    return min(penalty, 1.0)  # Cap at 1.0


def calculate_subscriber_tier(subscriber_count: int) -> str:
    """
    Classify channel by subscriber count.

    Tiers:
    - nano: < 10K
    - micro: 10K - 100K
    - mid: 100K - 1M
    - macro: 1M - 10M
    - mega: > 10M
    """
    if subscriber_count < TIER_NANO_MAX:
        return "nano"
    elif subscriber_count < TIER_MICRO_MAX:
        return "micro"
    elif subscriber_count < TIER_MID_MAX:
        return "mid"
    elif subscriber_count < TIER_MACRO_MAX:
        return "macro"
    else:
        return "mega"


def calculate_upload_frequency(publish_dates: list[str]) -> float:
    """
    Calculate average videos per month from publish dates.

    Args:
        publish_dates: List of ISO format date strings

    Returns:
        Videos per month (rounded to 2 decimals), or 0.0 if insufficient data
    """
    if len(publish_dates) < 2 or dateutil_parser is None:
        return 0.0

    try:
        dates = [dateutil_parser.parse(d) for d in publish_dates]
        dates.sort()
        time_span_days = (dates[-1] - dates[0]).total_seconds() / 86400

        # Handle edge case: same-second uploads
        time_span_days = max(time_span_days, 1 / 24)  # Minimum 1 hour span

        upload_frequency = (len(dates) / time_span_days) * 30
        return round(upload_frequency, 2)
    except Exception:
        return 0.0


def calculate_engagement_rate(videos: list[dict]) -> float:
    """
    Calculate average engagement rate across videos.

    Engagement = (likes + comments) / views

    Args:
        videos: List of video dicts with video_views, video_likes, video_comments

    Returns:
        Average engagement rate (rounded to 4 decimals)

    Note: FIX - includes 0-view videos using max(views, 1) to avoid bias
    """
    if not videos:
        return 0.0

    engagement_rates = []

    for video in videos:
        views = video.get('video_views', 0)
        likes = video.get('video_likes', 0)
        comments = video.get('video_comments', 0)

        # FIX: Include 0-view videos instead of skipping them
        engagement = (likes + comments) / max(views, 1)
        engagement_rates.append(engagement)

    if not engagement_rates:
        return 0.0

    return round(sum(engagement_rates) / len(engagement_rates), 4)


# ============================================================================
# TOPIC EXTRACTION
# ============================================================================

def extract_topics(
    videos: list[dict],
    stopwords: set,
    name_tokens: set,
    n_videos: int
) -> tuple[list[str], list[str], list[str]]:
    """
    Extract topics from video titles, tags, and descriptions.

    Args:
        videos: List of video dicts from get_video_details()
        stopwords: Language-specific stopwords
        name_tokens: Channel name tokens to exclude
        n_videos: Total number of videos for frequency calculation

    Returns:
        Tuple of (primary_keywords, secondary_keywords, common_tags)
    """
    min_doc_freq = max(2, math.ceil(MIN_DOC_FREQUENCY_RATIO * n_videos))

    # Count document frequency (how many videos mention each term)
    title_unigram_docs = Counter()
    title_bigram_docs = Counter()
    tag_docs = Counter()
    desc_docs = Counter()

    for idx, video in enumerate(videos):
        title = video.get('video_title', '')

        # Title unigrams
        title_tokens = set(t for t in tokenize(title, stopwords) if t not in name_tokens)
        title_unigram_docs.update(title_tokens)

        # Title bigrams
        bigrams = set(extract_bigrams(list(title_tokens), stopwords))
        title_bigram_docs.update(bigrams)

        # Tags
        tags = video.get('video_tags', [])
        clean_tags = set(
            tag.lower().strip()
            for tag in tags
            if tag and not any(nt in tag.lower() for nt in name_tokens)
        )
        tag_docs.update(clean_tags)

        # Descriptions (only first N videos, less weight)
        if idx < MAX_VIDEOS_FOR_DESC:
            desc = video.get('video_description', '')[:MAX_DESCRIPTION_CHARS]
            desc_tokens = set(t for t in tokenize(desc, stopwords) if t not in name_tokens)
            desc_docs.update(desc_tokens)

    # Score and rank terms
    scored_terms = []

    # Tags (highest weight - most accurate signal)
    for term, doc_freq in tag_docs.items():
        if doc_freq >= min_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * WEIGHT_TAGS * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'tag'))

    # Title bigrams (high weight - specific topics)
    for term, doc_freq in title_bigram_docs.items():
        if doc_freq >= min_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * WEIGHT_BIGRAMS * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'bigram'))

    # Title unigrams (medium weight)
    for term, doc_freq in title_unigram_docs.items():
        if doc_freq >= min_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * WEIGHT_UNIGRAMS * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'unigram'))

    # Description tokens (low weight - noisy)
    min_desc_doc_freq = max(2, math.ceil(MIN_DESC_DOC_FREQUENCY_RATIO * MAX_VIDEOS_FOR_DESC))
    for term, doc_freq in desc_docs.items():
        if doc_freq >= min_desc_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * WEIGHT_DESCRIPTION * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'description'))

    # Sort by score
    scored_terms.sort(key=lambda x: x[1], reverse=True)

    # Select best terms
    primary_keywords = []  # Multi-word phrases
    secondary_keywords = []  # Single words
    seen = set()

    for term, score, source in scored_terms:
        if term in seen:
            continue

        seen.add(term)

        if ' ' in term:  # Multi-word phrase
            if len(primary_keywords) < MAX_PRIMARY_KEYWORDS:
                primary_keywords.append(term)
        else:
            if len(secondary_keywords) < MAX_SECONDARY_KEYWORDS:
                secondary_keywords.append(term)

        if len(primary_keywords) >= MAX_PRIMARY_KEYWORDS and len(secondary_keywords) >= MAX_SECONDARY_KEYWORDS:
            break

    # Top tags (separate list)
    common_tags = [tag for tag, _ in tag_docs.most_common(MAX_COMMON_TAGS)]

    return primary_keywords, secondary_keywords, common_tags


def generate_description_summary(
    channel_name: str,
    channel_description: str,
    sample_titles: list[str],
    language: str,
    gemini_model
) -> str:
    """
    Generate AI summary of channel content using Gemini.

    Args:
        channel_name: Channel name
        channel_description: Channel description (first 500 chars)
        sample_titles: Recent video titles
        language: Detected language code
        gemini_model: Configured Gemini model instance

    Returns:
        Summary string or empty string on failure
    """
    if not gemini_model:
        return ""

    try:
        summary_prompt = f"""
Analyze this YouTube channel in 2-3 sentences:

Channel: {channel_name}
Description: {channel_description[:500]}
Recent video titles: {', '.join(sample_titles[:5])}

What is this channel's main content focus?
Answer in the same language as the channel ({language}).
"""

        response = gemini_model.generate_content(summary_prompt)
        return response.text.strip()
    except Exception:
        return ""


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_seed_channel(
    youtube_service,
    channel_id: str,
    max_videos: int = 50,
    gemini_model=None,
    on_progress: Callable[[str, float], None] | None = None,
    on_api_call: Callable[[str], None] | None = None,
    transcription_config: TranscriptionConfig | None = None,
    transcript_fetcher=None,
    transcript_store=None,
) -> SeedAnalysisResult:
    """
    Extract comprehensive profile from seed channel.

    This function is Streamlit-agnostic and uses callbacks for progress updates
    and API call tracking, following the pattern in youtube_api.py.

    Args:
        youtube_service: Authenticated YouTube Data API client
        channel_id: YouTube channel ID
        max_videos: Maximum videos to analyze (default 50)
        gemini_model: Optional configured Gemini model for AI summary
        on_progress: Callback for progress updates (message, percentage 0.0-1.0)
        on_api_call: Callback for API call tracking

    Returns:
        SeedAnalysisResult with profile or error
    """
    api_calls = 0
    warnings = []

    def progress(msg: str, pct: float):
        if on_progress:
            on_progress(msg, pct)

    # ========================================================================
    # STEP 1: Get channel metadata using core function
    # ========================================================================

    progress("Fetching channel metadata...", 0.1)

    try:
        stats_result = get_channel_stats(
            youtube_service,
            channel_ids=[channel_id],
            on_api_call=on_api_call
        )
        api_calls += stats_result.api_calls

        if not stats_result.stats:
            return SeedAnalysisResult(
                profile=None,
                error="Channel not found. The channel may be private, deleted, or the URL format may be incorrect.",
                api_calls=api_calls
            )

        channel_stats = stats_result.stats[0]

    except Exception as e:
        return SeedAnalysisResult(
            profile=None,
            error=f"Failed to fetch channel: {e}",
            api_calls=api_calls
        )

    # Extract channel info
    channel_name = channel_stats.get('description', '')  # Note: This might need adjustment
    subscriber_count = channel_stats.get('subscribers', 0)
    uploads_playlist_id = channel_stats.get('uploads_playlist_id')
    channel_description = channel_stats.get('description', '')
    topic_categories = channel_stats.get('topic_categories', [])
    channel_keywords = channel_stats.get('channel_keywords', [])
    default_language = channel_stats.get('default_language', '')

    # We need channel name from snippet - get_channel_stats doesn't return title
    # Make a supplementary call if needed, or use the channel_id
    # Actually, let's check if we need to make a direct call for the name
    # Looking at the stats, we don't have channel_name directly

    # Let's make a minimal call to get the channel name
    try:
        name_response = youtube_service.channels().list(
            part="snippet",
            id=channel_id
        ).execute()
        if on_api_call:
            on_api_call('youtube_channel')
        api_calls += 1

        if name_response.get('items'):
            snippet = name_response['items'][0]['snippet']
            channel_name = snippet.get('title', 'Unknown')
            channel_description = snippet.get('description', channel_description)
            category_id = snippet.get('categoryId', 'Unknown')
        else:
            channel_name = 'Unknown'
            category_id = 'Unknown'
    except Exception:
        channel_name = 'Unknown'
        category_id = 'Unknown'

    # Build channel name stopwords (don't extract brand name as topic)
    name_tokens = set(re.findall(r'\b[a-záéíóúñü]+\b', channel_name.lower()))
    name_tokens |= {"oficial", "official", "canal", "channel"}

    progress(f"Found: {channel_name} ({subscriber_count:,} subscribers)", 0.2)

    # ========================================================================
    # STEP 2: Get video details using core function
    # ========================================================================

    if not uploads_playlist_id:
        return SeedAnalysisResult(
            profile=None,
            error="Could not find uploads playlist for this channel.",
            api_calls=api_calls
        )

    progress("Fetching video details...", 0.3)

    try:
        video_result = get_video_details(
            youtube_service,
            channel_data=[{
                'channel_id': channel_id,
                'uploads_playlist_id': uploads_playlist_id,
                'channel_title': channel_name
            }],
            max_videos_per_channel=max_videos,
            on_api_call=on_api_call
        )
        api_calls += video_result.api_calls
        warnings.extend(video_result.warnings)

        videos = video_result.videos

    except Exception as e:
        return SeedAnalysisResult(
            profile=None,
            error=f"Failed to fetch videos: {e}",
            api_calls=api_calls
        )

    if not videos:
        return SeedAnalysisResult(
            profile=None,
            error="No videos found in channel. This channel may be empty or have all videos set to private.",
            api_calls=api_calls
        )

    progress(f"Analyzing {len(videos)} videos...", 0.5)

    # ========================================================================
    # STEP 3: Calculate engagement metrics
    # ========================================================================

    avg_engagement = calculate_engagement_rate(videos)

    # Extract publish dates for frequency calculation
    publish_dates = [v.get('published_at', '') for v in videos if v.get('published_at')]
    upload_frequency = calculate_upload_frequency(publish_dates)

    # ========================================================================
    # STEP 4: Language detection
    # ========================================================================

    sample_titles = [v.get('video_title', '') for v in videos]

    # Use default_language hint if available, otherwise detect
    if default_language in ('en', 'es'):
        detected_language = default_language
    else:
        detected_language = detect_language(sample_titles)

    stopwords = get_stopwords(detected_language)

    progress(f"Language detected: {detected_language.upper()}", 0.6)

    # ========================================================================
    # STEP 5: Topic extraction
    # ========================================================================

    progress("Extracting topics...", 0.7)

    primary_keywords, secondary_keywords, common_tags = extract_topics(
        videos=videos,
        stopwords=stopwords,
        name_tokens=name_tokens,
        n_videos=len(videos)
    )

    # Enrich common_tags with channel-level metadata — zero extra API calls.
    # topic_categories (YouTube's own topic taxonomy) and channel_keywords
    # (from brandingSettings) are both already fetched; folding them in
    # makes Jaccard tag-overlap more discriminating for seed mode.
    meta_tags = [
        t.lower().strip()
        for t in (topic_categories + channel_keywords)
        if t and len(t.strip()) >= MIN_TOKEN_LENGTH and t.strip().lower() not in name_tokens
    ]
    if meta_tags:
        common_tags = list(dict.fromkeys(common_tags + meta_tags))[:MAX_COMMON_TAGS]

    # Enrich secondary_keywords with tokenized channel_keywords.
    existing_kws = set(primary_keywords + secondary_keywords)
    for kw in channel_keywords:
        for token in tokenize(kw, stopwords):
            if token not in existing_kws and len(secondary_keywords) < MAX_SECONDARY_KEYWORDS:
                secondary_keywords.append(token)
                existing_kws.add(token)

    # ========================================================================
    # STEP 6: AI summary (optional)
    # ========================================================================

    description_summary = ""
    if gemini_model:
        progress("Generating AI summary...", 0.85)
        description_summary = generate_description_summary(
            channel_name=channel_name,
            channel_description=channel_description,
            sample_titles=sample_titles,
            language=detected_language,
            gemini_model=gemini_model
        )
        if description_summary:
            progress("AI-enhanced analysis complete", 0.95)

    # ========================================================================
    # STEP 6b: Transcript niche extraction (seed mode, optional)
    # ========================================================================

    transcript_niche_summary: dict = {}
    tc = transcription_config if transcription_config is not None else TranscriptionConfig()

    if tc.enabled and gemini_model:
        try:
            # Filter to non-Shorts videos (duration_seconds must be >= 60 or unknown)
            candidate_videos = [
                v for v in videos
                if not tc.skip_shorts
                or v.get('duration_seconds', 9999) >= 60
            ]
            fetch_videos = candidate_videos[:tc.max_videos]
            video_ids = [v['video_id'] for v in fetch_videos if v.get('video_id')]

            if video_ids:
                fetcher = transcript_fetcher
                if fetcher is None:
                    import os as _os
                    proxy_url = _os.getenv("TRANSCRIPT_PROXY_URL")
                    fetcher = YouTubeTranscriptFetcher(
                        channel_id=channel_id,
                        proxy_url=proxy_url,
                    )

                progress("Fetching transcripts for niche analysis...", 0.88)

                transcript_results = fetch_transcripts_parallel(
                    video_ids=video_ids,
                    fetcher=fetcher,
                    config=tc,
                    language_pref=detected_language,
                    on_progress=on_progress,
                    on_api_call=on_api_call,
                )

                # Optionally persist results via store
                if transcript_store is not None:
                    for tr in transcript_results:
                        try:
                            transcript_store.save_transcript(tr)
                        except Exception:
                            pass

                rate_limited = [r for r in transcript_results if r.status == "rate_limited"]
                if len(rate_limited) >= 3:
                    warnings.append(
                        "Transcript service unavailable; using metadata-only analysis."
                    )
                else:
                    niche_result = extract_niche_summary(
                        transcripts=transcript_results,
                        gemini_model=gemini_model,
                        on_api_call=on_api_call,
                        corpus_chars=tc.corpus_chars,
                    )
                    transcript_niche_summary = niche_result.summary

                    if transcript_niche_summary:
                        progress("Transcript niche analysis complete", 0.95)

        except Exception as exc:
            warnings.append(f"Transcript analysis skipped: {exc}")

    # ========================================================================
    # STEP 7: Build profile
    # ========================================================================

    subscriber_tier = calculate_subscriber_tier(subscriber_count)

    profile = SeedProfile(
        channel_id=channel_id,
        channel_name=channel_name,
        subscriber_count=subscriber_count,
        subscriber_tier=subscriber_tier,
        category=category_id,
        language=detected_language,
        upload_frequency=upload_frequency,
        avg_engagement_rate=avg_engagement,
        primary_keywords=primary_keywords[:MAX_PRIMARY_KEYWORDS],
        secondary_keywords=secondary_keywords[:MAX_SECONDARY_KEYWORDS],
        common_tags=common_tags[:MAX_COMMON_TAGS],
        recent_titles=sample_titles[:MAX_RECENT_TITLES],
        description_summary=description_summary,
        topic_categories=topic_categories,
        channel_keywords=channel_keywords,
        transcript_niche_summary=transcript_niche_summary,
    )

    progress(
        f"Analysis complete: {len(primary_keywords)} phrases, {len(common_tags)} tags, {len(secondary_keywords)} keywords",
        1.0
    )

    return SeedAnalysisResult(
        profile=profile,
        warnings=warnings,
        api_calls=api_calls
    )

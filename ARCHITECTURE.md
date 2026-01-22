# CCSeeker Architecture Documentation

## System Overview

CCSeeker is an AI-powered YouTube creator discovery tool that automates finding niche content creators. It reduces manual search time from hours to minutes through intelligent search and ranking algorithms.

**Core Design Philosophy:**
- Minimize API calls through intelligent caching
- Provide two distinct search modes for different user scenarios
- Make internal workings transparent through debug tooling
- Fail gracefully when APIs are unavailable or quota is exhausted
- Separate pure business logic from UI for testability

**Tech Stack:**
- **Streamlit**: UI framework with built-in state management and caching
- **YouTube Data API v3**: Channel and video metadata retrieval
- **Google Gemini AI**: Topic extraction, relevance scoring, and content generation
- **Pandas**: Data transformation and filtering
- **pytest**: Unit testing with mocked API clients

---

## Architecture: Layered Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│                         (app/main.py)                           │
│    Streamlit UI, user interactions, session state, rendering    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CACHE LAYER                              │
│                        (app/cache/)                              │
│      Streamlit @cache_data wrappers, TTL management             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                          CORE LAYER                              │
│                         (app/core/)                              │
│   Pure business logic: pipeline, APIs, relevance, validation    │
│              (Streamlit-agnostic, unit testable)                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                          │
│              YouTube Data API v3  │  Google Gemini AI           │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Presentation** | `app/main.py` | UI rendering, user input, session state, progress display |
| **Cache** | `app/cache/` | Streamlit caching wrappers, TTL configuration |
| **Core** | `app/core/` | Pure business logic, API wrappers, pipeline orchestration |
| **Utilities** | `app/*.py` | Seed analysis, similarity scoring, debug tracking |

---

## Project Structure

```
CCSeeker/
├── app/
│   ├── core/                     # Pure business logic (Streamlit-agnostic)
│   │   ├── __init__.py           # Public API exports (~30 functions/classes)
│   │   ├── query_utils.py        # Query validation, URL parsing, channel ID resolution
│   │   ├── relevance.py          # Keyword relevance scoring
│   │   ├── youtube_api.py        # YouTube Data API wrappers
│   │   ├── gemini_api.py         # Gemini AI API wrappers
│   │   └── pipeline.py           # Search pipeline orchestration
│   │
│   ├── cache/                    # Centralized caching layer
│   │   ├── __init__.py           # Cache exports and TTL constants
│   │   └── cache_layer.py        # Streamlit @cache_data wrappers
│   │
│   ├── main.py                   # Streamlit UI (~1467 lines)
│   ├── seed_topics_v2.py         # Seed channel topic extraction
│   ├── similarity_engine.py      # Multi-factor similarity scoring
│   ├── debug_tracker.py          # API usage tracking, quota monitoring
│   ├── feedback_tracker.py       # User feedback collection
│   └── smart_cache.py            # Per-channel video caching (24h TTL)
│
├── tests/                        # Unit test suite
│   ├── test_query_utils.py       # Query validation tests
│   ├── test_relevance.py         # Relevance scoring tests
│   ├── test_youtube_api.py       # YouTube API wrapper tests
│   ├── test_gemini_api.py        # Gemini API wrapper tests
│   └── test_pipeline.py          # Pipeline integration tests
│
├── docs/                         # Documentation assets
├── .streamlit/config.toml        # Streamlit configuration
├── requirements.txt              # Python dependencies
└── README.md                     # User guide
```

---

## Core Layer Design (`app/core/`)

The core layer contains pure business logic extracted from the original monolithic `main.py`. These modules are:

- **Streamlit-agnostic**: No `st.*` calls, no session state access
- **Testable**: Can be unit tested with mocked dependencies
- **Callback-based**: Progress and warnings communicated via callbacks

### Module Overview

| Module | Lines | Key Exports |
|--------|-------|-------------|
| `query_utils.py` | 199 | `validate_and_truncate_query()`, `extract_identifier_from_url()`, `resolve_channel_id()` |
| `relevance.py` | 115 | `calculate_keyword_relevance()` |
| `youtube_api.py` | 483 | `SearchResult`, `ChannelStatsResult`, `VideoDetailsResult`, `search_channels_hybrid()`, `get_channel_stats()`, `get_video_details()` |
| `gemini_api.py` | 290 | `OutreachDraft`, `SummaryResult`, `generate_ai_relevance_score()`, `generate_summary()`, `generate_outreach_drafts()` |
| `pipeline.py` | 753 | `PipelineResult`, `PipelineConfig`, `run_search_pipeline()` |

### Design Patterns

**1. Structured Results (Dataclasses)**
```python
@dataclass
class PipelineResult:
    channels_df: pd.DataFrame
    display_columns: list[str]
    column_explanations: dict[str, str]
    search_log: list[str]
    timings: dict[str, float]
    warnings: list[str]
    error: str | None = None
    ai_summary: str | None = None
```

**2. Callback Pattern for Progress**
```python
def run_search_pipeline(
    youtube_service,
    query: str,
    config: PipelineConfig,
    on_progress: Callable[[str, float], None] | None = None,  # UI updates
    on_api_call: Callable[[str], None] | None = None,         # Tracking
) -> PipelineResult:
```

**3. Protocol for Dependency Injection**
```python
class CacheFunctions(Protocol):
    def get_channel_stats_cached(self, channel_ids: tuple) -> list[dict]: ...
    def get_video_details_cached(self, channel_ids: tuple, max_videos: int) -> list[dict]: ...
    def search_channels_cached(self, query: str, region: str, max_videos: int) -> list[dict]: ...
```

---

## Search Pipeline Flow

### Keyword Mode

```
User Input: "manga, anime"
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 1: Query Validation                │
│ • Truncate to max 2 terms               │
│ • Strip quotes, normalize               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 2: Hybrid Search                   │
│ A) Video search → extract channel IDs   │
│ B) Channel name search → bonus matches  │
│ C) Score: video_matches*10 + name*5     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 3: Fetch Channel Statistics        │
│ • Subscribers, views, country           │
│ • Channel age, uploads playlist ID      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 4: Apply Filters                   │
│ • Minimum subscribers                   │
│ • Country filter                        │
│ • Match score threshold                 │
│ • Cap at 50 channels                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 5: Deep Analysis                   │
│ • Fetch 10 videos per channel           │
│ • Calculate engagement rates            │
│ • Apply recency filter                  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 6: Relevance Scoring               │
│ A) Algorithmic: keyword match in        │
│    titles (2x weight) + tags (1x)       │
│ B) AI: Gemini semantic analysis         │
│ C) Blend: 80% algorithmic + 20% AI      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ STEP 7: AI Summary (optional)           │
│ • Generate overview of top 5 channels   │
└─────────────────────────────────────────┘
         │
         ▼
        Results
```

### Seed Mode (Additional Steps)

```
User Input: youtube.com/@channel
         │
         ▼
┌─────────────────────────────────────────┐
│ SEED ANALYSIS (seed_topics_v2.py)       │
│ 1. Detect language (EN/ES stopwords)    │
│ 2. Extract bigrams from titles          │
│ 3. Extract tags from videos             │
│ 4. Apply penalty system for noise       │
│ 5. Gemini refinement (optional)         │
│                                         │
│ Output: seed_profile dict               │
│ • primary_keywords (phrases)            │
│ • secondary_keywords (words)            │
│ • common_tags                           │
│ • engagement_rate, upload_frequency     │
└─────────────────────────────────────────┘
         │
         ▼
      [Search Pipeline Steps 1-5]
         │
         ▼
┌─────────────────────────────────────────┐
│ SIMILARITY SCORING                      │
│ (similarity_engine.py)                  │
│                                         │
│ For each candidate channel:             │
│ ┌─────────────────────────────────────┐ │
│ │ Tag Overlap (30 pts)                │ │
│ │ Jaccard(candidate_tags, seed_tags)  │ │
│ ├─────────────────────────────────────┤ │
│ │ Keyword Overlap (30 pts)            │ │
│ │ Jaccard(candidate_kw, seed_kw)      │ │
│ ├─────────────────────────────────────┤ │
│ │ Subscriber Similarity (15 pts)      │ │
│ │ min(ratio, inverse_ratio)           │ │
│ ├─────────────────────────────────────┤ │
│ │ Engagement Similarity (17 pts)      │ │
│ │ 17 - (diff * 170)                   │ │
│ ├─────────────────────────────────────┤ │
│ │ Upload Frequency (8 pts)            │ │
│ │ min(ratio, inverse_ratio) * 8       │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Algorithmic Score: 0-100 points         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ GEMINI "VIBE" ANALYSIS                  │
│ (Top 10 channels only)                  │
│                                         │
│ Evaluates:                              │
│ • Topic overlap                         │
│ • Content style                         │
│ • Target audience                       │
│ • Production approach                   │
│                                         │
│ Returns: gemini_score (0-10)            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ FINAL BLENDED SCORE                     │
│                                         │
│ final = 0.8 * algorithmic + 0.2 * AI    │
└─────────────────────────────────────────┘
```

---

## Scoring Algorithms

### Keyword Mode: Relevance Score

The relevance score measures how well a channel's content matches the search query.

```python
def calculate_keyword_relevance(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    1. Parse query into terms (comma-separated)
    2. For each video:
       - title_match = query term found in title (weight: 2.0)
       - tags_match = query term found in tags (weight: 1.0)
       - video_score = (2.0 * title_match + 1.0 * tags_match) / 3.0
    3. Channel relevance = mean(video_scores)

    Returns: 0.0 to 1.0
    """
```

**AI Enhancement:**
```python
# Gemini evaluates video titles semantically
ai_score = generate_ai_relevance_score(model, channel_data, query)  # 0.0-1.0

# Blend: 80% keyword matching + 20% AI
final_relevance = 0.8 * algorithmic_relevance + 0.2 * ai_score
```

### Seed Mode: Similarity Score

The similarity score compares a candidate channel to the seed channel across multiple factors.

| Factor | Points | Calculation | Signal Quality |
|--------|--------|-------------|----------------|
| **Tag Overlap** | 30 | Jaccard similarity on video tags | High - creators consciously choose tags |
| **Keyword Overlap** | 30 | Jaccard similarity on title keywords | High - reflects content focus |
| **Subscriber Tier** | 15 | min(candidate/seed, seed/candidate) | Medium - prevents size mismatch |
| **Engagement Rate** | 17 | 17 - (abs_diff * 170) | Medium - audience quality indicator |
| **Upload Frequency** | 8 | ratio * 8 | Low - nice-to-have |

**Jaccard Similarity:**
```python
def jaccard_similarity(set1: set, set2: set) -> float:
    """J(A, B) = |A ∩ B| / |A ∪ B|"""
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0
```

**Gemini "Vibe" Analysis (top 10 channels):**
```python
# Evaluates content similarity beyond keyword matching
gemini_result = gemini_similarity_analysis(candidate, seed_profile, api_key)
# Returns: gemini_score (0-10), gemini_reason

# Final blend
final_similarity = 0.8 * algorithmic_score + 0.2 * (gemini_score * 10)
```

---

## Caching Architecture

### Cache TTLs

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Channel stats | 7 days | Subscriber counts change slowly |
| Search results | 3 days | Moderate freshness needed |
| Video details | 24 hours | New uploads appear daily |

### Per-Channel Video Caching

```python
# Problem: Popular channels appear in multiple searches
# Solution: Cache by channel_id, not by search query

@st.cache_data(ttl=86400)  # 24 hours
def get_channel_videos(channel_id: str, max_videos: int):
    """Cache individual channel's videos independently"""
    pass

# Search 1: "manga" → Fetches channels A, B, C videos
# Search 2: "anime" → Cache HIT for B, C; only fetch D
```

### Cache Key Strategy

```python
# Include max_videos in key (seed uses 50, keyword uses 10)
def _make_cache_key(channel_id: str, max_videos: int) -> str:
    return f"ch_vids_{channel_id}_{max_videos}"
```

### CacheFunctionsAdapter

The pipeline accepts a `CacheFunctions` protocol for dependency injection:

```python
class CacheFunctionsAdapter:
    """Adapter connecting core pipeline to Streamlit cache layer"""

    def get_channel_stats_cached(self, channel_ids: tuple) -> list[dict]:
        return cache_layer.get_channel_stats_cached(channel_ids)

    def get_video_details_cached(self, channel_ids: tuple, max_videos: int) -> list[dict]:
        return cache_layer.get_video_details_cached(channel_ids, max_videos)
```

---

## Debug & Monitoring System

### API Call Tracking

```python
def track_api_call(api_name: str):
    """
    Tracked operations:
    - youtube_search (100 units)
    - youtube_channel (1 unit)
    - youtube_video (1 unit)
    - youtube_playlist (1 unit)
    - gemini_summary
    - gemini_outreach
    - gemini_similarity
    """
```

### Quota Monitoring

```python
def calculate_youtube_quota_used() -> int:
    """
    YouTube API quota costs:
    - search: 100 units
    - channels/videos/playlists: 1 unit each

    Daily limit: 10,000 units (free tier)
    """
    return (
        data['youtube_search_calls'] * 100 +
        data['youtube_channel_calls'] * 1 +
        data['youtube_video_calls'] * 1 +
        data['youtube_playlist_calls'] * 1
    )
```

### Performance Timing

The debug sidebar displays timing for each pipeline step:
- Search
- Channel stats fetch
- Video details fetch
- AI relevance scoring
- Similarity calculation
- AI generation
- **Total time**

Plus bottleneck identification: shows which step consumed the most time.

### Daily Quota Persistence

```python
# Quota tracking persists across browser refreshes
daily_quota = {
    'date': '2024-01-15',
    'youtube_units': 1500,
    'youtube_calls': 25,
    'gemini_calls': 10,
    'gemini_cost_usd': 0.0012
}
# Stored in .daily_quota.json, resets at midnight PT
```

---

## Testing Architecture

### Test Strategy

All core modules are tested with mocked API clients - no actual API calls needed.

```python
@pytest.fixture
def mock_youtube():
    """Create mock YouTube API with reasonable defaults"""
    youtube = Mock()
    youtube.search().list().execute.return_value = {...}
    youtube.channels().list().execute.return_value = {...}
    return youtube
```

### Test Coverage

| Module | Tests | Coverage Focus |
|--------|-------|----------------|
| `test_query_utils.py` | 21 | Query validation, URL parsing, edge cases |
| `test_relevance.py` | 13 | Keyword matching, weights, empty inputs |
| `test_youtube_api.py` | ~20 | Search results, channel stats, error handling |
| `test_gemini_api.py` | ~15 | AI scoring, summary generation, API failures |
| `test_pipeline.py` | ~25 | Full pipeline, filters, early exits, callbacks |

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_pipeline.py

# Verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=app/core
```

---

## Seed Topic Extraction (`seed_topics_v2.py`)

### Language Detection

```python
def detect_language(texts: list[str]) -> str:
    """
    Count stopword hits: EN vs ES
    Returns: 'en' or 'es'

    Note: Other languages fall back to English stopwords,
    which may affect topic extraction quality.
    """
```

### Penalty System

Soft penalties replace hard blocking to allow context-dependent terms:

| Pattern | Penalty | Rationale |
|---------|---------|-----------|
| Years (2024) | 0.5 | Time-sensitive but sometimes relevant |
| Numbers (ep5) | 0.3 | Episode numbers are common noise |
| Months | 0.4 | Seasonal content markers |
| Promo words (subscribe) | 0.3 | Call-to-action noise |
| Event words (webinar) | 0.5 | One-time event content |

```python
def calculate_term_penalty(term: str) -> float:
    """Returns 0.0 (perfect) to 1.0 (remove)"""
    penalty = 0.0
    if has_year: penalty += 0.5
    if has_number: penalty += 0.3
    if is_month: penalty += 0.4
    return min(penalty, 1.0)

# Applied to scoring
score = doc_freq * weight * (1.0 - penalty)
```

### Topic Scoring Weights

| Source | Weight | Rationale |
|--------|--------|-----------|
| Tags | 2.0x | Most accurate signal - creator-chosen |
| Bigrams (title phrases) | 1.6x | Strong content indicator |
| Unigrams (title words) | 1.0x | Baseline signal |
| Description tokens | 0.5x | Noisy, low weight |

---

## API Quotas & Costs

### YouTube Data API

| Operation | Cost | Notes |
|-----------|------|-------|
| `search().list()` | 100 units | Most expensive |
| `channels().list()` | 1 unit | Batch up to 50 |
| `videos().list()` | 1 unit | Batch up to 50 |
| `playlistItems().list()` | 1 unit | Per request |

**Daily limit:** 10,000 units (free tier)

**Typical search cost:**
- 3 search calls × 100 = 300 units
- 4 channel stats call = 4 unit
- 50 video detail calls = 50 units
- 50 playlist calls = 50 units
- **Total: ~404 units per search**

### Gemini AI

| Tier | Rate Limits | Cost |
|------|-------------|------|
| Free | 15 RPM, 1M tokens/min | $0 |
| Paid | Higher limits | ~$0.0001/1K tokens |

---

## Performance & Efficiency

CCSeeker optimizes for two constraints: **API quota** (10,000 YouTube units/day) and **latency**. Performance was measured via the debug panel (January 2026).

### Measured Performance

#### Keyword Search Mode

| Scenario | Total Time | Bottleneck | Quota Used |
|----------|------------|------------|------------|
| Cold cache (no AI) | 9-10s | Video details (84-87%) | ~400 units |
| Warm cache (no AI) | 0.04s | Relevance filtering (46-48%) | ~100 units |
| Warm cache (with AI) | 17-19s | AI relevance (92-94%) | ~100 units |

<details>
<summary>Step-by-step breakdown (1 term, cold cache, no AI)</summary>

| Step | Time | % of Total |
|------|------|------------|
| Search | 1.2s | 12% |
| Channel stats | 0.2s | 2% |
| Video details | 8-9s | **84-87%** |
| Relevance scoring | <0.1s | <1% |
| **Total** | **9-10s** | 100% |

</details>

#### Seed-Based Search Mode (with AI and 2 terms)

| Step | Time | % of Total |
|------|------|------------|
| Search | 2.4s | 6% |
| Channel stats | 0.4s | 1% |
| Video details | 9.8s | 23% |
| AI relevance | 17.5s | **42%** |
| Similarity calculation | 8.1s | 19% |
| AI summary | 3.8s | 9% |
| **Total** | **~42s** | 100% |

*Seed channel: @t3dotgg*

### Key Findings

1. **Cache reduces latency by 99%** — Warm cache keyword search: 0.04s vs cold: 9-10s
2. **Cache reduces quota by 75%** — 100 units (warm) vs 400 units (cold)
3. **AI is the bottleneck when enabled** — 92-94% of keyword time, 42% of seed time
4. **Video details fetch is the bottleneck without AI** — Sequential API calls per channel

### Efficiency by Design

| Pattern | How It Saves Resources |
|---------|----------------------|
| **Filter-before-fetch** | Channels filtered by subscribers/country before expensive video fetches |
| **Entity-level caching** | Cache keyed by channel ID, not query — "cooking" and "recipes" share cache hits |
| **Batched channel stats** | Up to 50 channel IDs per API call (YouTube limit) |
| **Bounded work limits** | Max 2 search terms, 50 channels, 10 videos/channel — prevents runaway quota |
| **Quota monitoring** | Real-time tracking with warnings before exhaustion |

### Quota Budget

| Cache State | Quota Units | Searches/Day on 1 term (Free Tier) |
|-------------|-------------|--------------------------|
| Cold cache | 400 units | 25 searches |
| Warm cache | 100 units | 100 searches |

---

## Key Design Decisions

### 1. Layered Architecture
**Why:** Separates testable business logic from Streamlit-specific code. Core functions can be reused outside Streamlit (CLI, API, etc.).

### 2. Callback Pattern
**Why:** Allows core functions to report progress without depending on `st.progress()`. Tests can verify callbacks were called correctly.

### 3. Structured Results (Dataclasses)
**Why:** Type safety, IDE support, clear contracts between layers. Errors and warnings are part of the result, not exceptions.

### 4. Blended Scoring (80% Algorithmic + 20% AI)
**Why:** Algorithmic scoring is deterministic and fast. AI adds semantic understanding but is slower and costs money. Blend gets benefits of both.

### 5. Filter Before Fetch
**Why:** No point fetching videos for channels that don't meet subscriber threshold. Saves API quota.

### 6. Per-Channel Caching
**Why:** Same channel appears across different searches. Cache by channel ID, not search query, maximizes cache hits.

### 7. Soft Penalties vs Hard Blocks
**Why:** "2024" in a title might be noise ("2024 best movies") or relevant ("2024 predictions"). Penalties allow context rather than binary exclusion.

---

## Known Limitations

- **Language Support:** Stopwords and month detection only implemented for English and Spanish. Other languages fall back to English, affecting topic extraction quality.
- **YouTube API Quota:** 10K units/day limits search volume. Heavy users may hit quota mid-day.
- **Cache Staleness:** 24hr TTL for videos may show outdated data for rapidly changing channels.
- **Tag Dependency:** Similarity scoring works best when channels use descriptive tags. Channels without tags still score on keywords, subscribers, engagement, and upload frequency (70 points max).

---

## Future Considerations

- Add support for additional languages (French, German, Portuguese stopwords)
- Implement cache warming for frequently searched niches


---

**End of Document**

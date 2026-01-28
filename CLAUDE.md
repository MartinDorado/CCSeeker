# CLAUDE.md - CCSeeker Project Guide

## Project Overview

CCSeeker is an AI-powered YouTube creator discovery tool that automates finding niche content creators. It reduces manual search time from hours to minutes through intelligent search and ranking algorithms.

**Repository:** https://github.com/MartinDorado/CCseeker
**License:** Apache 2.0
**Author:** Martin Dorado

## Tech Stack

- **Python 3.11** - Core language
- **Streamlit 1.49.0** - Web UI framework
- **YouTube Data API v3** - Channel/video metadata
- **Google Gemini AI** - Topic extraction and content generation
- **Pandas** - Data processing
- **pytest** - Unit testing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
YOUTUBE_API_KEY=your_key
GEMINI_API_KEY=your_key  # Optional - core search works without it; enables AI features

# Run the application
streamlit run app/main.py

# Run tests
pytest tests/
```

## Project Structure

```
CCSeeker/
├── app/
│   ├── core/                     # Pure business logic (Streamlit-agnostic)
│   │   ├── __init__.py           # Public API exports (~33 functions/classes)
│   │   ├── query_utils.py        # Query validation, URL parsing, channel ID resolution
│   │   ├── relevance.py          # Keyword relevance scoring
│   │   ├── youtube_api.py        # YouTube Data API wrappers
│   │   ├── gemini_api.py         # Gemini AI API wrappers
│   │   ├── pipeline.py           # Search pipeline orchestration
│   │   ├── scoring_version.py    # Centralized scoring weights and version management
│   │   └── seed_topics.py        # Seed channel topic extraction and profiling
│   │
│   ├── cache/                    # Centralized caching layer
│   │   ├── __init__.py           # Cache exports and TTL constants
│   │   └── cache_layer.py        # Streamlit @cache_data wrappers
│   │
│   ├── analytics/                # ML and analytics module
│   │   ├── __init__.py           # Analytics exports (14 functions/classes)
│   │   ├── synthetic_data_generator.py  # Synthetic feedback generation
│   │   ├── ml_trainer.py         # ML model training (logistic regression, cross-validation)
│   │   ├── weight_optimizer.py   # Weight optimization algorithms
│   │   └── fabric_export.py      # Microsoft Fabric/Power BI export
│   │
│   ├── main.py                   # Streamlit UI and integration (~1675 lines)
│   ├── similarity_engine.py      # Multi-factor similarity scoring
│   ├── debug_tracker.py          # API usage tracking, quota monitoring
│   ├── feedback_tracker.py       # User feedback collection
│   └── smart_cache.py            # Per-channel video caching (24h TTL)
│
├── tests/                        # Unit test suite (262 tests total)
│   ├── test_query_utils.py       # 21 tests for query utilities
│   ├── test_relevance.py         # 13 tests for relevance scoring
│   ├── test_youtube_api.py       # 29 tests for YouTube API wrappers
│   ├── test_gemini_api.py        # 31 tests for Gemini API wrappers
│   ├── test_pipeline.py          # 26 tests for search pipeline
│   ├── test_seed_topics.py       # 46 tests for seed topic extraction
│   ├── test_analytics.py         # 27 tests for analytics module
│   ├── test_feedback_tracker.py  # 27 tests for feedback tracking
│   ├── test_scoring_version.py   # 26 tests for scoring version
│   └── test_performance.py       # 16 tests for performance benchmarks
│
├── docs/                         # Icons and screenshots
├── .streamlit/config.toml        # Streamlit configuration
├── requirements.txt              # Python dependencies
├── ARCHITECTURE.md               # Detailed technical documentation
└── README.md                     # User guide
```

## Architecture

### Design Principles

- **Separation of Concerns:** Core business logic in `app/core/` is Streamlit-agnostic
- **Testability:** Pure functions can be unit tested without UI dependencies
- **Callback Pattern:** Progress/warning updates use callbacks instead of direct `st.*` calls
- **Structured Results:** Dataclasses (`SearchResult`, `PipelineResult`, etc.) ensure type safety

### Two Search Modes

1. **Keyword Search** - Hybrid video content + channel name matching
2. **Seed-Based Search** - Extract topics from a seed channel to find similar creators

### Scoring Systems

**Keyword Mode - Relevance Score:**
1. Algorithmic relevance (keyword matching in titles/tags): 80%
2. AI semantic analysis (Gemini evaluates video titles): 20%

**Seed Mode - Similarity Score (100 points):**
| Factor | Points | Source |
|--------|--------|--------|
| Tag Overlap | 30 | Jaccard similarity on video tags |
| Keyword Overlap | 30 | Jaccard similarity on title keywords (bigrams + unigrams) |
| Subscriber Tier | 15 | Ratio-based scoring |
| Engagement Rate | 17 | Absolute difference penalty |
| Upload Frequency | 8 | Ratio-based scoring |

Final similarity = 80% algorithmic + 20% Gemini "vibe" analysis (when API key available)

## Key Modules

### Core Layer (`app/core/`)

| Module | Purpose |
|--------|---------|
| `query_utils.py` | `validate_and_truncate_query()`, `extract_identifier_from_url()`, `resolve_channel_id()` |
| `relevance.py` | `calculate_keyword_relevance()` - keyword match scoring |
| `youtube_api.py` | `search_channels_hybrid()`, `get_channel_stats()`, `get_video_details()` |
| `gemini_api.py` | `generate_ai_relevance_score()`, `generate_summary()`, `generate_outreach_drafts()` |
| `pipeline.py` | `run_search_pipeline()` - main search orchestration |
| `scoring_version.py` | Centralized scoring weights (`SCORING_VERSION`, weight constants, `get_weight_config()`) |
| `seed_topics.py` | `analyze_seed_channel()` - topic extraction from seed channels |

### Cache Layer (`app/cache/`)

| Function | Purpose |
|----------|---------|
| `get_channel_stats_cached()` | Cached channel statistics fetch |
| `get_video_details_cached()` | Cached video details fetch |
| `search_channels_cached()` | Cached search results |
| `search_channels_hybrid_cached()` | Cached hybrid search results |
| `CacheFunctionsAdapter` | Adapter for pipeline integration |

### Application Layer (`app/`)

| Module | Purpose |
|--------|---------|
| `main.py` | Streamlit UI, user interactions, result display |
| `similarity_engine.py` | `calculate_similarity_score()` - multi-factor channel comparison |
| `debug_tracker.py` | `track_api_call()`, quota monitoring, performance timing |
| `feedback_tracker.py` | `save_feedback()`, `get_feedback_stats()`, `export_feedback_csv()`, `get_negative_feedback_entries()` - user feedback persistence |

### Analytics Layer (`app/analytics/`)

| Module | Purpose |
|--------|---------|
| `synthetic_data_generator.py` | Generate synthetic feedback data for ML training |
| `ml_trainer.py` | Train ML models (logistic regression with cross-validation) |
| `weight_optimizer.py` | Optimize scoring weights based on feedback data |
| `fabric_export.py` | Export data to Microsoft Fabric/Power BI formats |

## API Quotas

- **YouTube Data API:** 10,000 units/day (search = 100 units, others = 1 unit)
- **Gemini:** 15 requests/min, 1M tokens/min (free tier)
- **Gemini Model:** `gemini-2.0-flash-lite` (configured in `app/main.py:343`)

## Environment Variables

API keys are loaded via `python-dotenv` with Streamlit Cloud fallback:

| Variable | Required | Purpose |
|----------|----------|---------|
| `YOUTUBE_API_KEY` | Yes | YouTube Data API v3 access |
| `GEMINI_API_KEY` | No | Enables AI features (summaries, semantic scoring, outreach emails, vibe analysis) |

**Loading mechanism** (`app/main.py:250-264` via `_get_secret()`):
1. Check `os.getenv()` first (local `.env` file)
2. Fall back to `st.secrets` for Streamlit Cloud deployment

## Persistence Files

These files are created at runtime and are git-ignored:

| File | Created By | Purpose |
|------|------------|---------|
| `.quota_cache.json` | `debug_tracker.py` | Daily API quota tracking, resets at midnight PT |
| `.feedback_data.json` | `feedback_tracker.py` | User feedback storage for analytics |

**Note:** On Streamlit Cloud, these files are ephemeral and reset on app restarts.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_query_utils.py

# Run with verbose output
pytest tests/ -v
```

### Code Style

- Type hints in function signatures
- Docstrings for public functions
- Dataclasses for structured results

### Git Workflow

- Feature branches (e.g., `feature/new-filter`, `fix/bug-name`)
- Pull request-based merging to main

## Common Development Tasks

| Task | Where to Look |
|------|---------------|
| Add a new search filter | `app/core/pipeline.py` - modify `run_search_pipeline()` |
| Change similarity weights | `app/core/scoring_version.py` - edit weight constants, bump `SCORING_VERSION` |
| Add new YouTube API call | `app/core/youtube_api.py` - add function, update `__init__.py` |
| Add new Gemini feature | `app/core/gemini_api.py` - add function, update `__init__.py` |
| Add caching for new function | `app/cache/cache_layer.py` - add cached wrapper |
| Track new API call type | `app/debug_tracker.py` - add to tracking |
| Add new test | `tests/test_<module>.py` - follow existing patterns |
| Add analytics feature | `app/analytics/` - add module, update `__init__.py` |
| Train/optimize ML model | `app/analytics/ml_trainer.py` or `weight_optimizer.py` |

## Modification Guidelines

Follow these rules to maintain architectural integrity:

1. **Core must remain Streamlit-agnostic** - No `st.*` calls in `app/core/`
2. **UI uses callbacks only** - Progress updates via `on_progress`, `on_api_call` callbacks
3. **Update tests when changing scoring** - Modify `tests/test_relevance.py` or `tests/test_pipeline.py`
4. **Update docs when changing constraints** - If you change max terms, max channels, TTLs, update README + ARCHITECTURE
5. **Export new functions from `__init__.py`** - When adding to `app/core/`, update `app/core/__init__.py`
6. **Bump SCORING_VERSION when changing weights** - Update version in `app/core/scoring_version.py` to invalidate cached scores

## Known Limitations

- **YouTube API quota:** 10K units/day (search = 100 units each)
- **Language support:** Stopwords and month detection only implemented for English and Spanish. Other languages fall back to English stopwords, which may affect topic extraction quality in seed mode.
- **Cache staleness:** 24hr TTL may show outdated data for rapidly changing channels

# CLAUDE.md - CCSeeker Project Guide

## Project Overview

CCSeeker is an AI-powered YouTube creator discovery tool that automates finding niche content creators. It reduces manual search time from hours to minutes through intelligent search and ranking algorithms.

**Repository:** https://github.com/MartinDorado/CCseeker
**License:** Apache 2.0
**Author:** Martin Dorado

## Tech Stack

- **Python 3.10+** - Core language
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
GEMINI_API_KEY=your_key  # Optional

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
│   ├── main.py                   # Streamlit UI and integration (~1467 lines)
│   ├── seed_topics_v2.py         # Seed channel topic extraction
│   ├── similarity_engine.py      # Multi-factor similarity scoring
│   ├── debug_tracker.py          # API usage tracking, quota monitoring
│   ├── feedback_tracker.py       # User feedback collection
│   └── smart_cache.py            # Legacy per-channel video caching
│
├── tests/                        # Unit test suite
│   ├── test_query_utils.py       # 21 tests for query utilities
│   ├── test_relevance.py         # 13 tests for relevance scoring
│   ├── test_youtube_api.py       # YouTube API wrapper tests
│   ├── test_gemini_api.py        # Gemini API wrapper tests
│   └── test_pipeline.py          # Search pipeline tests
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

### Cache Layer (`app/cache/`)

| Function | Purpose |
|----------|---------|
| `get_channel_stats_cached()` | Cached channel statistics fetch |
| `get_video_details_cached()` | Cached video details fetch |
| `search_channels_cached()` | Cached search results |
| `CacheFunctionsAdapter` | Adapter for pipeline integration |

### Application Layer (`app/`)

| Module | Purpose |
|--------|---------|
| `main.py` | Streamlit UI, user interactions, result display |
| `seed_topics_v2.py` | `analyze_seed_channel_v2()` - topic extraction from seed channels |
| `similarity_engine.py` | `calculate_similarity_score()` - multi-factor channel comparison |
| `debug_tracker.py` | `track_api_call()`, quota monitoring, performance timing |
| `feedback_tracker.py` | `save_feedback()` - user feedback collection |

## API Quotas

- **YouTube Data API:** 10,000 units/day (search = 100 units, others = 1 unit)
- **Gemini:** 15 requests/min, 1M tokens/min (free tier)

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
| Change similarity weights | `app/similarity_engine.py` - edit weight constants |
| Add new YouTube API call | `app/core/youtube_api.py` - add function, update `__init__.py` |
| Add new Gemini feature | `app/core/gemini_api.py` - add function, update `__init__.py` |
| Add caching for new function | `app/cache/cache_layer.py` - add cached wrapper |
| Track new API call type | `app/debug_tracker.py` - add to tracking |
| Add new test | `tests/test_<module>.py` - follow existing patterns |

## Known Limitations

- **YouTube API quota:** 10K units/day (search = 100 units each)
- **Language support:** Stopwords and month detection only implemented for English and Spanish. Other languages fall back to English stopwords, which may affect topic extraction quality in seed mode.
- **Cache staleness:** 24hr TTL may show outdated data for rapidly changing channels

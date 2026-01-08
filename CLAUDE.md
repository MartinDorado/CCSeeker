# CLAUDE.md - CCSeeker Project Guide

## Project Overview

CCSeeker is an AI-powered YouTube creator discovery tool that automates finding niche content creators. It reduces manual search time from 4-6 hours to minutes through intelligent search and ranking algorithms.

**Repository:** https://github.com/MartinDorado/CCseeker
**License:** Apache 2.0
**Author:** Martín Dorado

## Tech Stack

- **Python 3.10+** - Core language
- **Streamlit 1.49.0** - Web UI framework
- **YouTube Data API v3** - Channel/video metadata
- **Google Gemini AI** - Topic extraction and content generation
- **Pandas** - Data processing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
YOUTUBE_API_KEY=your_key
GEMINI_API_KEY=your_key  # Optional

# Run the application
streamlit run app/main.py
```

## Project Structure

```
CCSeeker/
├── app/
│   ├── main.py              # Main UI and search orchestration (~2000 lines)
│   ├── seed_topics_v2.py    # Seed channel topic extraction
│   ├── similarity_engine.py # Multi-factor similarity scoring
│   ├── smart_cache.py       # Per-channel video caching (24h TTL)
│   └── debug_tracker.py     # API usage tracking
├── docs/                    # Icons and screenshots
├── .streamlit/config.toml   # Streamlit configuration
├── requirements.txt         # Python dependencies
├── ARCHITECTURE.md          # Detailed technical documentation
└── README.md                # User guide
```

## Key Concepts

### Two Search Modes

1. **Keyword Search** - Hybrid video content + channel name matching
2. **Seed-Based Search** - Extract topics from a seed channel to find similar creators

### Similarity Scoring (100-point scale)

- Tag Overlap: 30%
- Keyword Matching: 30%
- Subscriber Tier: 15%
- Engagement Rate: 17%
- Upload Frequency: 8%

### Important Constants

```python
MAX_SEARCH_TERMS = 2              # Max comma-separated terms
MAX_SEARCH_RESULTS = 50           # Channels per search
MAX_VIDEOS_PER_CHANNEL = 10       # For keyword mode analysis
MAX_VIDEOS_PER_SEED = 50          # For seed profile analysis
MIN_RELEVANCE_SCORE = 0.15        # 15% keyword match required
```

## API Quotas

- **YouTube Data API:** 10,000 units/day (search = 100 units, others = 1 unit)
- **Gemini:** 15 requests/min, 1M tokens/min (free tier)

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `run_search()` | main.py | Main pipeline coordinator |
| `search_channels_hybrid()` | main.py | Two-phase search (video + channel name) |
| `analyze_seed_channel_v2()` | seed_topics_v2.py | Extract topics from seed channel |
| `calculate_similarity_score()` | similarity_engine.py | Multi-factor scoring |
| `get_video_details_cached()` | smart_cache.py | Cached video fetcher |

## Development Notes

### Code Style
- Section comments with line ranges in main.py (see TABLE OF CONTENTS)
- Type hints in function signatures
- Docstrings explain function behavior

### Error Handling
- Graceful degradation on API errors
- Empty lists/dicts returned on failures (not exceptions)
- User-friendly error messages in Streamlit UI

### Testing
- Currently manual testing only
- Future: Unit tests planned for similarity_engine.py, seed_topics_v2.py, smart_cache.py

### Git Workflow
- Feature branches (e.g., `refactor/similarity-engine`)
- Pull request-based merging to main

## Known Limitations

- Monolithic main.py (~2000 lines) - refactoring planned
- YouTube API quota limits (10K units/day)
- Seed analysis optimized for EN/ES content
- Similarity accuracy depends on channels using tags
- 24hr cache TTL may show stale data for rapidly changing channels

## Common Tasks

- **Adding search filters:** Modify filter pipeline in `run_search()`
- **Adjusting similarity weights:** Edit constants in `similarity_engine.py`
- **Changing topic extraction:** Modify penalty system in `seed_topics_v2.py`
- **Adding AI features:** Use Gemini client pattern in `main.py`

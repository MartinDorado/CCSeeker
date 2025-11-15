# CCSeeker 🔍

<div align="center">

![CCSeeker Logo](docs/appicons/app-icon-192x192.png)

**Discover Niche YouTube Creators**

*AI-powered creator discovery tool with intelligent search and similarity ranking*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Tech Stack](#-tech-stack) • [Architecture](ARCHITECTURE.md)

</div>

---

## 🎯 The Problem

Digital marketers spend hours manually searching for niche content creators on YouTube:
- **Time-intensive**: Manual channel discovery takes 4-6 hours per campaign
- **Limited tools**: Existing solutions are expensive ($50-200/month) or lack depth
- **Knowledge gap**: Finding creators when you don't know the exact terminology of the niche is difficult 

## 💡 The Solution

CCSeeker automates niche creator discovery with two intelligent search approaches:

1. **🔑 Keyword Search** - Search by topic using hybrid video + channel name matching
2. **📺 Channel-as-Seed** - Find similar creators by analyzing an example channel's content

The system ranks results by relevance/similarity, tracks API usage to stay within free quotas, and optionally generates AI-powered summaries and outreach emails.

---

## ✨ Features

### 🧠 Dual Search Modes

<details open>
<summary><strong>Keyword Search Mode</strong></summary>

![Keyword Search Interface](docs/screenshots/screenshot_keyword_search.jpg)

- **Multi-term support**: Search with up to 2 comma-separated topics
- **Prioritize region**: Where channels are more relevant
- **Hybrid matching**: Finds channels by video content AND channel names
- **Smart ranking**: Channels with 8 relevant videos (80 pts) outrank those with keyword only in name (5 pts)
- **Visual term counter**: Real-time feedback on query validity

### 🔍 Advanced Filtering

![Filtering Options](docs/screenshots/screenshot_filtering.jpg)

- **Subscriber threshold**: Set minimum audience size
- **Geographic targeting**: Filter by channel country
- **Activity filter**: Only show channels with uploads in last X months
- **Relevance threshold**: Automatically excludes channels with <15% keyword match

### Search Results
Results table shows relevance scores, subscriber counts, engagement rates, and more.

![Search Results](docs/screenshots/screenshot_keywords_display_df.jpg)

### 🤖 AI-Powered Features (Optional)

- **Channel Summaries**: Auto-generated overviews using Google Gemini
![Channel Summaries](docs/screenshots/screenshot_keywords_ai_summary.jpg)
- **Outreach Emails**: Personalized drafts in English or Spanish
![Outreach Emails](docs/screenshots/screenshot_keywords_ai_outreach_emails.jpg)

</details

</details>

<details open>
<summary><strong>Channel-as-Seed Mode</strong></summary>

![Channel-as-Seed Interface](docs/screenshots/screenshot_seed_mode.jpg)

- **Paste any YouTube channel URL** and the system:
  - Analyzes 50 recent videos
  - Detects content language (EN/ES)
  - Calculates engagement patterns and upload frequency
  ![Seed Channel Profile](docs/screenshots/screenshot_seed_channel_profile.jpg) 

- **Topic Extraction**: Identifies niche keywords from video titles and tags
  ![Seed Topic Extraction & AI channel summary](docs/screenshots/screenshot_seed_topic_extraction_and_ai_channel_summary.jpg)
  ![Filtering + Search Options](docs/screenshots/screenshot_seed_filtering_and_search_options.jpg)

- **Optional AI enhancement**: Gemini can analyze top 10 matches for "vibe" similarity
![Seed AI Generated Summary](docs/screenshots/screenshot_seed_ai_generated_summary.jpg)

- **Seed detailed match analysis**: Deep dive into why channels match the seed.
![Seed Detailed Match Analysis](docs/screenshots/screenshot_seed_detailed_match_analysis.jpg)

- **Multi-signal similarity scoring** (100-point scale):
  - Tag overlap (30%) - Jaccard similarity
  - Keyword matching (30%)
  - Subscriber tier (15%) - prevents 10M vs 10K mismatches
  - Engagement rate (17%)
  - Upload frequency (8%)
  ![Seed Similarity Score](docs/screenshots/screenshot_seed_similarity_score.jpg)
</details>

### 📊 Debug & Monitoring System

<details>
<summary><strong>Real-Time API Tracking</strong></summary>

### Debug Panel
*Collapsible sidebar provides transparency into API usage and performance metrics*

Toggle debug mode to see:

![Debug Sidebar](docs/screenshots/screenshot_debug_collapsed.jpg)

**API Call Summary**
![API Calls](docs/screenshots/screenshot_api_summary.jpg)
![API Calls2](docs/screenshots/screenshot_api_summary2.jpg)
- Tracks YouTube (search, channels, videos, playlists)
- Tracks Gemini (summary, outreach, similarity)
- Shows quota units consumed
- Estimates costs

**Performance Timing**
![Performance Timing](docs/screenshots/screenshot_performance_timing.jpg)
- Measures each pipeline stage
- Identifies bottlenecks
- Total execution time

**Quota Efficiency**
![Quota Efficiency](docs/screenshots/screenshot_quota_efficiency.jpg)
- Compares current vs baseline usage
- Tracks total runs for same search
- Shows cache effectiveness

</details>

---

## 🚀 Tech Stack

| Technology | Purpose | Why This Choice |
|------------|---------|-----------------|
| **Python 3.10+** | Core language | Rich ecosystem for data processing |
| **Streamlit** | Web UI framework | Built-in caching, state management, rapid iteration |
| **YouTube Data API v3** | Channel/video data | Official API, ToS-compliant (no scraping) |
| **Google Gemini AI** | Topic extraction & content generation | Free tier (15 RPM, 1M tokens/min) |
| **Pandas** | Data processing | Efficient filtering and transformations |
| **Custom caching layer** | API optimization | Per-channel caching reduces redundant fetches |

### Key Design Decisions

**Per-Channel Caching**
- Traditional approach: Cache entire search results → duplicates videos from popular channels
- CCSeeker approach: Cache each channel's videos independently (24hr TTL)
- Benefit: Popular channels appear in multiple searches → reuse cached data

**Two Search Modes**
- **Keywords**: When you know the niche ("vegan cooking")
- **Seed**: When you have an example, don't know niche's terminology, and just want to find simmilar channels.  

**Filter Before Fetching Videos**
- Apply subscriber/country filters BEFORE analyzing videos
- Saves API quota - no point fetching 10 videos from a 500-sub channel if minimum is 10K

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- [YouTube Data API v3 key](https://console.cloud.google.com/apis/credentials)
- [Google Gemini API key](https://makersuite.google.com/app/apikey) (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/MartinDorado/CCSeeker.git
cd ccseeker

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your keys:
# YOUTUBE_API_KEY=your_youtube_key_here
# GEMINI_API_KEY=your_gemini_key_here  # optional
```

### Run

```bash
streamlit run app/app_seed_gemini_hardened.py
```

App opens at `http://localhost:8501`

---

## 📖 Usage

### Quick Start: Keyword Search

1. Select **🔑 Keywords** mode
2. Enter 1-2 search terms (e.g., "manga, anime")
3. Relevant in: Country (optional) 
3. Set filters:
   - Minimum subscribers (default: 10,000)
   - Channel's origin country (optional)
   - Recent activity (default: 18 months)
4. Click **Find Creators**

Results show:
- Relevance score (% of videos matching keywords)
- Subscriber count
- Average views per video
- Engagement rate
- Country

### Quick Start: Seed-Based Discovery

1. Select **📺 Channel-as-Seed** mode
2. Paste YouTube channel URL
3. Click **Analyze Seed**
4. Review extracted topics and AI summary
5. Optionally edit generated search query
6. Set filters and click **Find Similar Channels**

Results ranked by similarity score (0-100) with match reasons.

### Optional: AI Features

**Generate Summary** (after search completes)
- Scroll to AI Generated Summary section
- Automatically creates overview of top channels

**Create Outreach Emails**
- Select language (English/Español)
- Click **Generate Outreach Drafts**
- Get personalized email templates for TOP 3. 

---

## 🗂️ Project Structure

```
CCSeeker/
├── app/
│   ├── app_seed_gemini_hardened.py  # Main UI & search orchestration
│   ├── seed_topics_v2.py            # Seed channel analysis engine
│   ├── similarity_engine.py         # Multi-factor similarity scoring
│   ├── smart_cache.py               # Per-channel caching layer
│   ├── debug_tracker.py             # API tracking & performance monitoring
│   └── theme_ccseeker_dark.css      # Custom dark theme
├── docs/
|   |── appicons/                    # App icons (192x192, 512x512)
|   |── screenshots/                 # Multiple screenshots from the app
|                    
├── .streamlit/                       # Streamlit config
├── requirements.txt                  # Python dependencies
├── .env.example                      # API key template
├── ARCHITECTURE.md                   # Technical deep dive
└── README.md                         # This file
```

**Key Functions:**

| Function | Purpose | File |
|----------|---------|------|
| `run_search()` | Pipeline coordinator - calls all other functions | app_seed_gemini_hardened.py |
| `search_channels_hybrid()` | YouTube hybrid search (video + channel name) | app_seed_gemini_hardened.py |
| `analyze_seed_channel_v2()` | Extract topics from seed channel | seed_topics_v2.py |
| `calculate_similarity_score()` | Multi-factor similarity algorithm | similarity_engine.py |
| `get_channel_videos()` | Cached video fetcher | smart_cache.py |
| `track_api_call()` | Debug mode API tracking | debug_tracker.py |


---

## 🔧 Configuration

Key constants in `app/app_seed_gemini_hardened.py`:

```python
MAX_SEARCH_TERMS = 2              # Maximum comma-separated terms
MAX_SEARCH_RESULTS = 50           # Channels returned per search
MAX_VIDEOS_PER_CHANNEL = 10       # Videos analyzed for relevance
MAX_VIDEOS_PER_SEED = 50          # Videos analyzed for seed profile
MIN_RELEVANCE_SCORE = 0.15        # 15% keyword match required

# Cache TTLs (in seconds)
CACHE_TTL_CHANNEL_STATS = 604800   # 1 week - used in get_channel_stats_cached()
CACHE_TTL_VIDEO_DETAILS = 259200   # 3 days - used in get_video_details_cached()
CACHE_TTL_SEARCH_RESULTS = 259200  # 3 days - used in search_channels_multi_term_cached()

# Filtering Thresholds
MIN_RELEVANCE_SCORE = 0.15         # 15% keyword match required
DEFAULT_MIN_SUBSCRIBERS = 10000
DEFAULT_MONTHS_RECENT = 18

```

**Similarity Weights** (in `similarity_engine.py`):
```python
# Total: 100 points
tag_overlap: 30        # Video tags Jaccard similarity
keyword_match: 30      # Title/description keyword presence
subscriber_tier: 15    # Subscriber count ratio
engagement_rate: 17    # (Likes + comments) / views
upload_frequency: 8    # Videos per month comparison
```

---

## 📊 API Quotas & Costs

### YouTube Data API v3 (Free Tier)
- **Daily Quota**: 10,000 units
- **Cost per operation**:
  - Search: 100 units
  - Channels: 1 unit
  - Videos: 1 unit
  - Playlists: 1 unit

**Typical search cost**: Varies based on:
- Number of search terms (1-2)
- Channels found (capped at 50)
- Cache hits (reduces repeat fetches)

Enable debug mode to see real-time usage.

### Google Gemini API (Free Tier)
- **Rate Limits**: 15 requests/min, 1M tokens/min
- **Cost**: Free
- **Paid tier**: ~$0.10-0.30 per 1M tokens (if needed)

---

## 🔒 Security & Best Practices

- ✅ API keys stored in `.env` (git-ignored)
- ✅ Graceful error handling for API failures
- ✅ Input validation (query truncation, URL parsing)
- ✅ Rate limiting awareness via debug panel
- ✅ No web scraping (ToS-compliant API usage)

---

## 🚧 Known Limitations

- **YouTube API Quota**: 10K units/day limits search volume
- **Seed Analysis Language**: Best results with English/Spanish content
- **Similarity Accuracy**: Depends on channels actually using tags
- **No Historical Data**: Can't analyze deleted videos or past performance
- **Cache Invalidation**: 24hr TTL may show stale data for rapidly changing channels

---

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Martín Dorado**
- Portfolio: [yourwebsite.com](https://yourwebsite.com)
- LinkedIn: [linkedin.com/in/martin-dorado](https://www.linkedin.com/in/martin-dorado/)
- GitHub: [@MartinDorado](https://github.com/MartinDorado)

---


## 📚 Additional Resources

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep dive into system design and algorithms
- **[YouTube API Docs](https://developers.google.com/youtube/v3)** - Official API reference
- **[Gemini API Guide](https://ai.google.dev/docs)** - AI integration documentation

---

<div align="center">

</div>

# CCSeeker Architecture Documentation

## System Overview

CCSeeker is a YouTube creator discovery tool that solves a specific problem: finding niche content creators when you either know the keywords (traditional search) or have an example channel but don't know what to search for (seed-based discovery).

**Core Design Philosophy:**
- Minimize API calls through intelligent caching
- Provide two distinct search modes for different user scenarios
- Make internal workings transparent through debug tooling
- Fail gracefully when APIs are unavailable or quota is exhausted

**Tech Stack:**
- **Streamlit**: UI framework with built-in state management and caching
- **YouTube Data API v3**: Channel and video metadata retrieval
- **Google Gemini AI**: Optional topic extraction and content generation
- **Pandas**: Data transformation and filtering
- **Custom caching layer**: Per-channel video storage

---

## Architecture: Top-Down View

```
User Input (Keyword OR Seed URL)
         ↓
   Search Mode Router
         ↓
    ┌────┴────┐
    ↓         ↓
Keyword    Seed Analysis
Search     (extract topics)
    ↓         ↓
    └────┬────┘
         ↓
   YouTube API Search
         ↓
   Filter & Rank Pipeline
         ↓
   Display Results + Optional AI Features
```
## Design Evolution

**Initial Constraints:**
- First production application 
- Uncertain scope during development (started as keyword search, evolved to include AI-powered seed analysis)
- Rapid as possible prototyping with AI assistance to validate product-market fit.


**Architectural Decision: Monolithic Structure**

The main application file (~2000 lines) consolidates UI, business logic, and orchestration in a single module. This was a deliberate choice to:

1. **Maintain coherent data flow** - Streamlit's reactive model makes state management across modules complex
2. **Minimize integration points** - Fewer files = fewer places for API quota tracking to break
3. **Enable rapid iteration** - Changes to search logic often require UI adjustments; co-location speeds this up

**Trade-offs Accepted:**
- Harder for external contributors to navigate (mitigated by section comments and docstrings)
- Some functions are far from ideal length 
- Testing requires more setup due to tight coupling

## 🔧 Post-Launch Refactoring Roadmap

This project started life as an MVP, optimized for getting a working tool into my hands
as fast as possible. As a result, a lot of logic currently lives inside the Streamlit app
(`main.py`), and most testing has been manual.

The refactoring plan is intentionally incremental and test-first:

1. **Stabilize current behaviour**
   - Document a manual QA checklist for both search modes (keywords + seed).
   - Add a few tiny smoke tests for pure functions (e.g. similarity scoring).

2. **Add unit tests for core logic**
   - Cover `similarity_engine.py` (Jaccard similarity, scoring breakdown).
   - Cover parts of `seed_topics_v2.py` (topic extraction, penalty system).
   - Add small tests for cache-key generation and tag handling in `smart_cache.py`.

3. **Split main.py into smaller modules (no behaviour changes)**
   - `config.py` – constants and configuration notes.
   - `youtube_client.py` / `gemini_client.py` – external API clients and helpers.
   - `search_pipeline.py` – `run_search` and related search/relevance logic.
   - `ui_seed.py`, `ui_results.py` – seed profile panel and results/analysis UI.

4. **Gradually increase test coverage**
   - Add tests around the search pipeline with mocked YouTube/Gemini clients.
   - Add regression tests for edge cases (no results, quota errors, missing data).

---

## Mode 1: Keyword Search

**Entry Point:** `run_search()` in `main.py`

### Flow

**1. Query Validation & Multi-Term Handling**

```python
def search_channels_multi_term(query: str, region_code: str, 
                               max_videos_per_term: int = 100):
    """
    Handle comma-separated queries with OR logic.
    Example: "manga, anime" → search both, merge results
    """
    terms = [t.strip() for t in query.split(',') if t.strip()]
    
    # Safety: enforce 2-term maximum
    if len(terms) > 2:
        terms = terms[:2]  # Auto-truncate
```

User enters "manga, anime, gaming" → System auto-truncates to "manga, anime" and warns user.

**2. Hybrid Search Strategy**

```python
def search_channels_hybrid(query: str, region_code: str):
    """
    Two-phase search:
    A) Primary: Find channels by VIDEO content
    B) Secondary: Find channels by NAME
    C) Combine and rank by match score
    """
```

**Phase A - Video Content Search:**
```python
# Search for videos matching query
video_response = youtube.search().list(
    q=query,
    type='video',
    maxResults=50, #Is 50 for being able to fetch a nextPageToken. MAX_VIDEOS_PER_TERM = 100
    order='relevance'
).execute()

# Extract channels from videos
for item in video_response['items']:
    channel_id = item['snippet']['channelId']
    all_channels[channel_id]['video_matches'] += 1
```

**Phase B - Channel Name Search:**
```python
# Search for channels by name
channel_response = youtube.search().list(
    q=query,
    type='channel',
    maxResults=50
).execute()

# Mark channels found by name
for item in channel_response['items']:
    channel_id = item['id']['channelId']
    all_channels[channel_id]['name_match'] = True
```

**Phase C - Scoring & Sorting:**
```python
# Calculate match scores
for channel_id, data in all_channels.items():
    match_score = data['video_matches'] * 10  # Video matches = 10 pts each
    if data['name_match']:
        match_score += 5  # Name match bonus = 5 pts
    
    ranked_channels.append({
        'channel_id': channel_id,
        'match_score': match_score
    })

# Sort by relevance (match_score descending)
ranked_channels.sort(key=lambda x: x['match_score'], reverse=True)
```

**Why this approach?** A channel with 8 relevant videos (80 pts) outranks one that merely has the keyword in its name (5 pts).

**3. Channel Statistics Fetch**

```python
def get_channel_stats(youtube_service, channel_ids):
    """
    Fetch comprehensive metadata for channels.
    API call: channels.list() - costs 1 unit per 50 channels
    """
    for item in response['items']:
        # Calculate derived metrics
        videos_count = int(statistics.get("videoCount", 0))
        total_views = int(statistics.get("viewCount", 0))
        avg_views = total_views / videos_count if videos_count > 0 else 0
        
        # Calculate channel age
        published_at = snippet.get("publishedAt")
        channel_age_days = (datetime.now() - parse(published_at)).days
        
        stats_data.append({
            "channel_id": item["id"],
            "country": snippet.get("country", "N/A"),
            "subscribers": int(statistics.get("subscriberCount", 0)),
            "views": total_views,
            "videos": videos_count,
            "uploads_playlist_id": uploads_id,
            "avg_views_per_video": avg_views,
            "channel_age_days": channel_age_days
        })
```

**4. Filter Pipeline (BEFORE video fetching)**

```python
# Step 3: Apply filters FIRST (saves API quota)
filtered_channels = enriched_channel_data[
    enriched_channel_data['subscribers'] >= min_subs_input
].copy()

# Optional: strict country filter
if country_filter_input:
    filtered_channels = filtered_channels[
        filtered_channels['country'] == country_filter_input.upper()
    ]
```

**Design decision:** Filter by subscribers/country BEFORE fetching videos. No point analyzing a 500-subscriber channel if minimum is 10K.

**5. Quality Selection & Deep Analysis**

```python
# Step 4: Cap at 50 channels, require minimum relevance
MIN_MATCH_SCORE = 10
quality_channels = filtered_channels[
    filtered_channels['match_score'] >= MIN_MATCH_SCORE
].head(50)

# Step 5: Fetch 10 videos per channel for relevance analysis
video_data = get_video_details_cached(
    channel_ids_tuple, 
    max_videos=10  # Deep analysis depth
)
```

**6. Relevance Scoring**

```python
def calculate_keyword_relevance(df, query):
    """
    Compute % of channel's videos that mention query terms.
    
    Returns: relevance_score (0.0 to 1.0)
    """
    # Match query terms against video titles + tags
    pattern = '(?:' + '|'.join(escaped_terms) + ')'
    df['is_relevant'] = df['combined_text'].str.contains(
        pattern, case=False
    )
    
    # Average across channel's videos
    relevance = df.groupby('channel_id')['is_relevant'].mean()
```

Example: Channel with 7/10 recent videos about "manga" → relevance_score = 0.70 (70%)

---

## Mode 2: Seed-Based Search

**Entry Point:** User pastes channel URL → `analyze_seed_channel_v2()` → similarity ranking

### Seed Analysis Pipeline

**1. Extract Seed Profile**

```python
def analyze_seed_channel_v2(youtube_service, channel_id, 
                            max_videos=50, user_banned_words=None):
    """
    Build comprehensive profile from seed channel:
    - Primary keywords (multi-word phrases from titles)
    - Secondary keywords (single words)
    - Common tags (from video metadata)
    - Engagement rate, upload frequency, subscriber tier
    """
```

**Step A - Language Detection:**
```python
def detect_language(texts: list[str]) -> str:
    """Count EN vs ES stopword hits"""
    return 'es' if es_hits > en_hits else 'en'

stopwords = get_stopwords(detected_language)
```

**Step B - Topic Extraction:**
```python
# Extract bigrams (2-word phrases)
for video in videos:
    tokens = tokenize(title, stopwords)  # Remove stopwords
    bigrams = extract_bigrams(tokens)    # "healthy" + "vegan" → "healthy vegan"

# Count document frequency (how many videos mention each term)
min_doc_freq = ceil(0.20 * n_videos)  # Must appear in 20% of videos

# Score with penalty system
for term, doc_freq in tag_docs.items():
    penalty = calculate_term_penalty(term, user_banned_words)
    score = doc_freq * 2.0 * (1.0 - penalty)  # Tags = 2x weight
```

**Penalty System** (replaces hard blocking):
```python
def calculate_term_penalty(term: str, user_banned: set) -> float:
    """
    Returns 0.0 (perfect) to 1.0 (remove)
    
    - User-banned words: +0.9
    - Years (2024): +0.5
    - Numbers (ep5): +0.3
    - Months (January): +0.4
    - Promo words (subscribe): +0.3
    """
```

**Output:**
```python
seed_profile = {
    'channel_name': str,
    'subscriber_count': int,
    'language': 'en' | 'es',
    'upload_frequency': float,  # videos/month
    'avg_engagement_rate': float,
    
    'primary_keywords': [str],    # Top 5 phrases
    'secondary_keywords': [str],  # Top 10 words
    'common_tags': [str],         # Top 15 tags
    'recent_titles': [str]
}
```

**2. Similarity Scoring**

```python
def calculate_similarity_score(candidate: dict, 
                               seed_profile: dict) -> dict:
    """
    Multi-factor scoring (100 points total)
    
    Returns: {
        'total_score': float,
        'match_reasons': [str],
        'breakdown': dict  # if debug=True
    }
    """
```

**The Algorithm:**

```python
# Factor 1: Tag Overlap (30 points)
candidate_tags = set(candidate['tags'])
seed_tags = set(seed_profile['common_tags'])
tag_overlap = jaccard_similarity(candidate_tags, seed_tags)
score += tag_overlap * 30

# Factor 2: Keyword Match (30 points)
candidate_kw = set(candidate['keywords'])
seed_kw = set(seed_profile['primary_keywords'] + 
              seed_profile['secondary_keywords'])
keyword_overlap = jaccard_similarity(candidate_kw, seed_kw)
score += keyword_overlap * 30

# Factor 3: Subscriber Similarity (15 points)
ratio = min(candidate_subs / seed_subs, seed_subs / candidate_subs)
score += ratio * 15

# Factor 4: Engagement Rate (17 points)
engagement_diff = abs(candidate_engagement - seed_engagement)
score += max(0, 17 - (engagement_diff * 170))

# Factor 5: Upload Frequency (8 points)
freq_ratio = min(candidate_freq / seed_freq, seed_freq / candidate_freq)
score += freq_ratio * 8
```

**Jaccard Similarity:**
```python
def jaccard_similarity(set1: set, set2: set) -> float:
    """
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Example:
    A = {manga, anime, review}
    B = {manga, anime, shonen}
    Intersection = 2, Union = 4
    Score = 0.5
    """
    return len(set1 & set2) / len(set1 | set2)
```

**Why these weights?**
- **Tags (30%)**: Most reliable - creators consciously choose these
- **Keywords (30%)**: Content focus indicator from titles
- **Subscriber tier (15%)**: Prevents 10M channel matching with 10K
- **Engagement (17%)**: Audience quality matters
- **Upload frequency (8%)**: Nice-to-have, not critical

---

## Caching Architecture

**Problem:** Popular channels appear in multiple searches. Traditional per-search caching duplicates video fetches.

**Solution:** Per-channel caching with 24-hour TTL

```python
class ChannelVideoCache:
    @staticmethod
    @st.cache_data(ttl=86400)  # 24 hours
    def get_channel_videos(channel_id: str, 
                          uploads_playlist_id: str,
                          max_videos: int,
                          _youtube_service):
        """
        Cache individual channel's videos independently.
        
        Key insight: Channel content doesn't change hourly,
        but search queries do.
        """
```

**How it works:**

```
Search 1: "manga" → Finds channels A, B, C
  ├─ Fetch A's videos → Cache[A] = [...videos]
  ├─ Fetch B's videos → Cache[B] = [...videos]
  └─ Fetch C's videos → Cache[C] = [...videos]

Search 2: "anime" → Finds channels B, C, D
  ├─ Cache HIT for B → 2 API calls saved
  ├─ Cache HIT for C → 2 API calls saved
  └─ Fetch D's videos → Cache[D] = [...videos]
```

**Cache key strategy:**
```python
def _make_cache_key(channel_id: str, max_videos: int) -> str:
    return f"ch_vids_{channel_id}_{max_videos}"
```

Why include `max_videos`? Seed analysis uses 50 videos, keyword search uses 10. Different cache entries.

---

## Debug & Monitoring System

**Purpose:** Make API usage visible to help users stay within quotas.

### API Call Tracking

```python
def track_api_call(api_name: str):
    """
    Increment counter for specific API.
    
    Tracked operations:
    - youtube_search
    - youtube_channel
    - youtube_video
    - youtube_playlist
    - gemini_summary
    - gemini_outreach
    - gemini_similarity
    """
    st.session_state.debug_data[f'{api_name}_calls'] += 1
```

**Called from:**
```python
# In search function
response = youtube.search().list(...).execute()
debug_tracker.track_api_call('youtube_search')  # Log the call

# In seed analysis
response = model.generate_content(prompt)
debug_tracker.track_api_call('gemini_summary')
```

### Quota Calculation

```python
def calculate_youtube_quota_used():
    """
    YouTube API quota costs:
    - search: 100 units
    - channels: 1 unit
    - videos: 1 unit
    - playlistItems: 1 unit
    
    Daily limit: 10,000 units (free tier)
    """
    total = 0
    total += data['youtube_search_calls'] * 100
    total += data['youtube_channel_calls'] * 1
    total += data['youtube_video_calls'] * 1
    total += data['youtube_playlist_calls'] * 1
    return total
```

### Performance Timing

```python
@time_operation('search')
def run_search(...):
    # Function executes
    # Timing recorded automatically
```

Tracks:
- Search time
- Channel stats fetch
- Video details fetch
- Similarity calculation
- AI generation
- **Total time**

---

## Data Flow: Complete Picture

```
1. USER INPUT
   Keywords: "manga, anime" OR Seed URL: youtube.com/@channel

2. SEARCH PHASE
   ├─ Keyword → Hybrid search (video + name)
   └─ Seed → Extract profile → Generate keywords
   
3. API CALLS (with caching)
   ├─ Search API (100 units each)
   ├─ Channels API (1 unit per 50)
   └─ Videos API (1 unit per channel) ← CACHED

4. FILTER PIPELINE
   ├─ Subscriber threshold
   ├─ Country filter
   ├─ Match score minimum (10)
   └─ Cap at 50 channels

5. DEEP ANALYSIS
   ├─ Fetch 10 videos per channel
   ├─ Calculate relevance score
   ├─ Calculate engagement rate
   └─ [Optional] Similarity ranking

6. AI ENHANCEMENT (optional)
   ├─ Generate summary (Gemini)
   └─ Create outreach emails (Gemini)

7. DISPLAY
   ├─ Results table
   ├─ Match explanations
   └─ Debug metrics (if enabled)
```

---

## Key Design Decisions

**1. Two Search Modes**
- **Why:** Different user needs. Sometimes you know the niche ("vegan cooking"), sometimes you have an example channel but don't know what terms to search.

**2. Hybrid Search (video + name)**
- **Why:** Channel names are often vague ("JohnSmith"). Video content is more descriptive.

**3. Filter Before Fetching Videos**
- **Why:** Saves API quota. No point analyzing channels that don't meet minimum criteria.

**4. Per-Channel Caching**
- **Why:** Popular channels appear in multiple searches. Cache their videos once, reuse across searches.

**5. Conservative Caps**
- **Why:** Free API tier has limits. Cap at 50 channels, 10 videos each = predictable quota usage.

**6. Soft Penalties vs Hard Blocks**
- **Why:** "2024" might be noise in titles, but "2024 trends" could be relevant. Penalties allow context.

---

## Technology Choices

**Streamlit:**
- Built-in caching (`@st.cache_data`)
- Session state management
- Rapid UI iteration
- Tradeoff: Limited control over page layout

**YouTube API:**
- Direct access to metadata
- No scraping (ToS compliant)
- Tradeoff: Quota limits (10K units/day free)

**Gemini AI:**
- Free tier (15 RPM, 1M tokens/min)
- Quality topic extraction
- Tradeoff: Rate limits, requires API key

**Pandas:**
- Fast filtering/sorting
- Natural for tabular results
- Tradeoff: Memory usage for large datasets

---

## File Structure

```
app/
├── app_seed_gemini_hardened.py  # Main UI & search orchestration
├── seed_topics_v2.py            # Seed channel analysis
├── similarity_engine.py         # Similarity scoring algorithms
├── smart_cache.py               # Per-channel caching layer
└── debug_tracker.py             # Monitoring & quota tracking
```

**Function responsibility:**
- `run_search()`: Pipeline coordinator
- `search_channels_hybrid()`: YouTube search logic
- `analyze_seed_channel_v2()`: Topic extraction
- `calculate_similarity_score()`: Ranking algorithm
- `get_channel_videos()`: Cached video fetcher

---

**End of Document**

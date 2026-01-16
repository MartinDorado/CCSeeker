"""
seed_topics_v2.py - Redesigned seed channel analyzer
Clean, accurate, no translation bloat
"""

import re
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Optional
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from . import debug_tracker
except ImportError:
    import debug_tracker

# ============================================================================
# LANGUAGE DETECTION & STOPWORDS
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


def detect_language(texts: list[str]) -> str:
    """
    Simple language detection: count EN vs ES stopword hits
    
    Returns: 'es' or 'en'
    """
    en_hits = 0
    es_hits = 0
    
    for text in texts:
        words = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())
        for word in words:
            if word in STOPWORDS_EN:
                en_hits += 1
            if word in STOPWORDS_ES:
                es_hits += 1
    
    return 'es' if es_hits > en_hits else 'en'


def get_stopwords(language: str) -> set:
    """Get stopwords for detected language"""
    if language == 'es':
        return STOPWORDS_ES | STOPWORDS_COMMON
    return STOPWORDS_EN | STOPWORDS_COMMON


# ============================================================================
# TOKENIZATION & CLEANING
# ============================================================================

def tokenize(text: str, stopwords: set, min_length: int = 3) -> list[str]:
    """
    Extract clean tokens from text
    
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
    Create meaningful two-word phrases
    
    Example: ['healthy', 'vegan', 'recipes'] → ['healthy vegan', 'vegan recipes']
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
# PENALTY SYSTEM (replaces looks_contextual)
# ============================================================================

def calculate_term_penalty(term: str) -> float:
    """
    Calculate penalty score for a term (0.0 = perfect, 1.0 = remove)

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
        penalty += 0.5
    elif any(char.isdigit() for t in tokens for char in t):  # "ep5"
        penalty += 0.3
    
    # Months (medium penalty)
    if tokens & MONTHS_EN or tokens & MONTHS_ES:
        penalty += 0.4
    
    # Promotional language (light penalty)
    if tokens & PROMO_WORDS:
        penalty += 0.3
    
    # Event-specific terms (medium penalty)
    if tokens & EVENT_WORDS:
        penalty += 0.5
    
    return min(penalty, 1.0)  # Cap at 1.0


# ============================================================================
# SEED CHANNEL ANALYSIS - CORE FUNCTION
# ============================================================================

def analyze_seed_channel_v2(
    youtube_service,
    channel_id: str,
    max_videos: int = 50,
    gemini_api_key: Optional[str] = None
) -> dict:
    """
    Extract comprehensive profile from seed channel
    
    Returns:
    {
        'channel_id': str,
        'channel_name': str,
        'subscriber_count': int,
        'subscriber_tier': str,
        'category': str,
        'language': str,
        'upload_frequency': float,  # videos per month
        'avg_engagement_rate': float,
        
        'primary_keywords': list[str],  # Top 5 phrases
        'secondary_keywords': list[str],  # Top 10 words
        'common_tags': list[str],  # Top 15 tags
        
        'recent_titles': list[str],
        'description_summary': str
    }
    """

    # Use st.status for consolidated progress display
    status = st.status("🔍 Analyzing seed channel...", expanded=True)

    # ========================================================================
    # STEP 1: Get channel metadata
    # ========================================================================

    try:
        channel_response = youtube_service.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()

        debug_tracker.track_api_call('youtube_channel')

        if not channel_response.get('items'):
            status.update(label="❌ Channel not found", state="error")
            st.error("Channel not found. The channel may be private, deleted, or the URL format may be incorrect.")
            return None
        
        channel = channel_response['items'][0]
        snippet = channel['snippet']
        stats = channel['statistics']
        content_details = channel['contentDetails']
        
    except Exception as e:
        status.update(label="❌ Failed to fetch channel", state="error")
        st.error(f"Failed to fetch channel: {e}")
        return None
    
    # Extract basic info
    channel_name = snippet.get('title', 'Unknown')
    subscriber_count = int(stats.get('subscriberCount', 0))
    total_views = int(stats.get('viewCount', 0))
    video_count = int(stats.get('videoCount', 0))
    uploads_playlist = content_details['relatedPlaylists']['uploads']
    
    # Channel description
    channel_description = snippet.get('description', '')
    
    # Build channel name stopwords (don't extract brand name as topic)
    name_tokens = set(re.findall(r'\b[a-záéíóúñü]+\b', channel_name.lower()))
    name_tokens |= {"oficial", "official", "canal", "channel"}

    status.write(f"📺 **{channel_name}** ({subscriber_count:,} subscribers, {video_count} videos)")
    
    # ========================================================================
    # STEP 2: Get recent videos (titles, tags, descriptions)
    # ========================================================================
    
    try:
        playlist_response = youtube_service.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist,
            maxResults=min(50, max_videos)
        ).execute()
        
        debug_tracker.track_api_call('youtube_playlist')

        video_ids = [
            item['snippet']['resourceId']['videoId']
            for item in playlist_response.get('items', [])
        ]
        
    except Exception as e:
        status.update(label="❌ Failed to fetch videos", state="error")
        st.error(f"Failed to fetch videos: {e}")
        return None

    if not video_ids:
        status.update(label="❌ No videos found", state="error")
        st.error("No videos found in channel. This channel may be empty or have all videos set to private.")
        return None
    
    # Fetch detailed video info (to get tags, descriptions, stats)
    try:
        videos_response = youtube_service.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        ).execute()

        debug_tracker.track_api_call('youtube_video')
        
        videos = videos_response.get('items', [])
        
    except Exception as e:
        status.update(label="❌ Failed to fetch video details", state="error")
        st.error(f"Failed to fetch video details: {e}")
        return None
    
    # ========================================================================
    # STEP 3: Calculate engagement metrics
    # ========================================================================
    
    engagement_rates = []
    publish_dates = []
    
    for video in videos:
        video_stats = video.get('statistics', {})
        views = int(video_stats.get('viewCount', 0))
        likes = int(video_stats.get('likeCount', 0))
        comments = int(video_stats.get('commentCount', 0))
        
        if views > 0:
            engagement = (likes + comments) / views
            engagement_rates.append(engagement)
        
        # Track publish dates for frequency calculation
        published = video['snippet'].get('publishedAt')
        if published:
            publish_dates.append(published)
    
    avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0.0
    
    # Calculate upload frequency (videos per month)
    upload_frequency = 0.0
    if len(publish_dates) >= 2:
        try:
            from dateutil import parser
            dates = [parser.parse(d) for d in publish_dates]
            dates.sort()
            time_span_days = (dates[-1] - dates[0]).total_seconds() / 86400
            if time_span_days <= 0:
                time_span_days = 1 / 24  # fallback: treat as at least an hour
            upload_frequency = (len(dates) / time_span_days) * 30

        except:
            pass
    
    # ========================================================================
    # STEP 4: Language detection
    # ========================================================================
    
    sample_titles = [v['snippet']['title'] for v in videos]
    detected_language = detect_language(sample_titles)
    stopwords = get_stopwords(detected_language)

    status.write(f"🌍 Language detected: **{detected_language.upper()}**")
    status.write(f"🎬 Analyzing {len(videos)} videos...")
    
    # ========================================================================
    # STEP 5: Topic extraction from titles
    # ========================================================================
    
    all_title_tokens = []
    all_title_bigrams = []
    
    for video in videos:
        title = video['snippet']['title']
        
        # Extract tokens (excluding channel name)
        tokens = [
            t for t in tokenize(title, stopwords)
            if t not in name_tokens
        ]
        
        all_title_tokens.extend(tokens)
        
        # Extract bigrams
        bigrams = extract_bigrams(tokens, stopwords)
        all_title_bigrams.extend(bigrams)
    
    # ========================================================================
    # STEP 6: Tag extraction
    # ========================================================================
    
    all_tags = []
    
    for video in videos:
        tags = video['snippet'].get('tags', [])
        # Clean tags (lowercase, remove channel name)
        cleaned_tags = [
            tag.lower().strip()
            for tag in tags
            if tag and not any(nt in tag.lower() for nt in name_tokens)
        ]
        all_tags.extend(cleaned_tags)
    
    # ========================================================================
    # STEP 7: Description keyword extraction (optional, light weight)
    # ========================================================================
    
    all_desc_tokens = []
    
    for video in videos[:10]:  # Only first 10 videos (descriptions are noisy)
        desc = video['snippet'].get('description', '')
        # Only first 200 chars of description
        desc_snippet = desc[:200]
        
        tokens = [
            t for t in tokenize(desc_snippet, stopwords)
            if t not in name_tokens
        ]
        all_desc_tokens.extend(tokens)
    
    # ========================================================================
    # STEP 8: Score and rank terms
    # ========================================================================
    
    # Count document frequency (how many videos mention each term)
    n_videos = len(videos)
    min_doc_freq = max(2, math.ceil(0.20 * n_videos))  # Must appear in 20% of videos
    
    # Count occurrences per video (for document frequency)
    title_unigram_docs = Counter()
    title_bigram_docs = Counter()
    tag_docs = Counter()
    desc_docs = Counter()
    
    for video in videos:
        # Title unigrams
        title = video['snippet']['title']
        title_tokens = set(t for t in tokenize(title, stopwords) if t not in name_tokens)
        title_unigram_docs.update(title_tokens)
        
        # Title bigrams
        bigrams = set(extract_bigrams(list(title_tokens), stopwords))
        title_bigram_docs.update(bigrams)
        
        # Tags
        tags = video['snippet'].get('tags', [])
        clean_tags = set(
            tag.lower().strip()
            for tag in tags
            if tag and not any(nt in tag.lower() for nt in name_tokens)
        )
        tag_docs.update(clean_tags)
    
    # For descriptions (less weight)
    for video in videos[:10]:
        desc = video['snippet'].get('description', '')[:200]
        desc_tokens = set(t for t in tokenize(desc, stopwords) if t not in name_tokens)
        desc_docs.update(desc_tokens)
    
    # ========================================================================
    # STEP 9: Apply scoring with penalties
    # ========================================================================
    
    scored_terms = []
    
    # Tags (highest weight - most accurate signal)
    for term, doc_freq in tag_docs.items():
        if doc_freq >= min_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * 2.0 * (1.0 - penalty)  # Tags worth 2x
            if score > 0:
                scored_terms.append((term, score, 'tag'))
    
    # Title bigrams (high weight - specific topics)
    for term, doc_freq in title_bigram_docs.items():
        if doc_freq >= min_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * 1.6 * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'bigram'))
    
    # Title unigrams (medium weight)
    for term, doc_freq in title_unigram_docs.items():
        if doc_freq >= min_doc_freq:
            penalty = calculate_term_penalty(term)
            score = doc_freq * 1.0 * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'unigram'))
    
    # Description tokens (low weight - noisy)
    for term, doc_freq in desc_docs.items():
        if doc_freq >= max(2, math.ceil(0.15 * 10)):  # Lower threshold for descriptions
            penalty = calculate_term_penalty(term)
            score = doc_freq * 0.5 * (1.0 - penalty)
            if score > 0:
                scored_terms.append((term, score, 'description'))
    
    # Sort by score
    scored_terms.sort(key=lambda x: x[1], reverse=True)
    
    # ========================================================================
    # STEP 10: Select best terms
    # ========================================================================
    
    # Prefer phrases over single words
    primary_keywords = []  # Multi-word phrases
    secondary_keywords = []  # Single words
    
    seen = set()
    
    for term, score, source in scored_terms:
        if term in seen:
            continue
        
        seen.add(term)
        
        if ' ' in term:  # Multi-word phrase
            primary_keywords.append(term)
        else:
            secondary_keywords.append(term)
        
        if len(primary_keywords) >= 5 and len(secondary_keywords) >= 10:
            break
    
    # Top tags (separate list)
    common_tags = [tag for tag, _ in tag_docs.most_common(15)]
    
    # ========================================================================
    # STEP 11: Gemini refinement (optional)
    # ========================================================================
    
    description_summary = ""
    
    if gemini_api_key and genai:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            
            # Summarize channel description
            summary_prompt = f"""
Analyze this YouTube channel in 2-3 sentences:

Channel: {channel_name}
Description: {channel_description[:500]}
Recent video titles: {', '.join(sample_titles[:5])}

What is this channel's main content focus?
Answer in the same language as the channel ({detected_language}).
"""
            
            response = model.generate_content(summary_prompt)
            description_summary = response.text.strip()

            status.write("✨ AI-enhanced analysis complete")

        except Exception as e:
            status.write(f"⚠️ AI analysis skipped: {e}")
    
    # ========================================================================
    # STEP 12: Calculate subscriber tier
    # ========================================================================
    
    if subscriber_count < 10_000:
        tier = "nano"
    elif subscriber_count < 100_000:
        tier = "micro"
    elif subscriber_count < 1_000_000:
        tier = "mid"
    elif subscriber_count < 10_000_000:
        tier = "macro"
    else:
        tier = "mega"
    
    # ========================================================================
    # FINAL: Return complete profile
    # ========================================================================
    
    profile = {
        'channel_id': channel_id,
        'channel_name': channel_name,
        'subscriber_count': subscriber_count,
        'subscriber_tier': tier,
        'category': snippet.get('categoryId', 'Unknown'),  # Note: categoryId is numeric
        'language': detected_language,
        'upload_frequency': round(upload_frequency, 2),
        'avg_engagement_rate': round(avg_engagement, 4),
        
        'primary_keywords': primary_keywords[:5],
        'secondary_keywords': secondary_keywords[:10],
        'common_tags': common_tags[:15],
        
        'recent_titles': sample_titles[:20],
        'description_summary': description_summary
    }

    # Update status to complete state
    status.update(
        label=f"✅ Analysis complete: {len(primary_keywords)} phrases, {len(common_tags)} tags, {len(secondary_keywords)} keywords",
        state="complete",
        expanded=False
    )

    return profile

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import re
from googleapiclient.errors import HttpError
from collections import Counter

# --- Configuration ---
API_KEY = "REDACTED_YOUTUBE_KEY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# --- Helper & API Functions ---

def extract_channel_id_from_url(url):
    """Extracts the channel ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com/channel/)([^/?&]+)',
        r'(?:youtube\.com/@)([^/?&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def Youtube(youtube_service, query, region_code, total_results_to_fetch):
    # ... (This function remains the same)
    channels = []
    next_page_token = None
    search_params = {'q': query, 'part': 'snippet', 'type': 'channel'}
    if region_code:
        search_params['regionCode'] = region_code
    while len(channels) < total_results_to_fetch:
        search_params['maxResults'] = min(50, total_results_to_fetch - len(channels))
        search_params['pageToken'] = next_page_token
        request = youtube_service.search().list(**search_params)
        response = request.execute()
        for item in response.get("items", []):
            channels.append({"channel_id": item["id"]["channelId"], "channel_title": item["snippet"]["title"]})
        next_page_token = response.get('nextPageToken')
        if not next_page_token: break
    return channels

def get_channel_stats(youtube_service, channel_ids):
    # ... (This function remains the same)
    stats_data = []
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i + 50]
        request = youtube_service.channels().list(part="snippet,statistics,contentDetails", id=",".join(chunk))
        response = request.execute()
        for item in response.get("items", []):
            content_details = item.get('contentDetails', {})
            related_playlists = content_details.get('relatedPlaylists', {})
            uploads_id = related_playlists.get('uploads')
            if uploads_id:
                stats_data.append({
                    "channel_id": item["id"], "country": item["snippet"].get("country", "N/A"),
                    "subscribers": int(item["statistics"].get("subscriberCount", 0)),
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "videos": int(item["statistics"].get("videoCount", 0)),
                    "uploads_playlist_id": uploads_id
                })
    return stats_data

def get_video_details(youtube_service, channel_data, max_videos_per_channel):
    # ... (This function is now also used by the seed analysis)
    all_video_details = []
    for channel in channel_data:
        playlist_id = channel["uploads_playlist_id"]
        try:
            request = youtube_service.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=max_videos_per_channel)
            response = request.execute()
        except HttpError:
            st.warning(f"Could not fetch videos for '{channel['channel_title']}': Playlist not found or private.")
            continue
        video_ids = [item["snippet"]["resourceId"]["videoId"] for item in response.get("items", [])]
        if not video_ids: continue
        video_request = youtube_service.videos().list(part="snippet,statistics", id=",".join(video_ids))
        video_response = video_request.execute()
        for item in video_response.get("items", []):
            all_video_details.append({
                "channel_id": channel["channel_id"], "video_id": item["id"],
                "video_title": item["snippet"]["title"], "published_at": item["snippet"]["publishedAt"],
                "video_views": int(item["statistics"].get("viewCount", 0)),
                "video_likes": int(item["statistics"].get("likeCount", 0)),
                "video_comments": int(item["statistics"].get("commentCount", 0)),
                "video_tags": item["snippet"].get("tags", []),
            })
    return all_video_details

def calculate_keyword_relevance(df, query):
    # ... (This function remains the same)
    if df.empty:
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])
    keywords = [word.strip() for word in re.split('OR|AND', query, flags=re.IGNORECASE)]
    pattern = '|'.join(keywords)
    df['is_relevant'] = df['video_title'].str.contains(pattern, case=False, na=False)
    relevance = df.groupby('channel_id')['is_relevant'].mean().reset_index()
    relevance = relevance.rename(columns={'is_relevant': 'relevance_score'})
    return relevance

# --- NEW: Function to resolve a channel handle (@username) to an ID ---
def get_channel_id_from_handle(youtube_service, handle):
    """
    Uses the search API to find a channel ID from a given handle (e.g., GoogleDevelopers).
    The YouTube Data API v3 does not have a direct endpoint to resolve a handle.
    The recommended workaround is to use the search endpoint.
    """
    try:
        # The handle from the URL does not include '@', so we search for it directly.
        request = youtube_service.search().list(
            q=handle, part="id", type="channel", maxResults=1
        )
        response = request.execute()
        items = response.get("items", [])
        if items:
            # The search result for a handle is usually very accurate.
            # We take the first result.
            return items[0]["id"]["channelId"]
        return None
    except HttpError as e:
        st.error(f"API Error while resolving handle '@{handle}': {e}")
        return None

# --- NEW: Channel-as-Seed Function ---
def analyze_seed_channel(youtube_service, seed_channel_id):
    """Analyzes a seed channel and generates a new search query."""
    st.info(f"Analyzing seed channel: {seed_channel_id}...")
    
    # We need to get the seed channel's uploads playlist ID first
    seed_stats = get_channel_stats(youtube_service, [seed_channel_id])
    if not seed_stats:
        st.error("Could not retrieve stats for the seed channel. It may be an invalid ID or username.")
        return None

    # Fetch a good number of videos to analyze for topics
    seed_videos = get_video_details(youtube_service, seed_stats, max_videos_per_channel=20)
    if not seed_videos:
        st.error("Could not retrieve videos from the seed channel to analyze.")
        return None

    # Extract keywords from titles and tags
    all_keywords = []
    for video in seed_videos:
        # Add words from title
        all_keywords.extend(re.split(r'\s|\|', video['video_title']))
        # Add tags
        all_keywords.extend(video['video_tags'])
    
    # Clean and count keywords
    cleaned_keywords = [k.lower() for k in all_keywords if len(k) > 3 and k.lower() not in ['video', 'review']]
    if not cleaned_keywords:
        st.error("Could not extract meaningful keywords from the seed channel.")
        return None

    # Find the top 5 most common keywords
    top_keywords = [word for word, count in Counter(cleaned_keywords).most_common(5)]
    
    # Create a new search query
    new_query = " OR ".join(f'"{k}"' for k in top_keywords)
    st.success(f"Generated new query from seed channel: {new_query}")
    return new_query

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("🤖 YouTube Creator Search Agent")

with st.expander("ℹ️ Search Tips and Best Practices"):
    st.info("""... (search tips from before) ...""")

# --- FIX: The search method selector is moved *outside* the form. ---
# This allows the UI to update instantly when the user changes the method,
# because changing a widget outside a form triggers an immediate script rerun.
st.header("1. Search Method")
search_method = st.radio("Choose your search method:", ("Keywords", "Channel-as-Seed"), horizontal=True)

with st.form("search_form"):
    # The input fields are now dynamically displayed inside the form based on the selection made above.
    if search_method == "Keywords":
        st.info("💡 Enter search terms like 'Game Reviews' or 'Python Tutorial AND Beginner'.")
        query_input = st.text_input("Search Keywords", "Manga OR Anime")
        # We define a blank variable here so the script doesn't crash later
        seed_url_input = "" 
    else:
        st.info("💡 Enter the full URL of a YouTube channel to find similar creators.")
        seed_url_input = st.text_input("YouTube Channel URL (the 'seed')", "https://www.youtube.com/@YourFavoriteChannel")
        # We define a blank variable here so the script doesn't crash later
        query_input = "" 
        with st.expander("How does Channel-as-Seed work?"):
            st.markdown("""
                This method discovers new channels based on a single example channel you provide.
                1.  **Analyze:** The agent fetches the latest videos from the URL you enter.
                2.  **Learn:** It extracts the most common topics and keywords from that channel's video titles and tags.
                3.  **Discover:** It then uses those learned keywords to launch a new, highly specific search to find other channels with similar content.
            """)

    region_input = st.text_input("Search Region (for search bias)", "AR", help="Leave blank for global search.")

    st.header("2. Filtering Criteria")
    c1, c2, c3 = st.columns(3)
    with c1: min_subs_input = st.number_input("Minimum Subscribers", value=10000, help="Set to 0 to ignore.")
    with c2: country_filter_input = st.text_input("Channel Country (strict filter)", "AR", help="Leave blank to ignore.")
    with c3:
        use_date_filter = st.checkbox("Filter by Date", value=True)
        months_ago_input = st.number_input("Published within last (months)", value=18, disabled=not use_date_filter)

    submitted = st.form_submit_button("Find Creators")

# --- Main Execution Logic ---
if submitted:
    if not API_KEY or API_KEY == "PASTE_YOUR_API_KEY_HERE":
        st.error("Please paste your YouTube API key into the script.")
    else:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        final_query = ""

        # --- MODIFICATION: Handle different URL types for Channel-as-Seed ---
        if search_method == "Channel-as-Seed":
            identifier = extract_channel_id_from_url(seed_url_input)
            seed_channel_id = None  # Initialize to None

            if not identifier:
                st.error("Could not extract a valid Channel ID or Username from the URL. Please use a standard YouTube channel URL (e.g., youtube.com/channel/ID or youtube.com/@username).")
            # If it looks like a channel ID, use it directly.
            elif identifier.startswith("UC"):
                seed_channel_id = identifier
                st.info(f"Found Channel ID: {seed_channel_id}")
            # Otherwise, assume it's a handle and try to resolve it to an ID.
            else:
                with st.spinner(f"Resolving handle '@{identifier}'..."):
                    seed_channel_id = get_channel_id_from_handle(youtube, identifier)
                if not seed_channel_id:
                    st.error(f"Could not find a channel ID for the handle '@{identifier}'. Please check the URL and try again.")
                else:
                    st.success(f"Resolved handle '@{identifier}' to Channel ID: {seed_channel_id}")
            if seed_channel_id:
                final_query = analyze_seed_channel(youtube, seed_channel_id)
        else:
            # Use the query from the text box
            final_query = query_input
        
        # --- Proceed with the rest of the logic only if we have a valid query ---
        if final_query:
            # The rest of the execution logic is the same as before,
            # but it uses `final_query` instead of `query_input`.
            with st.spinner("Step 1/4: Searching for channels..."):
                initial_channels = Youtube(youtube, final_query, region_input, total_results_to_fetch=50)
            
            # ... (the rest of the script continues as before)
            if not initial_channels:
                st.error("Search did not return any channels.")
            else:
                df_initial = pd.DataFrame(initial_channels)
                with st.expander("See raw channels found"):
                    st.dataframe(df_initial)

                with st.spinner("Step 2/4: Fetching channel statistics..."):
                    channel_statistics = get_channel_stats(youtube, df_initial['channel_id'].tolist())

                if not channel_statistics:
                    st.warning("Could not retrieve detailed stats for the found channels.")
                else:
                    df_stats = pd.DataFrame(channel_statistics)
                    enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')

                    with st.spinner("Step 3/4: Fetching recent video details..."):
                        video_data = get_video_details(youtube, enriched_channel_data.to_dict('records'), max_videos_per_channel=10)

                    if not video_data:
                        st.warning("Could not retrieve any video details from the channels.")
                        st.dataframe(enriched_channel_data.sort_values(by="subscribers", ascending=False))
                    else:
                        with st.spinner("Step 4/4: Analyzing and filtering results..."):
                            df_videos = pd.DataFrame(video_data)
                            relevance_scores = calculate_keyword_relevance(df_videos.copy(), final_query)
                            df_videos['published_at'] = pd.to_datetime(df_videos['published_at'])
                            df_videos['engagement_rate'] = (df_videos['video_likes'] + df_videos['video_comments']) / (df_videos['video_views'] + 1)
                            df_full = pd.merge(df_videos, enriched_channel_data, on='channel_id')
                            
                            filtered_df = df_full[df_full['subscribers'] >= min_subs_input]
                            
                            if use_date_filter:
                                date_cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=months_ago_input)
                                filtered_df = filtered_df[filtered_df['published_at'] >= date_cutoff]
                            
                            if country_filter_input:
                                filtered_df = filtered_df[filtered_df['country'] == country_filter_input.upper()]

                        if filtered_df.empty:
                            st.error("No channels matched all your filtering criteria after full analysis.")
                        else:
                            avg_engagement = filtered_df.groupby('channel_id')['engagement_rate'].mean().reset_index()
                            final_channels = pd.merge(enriched_channel_data, avg_engagement, on='channel_id')
                            final_channels = pd.merge(final_channels, relevance_scores, on='channel_id', how='left')
                            
                            top_channels = final_channels.sort_values(by=['relevance_score', 'subscribers'], ascending=False)
                            
                            st.success(f"Analysis Complete! Found {len(top_channels)} channels matching your criteria.")
                            st.info("ℹ️ Results are sorted by Keyword Relevance score, then by subscriber count.")
                            
                            top_channels['relevance_score'] = top_channels['relevance_score'].map('{:.0%}'.format)
                            top_channels['engagement_rate'] = top_channels['engagement_rate'].map('{:.2%}'.format)
                            
                            st.dataframe(top_channels[['channel_title', 'relevance_score', 'subscribers', 'country', 'engagement_rate']])
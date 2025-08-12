import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import re
from googleapiclient.errors import HttpError
from collections import Counter
import google.generativeai as genai
import os
from dotenv import load_dotenv

try:
    import pycountry
    COUNTRY_OPTIONS_BASE = [f"{c.name} ({c.alpha_2})" for c in pycountry.countries]
except Exception:
    COUNTRY_OPTIONS_BASE = [
        "Argentina (AR)",
        "Australia (AU)",
        "Canada (CA)",
        "France (FR)",
        "Germany (DE)",
        "India (IN)",
        "Japan (JP)",
        "United Kingdom (GB)",
        "United States (US)",
    ]

# --- Securely Load API Keys ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Constants ---
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
    """Channel search with proper parts so channelId is present."""
    channels = []
    next_page_token = None
    # IMPORTANT: include id so item["id"]["channelId"] exists
    search_params = {'q': query, 'part': 'id,snippet', 'type': 'channel'}
    if region_code:
        search_params['regionCode'] = region_code
    while len(channels) < total_results_to_fetch:
        search_params['maxResults'] = min(50, total_results_to_fetch - len(channels))
        if next_page_token:
            search_params['pageToken'] = next_page_token
        response = youtube_service.search().list(**search_params).execute()
        for item in response.get("items", []):
            channel_id = item.get("id", {}).get("channelId")
            if not channel_id:
                continue
            channels.append({
                "channel_id": channel_id,
                "channel_title": item.get("snippet", {}).get("title", "Unknown")
            })
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return channels

def get_channel_stats(youtube_service, channel_ids):
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

# --- Function to resolve a channel handle (@username) to an ID ---
def get_channel_id_from_handle(youtube_service, handle):
    """
    Uses the search API to find a channel ID from a given handle (e.g., GoogleDevelopers).
    The YouTube Data API v3 does not have a direct endpoint to resolve a handle.
    The recommended workaround is to use the search endpoint.
    """
    try:
        request = youtube_service.search().list(
            q=handle, part="id", type="channel", maxResults=1
        )
        response = request.execute()
        items = response.get("items", [])
        if items:
            return items[0]["id"]["channelId"]
        return None
    except HttpError as e:
        st.error(f"API Error while resolving handle '@{handle}': {e}")
        return None

# --- Channel-as-Seed Function ---
def analyze_seed_channel(youtube_service, seed_channel_id):
    """Analyzes a seed channel and generates a new search query."""
    st.info(f"Analyzing seed channel: {seed_channel_id}...")
    seed_stats = get_channel_stats(youtube_service, [seed_channel_id])
    if not seed_stats:
        st.error("Could not retrieve stats for the seed channel. It may be an invalid ID or username.")
        return None
    seed_videos = get_video_details(youtube_service, seed_stats, max_videos_per_channel=20)
    if not seed_videos:
        st.error("Could not retrieve videos from the seed channel to analyze.")
        return None
    all_keywords = []
    for video in seed_videos:
        all_keywords.extend(re.split(r'\s|\|', video['video_title']))
        all_keywords.extend(video['video_tags'])
    cleaned_keywords = [k.lower() for k in all_keywords if len(k) > 3 and k.lower() not in ['video', 'review']]
    if not cleaned_keywords:
        st.error("Could not extract meaningful keywords from the seed channel.")
        return None
    top_keywords = [word for word, count in Counter(cleaned_keywords).most_common(5)]
    new_query = " OR ".join(f'"{k}"' for k in top_keywords)
    st.success(f"Generated new query from seed channel: {new_query}")
    return new_query

# --- NEW: Generative AI Summary Function ---
def generate_summary(df_results, query):
    """Formats the data and calls the Gemini API to generate a summary."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        top_5_df = df_results.head(5)
        data_string = ""
        for _, row in top_5_df.iterrows():
            data_string += f"- Channel: {row['channel_title']}\n"
            data_string += f"  - Subscribers: {row['subscribers']:,}\n"
            data_string += f"  - Country: {row['country']}\n"
            data_string += f"  - Relevance Score: {row['relevance_score']}\n"
            data_string += f"  - Avg. Engagement Rate: {row['engagement_rate']}\n\n"

        prompt = f"""
        You are an expert marketing analyst. Provide a concise summary of the top YouTube channels for the query "{query}".
        Base your analysis ONLY on the data below and highlight 2–3 standout channels and why.
        Data:
        {data_string}
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("🤖 YouTube Creator Search Agent")

with st.expander("ℹ️ Search Tips and Best Practices"):
    st.info("""... (search tips from before) ...""")

# The search method selector is outside the form to allow instant UI updates.
st.header("1. Search Method")
search_method = st.radio("Choose your search method:", ("Keywords", "Channel-as-Seed"), horizontal=True)

with st.form("search_form"):
    # Inputs change depending on the selected method
    if search_method == "Keywords":
        st.info("💡 Enter search terms like 'Game Reviews' or 'Python Tutorial AND Beginner'.")
        query_input = st.text_input("Search Keywords", "Manga OR Anime")
        seed_url_input = ""  # keep defined
    else:
        st.info("💡 Enter the full URL of a YouTube channel to find similar creators.")
        seed_url_input = st.text_input("YouTube Channel URL (the 'seed')", "https://www.youtube.com/@YourFavoriteChannel")
        query_input = ""  # keep defined
        with st.expander("How does Channel-as-Seed work?"):
            st.markdown("""
                This method discovers new channels based on a single example channel you provide.
                1.  **Analyze:** The agent fetches the latest videos from the URL you enter.
                2.  **Learn:** It extracts the most common topics and keywords from that channel's video titles and tags.
                3.  **Discover:** It then uses those learned keywords to launch a new, highly specific search to find other channels with similar content.
            """)

    country_options = COUNTRY_OPTIONS_BASE.copy()
    country_options.sort()
    country_options.insert(0, "Global")
    default_index = next((i for i, v in enumerate(country_options) if v.endswith("(AR)")), 0)
    selected_country = st.selectbox(
        "Search Region (for search bias)",
        country_options,
        index=default_index,
        help="Start typing a country name or its two-letter code.",
    )
    region_input = "" if selected_country == "Global" else selected_country.split("(")[-1][:2]

    st.header("2. Filtering Criteria")
    c1, c2, c3 = st.columns(3)
    with c1: min_subs_input = st.number_input("Minimum Subscribers", value=10000, help="Set to 0 to ignore.")
    with c2:
        country_filter_options = COUNTRY_OPTIONS_BASE.copy()
        country_filter_options.sort()
        country_filter_options.insert(0, "Any Country")
        default_filter_index = next((i for i, v in enumerate(country_filter_options) if v.endswith("(AR)")), 0)

        selected_country_filter = st.selectbox(
            "Channel Country (strict filter)",
            country_filter_options,
            index=default_filter_index,
            help="Start typing a country name or its two-letter code to filter."
        )
        country_filter_input = ""
        if selected_country_filter != "Any Country":
            match = re.search(r'\((\w{2})\)', selected_country_filter)
            if match:
                country_filter_input = match.group(1)
    with c3:
        use_date_filter = st.checkbox("Filter by Date", value=True)
        months_ago_input = st.number_input("Published within last (months)", value=18, disabled=not use_date_filter)

    # --- NEW: Checkbox for AI Summary (inside the form) ---
    generate_summary_checkbox = st.checkbox("Generate AI Summary of Top Results")

    submitted = st.form_submit_button("Find Creators")

# --- Main Execution Logic ---
if submitted:
    if not YOUTUBE_API_KEY:
        st.error("Please ensure your YOUTUBE_API_KEY is set in your .env file.")
    else:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        final_query = ""

        if search_method == "Channel-as-Seed":
            identifier = extract_channel_id_from_url(seed_url_input)
            seed_channel_id = None

            if not identifier:
                st.error("Could not extract a valid Channel ID or Username from the URL. Use youtube.com/channel/ID or youtube.com/@username")
                final_query = None
            elif identifier.startswith("UC"):
                seed_channel_id = identifier
                st.info(f"Found Channel ID: {seed_channel_id}")
            else:
                with st.spinner(f"Resolving handle '@{identifier}'..."):
                    seed_channel_id = get_channel_id_from_handle(youtube, identifier)
                if not seed_channel_id:
                    st.error(f"Could not find a channel ID for the handle '@{identifier}'.")
                    final_query = None
                else:
                    st.success(f"Resolved handle '@{identifier}' to Channel ID: {seed_channel_id}")

            if seed_channel_id:
                final_query = analyze_seed_channel(youtube, seed_channel_id)
        else:
            final_query = query_input

        # --- Proceed only if we have a valid query ---
        if final_query:
            with st.spinner("Step 1/4: Searching for channels..."):
                initial_channels = Youtube(youtube, final_query, region_input, total_results_to_fetch=50)

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

                            # --- AI Summary BEFORE formatting numbers for the table ---
                            if generate_summary_checkbox:
                                if not GEMINI_API_KEY:
                                    st.error("Please ensure your GEMINI_API_KEY is set in your .env file to generate a summary.")
                                else:
                                    with st.spinner("Generating AI Summary..."):
                                        summary_df = top_channels.copy()
                                        summary_df['relevance_score'] = summary_df['relevance_score'].map('{:.0%}'.format)
                                        summary_df['engagement_rate'] = summary_df['engagement_rate'].map('{:.2%}'.format)
                                        summary_text = generate_summary(summary_df, final_query)
                                        st.subheader("📝 AI Generated Summary")
                                        st.markdown(summary_text)

                            # Format for display
                            top_channels['relevance_score'] = top_channels['relevance_score'].map('{:.0%}'.format)
                            top_channels['engagement_rate'] = top_channels['engagement_rate'].map('{:.2%}'.format)

                            st.dataframe(top_channels[['channel_title', 'relevance_score', 'subscribers', 'country', 'engagement_rate']])
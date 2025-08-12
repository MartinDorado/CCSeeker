import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import re
from googleapiclient.errors import HttpError
from collections import Counter
import os
from dotenv import load_dotenv
# --- ADDED FOR GENAI SUMMARY ---
import google.generativeai as genai

# --- Securely Load API Keys ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# --- ADDED FOR GENAI SUMMARY ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Constants ---
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# --- All API and Data Functions ---

def extract_channel_id_from_url(url):
    patterns = [ r'(?:youtube\.com/channel/)([^/?&]+)', r'(?:youtube\.com/@)([^/?&]+)' ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def Youtube(youtube_service, query, region_code, total_results_to_fetch):
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
    all_video_details = []
    for channel in channel_data:
        playlist_id = channel["uploads_playlist_id"]
        try:
            request = youtube_service.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=max_videos_per_channel)
            response = request.execute()
        except HttpError as e:
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
    if df.empty:
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])
    keywords = [word.strip() for word in re.split('OR|AND', query, flags=re.IGNORECASE)]
    pattern = '|'.join(keywords)
    df['is_relevant'] = df['video_title'].str.contains(pattern, case=False, na=False)
    relevance = df.groupby('channel_id')['is_relevant'].mean().reset_index()
    relevance = relevance.rename(columns={'is_relevant': 'relevance_score'})
    return relevance

def analyze_seed_channel(youtube_service, seed_channel_id):
    st.info(f"Analyzing seed channel: {seed_channel_id}...")
    seed_stats = get_channel_stats(youtube_service, [seed_channel_id])
    if not seed_stats:
        st.error("Could not retrieve stats for the seed channel. It may be an invalid ID.")
        return None
    seed_videos = get_video_details(youtube_service, seed_stats, max_videos_per_channel=20)
    if not seed_videos:
        st.error("Could not retrieve videos from the seed channel to analyze.")
        return None
    all_keywords = []
    for video in seed_videos:
        all_keywords.extend(re.split(r'\s|\|', video['video_title']))
        all_keywords.extend(video['video_tags'])
    cleaned_keywords = [k.lower() for k in all_keywords if len(k) > 3 and k.lower() not in ['video', 'review', 'and', 'the']]
    if not cleaned_keywords:
        st.error("Could not extract meaningful keywords from the seed channel.")
        return None
    top_keywords = [word for word, count in Counter(cleaned_keywords).most_common(5)]
    new_query = " OR ".join(f'"{k}"' for k in top_keywords)
    st.success(f"Generated new query from seed channel: {new_query}")
    return new_query

# --- ADDED FOR GENAI SUMMARY ---
def generate_summary(df_results, query):
    """Formats the data and calls the Gemini API to generate a summary."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        top_5_df = df_results.head(5)
        
        data_string = ""
        for index, row in top_5_df.iterrows():
            data_string += f"- Channel: {row['channel_title']}\n"
            data_string += f"  - Subscribers: {row['subscribers']:,}\n"
            data_string += f"  - Country: {row['country']}\n"
            data_string += f"  - Relevance Score: {row['relevance_score']}\n"
            data_string += f"  - Avg. Engagement Rate: {row['engagement_rate']}\n\n"

        prompt = f"""
        You are an expert marketing analyst. Your task is to provide a concise summary of the top YouTube channels found for a specific search query.
        Based ONLY on the data provided below, write a brief, professional summary highlighting the top 2-3 most promising channels and why they stand out.
        **Search Query:** "{query}"
        **Data:**
        {data_string}
        **Summary:**
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🤖 YouTube Creator Search Agent")

st.header("1. Search Method")
search_method = st.radio(
    "Choose your search method:",
    ("Keywords", "Channel-as-Seed"),
    key="search_method",
    horizontal=True,
    label_visibility="collapsed"
)

with st.form("search_form"):
    if 'search_method' not in st.session_state:
        st.session_state.search_method = "Keywords"

    if st.session_state.search_method == "Keywords":
        query_input = st.text_input("Search Keywords", "Manga OR Anime")
        seed_url_input = "" 
    else: # Channel-as-Seed
        seed_url_input = st.text_input("YouTube Channel URL (the 'seed')", "https://www.youtube.com/@YourFavoriteChannel")
        query_input = "" 
        with st.expander("How does Channel-as-Seed work?"):
            st.markdown("""This method discovers new channels based on an example...""")

    region_input = st.text_input("Search Region (biases results)", "AR", help="Leave blank for global search.")
    
    st.header("2. Filtering Criteria")
    c1, c2, c3 = st.columns(3)
    with c1: min_subs_input = st.number_input("Minimum Subscribers", value=10000)
    with c2: country_filter_input = st.text_input("Channel Country (strict filter)", "AR", help="Leave blank to ignore.")
    with c3:
        use_date_filter = st.checkbox("Filter by Date", value=True)
        months_ago_input = st.number_input("Published within last (months)", value=18, disabled=not use_date_filter)
    
    # --- ADDED FOR GENAI SUMMARY ---
    st.header("3. Enhancements")
    generate_summary_checkbox = st.checkbox("Generate AI Summary of Top Results")
    
    submitted = st.form_submit_button("Find Creators")

# --- Main Execution Logic ---
if submitted:
    if not YOUTUBE_API_KEY:
        st.error("Please provide your YouTube API key in the .env file.")
    elif generate_summary_checkbox and not GEMINI_API_KEY:
        st.error("Please provide your Gemini API key in the .env file to generate a summary.")
    else:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        final_query = ""

        if st.session_state.search_method == "Channel-as-Seed":
            if not seed_url_input: st.error("Please provide a seed channel URL.")
            else:
                seed_channel_id = extract_channel_id_from_url(seed_url_input)
                if not seed_channel_id: st.error("Could not extract a valid Channel ID or Username from the URL.")
                else: final_query = analyze_seed_channel(youtube, seed_channel_id)
        else: # Keyword search
            if not query_input: st.error("Please provide search keywords.")
            else: final_query = query_input
        
        if final_query:
            with st.spinner("Step 1/4: Searching for channels..."):
                initial_channels = Youtube(youtube, final_query, region_input, total_results_to_fetch=50)
            if not initial_channels: st.error("Search did not return any channels.")
            else:
                df_initial = pd.DataFrame(initial_channels)
                with st.expander("See raw channels found"): st.dataframe(df_initial)
                with st.spinner("Step 2/4: Fetching channel statistics..."):
                    channel_statistics = get_channel_stats(youtube, df_initial['channel_id'].tolist())
                if not channel_statistics: st.warning("Could not retrieve detailed stats for the found channels.")
                else:
                    df_stats = pd.DataFrame(channel_statistics)
                    enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')
                    with st.spinner("Step 3/4: Fetching recent video details..."):
                        video_data = get_video_details(youtube, enriched_channel_data.to_dict('records'), max_videos_per_channel=10)
                    if not video_data:
                        st.warning("Could not retrieve video details. Displaying channel data only.")
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
                            
                            display_df = top_channels.copy()
                            display_df['relevance_score'] = display_df['relevance_score'].map('{:.0%}'.format)
                            display_df['engagement_rate'] = display_df['engagement_rate'].map('{:.2%}'.format)
                            
                            # --- ADDED FOR GENAI SUMMARY ---
                            if generate_summary_checkbox:
                                with st.spinner("Generating AI Summary..."):
                                    summary_text = generate_summary(display_df, final_query)
                                    st.subheader("📝 AI Generated Summary")
                                    st.markdown(summary_text)
                            
                            st.subheader("📊 Detailed Results")
                            st.dataframe(display_df[['channel_title', 'relevance_score', 'subscribers', 'country', 'engagement_rate']])
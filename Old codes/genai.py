import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import re
from googleapiclient.errors import HttpError
from collections import Counter
import google.generativeai as genai # New Import

# --- Configuration ---
YOUTUBE_API_KEY = "AIzaSyAFn5Ky0fAY3V8Vhu_ITl8_4HUZTNi3HFA"
# --- NEW: Add your Gemini API Key here ---
GEMINI_API_KEY = "AIzaSyAgaInyRL0MMYwy8_-OCf7A2UkXVz2_RGU"

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# --- All API and Data Functions ---
# (The functions Youtube, get_channel_stats, get_video_details, 
# and calculate_keyword_relevance remain the same as your last version)
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
            if e.resp.status == 404:
                st.warning(f"Could not fetch videos for '{channel['channel_title']}': Playlist not found.")
            else:
                st.warning(f"An API error occurred for channel '{channel['channel_title']}': {e}")
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

# --- NEW: Generative AI Summary Function ---
def generate_summary(df_results, query):
    """Formats the data and calls the Gemini API to generate a summary."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # --- FIX IS HERE: Use the correct model name from your list ---
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # 1. Format the data for the prompt
        top_5_df = df_results.head(5)
        
        data_string = ""
        for index, row in top_5_df.iterrows():
            data_string += f"- Channel: {row['channel_title']}\n"
            data_string += f"  - Subscribers: {row['subscribers']:,}\n"
            data_string += f"  - Country: {row['country']}\n"
            data_string += f"  - Relevance Score: {row['relevance_score']}\n"
            data_string += f"  - Avg. Engagement Rate: {row['engagement_rate']}\n\n"

        # 2. Create the prompt
        prompt = f"""
        You are an expert marketing analyst. Your task is to provide a concise summary of the top YouTube channels found for a specific search query.
        Based ONLY on the data provided below, write a brief, professional summary highlighting the top 2-3 most promising channels and why they stand out.

        **Search Query:** "{query}"

        **Data:**
        {data_string}

        **Summary:**
        """

        # 3. Call the API
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("🤖 YouTube Creator Search Agent")

# ... (UI code is the same as your last version, with one addition)
with st.form("search_form"):
    st.header("1. Search Criteria")
    c1, c2 = st.columns(2)
    with c1: query_input = st.text_input("Search Keywords", "Manga OR Anime")
    with c2: region_input = st.text_input("Search Region (for search bias)", "AR", help="Leave blank for global search.")

    st.header("2. Filtering Criteria")
    c1, c2, c3 = st.columns(3)
    with c1: min_subs_input = st.number_input("Minimum Subscribers", value=10000, help="Set to 0 to ignore.")
    with c2: country_filter_input = st.text_input("Channel Country (strict filter)", "AR", help="Leave blank to ignore.")
    with c3:
        use_date_filter = st.checkbox("Filter by Date", value=True)
        months_ago_input = st.number_input("Published within last (months)", value=18, disabled=not use_date_filter)
    
    # --- NEW: Checkbox for AI Summary ---
    st.header("3. Enhancements")
    generate_summary_checkbox = st.checkbox("Generate AI Summary of Top Results")


    submitted = st.form_submit_button("Find Creators")

if submitted:
    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "PASTE_YOUR_YOUTUBE_API_KEY_HERE":
        st.error("Please paste your YouTube API key into the script.")
    # --- NEW: Check for Gemini key if the box is checked ---
    elif generate_summary_checkbox and (not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE"):
        st.error("Please paste your Gemini API key into the script to generate a summary.")
    else:
        # ... (The main logic for searching and filtering is the same)
        with st.spinner("Step 1/4: Searching for channels..."):
            youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
            initial_channels = Youtube(youtube, query_input, region_input, total_results_to_fetch=50)

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
                        relevance_scores = calculate_keyword_relevance(df_videos.copy(), query_input)
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
                        
                        # --- NEW: Call the summary function if the box is checked ---
                        if generate_summary_checkbox:
                            with st.spinner("Generating AI Summary..."):
                                # Re-format percentages for the AI model
                                summary_df = top_channels.copy()
                                summary_df['relevance_score'] = summary_df['relevance_score'].map('{:.0%}'.format)
                                summary_df['engagement_rate'] = summary_df['engagement_rate'].map('{:.2%}'.format)
                                
                                summary_text = generate_summary(summary_df, query_input)
                                st.subheader("📝 AI Generated Summary")
                                st.markdown(summary_text)

                        # Format for display
                        top_channels['relevance_score'] = top_channels['relevance_score'].map('{:.0%}'.format)
                        top_channels['engagement_rate'] = top_channels['engagement_rate'].map('{:.2%}'.format)
                        
                        st.subheader("📊 Detailed Results")
                        st.dataframe(top_channels[['channel_title', 'relevance_score', 'subscribers', 'country', 'engagement_rate']])
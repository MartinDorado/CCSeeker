import pandas as pd
from googleapiclient.discovery import build
import os

# --- Configuration ---
API_KEY = "REDACTED_YOUTUBE_KEY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MAX_VIDEOS_PER_CHANNEL = 5

# --- API Functions ---

# MODIFIED FUNCTION: Now handles pagination to get more than 50 results.
def Youtube(youtube_service, total_results_to_fetch=100):
    print(f"Step 1: Searching for up to {total_results_to_fetch} creator channels...")
    channels = []
    next_page_token = None
    
    while len(channels) < total_results_to_fetch:
        # We can only request a maximum of 50 at a time.
        results_per_page = min(50, total_results_to_fetch - len(channels))

        request = youtube_service.search().list(
            q="Manga OR Anime",
            part="snippet",
            type="channel",
            regionCode="AR",
            maxResults=results_per_page,
            pageToken=next_page_token # Use the token to get the next page.
        )
        response = request.execute()

        for item in response.get("items", []):
            channels.append({
                "channel_id": item["id"]["channelId"],
                "channel_title": item["snippet"]["title"]
            })

        # Check if there's another page of results.
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break # Exit the loop if there are no more pages.

    print(f"Found {len(channels)} channels.")
    return channels

def get_channel_stats(youtube_service, channel_ids):
    print("Step 2: Fetching channel statistics...")
    stats_data = []
    # Process IDs in chunks of 50, as that's the API limit for channels.list
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i + 50]
        request = youtube_service.channels().list(
            part="snippet,statistics", id=",".join(chunk)
        )
        response = request.execute()
        for item in response.get("items", []):
            stats_data.append({
                "channel_id": item["id"],
                # --- NEW LINE ADDED HERE ---
                "country": item["snippet"].get("country", "N/A"), # Use .get() in case it's not set
                "subscribers": int(item["statistics"].get("subscriberCount", 0)),
                "views": int(item["statistics"].get("viewCount", 0)),
                "videos": int(item["statistics"].get("videoCount", 0)),
                "channel_description": item["snippet"]["description"],
                "uploads_playlist_id": item["id"].replace("UC", "UU", 1)
            })
    print(f"Successfully fetched stats for {len(stats_data)} channels.")
    return stats_data

def get_video_details(youtube_service, channel_data):
    print("Step 3: Fetching recent video details for each channel...")
    all_video_details = []
    for channel in channel_data:
        playlist_id = channel["uploads_playlist_id"]
        try:
            request = youtube_service.playlistItems().list(
                part="snippet", playlistId=playlist_id, maxResults=MAX_VIDEOS_PER_CHANNEL
            )
            response = request.execute()
        except Exception as e:
            print(f"Could not fetch videos for channel {channel['channel_id']}: {e}")
            continue # Skip to the next channel if there's an error
            
        video_ids = [item["snippet"]["resourceId"]["videoId"] for item in response.get("items", [])]
        if not video_ids:
            continue

        video_request = youtube_service.videos().list(
            part="snippet,statistics", id=",".join(video_ids)
        )
        video_response = video_request.execute()

        for item in video_response.get("items", []):
            all_video_details.append({
                "channel_id": channel["channel_id"],
                "channel_title": channel["channel_title"],
                "video_id": item["id"],
                "video_title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "video_views": int(item["statistics"].get("viewCount", 0)),
                "video_likes": int(item["statistics"].get("likeCount", 0)),
                "video_comments": int(item["statistics"].get("commentCount", 0)),
                "video_tags": item["snippet"].get("tags", []),
            })
    return all_video_details

# --- Execution ---
# --- Execution ---
# --- Execution ---
if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("Error: Please paste your API key into the script.")
    else:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        
        initial_channels = Youtube(youtube, total_results_to_fetch=100)
        
        if initial_channels:
            df_initial = pd.DataFrame(initial_channels)
            channel_ids = df_initial['channel_id'].tolist()
            
            channel_statistics = get_channel_stats(youtube, channel_ids)
            df_stats = pd.DataFrame(channel_statistics)
            
            enriched_channel_data = pd.merge(df_initial, df_stats, on='channel_id')
            
            video_data = get_video_details(youtube, enriched_channel_data.to_dict('records'))

            if video_data:
                df_videos = pd.DataFrame(video_data)
                
                # --- FIX IS HERE ---
                # Add 'country' to the list of columns to include in the final merge.
                final_df = pd.merge(
                    df_videos,
                    enriched_channel_data[['channel_id', 'country', 'subscribers', 'views', 'videos']],
                    on='channel_id'
                )
                
                final_df.to_csv("creator_video_data.csv", index=False, encoding='utf-8')
                print("\nSuccess! Results saved to creator_video_data.csv")
            else:
                print("Could not retrieve video details.")
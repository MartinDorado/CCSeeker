import pandas as pd

# Load the data collected from Phase 2
try:
    df = pd.read_csv("creator_video_data.csv")
except FileNotFoundError:
    print("Error: 'creator_video_data.csv' not found. Make sure you have run the Phase 2 script successfully.")
    exit()

# --- Core Filtering Logic ---

# 1. Convert 'published_at' column to datetime objects for comparison.
df['published_at'] = pd.to_datetime(df['published_at'])

# 2. Calculate the engagement rate for each video.
df['engagement_rate'] = (df['video_likes'] + df['video_comments']) / (df['video_views'] + 1)

# 3. Define your filtering criteria.
MIN_SUBSCRIBERS = 10000
TARGET_COUNTRY = 'AR'
# --- NEW: Define the date cutoff for the last 18 months ---
date_cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=18)


# 4. Apply the filters to the DataFrame.
# --- NEW: Added the date filter to this block ---
filtered_df = df[
    (df['subscribers'] >= MIN_SUBSCRIBERS) &
    (df['country'] == TARGET_COUNTRY) &
    (df['published_at'] >= date_cutoff)
]

# --- NEW: Calculate Average Engagement Rate Per Channel ---
# We group the filtered data by channel and calculate the mean of the engagement rate.
avg_engagement_per_channel = filtered_df.groupby(['channel_id', 'channel_title', 'subscribers'])['engagement_rate'].mean().reset_index()


# --- Display Results ---
print(f"Original number of videos: {len(df)}")
print(f"Number of videos published in the last 18 months from relevant channels: {len(filtered_df)}")

# Display the top 5 channels sorted by their average engagement rate.
top_channels = avg_engagement_per_channel.sort_values(by='engagement_rate', ascending=False)

print("\nTop 5 Channels by Average Engagement Rate (Last 18 Months):")
print(top_channels.head(5))

# --- Save the new channel-focused results to a CSV file ---
top_channels.to_csv('filtered_channels_by_engagement.csv', index=False, encoding='utf-8')
print("\n✅ Full list of filtered channels saved to 'filtered_channels_by_engagement.csv'")
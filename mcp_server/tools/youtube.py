"""
mcp_server/tools/youtube.py - YouTube API operations
"""
import os
import re
from typing import Dict, List, Optional, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import asyncio
from concurrent.futures import ThreadPoolExecutor


class YouTubeTools:
    """Encapsulates all YouTube Data API operations."""
    
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        self._youtube = None
        self._executor = ThreadPoolExecutor(max_workers=5)
    
    @property
    def youtube(self):
        """Lazy initialization of YouTube client."""
        if not self._youtube and self.api_key:
            self._youtube = build("youtube", "v3", developerKey=self.api_key)
        return self._youtube
    
    async def search_channels(
        self,
        query: str,
        region_code: str = "",
        use_boolean: bool = False,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """Search for YouTube channels with optional boolean query support."""
        if not self.youtube:
            raise ValueError("YouTube API key not configured")
        
        if use_boolean and self._has_boolean_operators(query):
            return await self._search_boolean(query, region_code, max_results)
        else:
            return await self._search_simple(query, region_code, max_results)
    
    async def _search_simple(
        self, 
        query: str, 
        region_code: str, 
        max_results: int
    ) -> Dict[str, Any]:
        """Simple channel search."""
        loop = asyncio.get_event_loop()
        
        def _search():
            channels = []
            next_page_token = None
            fetched = 0
            
            while fetched < max_results:
                try:
                    request = self.youtube.search().list(
                        q=query,
                        type="channel",
                        part="id,snippet",
                        maxResults=min(50, max_results - fetched),
                        pageToken=next_page_token,
                        regionCode=region_code if region_code else None
                    )
                    response = request.execute()
                    
                    for item in response.get("items", []):
                        channels.append({
                            "channel_id": item["id"]["channelId"],
                            "title": item["snippet"]["title"],
                            "description": item["snippet"]["description"][:200],
                            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"]
                        })
                    
                    fetched += len(response.get("items", []))
                    next_page_token = response.get("nextPageToken")
                    
                    if not next_page_token:
                        break
                        
                except HttpError as e:
                    raise Exception(f"YouTube API error: {e}")
            
            return channels
        
        channels = await loop.run_in_executor(self._executor, _search)
        
        return {
            "query": query,
            "region": region_code or "global",
            "count": len(channels),
            "channels": channels
        }
    
    async def _search_boolean(
        self, 
        query: str, 
        region_code: str, 
        max_results: int
    ) -> Dict[str, Any]:
        """Boolean query search with AND/OR support."""
        # Parse query into DNF (Disjunctive Normal Form)
        dnf_clauses = self._parse_query_to_dnf(query)
        all_channels = {}
        
        for clause in dnf_clauses:
            # For AND clauses, we need intersection
            clause_results = []
            for term in clause:
                term_result = await self._search_simple(
                    term.strip('"'), 
                    region_code, 
                    max_results
                )
                clause_results.append({
                    ch["channel_id"]: ch 
                    for ch in term_result["channels"]
                })
            
            # Intersect results for AND
            if clause_results:
                intersection = set(clause_results[0].keys())
                for result in clause_results[1:]:
                    intersection &= set(result.keys())
                
                for channel_id in intersection:
                    all_channels[channel_id] = clause_results[0][channel_id]
        
        return {
            "query": query,
            "boolean_mode": True,
            "region": region_code or "global",
            "count": len(all_channels),
            "channels": list(all_channels.values())[:max_results]
        }
    
    async def analyze_seed_channel(
        self,
        channel_input: str,
        max_videos: int = 30,
        target_language: str = "auto",
        ignore_words: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze a seed channel to extract topics."""
        if not self.youtube:
            raise ValueError("YouTube API key not configured")
        
        # Resolve channel ID
        channel_id = await self.resolve_channel_id(channel_input)
        if not channel_id:
            raise ValueError(f"Could not resolve channel: {channel_input}")
        
        loop = asyncio.get_event_loop()
        
        def _analyze():
            # Get channel details
            channel_response = self.youtube.channels().list(
                part="snippet,contentDetails",
                id=channel_id
            ).execute()
            
            if not channel_response.get("items"):
                raise ValueError("Channel not found")
            
            channel_data = channel_response["items"][0]
            uploads_playlist = channel_data["contentDetails"]["relatedPlaylists"]["uploads"]
            
            # Get recent videos
            videos = []
            video_response = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist,
                maxResults=min(50, max_videos)
            ).execute()
            
            video_ids = [
                item["snippet"]["resourceId"]["videoId"]
                for item in video_response.get("items", [])
            ]
            
            if video_ids:
                # Get video details with tags
                videos_detail = self.youtube.videos().list(
                    part="snippet",
                    id=",".join(video_ids)
                ).execute()
                
                for video in videos_detail.get("items", []):
                    videos.append({
                        "title": video["snippet"]["title"],
                        "tags": video["snippet"].get("tags", []),
                        "description": video["snippet"]["description"][:500]
                    })
            
            # Extract topics (simplified version)
            topics = self._extract_topics(videos, ignore_words or [])
            
            return {
                "channel_id": channel_id,
                "channel_name": channel_data["snippet"]["title"],
                "videos_analyzed": len(videos),
                "topics": topics,
                "suggested_query": " OR ".join([f'"{t}"' if " " in t else t for t in topics[:6]])
            }
        
        return await loop.run_in_executor(self._executor, _analyze)
    
    async def get_channel_analytics(
        self,
        channel_ids: List[str],
        include_videos: bool = True,
        videos_per_channel: int = 10,
        calculate_engagement: bool = True
    ) -> Dict[str, Any]:
        """Get detailed analytics for channels."""
        if not self.youtube:
            raise ValueError("YouTube API key not configured")
        
        loop = asyncio.get_event_loop()
        
        def _get_analytics():
            # Batch request for channel stats
            channels_response = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=",".join(channel_ids[:50])  # API limit
            ).execute()
            
            analytics = []
            for channel in channels_response.get("items", []):
                channel_data = {
                    "channel_id": channel["id"],
                    "title": channel["snippet"]["title"],
                    "country": channel["snippet"].get("country", "N/A"),
                    "subscribers": int(channel["statistics"].get("subscriberCount", 0)),
                    "total_views": int(channel["statistics"].get("viewCount", 0)),
                    "total_videos": int(channel["statistics"].get("videoCount", 0)),
                }
                
                if include_videos and calculate_engagement:
                    # Get recent videos for engagement calculation
                    uploads_playlist = channel["contentDetails"]["relatedPlaylists"]["uploads"]
                    
                    try:
                        videos_response = self.youtube.playlistItems().list(
                            part="snippet",
                            playlistId=uploads_playlist,
                            maxResults=videos_per_channel
                        ).execute()
                        
                        video_ids = [
                            item["snippet"]["resourceId"]["videoId"]
                            for item in videos_response.get("items", [])
                        ]
                        
                        if video_ids:
                            videos_detail = self.youtube.videos().list(
                                part="statistics",
                                id=",".join(video_ids)
                            ).execute()
                            
                            engagement_rates = []
                            for video in videos_detail.get("items", []):
                                stats = video["statistics"]
                                views = int(stats.get("viewCount", 0))
                                likes = int(stats.get("likeCount", 0))
                                comments = int(stats.get("commentCount", 0))
                                
                                if views > 0:
                                    engagement_rates.append((likes + comments) / views)
                            
                            if engagement_rates:
                                channel_data["avg_engagement_rate"] = sum(engagement_rates) / len(engagement_rates)
                                channel_data["engagement_sample_size"] = len(engagement_rates)
                    
                    except HttpError:
                        channel_data["avg_engagement_rate"] = None
                        channel_data["engagement_sample_size"] = 0
                
                analytics.append(channel_data)
            
            return {
                "channels_analyzed": len(analytics),
                "analytics": analytics
            }
        
        return await loop.run_in_executor(self._executor, _get_analytics)
    
    async def resolve_channel_id(self, channel_input: str) -> Optional[str]:
        """Resolve various channel input formats to channel ID."""
        if not channel_input:
            return None
        
        # Direct channel ID
        if channel_input.startswith("UC") and len(channel_input) >= 20:
            return channel_input
        
        # Extract from URL
        url_patterns = [
            r'youtube\.com/channel/([^/?&]+)',
            r'youtube\.com/@([^/?&]+)',
            r'youtube\.com/c/([^/?&]+)',
            r'youtube\.com/user/([^/?&]+)'
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, channel_input)
            if match:
                identifier = match.group(1)
                if identifier.startswith("UC"):
                    return identifier
                # Need to search for the channel
                result = await self._search_simple(identifier, "", 1)
                if result["channels"]:
                    return result["channels"][0]["channel_id"]
        
        # Treat as @handle or channel name
        handle = channel_input.lstrip("@")
        result = await self._search_simple(handle, "", 1)
        if result["channels"]:
            return result["channels"][0]["channel_id"]
        
        return None
    
    def _has_boolean_operators(self, query: str) -> bool:
        """Check if query contains boolean operators."""
        return bool(re.search(r'\b(AND|OR)\b', query, re.IGNORECASE))
    
    def _parse_query_to_dnf(self, query: str) -> List[List[str]]:
        """Parse boolean query to Disjunctive Normal Form."""
        # Split by OR first
        or_clauses = re.split(r'\bOR\b', query, flags=re.IGNORECASE)
        dnf = []
        
        for clause in or_clauses:
            # Split by AND within each OR clause
            and_terms = re.split(r'\bAND\b', clause, flags=re.IGNORECASE)
            terms = [term.strip() for term in and_terms if term.strip()]
            if terms:
                dnf.append(terms)
        
        return dnf
    
    def _extract_topics(self, videos: List[Dict], ignore_words: List[str]) -> List[str]:
        """Extract top topics from video data."""
        from collections import Counter
        
        # Simple word frequency analysis
        word_counts = Counter()
        ignore_set = set(w.lower() for w in ignore_words)
        
        for video in videos:
            # Process title
            words = re.findall(r'\b[a-z]+\b', video["title"].lower())
            words = [w for w in words if len(w) > 3 and w not in ignore_set]
            word_counts.update(words)
            
            # Process tags
            for tag in video.get("tags", []):
                tag_words = re.findall(r'\b[a-z]+\b', tag.lower())
                tag_words = [w for w in tag_words if len(w) > 3 and w not in ignore_set]
                word_counts.update(tag_words)
        
        # Get top topics
        top_topics = [word for word, _ in word_counts.most_common(15)]
        
        # Try to form bigrams for better topics
        bigrams = []
        for video in videos[:10]:  # Sample first 10 videos
            title_words = video["title"].lower().split()
            for i in range(len(title_words) - 1):
                bigram = f"{title_words[i]} {title_words[i+1]}"
                if all(w not in ignore_set for w in bigram.split()):
                    bigrams.append(bigram)
        
        bigram_counts = Counter(bigrams)
        top_bigrams = [bg for bg, count in bigram_counts.most_common(5) if count > 2]
        
        return top_bigrams + top_topics[:10-len(top_bigrams)]
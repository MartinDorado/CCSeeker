"""
mcp_server/server.py - Production-ready MCP server
"""
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import hashlib
from dotenv import load_dotenv
load_dotenv()
from mcp.server import Server, NotificationOptions  
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from .tools.youtube import YouTubeTools
from .tools.ai_generation import AIGenerationTools
from .resources.cache import CacheManager


class CCSeekerMCPServer:
    """Production MCP server for YouTube creator discovery."""
    
    def __init__(self):
        self.server = Server("ccseeker")
        self.youtube_tools = YouTubeTools()
        self.ai_tools = AIGenerationTools()
        self.cache = CacheManager(ttl_seconds=3600)  # 1-hour cache
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Register all MCP handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List all available tools with proper schemas."""
            return [
                # Search tools
                types.Tool(
                    name="search_channels",
                    description="Search YouTube channels with advanced filtering",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (supports AND/OR operators)"
                            },
                            "region_code": {
                                "type": "string",
                                "description": "ISO country code (e.g., US, GB)",
                                "maxLength": 2
                            },
                            "use_boolean": {
                                "type": "boolean",
                                "description": "Enable boolean query parsing",
                                "default": False
                            },
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 50
                            }
                        },
                        "required": ["query"]
                    }
                ),
                
                # Analytics tools
                types.Tool(
                    name="analyze_seed_channel",
                    description="Extract topics from a seed YouTube channel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel_input": {
                                "type": "string",
                                "description": "Channel URL, @handle, or UC... ID"
                            },
                            "max_videos": {
                                "type": "integer",
                                "minimum": 5,
                                "maximum": 50,
                                "default": 30
                            },
                            "target_language": {
                                "type": "string",
                                "enum": ["auto", "en", "es", "pt", "fr", "de"],
                                "default": "auto"
                            },
                            "ignore_words": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Words to exclude from analysis"
                            }
                        },
                        "required": ["channel_input"]
                    }
                ),
                
                types.Tool(
                    name="get_channel_analytics",
                    description="Get detailed analytics for YouTube channels",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 50
                            },
                            "include_videos": {
                                "type": "boolean",
                                "default": True
                            },
                            "videos_per_channel": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "default": 10
                            },
                            "calculate_engagement": {
                                "type": "boolean",
                                "default": True
                            }
                        },
                        "required": ["channel_ids"]
                    }
                ),
                
                # Generation tools
                types.Tool(
                    name="generate_summary",
                    description="Generate AI summary of channel analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channels_data": {
                                "type": "array",
                                "description": "Channel analytics data"
                            },
                            "query_context": {
                                "type": "string",
                                "description": "Original search query for context"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["en", "es"],
                                "default": "en"
                            },
                            "max_channels": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "default": 5
                            }
                        },
                        "required": ["channels_data", "query_context"]
                    }
                ),
                
                types.Tool(
                    name="generate_outreach",
                    description="Generate personalized outreach emails",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 10
                            },
                            "campaign_context": {
                                "type": "string",
                                "description": "Campaign/collaboration context"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["en", "es"],
                                "default": "en"
                            },
                            "tone": {
                                "type": "string",
                                "enum": ["professional", "casual", "enthusiastic"],
                                "default": "professional"
                            }
                        },
                        "required": ["channel_names", "campaign_context"]
                    }
                ),
                
                # Batch operations
                types.Tool(
                    name="batch_analyze",
                    description="Analyze multiple channels in parallel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel_inputs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of channel URLs/handles/IDs"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["topics", "engagement", "full"],
                                "default": "full"
                            }
                        },
                        "required": ["channel_inputs"]
                    }
                ),
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            """Route tool calls to appropriate handlers with caching."""
            
            # Generate cache key
            cache_key = self._generate_cache_key(name, arguments)
            
            # Check cache for read operations
            if name in ["search_channels", "get_channel_analytics", "analyze_seed_channel"]:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "success",
                            "cached": True,
                            "data": cached_result
                        }, indent=2)
                    )]
            
            try:
                # Route to appropriate handler
                if name == "search_channels":
                    result = await self.youtube_tools.search_channels(**arguments)
                
                elif name == "analyze_seed_channel":
                    result = await self.youtube_tools.analyze_seed_channel(**arguments)
                
                elif name == "get_channel_analytics":
                    result = await self.youtube_tools.get_channel_analytics(**arguments)
                
                elif name == "generate_summary":
                    result = await self.ai_tools.generate_summary(**arguments)
                
                elif name == "generate_outreach":
                    result = await self.ai_tools.generate_outreach(**arguments)
                
                elif name == "batch_analyze":
                    result = await self._batch_analyze(**arguments)
                
                else:
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": f"Unknown tool: {name}"
                        })
                    )]
                
                # Cache successful results
                if name in ["search_channels", "get_channel_analytics", "analyze_seed_channel"]:
                    await self.cache.set(cache_key, result)
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "cached": False,
                        "data": result
                    }, indent=2)
                )]
                
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": str(e),
                        "tool": name
                    })
                )]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """List available resources (search history, saved campaigns)."""
            return [
                types.Resource(
                    uri="ccseeker://searches/history",
                    name="Search History",
                    description="Recent channel searches and results",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="ccseeker://campaigns/saved",
                    name="Saved Campaigns",
                    description="Saved outreach campaigns",
                    mimeType="application/json"
                ),
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource data."""
            if uri == "ccseeker://searches/history":
                return json.dumps(await self.cache.get_all_keys())
            elif uri == "ccseeker://campaigns/saved":
                # Implement campaign storage
                return json.dumps({"campaigns": []})
            return json.dumps({"error": "Resource not found"})
    
    def _generate_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate deterministic cache key for tool calls."""
        # Sort arguments for consistent hashing
        sorted_args = json.dumps(arguments, sort_keys=True)
        key_string = f"{tool_name}:{sorted_args}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _batch_analyze(self, channel_inputs: List[str], analysis_type: str) -> Dict:
        """Perform batch analysis on multiple channels."""
        tasks = []
        for channel_input in channel_inputs:
            if analysis_type in ["topics", "full"]:
                tasks.append(self.youtube_tools.analyze_seed_channel(
                    channel_input=channel_input,
                    max_videos=20
                ))
            if analysis_type in ["engagement", "full"]:
                # First resolve channel ID
                channel_id = await self.youtube_tools.resolve_channel_id(channel_input)
                if channel_id:
                    tasks.append(self.youtube_tools.get_channel_analytics(
                        channel_ids=[channel_id],
                        include_videos=True
                    ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "analyzed_count": len([r for r in results if not isinstance(r, Exception)]),
            "failed_count": len([r for r in results if isinstance(r, Exception)]),
            "results": [
                r if not isinstance(r, Exception) else {"error": str(r)}
                for r in results
            ]
        }
    
    async def run(self):
        """Run the MCP server with stdio transport."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ccseeker",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Entry point for the MCP server."""
    server = CCSeekerMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
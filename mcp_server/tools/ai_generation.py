"""
mcp_server/tools/ai_generation.py - AI-powered content generation
"""
import os
from typing import Dict, List, Any, Optional
import google.generativeai as genai
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AIGenerationTools:
    """Handles all AI generation tasks using Gemini."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    @property
    def model(self):
        """Lazy initialization of Gemini model."""
        if not self._model and self.api_key:
            self._model = genai.GenerativeModel('gemini-1.5-flash')
        return self._model
    
    async def generate_summary(
        self,
        channels_data: List[Dict],
        query_context: str,
        language: str = "en",
        max_channels: int = 5
    ) -> Dict[str, Any]:
        """Generate AI summary of channel analysis results."""
        if not self.model:
            raise ValueError("Gemini API key not configured")
        
        # Prepare data for summary
        top_channels = sorted(
            channels_data, 
            key=lambda x: x.get('subscribers', 0), 
            reverse=True
        )[:max_channels]
        
        # Format channel data for prompt
        channels_text = []
        for ch in top_channels:
            channels_text.append(
                f"- {ch['title']}: {ch.get('subscribers', 0):,} subscribers, "
                f"{ch.get('country', 'N/A')} country, "
                f"{ch.get('avg_engagement_rate', 0):.2%} engagement"
            )
        
        prompt = self._build_summary_prompt(
            channels_text, 
            query_context, 
            language
        )
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            response = self.model.generate_content(prompt)
            return response.text
        
        summary_text = await loop.run_in_executor(self._executor, _generate)
        
        return {
            "summary": summary_text,
            "channels_analyzed": len(channels_data),
            "channels_highlighted": len(top_channels),
            "language": language,
            "query_context": query_context
        }
    
    async def generate_outreach(
        self,
        channel_names: List[str],
        campaign_context: str,
        language: str = "en",
        tone: str = "professional"
    ) -> Dict[str, Any]:
        """Generate personalized outreach emails for creators."""
        if not self.model:
            raise ValueError("Gemini API key not configured")
        
        outreach_drafts = []
        
        for channel_name in channel_names[:10]:  # Limit to 10
            prompt = self._build_outreach_prompt(
                channel_name,
                campaign_context,
                language,
                tone
            )
            
            loop = asyncio.get_event_loop()
            
            def _generate():
                response = self.model.generate_content(prompt)
                return response.text
            
            try:
                draft = await loop.run_in_executor(self._executor, _generate)
                outreach_drafts.append({
                    "channel": channel_name,
                    "draft": draft,
                    "status": "success"
                })
            except Exception as e:
                outreach_drafts.append({
                    "channel": channel_name,
                    "draft": "",
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "drafts": outreach_drafts,
            "language": language,
            "tone": tone,
            "campaign_context": campaign_context
        }
    
    def _build_summary_prompt(
        self, 
        channels_text: List[str], 
        query_context: str,
        language: str
    ) -> str:
        """Build prompt for channel summary generation."""
        lang_instruction = {
            "en": "Write in clear, professional English.",
            "es": "Escribe en español claro y profesional."
        }.get(language, "Write in English.")
        
        return f"""
        {lang_instruction}
        
        You are a marketing analyst. Provide a concise summary of the top YouTube channels 
        found for the query "{query_context}".
        
        Highlight 2-3 standout channels and explain why they're good matches.
        Keep it under 200 words.
        
        Channel data:
        {chr(10).join(channels_text)}
        
        Focus on:
        1. Overall quality of results
        2. Standout channels and why
        3. Potential collaboration opportunities
        """
    
    def _build_outreach_prompt(
        self,
        channel_name: str,
        campaign_context: str,
        language: str,
        tone: str
    ) -> str:
        """Build prompt for outreach email generation."""
        lang_instructions = {
            "en": "Write in English.",
            "es": "Escribe en español."
        }.get(language, "Write in English.")
        
        tone_instructions = {
            "professional": "Use a professional but warm tone.",
            "casual": "Use a friendly, casual tone.",
            "enthusiastic": "Use an enthusiastic, energetic tone."
        }.get(tone, "Use a professional tone.")
        
        return f"""
        {lang_instructions} {tone_instructions}
        
        Write a short outreach email (max 150 words) to the YouTube creator "{channel_name}".
        
        Context: {campaign_context}
        
        Requirements:
        - Address them by their channel name
        - Mention something specific about their content (inferred from name)
        - Express interest in collaboration
        - Include a clear call to action
        - Be respectful of their time
        - No subject line needed
        
        Make it personal and genuine, not generic.
        """
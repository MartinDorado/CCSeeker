"""
gemini_api.py - Gemini AI API wrapper functions

Pure functions for interacting with the Gemini AI API.
These functions are Streamlit-agnostic and can be unit tested with mocked API clients.

Key design decisions:
- Functions accept model as a parameter (no global state)
- Callbacks are used for API tracking instead of direct session_state access
- Results are returned as data, not rendered directly
"""

import re
import pandas as pd
from typing import Callable
from dataclasses import dataclass, field


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class OutreachDraft:
    """A single outreach email draft."""
    channel_title: str
    draft_text: str


@dataclass
class SummaryResult:
    """Result from generating a summary."""
    text: str
    error: str | None = None


# ============================================================================
# AI RELEVANCE SCORING
# ============================================================================

def _build_relevance_prompt(channel_data: dict, query: str) -> str:
    """Build the enriched relevance prompt, capped at ~3 KB."""
    channel_title = channel_data.get("channel_title", "")
    video_titles = channel_data.get("video_titles", [])
    description = (channel_data.get("channel_description") or "").strip()
    topic_categories = channel_data.get("topic_categories") or []
    channel_keywords = channel_data.get("channel_keywords") or []
    video_descriptions = channel_data.get("video_descriptions") or []

    # Optional channel-level sections (omitted when empty)
    desc_line = f"\n- Description: {description[:400]}" if description else ""
    topics_line = (
        f"\n- YouTube topic categories: {', '.join(topic_categories)}"
        if topic_categories else ""
    )
    keywords_line = (
        f"\n- Channel keywords: {', '.join(channel_keywords[:10])}"
        if channel_keywords else ""
    )

    # Video lines: title + first description excerpt when available
    video_lines = []
    for i, title in enumerate(video_titles[:10]):
        desc_excerpt = ""
        if i < len(video_descriptions) and video_descriptions[i]:
            desc_excerpt = f" — {str(video_descriptions[i])[:120]}"
        video_lines.append(f"- {title}{desc_excerpt}")
    videos_str = "\n".join(video_lines) if video_lines else "(no videos)"

    prompt = (
        f'Act as a YouTube content analyst. Evaluate how relevant this channel is to the search query.\n\n'
        f'**Search Query:** "{query}"\n\n'
        f'**Channel:**\n'
        f'- Name: {channel_title}'
        f'{desc_line}{topics_line}{keywords_line}\n\n'
        f'**Recent Videos (titles + description excerpt):**\n'
        f'{videos_str}\n\n'
        f'**Instructions:**\n'
        f'Based on all the above, rate the channel\'s relevance 0–10 '
        f'(0 = completely irrelevant, 10 = perfect match). '
        f'Respond with a single integer only.\n\n'
        f'**Relevance Score (0-10):**'
    )

    # Defensive cap: keep prompt under ~3 KB by trimming description first
    if len(prompt) > 3000:
        description = description[:200]
        desc_line = f"\n- Description: {description}" if description else ""
        prompt = (
            f'Act as a YouTube content analyst. Evaluate how relevant this channel is to the search query.\n\n'
            f'**Search Query:** "{query}"\n\n'
            f'**Channel:**\n'
            f'- Name: {channel_title}'
            f'{desc_line}{topics_line}{keywords_line}\n\n'
            f'**Recent Videos (titles + description excerpt):**\n'
            f'{videos_str}\n\n'
            f'**Instructions:**\n'
            f'Based on all the above, rate the channel\'s relevance 0–10 '
            f'(0 = completely irrelevant, 10 = perfect match). '
            f'Respond with a single integer only.\n\n'
            f'**Relevance Score (0-10):**'
        )

    return prompt


def generate_ai_relevance_score(
    model,
    channel_data: dict,
    query: str,
    on_api_call: Callable[[str], None] | None = None,
) -> float:
    """
    Uses a Gemini model to score channel relevance using enriched channel signals.

    Args:
        model: An initialized Gemini model instance.
        channel_data: Dict with 'channel_title', 'video_titles' (required), and
            optionally 'channel_description', 'topic_categories', 'channel_keywords',
            'video_descriptions' (list of first-120-char excerpts, one per video).
        query: The user's original search query.
        on_api_call: Optional callback for tracking API calls.

    Returns:
        A relevance score between 0.0 and 1.0, or 0.0 on failure.

    Notes:
        - Processes up to 10 video titles (+ description excerpts when available)
        - Returns 0.0 if no video titles provided or on API error
        - Gracefully omits enriched sections when optional fields are missing
    """
    if not channel_data.get("video_titles"):
        return 0.0

    if on_api_call:
        on_api_call('gemini_relevance')

    prompt = _build_relevance_prompt(channel_data, query)

    try:
        response = model.generate_content(prompt)
        match = re.search(r'\d+', response.text)
        if match:
            score = int(match.group(0))
            return min(10, max(0, score)) / 10.0
        return 0.0
    except Exception:
        return 0.0


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_summary(
    model,
    df_results: pd.DataFrame,
    query: str,
    seed_channel_name: str | None = None,
    on_api_call: Callable[[str], None] | None = None,
) -> SummaryResult:
    """
    Generate a summary of the top YouTube channels using Gemini.

    Args:
        model: An initialized Gemini model instance.
        df_results: DataFrame with search results (must have channel_title, subscribers, etc.)
        query: The original search query.
        seed_channel_name: If provided, indicates seed-based search mode.
        on_api_call: Optional callback for tracking API calls.

    Returns:
        SummaryResult with text or error message.

    Notes:
        - Detects search mode based on seed_channel_name parameter
        - Processes top 5 results only
    """
    try:
        if on_api_call:
            on_api_call('gemini_summary')

        top_5_df = df_results.head(5)

        # Detect search mode
        is_seed_based = seed_channel_name is not None and 'similarity_score' in df_results.columns

        data_string = ""
        for _, row in top_5_df.iterrows():
            data_string += f"- Channel: {row['channel_title']}\n"
            data_string += f"  - Subscribers: {row['subscribers']:,}\n"
            data_string += f"  - Country: {row['country']}\n"

            if is_seed_based:
                # Seed-based mode: show BOTH similarity and relevance
                similarity_score = row.get('similarity_score', 'N/A')
                relevance_score = row.get('relevance_score', 'N/A')

                # Extract match reasons from similarity dict if available
                if 'similarity' in row and isinstance(row['similarity'], dict):
                    reasons = row['similarity'].get('match_reasons', [])
                    reasons_text = ', '.join(reasons[:2]) if reasons else 'See detailed analysis'
                else:
                    reasons_text = 'N/A'

                data_string += f"  - Similarity Score: {similarity_score}/100\n"
                data_string += f"  - Why Similar: {reasons_text}\n"
                data_string += f"  - Topic Focus (Relevance): {relevance_score}\n"
            else:
                # Keyword mode: show relevance score
                data_string += f"  - Relevance Score: {row['relevance_score']}\n"

            data_string += f"  - Avg. Engagement Rate: {row['engagement_rate']}\n\n"

        # Adjust prompt based on search mode
        if is_seed_based:
            prompt = f"""
You are an expert marketing analyst. Provide a concise summary of the top YouTube channels similar to "{seed_channel_name}".

These channels were found using similarity analysis based on content topics, tags, audience size, and engagement patterns.

For each channel, consider:
- **Similarity Score (0-100)**: Overall match to the seed channel
- **Topic Focus (Relevance %)**: How focused they are on the auto-generated keywords from the seed
- **Engagement Rate**: How interactive their audience is

Base your analysis ONLY on the data below and highlight 2-3 standout matches and why they're good fits for collaboration.

Data:
{data_string}
"""
        else:
            prompt = f"""
You are an expert marketing analyst. Provide a concise summary of the top YouTube channels for the query "{query}".

Base your analysis ONLY on the data below and highlight 2-3 standout channels and why.

Data:
{data_string}
"""

        response = model.generate_content(prompt)
        return SummaryResult(text=response.text)

    except Exception as e:
        return SummaryResult(text="", error=f"An error occurred while generating the summary: {e}")


# ============================================================================
# OUTREACH DRAFT GENERATION
# ============================================================================

def generate_outreach_drafts(
    model,
    top_channels_df: pd.DataFrame,
    original_query: str,
    limit: int = 3,
    retries: int = 2,
    language: str = "en",
    on_api_call: Callable[[str], None] | None = None,
) -> list[OutreachDraft]:
    """
    Generate short, friendly outreach email drafts for the top N channels using Gemini.

    Args:
        model: An initialized Gemini model instance.
        top_channels_df: DataFrame with 'channel_title' column.
        original_query: The original search query to reference.
        limit: Number of channels to process (default 3).
        retries: How many times to retry a failed API call (default 2).
        language: "en" for English or "es" for Spanish (default "en").
        on_api_call: Optional callback for tracking API calls.

    Returns:
        List of OutreachDraft objects with channel_title and draft_text.

    Notes:
        - Handles empty DataFrames gracefully
        - Retries on API failures
        - Supports English and Spanish
    """
    results: list[OutreachDraft] = []

    if top_channels_df is None or top_channels_df.empty:
        return results
    if 'channel_title' not in top_channels_df.columns:
        return results

    df = (
        top_channels_df[['channel_title']]
        .dropna(subset=['channel_title'])
        .copy()
    )
    df['channel_title'] = df['channel_title'].astype(str).str.strip()
    df = df[df['channel_title'] != ""].drop_duplicates(subset=['channel_title'])
    df = df.head(max(0, int(limit)))

    oq = (original_query or "").strip() or "my audience's interests"

    # Language instruction
    lang = (language or "en").lower()
    if lang.startswith("es"):
        lang_line = "Write the email in Spanish. Usa un espanol claro, neutro y profesional."
    else:
        lang_line = "Write the email in English in a clear, professional yet friendly tone."

    for _, row in df.iterrows():
        channel_name = row['channel_title']

        if on_api_call:
            on_api_call('gemini_outreach')

        prompt = f"""
Act as a marketing professional. Your task is to write a short, friendly, and professional outreach email to a YouTube creator.

**Instructions:**
- The tone should be respectful and concise.
- Mention the creator's channel name specifically.
- Reference the topic of my original search, which was "{oq}".
- The goal is to express interest in a potential collaboration.
- Do not use overly corporate language.
- {lang_line}

**Creator Channel Name:** {channel_name}

**Email Draft:**
"""

        draft_text = ""
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = model.generate_content(prompt)
                draft_text = (getattr(resp, "text", None) or str(resp)).strip()
                if draft_text.startswith("```"):
                    draft_text = draft_text.strip("`").strip()
                break
            except Exception as e:
                last_err = e
                continue

        if not draft_text and last_err:
            draft_text = f"(Error generating draft for '{channel_name}': {type(last_err).__name__}: {last_err})"

        results.append(OutreachDraft(
            channel_title=channel_name,
            draft_text=draft_text
        ))

    return results

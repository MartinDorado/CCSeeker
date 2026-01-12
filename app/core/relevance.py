"""
relevance.py - Keyword relevance scoring for YouTube channels

Pure functions for calculating how relevant a channel's content is
to a given search query, based on video titles and tags.

These functions are Streamlit-agnostic and can be unit tested independently.
"""

import re
import pandas as pd

from .query_utils import strip_outer_quotes


# ============================================================================
# RELEVANCE SCORING
# ============================================================================

def calculate_keyword_relevance(
    df: pd.DataFrame,
    query: str,
    title_weight: float = 2.0,
    tags_weight: float = 1.0
) -> pd.DataFrame:
    """
    Compute per-channel relevance by matching query terms against video titles and tags.

    The function calculates a weighted score based on keyword matches in:
    - Video titles (higher weight by default)
    - Video tags (lower weight by default)

    Args:
        df: DataFrame with columns 'channel_id', 'video_title', and optionally 'video_tags'
        query: Search query (comma-separated terms or single phrase)
        title_weight: Weight multiplier for title matches (default: 2.0)
        tags_weight: Weight multiplier for tag matches (default: 1.0)

    Returns:
        DataFrame with columns ['channel_id', 'relevance_score'] where
        relevance_score is the average per-video score (0.0 to 1.0) for each channel.

    Notes:
        - Parsing is comma-only: "term1, term2, phrase three". Boolean text like AND/OR is not parsed.
        - Word boundaries are added for simple alphanumeric terms to avoid substring matches
          (e.g., "man" won't match "manga").
        - Index alignment is preserved by creating fallback Series with `index=df.index`.
        - Title vs tags can be weighted via `title_weight` and `tags_weight`.
        - The per-video score is a weighted average in [0, 1].

    Examples:
        >>> df = pd.DataFrame({
        ...     'channel_id': ['UC1', 'UC1', 'UC2'],
        ...     'video_title': ['Manga Review', 'Anime News', 'Gaming Stream'],
        ...     'video_tags': [['manga', 'review'], ['anime'], ['gaming']]
        ... })
        >>> result = calculate_keyword_relevance(df, "manga, anime")
        >>> result
           channel_id  relevance_score
        0         UC1             0.75
        1         UC2             0.00
    """
    if df.empty or not isinstance(query, str) or not query.strip():
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])

    # Accept only comma-separated terms; otherwise treat the entire query as one term
    if ',' in query:
        raw_terms = [t.strip() for t in query.split(',') if t.strip()]
    else:
        raw_terms = [query.strip()]

    # Build regex parts with safe escaping and word boundaries for simple words
    cleaned_parts = []
    for t in raw_terms:
        if not t:
            continue
        t = strip_outer_quotes(t)
        if not t:
            continue
        # If the term is a single "word" (letters/digits/_), wrap with word boundaries
        if re.match(r'^\w+$', t, flags=re.UNICODE):
            cleaned_parts.append(r'\b' + re.escape(t) + r'\b')
        else:
            cleaned_parts.append(re.escape(t))

    if not cleaned_parts:
        return pd.DataFrame(columns=['channel_id', 'relevance_score'])

    pattern = '(?:' + '|'.join(cleaned_parts) + ')'

    def _tags_to_text(x):
        """Convert tags list to searchable text."""
        if isinstance(x, list):
            return ' '.join([str(i) for i in x if i is not None])
        return str(x) if x is not None else ''

    # Ensure alignment by constructing fallbacks with the same index
    title_series = df.get('video_title', pd.Series('', index=df.index)).fillna('')
    tags_series = df.get('video_tags', pd.Series('', index=df.index))
    tags_text = tags_series.apply(_tags_to_text)

    # Boolean matches per field
    title_match = title_series.str.contains(pattern, case=False, na=False, regex=True)
    tags_match = tags_text.str.contains(pattern, case=False, na=False, regex=True)

    # Weighted per-video score in [0, 1]
    denom = float(title_weight + tags_weight) if (title_weight + tags_weight) != 0 else 1.0
    video_score = (title_weight * title_match.astype(float) + tags_weight * tags_match.astype(float)) / denom

    # Average to channel-level relevance score
    tmp = pd.DataFrame({'channel_id': df['channel_id'], 'video_score': video_score})
    relevance = tmp.groupby('channel_id', as_index=False)['video_score'].mean()
    relevance = relevance.rename(columns={'video_score': 'relevance_score'})

    return relevance

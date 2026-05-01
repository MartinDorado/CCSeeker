"""
Canned transcript fixtures for unit tests.

Five blobs covering the main result statuses and content types.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.transcription import TranscriptResult

# 1. English educational transcript (status=ok)
TRANSCRIPT_EN_EDUCATIONAL = TranscriptResult(
    video_id="vid_en_edu_1",
    channel_id="UC_test_channel",
    language="en",
    text=(
        "Welcome to today's lesson on machine learning. "
        "We'll explore gradient descent and how neural networks learn from data. "
        "First, let's understand what a loss function is and why we want to minimize it. "
        "The gradient tells us the direction of steepest ascent, so we move in the opposite direction. "
        "This iterative process is called training and it adjusts the model weights. "
        "By the end of this tutorial you'll be able to implement your own simple neural network."
    ),
    status="ok",
)

# 2. Spanish vlog transcript (status=ok)
TRANSCRIPT_ES_VLOG = TranscriptResult(
    video_id="vid_es_vlog_1",
    channel_id="UC_test_channel",
    language="es",
    text=(
        "Hola a todos, bienvenidos de vuelta al canal. "
        "Hoy les voy a contar mi experiencia viajando por toda España en tren. "
        "Empezamos en Madrid y fuimos hacia Barcelona, pasando por Valencia. "
        "La comida local fue increíble, especialmente la paella valenciana. "
        "En el próximo video les muestro los mejores lugares para visitar en cada ciudad."
    ),
    status="ok",
)

# 3. Empty transcript (status=ok but empty text — simulates very short video)
TRANSCRIPT_EMPTY = TranscriptResult(
    video_id="vid_empty_1",
    channel_id="UC_test_channel",
    language=None,
    text="",
    status="ok",
)

# 4. Disabled captions (status=disabled)
TRANSCRIPT_DISABLED = TranscriptResult(
    video_id="vid_disabled_1",
    channel_id="UC_test_channel",
    language=None,
    text="",
    status="disabled",
    error_message="TranscriptsDisabled: Transcripts are disabled for this video.",
)

# 5. Short fragment — simulates a YouTube Shorts-style video
TRANSCRIPT_SHORTS_FRAGMENT = TranscriptResult(
    video_id="vid_shorts_1",
    channel_id="UC_test_channel",
    language="en",
    text="Quick tip: always use list comprehensions in Python for cleaner code!",
    status="ok",
)

# Convenience: all fixtures as a list
ALL_FIXTURES = [
    TRANSCRIPT_EN_EDUCATIONAL,
    TRANSCRIPT_ES_VLOG,
    TRANSCRIPT_EMPTY,
    TRANSCRIPT_DISABLED,
    TRANSCRIPT_SHORTS_FRAGMENT,
]

# Convenience: mapping video_id → TranscriptResult (for FakeTranscriptFetcher)
FAKE_RESPONSES: dict[str, TranscriptResult] = {
    t.video_id: t for t in ALL_FIXTURES
}

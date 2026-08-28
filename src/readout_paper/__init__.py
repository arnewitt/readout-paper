"""Local text-to-speech for reading papers out loud."""

from .tts import (
    DEFAULT_VOICE,
    REPO_ID,
    SAMPLE_RATE,
    VOICES,
    synth_chunks,
    synth_to_wav,
    wav_duration,
)

__all__ = [
    "DEFAULT_VOICE",
    "REPO_ID",
    "SAMPLE_RATE",
    "VOICES",
    "synth_chunks",
    "synth_to_wav",
    "wav_duration",
]

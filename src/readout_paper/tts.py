"""Local text-to-speech with Kokoro-82M.

Thin wrapper around kokoro's KPipeline: text in, 24 kHz mono WAV out.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from kokoro import KPipeline

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

# A few known-good voices per language code. Not exhaustive -- the full set
# lives under voices/ in the hexgrad/Kokoro-82M repo.
VOICES: dict[str, tuple[str, ...]] = {
    "a": ("af_heart", "af_bella", "af_nicole", "am_michael", "am_puck"),
    "b": ("bf_emma", "bf_isabella", "bm_george", "bm_fable"),
    "e": ("ef_dora", "em_alex"),
    "f": ("ff_siwis",),
    "h": ("hf_alpha", "hm_omega"),
    "i": ("if_sara", "im_nicola"),
    "j": ("jf_alpha", "jm_kumo"),
    "p": ("pf_dora", "pm_alex"),
    "z": ("zf_xiaobei", "zm_yunjian"),
}

LANG_NAMES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

_pipelines: dict[tuple[str, str | None], "KPipeline"] = {}


def get_pipeline(lang_code: str, device: str | None = None) -> "KPipeline":
    """Return a cached KPipeline for this language, loading the model on first use."""
    key = (lang_code, device)
    if key not in _pipelines:
        # Imported lazily: kokoro pulls in torch and spacy, which is slow.
        from kokoro import KPipeline

        _pipelines[key] = KPipeline(lang_code=lang_code, repo_id=REPO_ID, device=device)
    return _pipelines[key]


def synth_chunks(
    text: str,
    voice: str = DEFAULT_VOICE,
    lang_code: str | None = None,
    speed: float = 1.0,
    device: str | None = None,
) -> Iterator[np.ndarray]:
    """Yield float32 mono audio chunks at SAMPLE_RATE, one per phonemized segment."""
    pipeline = get_pipeline(lang_code or voice[0], device)
    for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        if result.audio is None:
            continue
        yield result.audio.detach().cpu().numpy()


def synth_to_wav(
    text: str,
    out_path: str | Path,
    voice: str = DEFAULT_VOICE,
    lang_code: str | None = None,
    speed: float = 1.0,
    device: str | None = None,
) -> Path:
    """Synthesize `text` and write it to `out_path` as a 16-bit mono WAV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        for chunk in synth_chunks(text, voice, lang_code, speed, device):
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            wav.writeframes(pcm.tobytes())
            frames += len(pcm)

    if frames == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Kokoro produced no audio for this input -- it may be empty, "
            "punctuation only, or not in the selected language."
        )
    return out_path


def wav_duration(path: str | Path) -> float:
    """Duration of a WAV file in seconds."""
    with wave.open(str(path)) as wav:
        return wav.getnframes() / wav.getframerate()

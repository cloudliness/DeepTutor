"""Voice services — text-to-speech and speech-to-text.

Public facade used by the API router and the config test runner. Config is
resolved from the model catalog (``services.tts`` / ``services.stt``) exactly
like embedding/LLM, so voice providers are configured through the same
Settings catalog UI.
"""

from __future__ import annotations

import logging
from typing import Any

from deeptutor.services.voice.adapters import get_stt_adapter, get_tts_adapter
from deeptutor.services.voice.audio import concat_speech_audio
from deeptutor.services.voice.base import (
    BaseTTSAdapter,
    TranscriptCue,
    VoiceProviderError,
    strip_markdown_for_speech,
    synthesize_with_timeout,
)
from deeptutor.services.voice.config import STTConfig, TTSConfig
from deeptutor.services.voice.speech_chunks import (
    chunk_limit,
    chunk_speech_text,
    halve_speech_text,
    soften_for_speech,
)

logger = logging.getLogger(__name__)


def _math_speak_from_settings() -> bool:
    """Read the Math Speak UI preference; default on if settings are unavailable."""
    try:
        from deeptutor.services.settings.interface_settings import get_ui_settings

        return bool(get_ui_settings().get("voice_math_speak", True))
    except Exception:
        return True


async def synthesize_speech(
    text: str,
    *,
    catalog: dict[str, Any] | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    strip_markdown: bool = True,
    math_speak: bool | None = None,
) -> tuple[bytes, str]:
    """Synthesize ``text`` using the active TTS catalog selection.

    Returns ``(audio_bytes, content_type)``. ``voice`` / ``response_format``
    override the catalog defaults for this call. ``math_speak`` defaults to
    the Settings › Text-to-Speech preference (on).
    """
    from deeptutor.services.config.provider_runtime import resolve_tts_runtime_config

    config = resolve_tts_runtime_config(catalog=catalog)
    if voice:
        config.voice = voice
    if response_format:
        config.response_format = response_format
    speak_math = _math_speak_from_settings() if math_speak is None else math_speak
    # Do not truncate here. A cap used to drop the rest of a long answer, so
    # read-aloud stopped after the first window. Pieces are synthesized below.
    prepared = (
        strip_markdown_for_speech(text, max_chars=0, math_speak=speak_math)
        if strip_markdown
        else text.strip()
    )
    prepared = soften_for_speech(prepared)
    if not prepared:
        raise VoiceProviderError("Nothing to speak after cleaning the text.")
    if config.adapter == "edge":
        from deeptutor.services.voice.adapters.edge import edge_voice_for_text

        # One voice for the whole passage, so a short later piece does not
        # switch language halfway through.
        config.voice = edge_voice_for_text(prepared, config.voice)
    chunks = chunk_speech_text(
        prepared,
        max_chars=chunk_limit(config.max_input_chars, config.adapter),
    )
    if not chunks:
        raise VoiceProviderError("Nothing to speak after cleaning the text.")
    if len(chunks) > 1:
        logger.debug(
            "Synthesizing speech in %s chunks (%s characters)",
            len(chunks),
            len(prepared),
        )
    adapter = get_tts_adapter(config.adapter)
    parts: list[bytes] = []
    content_type = "application/octet-stream"
    for chunk in chunks:
        audio, content_type = await _synthesize_piece(adapter, chunk, config)
        parts.append(audio)
    return concat_speech_audio(parts, content_type), content_type


def _edge_returned_no_audio(exc: VoiceProviderError) -> bool:
    return "no audio" in str(exc).lower()


async def _synthesize_piece(
    adapter: BaseTTSAdapter,
    text: str,
    config: TTSConfig,
) -> tuple[bytes, str]:
    """Synthesize one piece, splitting an Edge no-audio failure once."""
    try:
        audio, content_type = await synthesize_with_timeout(adapter, text, config)
    except VoiceProviderError as exc:
        if config.adapter != "edge" or not _edge_returned_no_audio(exc):
            raise
        left, right = halve_speech_text(text)
        if not right:
            raise
        logger.debug("Edge returned no audio; retrying the passage in two shorter pieces")
        first, content_type = await synthesize_with_timeout(adapter, left, config)
        second, second_type = await synthesize_with_timeout(adapter, right, config)
        if not first or not second:
            raise VoiceProviderError("The speech provider returned no audio.") from exc
        merged = concat_speech_audio([first, second], content_type or second_type)
        return merged, content_type or second_type
    if not audio:
        raise VoiceProviderError("The speech provider returned no audio.")
    return audio, content_type


async def transcribe_audio(
    audio: bytes,
    *,
    catalog: dict[str, Any] | None = None,
    filename: str = "audio.webm",
    content_type: str = "application/octet-stream",
    language: str | None = None,
) -> str:
    """Transcribe ``audio`` using the active STT catalog selection."""
    from deeptutor.services.config.provider_runtime import resolve_stt_runtime_config

    config = resolve_stt_runtime_config(catalog=catalog)
    if language:
        config.language = language
    adapter = get_stt_adapter(config.adapter)
    return await adapter.transcribe(audio, config, filename=filename, content_type=content_type)


async def transcribe_audio_cues(
    audio: bytes,
    *,
    catalog: dict[str, Any] | None = None,
    filename: str = "audio.webm",
    content_type: str = "application/octet-stream",
    language: str | None = None,
) -> list[TranscriptCue]:
    """Transcribe ``audio`` into timed cues using the active STT selection.

    Providers that cannot report timings answer with a single untimed cue, so
    the caller decides how to place it rather than losing the transcript.
    """
    from deeptutor.services.config.provider_runtime import resolve_stt_runtime_config

    config = resolve_stt_runtime_config(catalog=catalog)
    if language:
        config.language = language
    adapter = get_stt_adapter(config.adapter)
    return await adapter.transcribe_cues(
        audio, config, filename=filename, content_type=content_type
    )


__all__ = [
    "TranscriptCue",
    "VoiceProviderError",
    "TTSConfig",
    "STTConfig",
    "synthesize_speech",
    "transcribe_audio",
    "transcribe_audio_cues",
    "strip_markdown_for_speech",
]

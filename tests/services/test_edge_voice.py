"""Edge read-aloud speech: keyless catalog resolution and the edge-tts call, offline."""

import sys
import types

import pytest

from deeptutor.services.config.provider_runtime import TTS_PROVIDERS, resolve_tts_runtime_config
from deeptutor.services.voice import synthesize_speech
from deeptutor.services.voice.adapters.edge import edge_rate
from deeptutor.services.voice.base import VoiceProviderError


def catalog(**model):
    """The connection Settings stores for Edge: no API key."""
    return {
        "version": 1,
        "connections": [{"id": "speech", "provider": "edge"}],
        "services": {
            "tts": {
                "active_profile_id": "p",
                "active_model_id": "m",
                "profiles": [
                    {
                        "id": "p",
                        "provider_ref": {"connection_id": "speech", "binding": "edge"},
                        "models": [{"id": "m", "model": "edge-tts", **model}],
                    }
                ],
            }
        },
    }


@pytest.fixture
def fake_edge(monkeypatch):
    calls = []

    class Communicate:
        def __init__(self, text, voice, *, rate="+0%"):
            calls.append({"text": text, "voice": voice, "rate": rate})

        async def stream(self):
            yield {"type": "SentenceBoundary", "offset": 0, "duration": 1, "text": "hi"}
            yield {"type": "audio", "data": b"ID3"}
            yield {"type": "audio", "data": b"abc"}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    return calls


def test_edge_resolves_without_an_api_key() -> None:
    resolved = resolve_tts_runtime_config(catalog())
    assert resolved.provider_name == resolved.adapter == "edge"
    assert resolved.model == TTS_PROVIDERS["edge"].default_model
    assert resolved.response_format == "mp3"


@pytest.mark.asyncio
async def test_edge_speaks_with_the_configured_voice_and_speed(fake_edge) -> None:
    audio = await synthesize_speech(
        "Hello", catalog=catalog(voice="en-US-AvaNeural", speed=1.25), math_speak=False
    )
    assert audio == (b"ID3abc", "audio/mpeg")
    assert fake_edge == [{"text": "Hello", "voice": "en-US-AvaNeural", "rate": "+25%"}]


@pytest.mark.asyncio
async def test_edge_defaults_to_a_chinese_voice(fake_edge) -> None:
    await synthesize_speech("你好", catalog=catalog(), math_speak=False)
    assert fake_edge[0]["voice"] == "zh-CN-XiaoxiaoNeural"


def test_edge_rate_bounds() -> None:
    assert edge_rate(None) == "+0%"
    assert edge_rate(0.8) == "-20%"
    with pytest.raises(VoiceProviderError):
        edge_rate(3)


@pytest.mark.asyncio
async def test_empty_voice_follows_the_text_language(fake_edge) -> None:
    await synthesize_speech("The last ghost shows him the future.", catalog=catalog(), math_speak=False)
    assert fake_edge[0]["voice"] == "en-US-AriaNeural"


@pytest.mark.asyncio
async def test_long_markdown_is_chunked_and_nothing_is_dropped(fake_edge) -> None:
    paragraphs = [
        f"## Stave {index}\n\nThe ghost speaks **plainly** — about paragraph {index}, "
        "which is long enough to need its own breath."
        for index in range(6)
    ]
    audio = await synthesize_speech("\n\n".join(paragraphs), catalog=catalog(), math_speak=False)
    assert len(fake_edge) > 1
    assert all(call["voice"] == "en-US-AriaNeural" for call in fake_edge)
    assert all(len(call["text"]) <= 480 for call in fake_edge)
    spoken = " ".join(call["text"] for call in fake_edge)
    assert "Stave 0" in spoken and "Stave 5" in spoken
    assert "#" not in spoken and "**" not in spoken
    assert audio[0] == b"ID3abc" * len(fake_edge)
    assert audio[1] == "audio/mpeg"


@pytest.mark.asyncio
async def test_edge_no_audio_retries_the_passage_in_shorter_pieces(monkeypatch) -> None:
    calls = []

    class NoAudioReceived(Exception):
        pass

    class Communicate:
        def __init__(self, text, voice, *, rate="+0%"):
            calls.append(text)
            self.text = text

        async def stream(self):
            if len(self.text) > 80:
                raise NoAudioReceived("No audio was received.")
            yield {"type": "audio", "data": self.text[:1].encode()}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    text = "This opening clause is deliberately long enough to fail. The ending clause still has to be spoken."
    audio, content_type = await synthesize_speech(text, catalog=catalog(), math_speak=False)
    assert content_type == "audio/mpeg"
    assert audio
    assert len(calls) >= 3  # the long attempt, then both halves
    assert all(len(call) <= 80 for call in calls[1:])


@pytest.mark.asyncio
async def test_edge_value_error_is_a_provider_error_not_a_validation_failure(monkeypatch) -> None:
    class Communicate:
        def __init__(self, text, voice, *, rate="+0%"):
            raise ValueError("Maximum byte length is too small or invalid text structure near '&'")

        async def stream(self):
            if False:
                yield {}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    with pytest.raises(VoiceProviderError, match="invalid text structure"):
        await synthesize_speech("Hello", catalog=catalog(), math_speak=False)

"""Long read-aloud text is cleaned and split instead of truncated."""

import io
import wave

from deeptutor.services.voice.audio import concat_speech_audio
from deeptutor.services.voice.base import strip_markdown_for_speech
from deeptutor.services.voice.speech_chunks import (
    EDGE_SPEECH_CHUNK_CHARS,
    chunk_limit,
    chunk_speech_text,
    halve_speech_text,
    soften_for_speech,
)


def _words(text: str) -> list[str]:
    return [word.strip(".,;:!?。！？") for word in text.split() if word.strip(".,;:!?。！？")]


def test_chunks_cover_the_whole_passage_without_passing_the_cap() -> None:
    sentences = [f"Sentence {index} explains the next idea." for index in range(30)]
    text = "\n\n".join(sentences)
    chunks = chunk_speech_text(text, max_chars=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    spoken = " ".join(chunks)
    assert "Sentence 0" in spoken and "Sentence 29" in spoken


def test_chunks_prefer_sentence_boundaries() -> None:
    text = "Alpha is first. Beta is second. Gamma is third and a bit longer."
    chunks = chunk_speech_text(text, max_chars=40)
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert not any(chunk.endswith(" fir") or " firs" == chunk[-5:] for chunk in chunks)
    assert " ".join(_words(" ".join(chunks))) == " ".join(_words(text))


def test_chinese_sentences_split_without_spaces() -> None:
    text = "第一句讲完了。第二句也讲完了。第三句继续说下去不换行。"
    chunks = chunk_speech_text(text, max_chars=12)
    assert len(chunks) > 1
    assert all(len(chunk) <= 12 for chunk in chunks)
    assert "第一句" in chunks[0]
    assert "第三句" in "".join(chunks)


def test_a_single_long_token_is_cut_to_the_cap() -> None:
    chunks = chunk_speech_text("x" * 50, max_chars=16)
    assert chunks
    assert all(len(chunk) <= 16 for chunk in chunks)
    assert "".join(chunks) == "x" * 50


def test_edge_limit_is_paragraph_sized_and_other_providers_keep_theirs() -> None:
    assert chunk_limit(4096, "edge") == EDGE_SPEECH_CHUNK_CHARS
    assert chunk_limit(200, "edge") == 200
    assert chunk_limit(4096, "openai_compat") == 4096
    assert chunk_limit(0, "openai_compat") == 4096


def test_soften_turns_dashes_and_control_characters_into_pauses() -> None:
    assert soften_for_speech("Go on \u2014 what next?") == "Go on, what next?"
    assert "\x0b" not in soften_for_speech("Keep\x0bgoing")


def test_markdown_message_is_not_truncated_before_chunking() -> None:
    paragraphs = [
        f"## Section {index}\n\nThis is **paragraph {index}** with an em dash — and a list:\n- one\n- two"
        for index in range(8)
    ]
    prepared = soften_for_speech(strip_markdown_for_speech("\n\n".join(paragraphs), max_chars=0))
    chunks = chunk_speech_text(prepared, max_chars=EDGE_SPEECH_CHUNK_CHARS)
    spoken = "\n".join(chunks)
    assert len(prepared) > EDGE_SPEECH_CHUNK_CHARS
    assert all(len(chunk) <= EDGE_SPEECH_CHUNK_CHARS for chunk in chunks)
    assert "#" not in spoken and "**" not in spoken and "—" not in spoken
    assert "Section 0" in spoken and "Section 7" in spoken
    assert "paragraph 7" in spoken


def test_halve_keeps_both_sides() -> None:
    left, right = halve_speech_text("Alpha is first. Beta is second. Gamma is third.")
    assert left and right
    assert "Alpha" in left
    assert "Gamma" in right


def test_wav_chunks_concatenate_into_one_playable_file() -> None:
    def wav(frames: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(frames)
        return buffer.getvalue()

    first, second = b"\x00\x01" * 8, b"\x02\x03" * 8
    joined = concat_speech_audio([wav(first), wav(second)], "audio/wav")
    with wave.open(io.BytesIO(joined), "rb") as handle:
        assert handle.getnframes() == 16
        assert handle.readframes(handle.getnframes()) == first + second


def test_mp3_chunks_concatenate_in_order() -> None:
    assert concat_speech_audio([b"aaa", b"bbb"], "audio/mpeg") == b"aaabbb"

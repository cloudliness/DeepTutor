"""Split speakable text into provider-sized pieces without dropping the rest.

Full-message read-aloud used to cut the cleaned text at the provider cap, so
everything after the first window was silent. Edge's service also returns no
audio or sits until the request times out when one SSML payload is long.
Callers synthesize these pieces in order and concatenate the audio.
"""

from __future__ import annotations

import re

# Edge's websocket often returns no audio, or holds the socket until the
# request timeout, once a single payload gets long. Paragraph-sized pieces
# stay under that failure mode. Other providers keep their own cap.
EDGE_SPEECH_CHUNK_CHARS = 480

# English breaks on punctuation plus whitespace, so "3.14" stays intact.
# CJK punctuation often has no following space, so it breaks on its own.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])\s*")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,，、:：])\s*")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_DASH = re.compile(r"\s*[—–]\s*")


def soften_for_speech(text: str) -> str:
    """Drop characters that make a voice model fail or spell punctuation."""
    if not text:
        return ""
    out = _CONTROL.sub(" ", text)
    out = out.replace("\u00a0", " ").replace("\u200b", "")
    out = _DASH.sub(", ", out)
    out = _MULTI_SPACE.sub(" ", out)
    out = _MULTI_NL.sub("\n\n", out)
    return out.strip()


def chunk_limit(max_input_chars: int, adapter: str) -> int:
    """Largest piece one synthesis call should receive."""
    limit = max_input_chars if max_input_chars and max_input_chars > 0 else 4096
    if adapter == "edge":
        return max(1, min(limit, EDGE_SPEECH_CHUNK_CHARS))
    return max(1, limit)


def chunk_speech_text(text: str, *, max_chars: int) -> list[str]:
    """Pack ``text`` into pieces of at most ``max_chars``, in reading order.

    Prefers paragraph, then sentence, then clause, then whitespace. A single
    token longer than the cap is cut so a request is never over the limit.
    """
    cleaned = _CONTROL.sub(" ", text or "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
    if not cleaned:
        return []
    limit = max_chars if max_chars and max_chars > 0 else 1
    if len(cleaned) <= limit:
        return [cleaned]

    chunks: list[str] = []
    buffer = ""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    for paragraph in paragraphs:
        pieces = _pieces(paragraph, limit)
        for piece in pieces:
            if buffer and len(buffer) + 2 + len(piece) > limit:
                chunks.append(buffer)
                buffer = piece
            elif buffer:
                buffer = f"{buffer}\n\n{piece}"
            else:
                buffer = piece
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk.strip()]


def halve_speech_text(text: str) -> tuple[str, str]:
    """Split near the middle, on a sentence or space when one is close."""
    cleaned = (text or "").strip()
    if len(cleaned) < 2:
        return cleaned, ""
    midpoint = len(cleaned) // 2
    best_end = -1
    for marker in (". ", "! ", "? ", "。", "！", "？", "\n", " "):
        found = cleaned.rfind(marker, 0, midpoint + len(marker))
        if found >= len(cleaned) // 4:
            best_end = max(best_end, found + len(marker))
    if best_end < 0:
        best_end = midpoint
    left = cleaned[:best_end].strip()
    right = cleaned[best_end:].strip()
    if not left or not right:
        return cleaned, ""
    return left, right


def _pieces(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    if len(sentences) <= 1:
        clauses = [part.strip() for part in _CLAUSE_BOUNDARY.split(text) if part.strip()]
        if len(clauses) <= 1:
            return _hard_split(text, limit)
        return _pack(clauses, limit, separator=" ")
    return _pack(sentences, limit, separator=" ")


def _pack(parts: list[str], limit: int, *, separator: str) -> list[str]:
    packed: list[str] = []
    buffer = ""
    for part in parts:
        bits = _pieces(part, limit) if len(part) > limit else [part]
        for bit in bits:
            joiner = separator if buffer else ""
            if buffer and len(buffer) + len(joiner) + len(bit) > limit:
                packed.append(buffer)
                buffer = bit
            elif buffer:
                buffer = f"{buffer}{joiner}{bit}"
            else:
                buffer = bit
    if buffer:
        packed.append(buffer)
    return packed


def _hard_split(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    rest = text.strip()
    while rest:
        if len(rest) <= limit:
            parts.append(rest)
            break
        window = rest[:limit]
        cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit
        piece = rest[:cut].strip()
        rest = rest[cut:].strip()
        if piece:
            parts.append(piece)
        else:
            parts.append(rest[:limit])
            rest = rest[limit:].strip()
    return parts


__all__ = [
    "EDGE_SPEECH_CHUNK_CHARS",
    "chunk_limit",
    "chunk_speech_text",
    "halve_speech_text",
    "soften_for_speech",
]

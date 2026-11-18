/**
 * Break a message into read-aloud requests.
 *
 * The server cleans Markdown and splits anything still too long for the
 * provider. Doing a first split here lets playback start at the first
 * paragraph instead of waiting for the whole answer, and keeps fenced code
 * in one request so the server can drop it instead of speaking a fragment.
 */

const DEFAULT_SEGMENT_CHARS = 900

export function speechSegments(text: string, maxChars = DEFAULT_SEGMENT_CHARS): string[] {
  const source = text.replace(/\r\n/g, "\n").trim()
  if (!source) return []
  const limit = Math.max(1, maxChars)
  if (source.length <= limit) return [source]

  const segments: string[] = []
  let current = ""
  const flush = () => {
    const piece = current.trim()
    if (piece) segments.push(piece)
    current = ""
  }

  for (const block of splitBlocks(source)) {
    const pieces = block.startsWith("```") || block.length <= limit ? [block] : splitProse(block, limit)
    for (const piece of pieces) {
      if (current && current.length + 2 + piece.length > limit) flush()
      current = current ? `${current}\n\n${piece}` : piece
      if (current.length > limit) flush()
    }
  }
  flush()
  return segments.length ? segments : [source]
}

function splitBlocks(source: string): string[] {
  const blocks: string[] = []
  const fence = /```[\s\S]*?```/g
  let last = 0
  for (const match of source.matchAll(fence)) {
    const index = match.index ?? 0
    if (index > last) blocks.push(...proseBlocks(source.slice(last, index)))
    blocks.push(match[0])
    last = index + match[0].length
  }
  if (last < source.length) blocks.push(...proseBlocks(source.slice(last)))
  return blocks
}

function proseBlocks(source: string): string[] {
  return source
    .split(/\n{2,}/)
    .map(part => part.trim())
    .filter(Boolean)
}

function splitProse(text: string, maxChars: number): string[] {
  const sentences = text
    .split(/(?<=[.!?])\s+|(?<=[。！？])\s*/)
    .map(part => part.trim())
    .filter(Boolean)
  if (sentences.length <= 1) return hardSplit(text, maxChars)
  const packed: string[] = []
  let current = ""
  for (const sentence of sentences) {
    const bits = sentence.length > maxChars ? hardSplit(sentence, maxChars) : [sentence]
    for (const bit of bits) {
      if (current && current.length + 1 + bit.length > maxChars) {
        packed.push(current)
        current = bit
      } else {
        current = current ? `${current} ${bit}` : bit
      }
    }
  }
  if (current) packed.push(current)
  return packed
}

function hardSplit(text: string, maxChars: number): string[] {
  const parts: string[] = []
  let rest = text.trim()
  while (rest) {
    if (rest.length <= maxChars) {
      parts.push(rest)
      break
    }
    let cut = rest.lastIndexOf(" ", maxChars)
    if (cut < maxChars / 2) cut = maxChars
    const piece = rest.slice(0, cut).trim()
    rest = rest.slice(cut).trim()
    if (piece) parts.push(piece)
    else break
  }
  return parts
}

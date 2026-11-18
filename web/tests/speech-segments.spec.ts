import { expect, it } from 'vitest'
import { speechSegments } from '@/lib/speech-segments'

it('keeps a short selection as one request', () => {
  expect(speechSegments('Only this sentence.')).toEqual(['Only this sentence.'])
})

it('splits a long answer on paragraphs and keeps a code fence intact', () => {
  const code = '```py\nprint("keep this fence together")\n```'
  const paragraphs = Array.from({ length: 8 }, (_, index) => {
    return `Paragraph ${index} is a full thought about the ghost and the family.`
  })
  const text = `${paragraphs[0]}\n\n${code}\n\n${paragraphs.slice(1).join('\n\n')}`
  const segments = speechSegments(text, 180)
  expect(segments.length).toBeGreaterThan(1)
  expect(segments.some(segment => segment.includes('print("keep this fence together")'))).toBe(true)
  expect(segments.some(segment => segment.includes('Paragraph 0'))).toBe(true)
  expect(segments.some(segment => segment.includes('Paragraph 7'))).toBe(true)
  expect(segments.every(segment => segment.length <= 180 || segment.startsWith('```'))).toBe(true)
})

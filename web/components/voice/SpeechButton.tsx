'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, Play, Square, Volume2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import Tooltip from '@/shared/ui/Tooltip'
import { apiFetch, apiUrl } from '@/lib/api'
import { notify } from '@/lib/notifications'
import { speechSegments } from '@/lib/speech-segments'
import { useVoiceAutoplay } from '@/hooks/useVoiceAutoplay'
import { SPEECH_PLAYBACK_FAILURE_MESSAGE, SPEECH_TIMEOUT_MESSAGE } from '@/lib/voice-settings'

// Speaker button: synthesizes this one reply and plays it. Auto-play of later
// replies is a Settings preference (`autoPlayFresh`), not a first-click prompt.
let activePlayback: { owner: object; stop: () => void } | null = null
// 50 ms of playable unsigned 8-bit PCM silence; an empty WAV cannot unlock playback.
const SILENT_UNLOCK_WAV =
  'data:audio/wav;base64,UklGRrQBAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YZABAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA'

function detailText(detail: unknown): string {
  if (typeof detail === 'string') return detail.trim()
  if (Array.isArray(detail)) {
    return detail
      .map(item => {
        if (typeof item === 'string') return item.trim()
        if (item && typeof item === 'object') {
          const record = item as { msg?: unknown; message?: unknown }
          if (typeof record.msg === 'string') return record.msg.trim()
          if (typeof record.message === 'string') return record.message.trim()
        }
        return ''
      })
      .filter(Boolean)
      .join(' ')
  }
  if (detail && typeof detail === 'object') {
    const record = detail as { message?: unknown; msg?: unknown }
    if (typeof record.message === 'string' && record.message.trim()) return record.message.trim()
    if (typeof record.msg === 'string' && record.msg.trim()) return record.msg.trim()
  }
  return ''
}

async function ttsErrorMessage(resp: Response): Promise<string> {
  const readable = resp as Response & {
    text?: () => Promise<string>
    json?: () => Promise<unknown>
  }
  // Real responses expose text(). Tests often stub json() only, so try that
  // when there is no text body to read.
  if (typeof readable.text === 'function') {
    try {
      const raw = (await readable.text()).trim()
      if (!raw) return ''
      try {
        const body = JSON.parse(raw) as { detail?: unknown; message?: unknown }
        return detailText(body.detail) || detailText(body.message)
      } catch {
        if (raw.startsWith('<')) return ''
        return raw.slice(0, 400)
      }
    } catch {
      /* body already consumed, or the stub has no text() */
    }
  }
  if (typeof readable.json === 'function') {
    try {
      const body = (await readable.json()) as { detail?: unknown; message?: unknown }
      return detailText(body?.detail) || detailText(body?.message)
    } catch {
      /* non-JSON error body */
    }
  }
  return ''
}

export function PlayAudioButton({
  content,
  conversationKey,
  autoPlayFresh,
  selection = false,
  playbackRate = 1,
  className,
}: {
  content: string
  conversationKey?: string
  autoPlayFresh: boolean
  playbackRate?: number
  selection?: boolean
  className?: string
}) {
  const { t } = useTranslation()
  const { autoplayEnabled } = useVoiceAutoplay(conversationKey)
  const [state, setState] = useState<'idle' | 'loading' | 'playing'>('idle')
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rateRef = useRef(playbackRate)
  useEffect(() => {
    rateRef.current = playbackRate
    if (audioRef.current) audioRef.current.playbackRate = playbackRate
  }, [playbackRate])
  const urlRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const genRef = useRef(0)
  const autoPlayedRef = useRef(false)
  const playbackOwnerRef = useRef<object>({})
  const cancelWaitRef = useRef<(() => void) | null>(null)

  const cleanup = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current)
      urlRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    genRef.current += 1
    cancelWaitRef.current?.()
    cancelWaitRef.current = null
    cleanup()
    setState('idle')
    if (activePlayback?.owner === playbackOwnerRef.current) activePlayback = null
  }, [cleanup])

  const play = useCallback(async () => {
    activePlayback?.stop()
    const gen = ++genRef.current
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    activePlayback = { owner: playbackOwnerRef.current, stop }
    setState('loading')

    const segments = speechSegments(content)
    if (!segments.length) {
      notify(t(SPEECH_PLAYBACK_FAILURE_MESSAGE), { tone: 'error' })
      stop()
      return
    }

    // WebKit evaluates media playback eligibility when play() is called. Keep
    // this element unlocked in the click task, then replace its source after
    // synthesis completes rather than constructing a new audio element then.
    const audio = new Audio(SILENT_UNLOCK_WAV)
    audioRef.current = audio
    void Promise.resolve(audio.play()).catch(() => {
      /* Unlocking playback can fail on clients that already permit it. */
    })

    const fetchSegment = (segment: string) => {
      const pending = apiFetch(apiUrl('/api/voice/tts'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: segment }),
        signal: ac.signal,
      })
      // A segment prefetched for the next paragraph is abandoned when playback
      // stops. Swallow that rejection so it does not surface as unhandled.
      pending.catch(() => undefined)
      return pending
    }

    const failure = (status: number) =>
      t(status === 504 ? SPEECH_TIMEOUT_MESSAGE : SPEECH_PLAYBACK_FAILURE_MESSAGE) +
      ` (HTTP ${status})`

    try {
      let upcoming: Promise<Response> | null = fetchSegment(segments[0])
      for (let index = 0; index < segments.length; index += 1) {
        if (gen !== genRef.current) return
        const resp = await upcoming
        upcoming =
          index + 1 < segments.length ? fetchSegment(segments[index + 1]) : null
        if (gen !== genRef.current) return
        if (!resp.ok) {
          const detail = await ttsErrorMessage(resp)
          if (gen !== genRef.current) return
          notify(detail || failure(resp.status), { tone: 'error' })
          stop()
          return
        }
        const blob = await resp.blob()
        if (!blob.size) throw new Error('Speech returned empty audio')
        if (gen !== genRef.current) return
        const previous = urlRef.current
        const url = URL.createObjectURL(blob)
        urlRef.current = url
        if (previous) URL.revokeObjectURL(previous)
        const last = index === segments.length - 1
        audio.src = url
        audio.load()
        audio.playbackRate = rateRef.current
        const ended = new Promise<void>(resolve => {
          cancelWaitRef.current = resolve
          audio.onended = () => {
            if (gen === genRef.current && last) {
              setState('idle')
              cleanup()
              if (activePlayback?.owner === playbackOwnerRef.current) activePlayback = null
            }
            resolve()
          }
          audio.onerror = () => {
            if (gen === genRef.current) {
              notify(t(SPEECH_PLAYBACK_FAILURE_MESSAGE), { tone: 'error' })
              stop()
            }
            resolve()
          }
        })
        await audio.play()
        if (gen !== genRef.current) {
          audio.pause()
          return
        }
        setState('playing')
        await ended
        if (gen !== genRef.current) return
      }
    } catch (err) {
      if (gen !== genRef.current) return
      if (err instanceof Error && err.name === 'AbortError') return
      const message = err instanceof Error ? err.message.trim() : ''
      notify(
        message && !/^failed to fetch$/i.test(message) ? message : t(SPEECH_PLAYBACK_FAILURE_MESSAGE),
        { tone: 'error' }
      )
      stop()
    }
  }, [cleanup, content, stop, t])

  const handleClick = useCallback(() => {
    if (state === 'playing' || state === 'loading') {
      stop()
      return
    }
    void play()
  }, [play, state, stop])

  // Auto-play a freshly-generated reply when Settings auto-play is on.
  useEffect(() => {
    if (!autoPlayFresh || !autoplayEnabled) return
    if (autoPlayedRef.current) return
    if (!content.trim()) return
    autoPlayedRef.current = true
    const id = window.setTimeout(() => void play(), 0)
    return () => window.clearTimeout(id)
  }, [autoPlayFresh, autoplayEnabled, content, play])

  useEffect(
    () => () => {
      genRef.current += 1
      cancelWaitRef.current?.()
      cleanup()
      if (activePlayback?.owner === playbackOwnerRef.current) activePlayback = null
    },
    [cleanup]
  )

  return (
    <div className={`relative inline-flex ${selection ? 'h-full' : ''}`}>
      <Tooltip
        label={state !== 'idle' ? t('Stop') : t(selection ? 'Read aloud' : 'Play aloud')}
        side="top"
      >
        <button
          type="button"
          onClick={handleClick}
          aria-label={state !== 'idle' ? t('Stop') : t(selection ? 'Read aloud' : 'Play aloud')}
          aria-busy={state === 'loading'}
          className={
            className ??
            `inline-flex items-center justify-center rounded-md p-1 transition-colors ${
              state === 'playing'
                ? 'text-[var(--primary)]'
                : 'text-[var(--muted-foreground)] hover:bg-[var(--muted)]/50 hover:text-[var(--foreground)]'
            }`
          }
        >
          {state === 'loading' ? (
            <Loader2 size={15} strokeWidth={1.8} className="animate-spin" />
          ) : state === 'playing' ? (
            <Square size={13} strokeWidth={1.8} className="fill-current" />
          ) : selection ? (
            <Play size={15} strokeWidth={1.8} className="fill-current" />
          ) : (
            <Volume2 size={15} strokeWidth={1.5} />
          )}
        </button>
      </Tooltip>
    </div>
  )
}

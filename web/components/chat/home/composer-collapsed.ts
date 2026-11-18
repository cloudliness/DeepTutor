"use client";

import { browserStorage } from "@/shared/storage";

/**
 * Whether the chat composer is collapsed. One browser-local preference for
 * every composer, the same way the knowledge-base file tree remembers which
 * folders are open.
 */
export const COMPOSER_COLLAPSED_KEY = "deeptutor.composer-collapsed";
const EVENT = "deeptutor:composer-collapsed";

export function readComposerCollapsed(): boolean {
  return (
    typeof window !== "undefined" &&
    browserStorage.readRaw("local", COMPOSER_COLLAPSED_KEY) === "1"
  );
}

export function writeComposerCollapsed(collapsed: boolean): void {
  if (readComposerCollapsed() === collapsed) return;
  browserStorage.writeRaw("local", COMPOSER_COLLAPSED_KEY, collapsed ? "1" : "0");
  window.dispatchEvent(new Event(EVENT));
}

export function subscribeComposerCollapsed(onChange: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === COMPOSER_COLLAPSED_KEY) onChange();
  };
  window.addEventListener(EVENT, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(EVENT, onChange);
    window.removeEventListener("storage", onStorage);
  };
}

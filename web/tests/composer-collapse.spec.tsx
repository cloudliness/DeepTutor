import { createRef, type ComponentProps } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { MessageSquare } from "lucide-react";

import ChatComposer from "@/components/chat/home/ChatComposer";
import { COMPOSER_COLLAPSED_KEY } from "@/components/chat/home/composer-collapsed";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));
vi.mock("@/lib/workspace-drafts", () => ({
  currentWorkspaceDraftLocation: () => ({ workspaceId: "", pathname: "/chat" }),
  readWorkspaceDraft: async () => null,
  saveWorkspaceDraft: async () => undefined,
}));

function renderComposer(overrides: Partial<ComponentProps<typeof ChatComposer>> = {}) {
  const onSend = vi.fn();
  const focusInputRef = createRef<(() => void) | null>();
  const noop = () => undefined;
  const props: ComponentProps<typeof ChatComposer> = {
    composerRef: createRef(), capMenuRef: createRef(), capBtnRef: createRef(),
    spaceMenuRef: createRef(), spaceBtnRef: createRef(), dragCounter: { current: 0 },
    dragging: false, capMenuOpen: false, spaceMenuOpen: false, hasMessages: true,
    attachments: [], attachmentError: null,
    activeCap: { value: "", label: "Chat", description: "Chat", icon: MessageSquare,
      allowedTools: [] },
    knowledgeBases: [], llmOptions: [], activeLLMDefault: null, llmSelection: null,
    llmOptionsLoading: false, llmOptionsError: false,
    selectedNotebookRecords: [], selectedBookReferences: [], selectedHistorySessions: [],
    selectedAgentSessions: [], selectedQuestionEntries: [], notebookReferenceGroups: [],
    selectedPersona: null, selectedMemoryFiles: [], selectedKnowledgeBases: [],
    isStreaming: false, isVisualizeMode: false, capabilityNeedsConfig: false,
    capabilityConfigConfirmed: false, onRequestConfigConfirm: noop, capabilities: [],
    onSetCapMenuOpen: noop, onSetSpaceMenuOpen: noop, onToggleKB: noop,
    onSelectLLM: noop, onSelectNotebookPicker: noop, onSelectBookPicker: noop,
    onSelectHistoryPicker: noop, onSelectAgentsPicker: noop, onSelectQuestionBankPicker: noop,
    onSelectPersonaPicker: noop, onSelectMemoryPicker: noop, onClearPersona: noop,
    onToggleMemoryFile: noop, onSend, onRemoveAttachment: noop, onRemoveHistory: noop,
    onRemoveAgent: noop, onRemoveBookReference: noop, onRemoveNotebook: noop,
    onRemoveQuestion: noop, onDragEnter: noop, onDragLeave: noop, onDragOver: noop,
    onDrop: noop, onPaste: noop, onAddFiles: noop, onSelectCapability: noop,
    onCancelStreaming: noop,
    focusInputRef,
    ...overrides,
  };
  const view = render(<ChatComposer {...props} />);
  return { ...view, onSend, focusInputRef };
}

beforeEach(() => {
  localStorage.clear();
});

it("collapses the composer, remembers it, and expands again", async () => {
  const first = renderComposer();
  expect(screen.getByRole("textbox")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Collapse composer" }));
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Expand composer" })).toBeVisible();
  expect(localStorage.getItem(COMPOSER_COLLAPSED_KEY)).toBe("1");

  first.unmount();
  renderComposer();
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Expand composer" }));
  expect(await screen.findByRole("textbox")).toBeVisible();
  expect(localStorage.getItem(COMPOSER_COLLAPSED_KEY)).toBe("0");
});

it("expands when the reader starts typing, and when focus is requested", async () => {
  const { focusInputRef } = renderComposer();
  fireEvent.click(screen.getByRole("button", { name: "Collapse composer" }));
  fireEvent.keyDown(document.body, { key: "h" });
  const input = await screen.findByRole("textbox");
  await waitFor(() => expect(input).toHaveValue("h"));

  fireEvent.click(screen.getByRole("button", { name: "Collapse composer" }));
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  await act(async () => {
    focusInputRef.current?.();
  });
  expect(await screen.findByRole("textbox")).toBeVisible();
});

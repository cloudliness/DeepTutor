import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import KbDocumentList from "@/components/knowledge/KbDocumentList";

const filesApi = vi.hoisted(() => ({
  list: vi.fn(),
  remove: vi.fn(),
  create: vi.fn(),
  move: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("@/features/knowledge/api/files", () => ({
  listKnowledgeBaseFiles: filesApi.list,
  deleteKbFile: filesApi.remove,
  createKbFolder: filesApi.create,
  moveKbFile: filesApi.move,
}));

const folder = {
  name: "01-Basic-Arithmetic-Workbook",
  type: "folder" as const,
};

beforeEach(() => {
  filesApi.list.mockReset();
  filesApi.remove.mockReset().mockResolvedValue({ was_indexed: true });
  filesApi.create.mockReset();
  filesApi.move.mockReset();
});

it("confirms and deletes an empty folder from the files panel", async () => {
  filesApi.list
    .mockResolvedValueOnce([folder])
    .mockResolvedValueOnce([]);
  const onSelect = vi.fn();

  render(
    <KbDocumentList
      kbName="math"
      selectedFile={null}
      onSelect={onSelect}
      collapsed={false}
      onToggleCollapsed={vi.fn()}
    />,
  );

  expect(await screen.findByText("01-Basic-Arithmetic-Workbook")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Delete folder" }));
  expect(screen.getByText("Delete?")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Confirm delete" }));

  await waitFor(() =>
    expect(filesApi.remove).toHaveBeenCalledWith("math", "01-Basic-Arithmetic-Workbook"),
  );
  await waitFor(() =>
    expect(screen.queryByText("01-Basic-Arithmetic-Workbook")).not.toBeInTheDocument(),
  );
  expect(onSelect).not.toHaveBeenCalled();
});

it("clears a preview that lives inside the deleted folder", async () => {
  const entries = [
    folder,
    {
      name: "01-Basic-Arithmetic-Workbook/one.pdf",
      type: "file" as const,
      size: 12,
      modified: Date.now() / 1000,
    },
  ];
  filesApi.list.mockResolvedValue(entries);
  const onSelect = vi.fn();

  render(
    <KbDocumentList
      kbName="math"
      selectedFile="01-Basic-Arithmetic-Workbook/one.pdf"
      onSelect={onSelect}
      collapsed={false}
      onToggleCollapsed={vi.fn()}
    />,
  );

  await screen.findByText("01-Basic-Arithmetic-Workbook");
  fireEvent.click(screen.getByRole("button", { name: "Delete folder" }));
  fireEvent.click(screen.getByRole("button", { name: "Confirm delete" }));

  await waitFor(() => expect(filesApi.remove).toHaveBeenCalledTimes(1));
  expect(onSelect).toHaveBeenCalledWith(null);
});

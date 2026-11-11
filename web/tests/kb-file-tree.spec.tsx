import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import KbDocumentList from "@/components/knowledge/KbDocumentList";

const filesApi = vi.hoisted(() => ({
  list: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("@/features/knowledge/api/files", () => ({
  listKnowledgeBaseFiles: filesApi.list,
  deleteKbFile: vi.fn(),
  createKbFolder: vi.fn(),
  moveKbFile: vi.fn(),
}));

const folder = {
  name: "01-Basic-Arithmetic-Workbook",
  type: "folder" as const,
};
const file = {
  name: "01-Basic-Arithmetic-Workbook/00-Front-Matter.pdf",
  type: "file" as const,
  size: 20,
  modified: 1,
};

function renderTree(kbName = "math") {
  return render(
    <KbDocumentList
      kbName={kbName}
      selectedFile={null}
      onSelect={vi.fn()}
      collapsed={false}
      onToggleCollapsed={vi.fn()}
    />,
  );
}

beforeEach(() => {
  localStorage.clear();
  filesApi.list.mockReset().mockResolvedValue([folder, file]);
});

it("keeps folders collapsed until the user opens them, then restores that per KB", async () => {
  const first = renderTree();
  const folderLabel = await screen.findByText("01-Basic-Arithmetic-Workbook");
  expect(folderLabel).toHaveAttribute("title", "01-Basic-Arithmetic-Workbook");
  expect(screen.queryByText("00-Front-Matter.pdf")).not.toBeInTheDocument();

  fireEvent.click(folderLabel);
  const fileLabel = screen.getByText("00-Front-Matter.pdf");
  expect(fileLabel).toHaveAttribute("title", "00-Front-Matter.pdf");
  expect(localStorage.getItem("kb-files:math:expanded")).toBe(
    JSON.stringify(["01-Basic-Arithmetic-Workbook"]),
  );

  first.unmount();
  const restored = renderTree();
  expect(await screen.findByText("00-Front-Matter.pdf")).toBeInTheDocument();
  restored.unmount();

  renderTree("physics");
  expect(await screen.findByText("01-Basic-Arithmetic-Workbook")).toBeInTheDocument();
  expect(screen.queryByText("00-Front-Matter.pdf")).not.toBeInTheDocument();
});

it("resizes the file list from its edge and restores the width", async () => {
  const onToggle = vi.fn();
  const first = render(
    <KbDocumentList
      kbName="math"
      selectedFile={null}
      onSelect={vi.fn()}
      collapsed={false}
      onToggleCollapsed={onToggle}
    />,
  );
  await screen.findByText("01-Basic-Arithmetic-Workbook");
  const handle = screen.getByRole("separator", { name: "Resize file list" });
  expect(handle).toHaveAttribute("aria-valuenow", "220");
  fireEvent.keyDown(handle, { key: "ArrowRight" });
  expect(handle).toHaveAttribute("aria-valuenow", "236");
  expect(first.container.querySelector("aside")).toHaveStyle({ width: "236px" });
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  expect(onToggle).toHaveBeenCalledTimes(1);

  first.unmount();
  renderTree();
  await screen.findByText("01-Basic-Arithmetic-Workbook");
  const restored = screen.getByRole("separator", { name: "Resize file list" });
  expect(restored).toHaveAttribute("aria-valuenow", "236");
  expect(localStorage.getItem("kb-files:tree-width")).toBe("236");
});

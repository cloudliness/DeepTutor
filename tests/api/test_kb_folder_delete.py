"""File and folder deletion keeps the linked-folder ledger and the index aligned."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("llama_index.core")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

try:
    knowledge_router_module = importlib.import_module("deeptutor.api.routers.knowledge")
    router = knowledge_router_module.router
except Exception:  # pragma: no cover - optional heavy imports in lightweight envs
    knowledge_router_module = None
    router = None

pytestmark = pytest.mark.skipif(
    router is None, reason="deeptutor.api.routers.knowledge could not be imported"
)


@pytest.fixture(autouse=True)
def _disable_pocketbase(monkeypatch):
    monkeypatch.setattr("deeptutor.services.pocketbase_client.is_pocketbase_enabled", lambda: False)
    previous = getattr(Settings, "_embed_model", None)
    Settings.embed_model = MockEmbedding(embed_dim=4)
    yield
    Settings._embed_model = previous


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


def _manager(monkeypatch, tmp_path: Path):
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    manager = KnowledgeBaseManager(base_dir=str(tmp_path / "kbs"))
    monkeypatch.setattr(knowledge_router_module, "get_kb_manager", lambda: manager)
    return manager


def _node(raw_dir: Path, rel: str, text: str) -> TextNode:
    path = raw_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    node = TextNode(
        text=text,
        metadata={
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "document_key": rel,
        },
    )
    node.embedding = [1.0, 0.0, 0.0, 0.0]
    return node


def _seed(monkeypatch, tmp_path: Path):
    from deeptutor.utils.file_types import FileTypeRouter

    manager = _manager(monkeypatch, tmp_path)
    kb_dir = manager.base_dir / "math"
    raw_dir = kb_dir / "raw"
    raw_dir.mkdir(parents=True)
    manager.register_knowledge_base("math")
    manager.config["knowledge_bases"]["math"]["rag_provider"] = "llamaindex"
    manager.config["knowledge_bases"]["math"]["status"] = "ready"
    manager._save_config()

    source = tmp_path / "documents" / "math"
    workbook = source / "01-Basic-Arithmetic-Workbook"
    workbook.mkdir(parents=True)
    originals = {}
    for name in ("one.md", "two.md", "three.md"):
        path = workbook / name
        path.write_text(name, encoding="utf-8")
        originals[path] = path.read_bytes()
    nodes = [
        _node(raw_dir, f"01-Basic-Arithmetic-Workbook/{name}", name)
        for name in ("one.md", "two.md", "three.md")
    ]
    nodes.append(_node(raw_dir, "other.md", "other"))
    index = VectorStoreIndex(nodes=nodes, embed_model=Settings.embed_model, show_progress=False)
    version = kb_dir / "version-1"
    version.mkdir()
    index.storage_context.persist(persist_dir=str(version))
    (version / "meta.json").write_text(
        json.dumps(
            {
                "version": "version-1",
                "provider": "llamaindex",
                "signature": "test-signature",
                "state": "published",
                "readiness_verified": True,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    folder = manager.link_folder("math", str(source))
    collected = [
        str(path) for path in FileTypeRouter.collect_supported_files(source, recursive=True)
    ]
    manager.update_folder_sync_state("math", folder["id"], collected)
    metadata_path = kb_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["file_hashes"] = {
        "01-Basic-Arithmetic-Workbook/one.md": "one",
        "01-Basic-Arithmetic-Workbook/two.md": "two",
        "01-Basic-Arithmetic-Workbook/three.md": "three",
        "other.md": "other",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return manager, kb_dir, folder["id"], originals


def _active_doc_count(kb_dir: Path) -> int:
    from deeptutor.services.rag.index_probe import latest_ready_provider_version

    entry = latest_ready_provider_version(kb_dir, "llamaindex")
    assert entry is not None
    return int(entry["doc_count"])


def test_delete_one_synced_file_makes_the_next_sync_see_it(monkeypatch, tmp_path: Path) -> None:
    manager, kb_dir, folder_id, originals = _seed(monkeypatch, tmp_path)
    source = tmp_path / "documents" / "math" / "01-Basic-Arithmetic-Workbook" / "one.md"

    with TestClient(_build_app()) as client:
        deleted = client.delete(
            "/api/knowledge-bases/math/files/01-Basic-Arithmetic-Workbook/one.md"
        )
        listing = client.get("/api/knowledge-bases/math/files")
        missing = client.delete("/api/knowledge-bases/math/files/no-such-folder")

    assert deleted.status_code == 200
    assert deleted.json() == {
        "status": "ok",
        "path": "01-Basic-Arithmetic-Workbook/one.md",
        "was_indexed": True,
    }
    names = {entry["name"] for entry in listing.json()["files"]}
    assert "01-Basic-Arithmetic-Workbook/one.md" not in names
    assert "01-Basic-Arithmetic-Workbook/two.md" in names
    assert "01-Basic-Arithmetic-Workbook" in names
    assert missing.status_code == 404
    assert source.read_bytes() == originals[source]
    changes = manager.detect_folder_changes("math", folder_id)
    assert changes["new_files"] == [str(source)]
    assert _active_doc_count(kb_dir) == 3


def test_delete_folder_reports_removed_files_and_leaves_siblings(
    monkeypatch, tmp_path: Path
) -> None:
    manager, kb_dir, folder_id, originals = _seed(monkeypatch, tmp_path)

    with TestClient(_build_app()) as client:
        removed = client.delete("/api/knowledge-bases/math/files/01-Basic-Arithmetic-Workbook")
        listing = client.get("/api/knowledge-bases/math/files")

    assert removed.status_code == 200
    body = removed.json()
    assert body["status"] == "ok"
    assert body["type"] == "folder"
    assert body["path"] == "01-Basic-Arithmetic-Workbook"
    assert set(body["removed_files"]) == {
        "01-Basic-Arithmetic-Workbook/one.md",
        "01-Basic-Arithmetic-Workbook/two.md",
        "01-Basic-Arithmetic-Workbook/three.md",
    }
    assert {entry["name"] for entry in listing.json()["files"]} == {"other.md"}
    assert _active_doc_count(kb_dir) == 1
    for path, payload in originals.items():
        assert path.read_bytes() == payload
    assert manager.get_linked_folders("math")[0]["file_count"] == 0
    assert len(manager.detect_folder_changes("math", folder_id)["new_files"]) == 3

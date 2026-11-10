"""Deleting a raw source must drop its chunks from the active LlamaIndex.

FAISS cannot delete vectors in place. Removal republishes the remaining nodes
from the embeddings already stored, without calling the embedding provider.
"""

from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import QueryBundle, TextNode
import pytest

from deeptutor.services.rag.index_probe import latest_ready_provider_version
from deeptutor.services.rag.pipelines.llamaindex import storage, vector_store

pytest.importorskip("llama_index.core")

_DIM = 8
_DELETED = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_KEPT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _mock_embedding():
    # Reading Settings.embed_model resolves the OpenAI default. Touch the
    # private field so a machine without an API key can still run this.
    previous = getattr(Settings, "_embed_model", None)
    Settings.embed_model = MockEmbedding(embed_dim=_DIM)
    storage.clear_index_cache()
    yield
    Settings._embed_model = previous
    storage.clear_index_cache()


def _node(rel: str, text: str, vector: list[float], raw_dir: Path) -> TextNode:
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
    node.embedding = list(vector)
    return node


def _publish(kb_dir: Path, nodes: list[TextNode], *, faiss: bool) -> None:
    kwargs = {"nodes": nodes, "embed_model": Settings.embed_model, "show_progress": False}
    if faiss:
        context = vector_store.new_faiss_storage_context(_DIM)
        assert context is not None
        kwargs["storage_context"] = context
    index = VectorStoreIndex(**kwargs)
    version = kb_dir / "version-1"
    version.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(version))
    (version / "meta.json").write_text(
        json.dumps(
            {
                "version": "version-1",
                "provider": "llamaindex",
                "signature": "test-signature",
                "layout": "flat",
                "state": "published",
                "readiness_verified": True,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def _active_count(kb_dir: Path) -> int:
    entry = latest_ready_provider_version(kb_dir, "llamaindex")
    assert entry is not None
    return int(entry["doc_count"])


def _hits(kb_dir: Path, vector: list[float]) -> set[str]:
    entry = latest_ready_provider_version(kb_dir, "llamaindex")
    assert entry is not None
    index = vector_store.load_index(Path(entry["storage_path"]))
    found = index.as_retriever(similarity_top_k=8).retrieve(
        QueryBundle(query_str="q", embedding=vector)
    )
    return {hit.node.metadata.get("document_key", "") for hit in found}


@pytest.mark.parametrize("backend", ["simple", "faiss"])
def test_dropping_a_source_removes_its_chunks_and_keeps_the_rest(
    tmp_path: Path, backend: str
) -> None:
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("llama_index.vector_stores.faiss")
    kb_dir = tmp_path / "math"
    raw_dir = kb_dir / "raw"
    nodes = [
        _node("01-Basic-Arithmetic-Workbook/one.md", "deleted chunk", _DELETED, raw_dir),
        _node("01-Basic-Arithmetic-Workbook/one.md", "deleted chunk two", _DELETED, raw_dir),
        _node("01-Basic-Arithmetic-Workbook/two.md", "kept workbook", _KEPT, raw_dir),
        _node("other.md", "kept sibling", _KEPT, raw_dir),
    ]
    _publish(kb_dir, nodes, faiss=backend == "faiss")
    assert _active_count(kb_dir) == 4

    removed = storage.drop_sources_from_active_index(
        kb_dir, {"01-Basic-Arithmetic-Workbook/one.md"}
    )

    assert removed == 2
    assert _active_count(kb_dir) == 2
    assert "01-Basic-Arithmetic-Workbook/one.md" not in _hits(kb_dir, _DELETED)
    assert _hits(kb_dir, _KEPT) == {
        "01-Basic-Arithmetic-Workbook/two.md",
        "other.md",
    }
    assert (raw_dir / "01-Basic-Arithmetic-Workbook" / "one.md").is_file()


def test_dropping_a_folder_prefix_clears_every_file_under_it(tmp_path: Path) -> None:
    kb_dir = tmp_path / "math"
    raw_dir = kb_dir / "raw"
    nodes = [
        _node("01-Basic-Arithmetic-Workbook/one.md", "one", _DELETED, raw_dir),
        _node("01-Basic-Arithmetic-Workbook/two.md", "two", _DELETED, raw_dir),
        _node("other.md", "other", _KEPT, raw_dir),
    ]
    _publish(kb_dir, nodes, faiss=False)

    removed = storage.drop_sources_from_active_index(
        kb_dir,
        set(),
        directory_prefixes={"01-Basic-Arithmetic-Workbook"},
    )

    assert removed == 2
    assert _active_count(kb_dir) == 1
    assert _hits(kb_dir, _KEPT) == {"other.md"}


def test_dropping_every_source_publishes_an_empty_active_index(tmp_path: Path) -> None:
    kb_dir = tmp_path / "math"
    raw_dir = kb_dir / "raw"
    _publish(
        kb_dir,
        [_node("01-Basic-Arithmetic-Workbook/one.md", "only", _DELETED, raw_dir)],
        faiss=False,
    )

    removed = storage.drop_sources_from_active_index(
        kb_dir,
        set(),
        directory_prefixes={"01-Basic-Arithmetic-Workbook"},
    )

    assert removed == 1
    assert _active_count(kb_dir) == 0
    assert _hits(kb_dir, _DELETED) == set()


def test_source_absolute_paths_match_the_raw_relative_path(tmp_path: Path) -> None:
    """Chunks indexed from the linked source path, not from raw/, still drop.

    ``file_path`` on those nodes is ``/mnt/documents/.../01-Basic/a.pdf``. It
    does not resolve under raw/, so a raw-relative comparison used to remove
    nothing and leave the published version untouched.
    """
    kb_dir = tmp_path / "math"
    raw_dir = kb_dir / "raw"
    kept_path = raw_dir / "other.md"
    kept_path.parent.mkdir(parents=True)
    kept_path.write_text("other", encoding="utf-8")
    nodes = [
        TextNode(
            text="front",
            metadata={
                "file_name": "00-Front-Matter.pdf",
                "file_path": "/mnt/documents/math/01-Basic-Arithmetic-Workbook/00-Front-Matter.pdf",
            },
        ),
        TextNode(
            text="other",
            metadata={"file_name": "other.md", "file_path": str(kept_path.resolve()), "document_key": "other.md"},
        ),
    ]
    for node, vector in zip(nodes, (_DELETED, _KEPT)):
        node.embedding = list(vector)
    _publish(kb_dir, nodes, faiss=False)

    removed = storage.drop_sources_from_active_index(
        kb_dir, {"01-Basic-Arithmetic-Workbook/00-Front-Matter.pdf"}
    )

    assert removed == 1
    assert _active_count(kb_dir) == 1
    assert _hits(kb_dir, _KEPT) == {"other.md"}


def test_empty_raw_set_drops_chunks_whose_paths_do_not_match(tmp_path: Path) -> None:
    kb_dir = tmp_path / "math"
    (kb_dir / "raw").mkdir(parents=True)
    node = TextNode(
        text="orphan",
        metadata={"file_name": "mystery.pdf", "file_path": "/tmp/not-in-this-kb/mystery.pdf"},
    )
    node.embedding = list(_DELETED)
    _publish(kb_dir, [node], faiss=False)

    removed = storage.republish_index_for_raw_files(kb_dir, set())

    assert removed == 1
    assert _active_count(kb_dir) == 0


def test_drop_is_a_no_op_when_the_index_has_no_matching_source(tmp_path: Path) -> None:
    kb_dir = tmp_path / "math"
    raw_dir = kb_dir / "raw"
    _publish(
        kb_dir,
        [_node("other.md", "other", _KEPT, raw_dir)],
        faiss=False,
    )
    before = list(kb_dir.glob("version-*"))

    assert storage.drop_sources_from_active_index(kb_dir, {"missing.md"}) == 0
    assert list(kb_dir.glob("version-*")) == before
    assert _active_count(kb_dir) == 1

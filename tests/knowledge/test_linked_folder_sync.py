"""Persistence of linked-folder sync state (``metadata.json``).

``update_folder_sync_state()`` used to mutate the loaded metadata dict and
return without writing it back, so ``last_sync``/``synced_files`` silently
vanished while the API layer logged success. These tests pin the write-back.
The atomicity contract of the shared writer lives in
``tests/services/test_file_io.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deeptutor.api.routers.knowledge import LinkedFolderInfo
from deeptutor.knowledge import manager as manager_module
from deeptutor.knowledge.add_documents import remove_raw_document, remove_raw_tree
from deeptutor.knowledge.manager import KnowledgeBaseManager
from deeptutor.utils.file_types import FileTypeRouter


def _manager_with_linked_folder(tmp_path: Path) -> tuple[KnowledgeBaseManager, Path, str, Path]:
    """A registered KB with one linked folder containing one markdown file."""
    manager = KnowledgeBaseManager(base_dir=str(tmp_path / "kbs"))

    kb_dir = manager.base_dir / "kb"
    kb_dir.mkdir()
    manager.register_knowledge_base("kb")

    source = tmp_path / "notes"
    source.mkdir()
    doc = source / "note.md"
    doc.write_text("hello", encoding="utf-8")

    folder_info = manager.link_folder("kb", str(source))
    assert folder_info["last_sync"] is None
    return manager, kb_dir / "metadata.json", folder_info["id"], doc


def test_update_folder_sync_state_persists_to_disk(tmp_path: Path) -> None:
    manager, metadata_file, folder_id, doc = _manager_with_linked_folder(tmp_path)

    manager.update_folder_sync_state("kb", folder_id, [str(doc)])

    on_disk = json.loads(metadata_file.read_text(encoding="utf-8"))
    folder = on_disk["linked_folders"][0]
    assert "last_sync" in folder
    assert str(doc) in folder["synced_files"]
    assert folder["file_count"] == 1

    # A fresh manager (new process in real life) must see the sync state too.
    reloaded = KnowledgeBaseManager(base_dir=str(manager.base_dir))
    folder = reloaded.get_linked_folders("kb")[0]
    assert "last_sync" in folder
    assert str(doc) in folder["synced_files"]

    info = LinkedFolderInfo(**folder)
    assert info.last_sync == folder["last_sync"]


def test_update_folder_sync_state_unknown_folder_writes_nothing(tmp_path: Path) -> None:
    manager, metadata_file, _folder_id, doc = _manager_with_linked_folder(tmp_path)
    before = metadata_file.read_bytes()

    manager.update_folder_sync_state("kb", "no-such-id", [str(doc)])

    assert metadata_file.read_bytes() == before


def test_sync_snapshot_preserves_change_made_during_indexing(tmp_path: Path) -> None:
    manager, _metadata_file, folder_id, doc = _manager_with_linked_folder(tmp_path)
    staged_mtime = "2026-09-01T10:00:00"
    manager.update_folder_sync_state("kb", folder_id, [str(doc)], {str(doc): staged_mtime})

    changes = manager.detect_folder_changes("kb", folder_id)
    assert changes["modified_files"] == [str(doc)]


def test_empty_successful_sync_records_last_sync_without_changing_file_state(
    tmp_path: Path,
) -> None:
    manager, metadata_file, folder_id, doc = _manager_with_linked_folder(tmp_path)
    manager.update_folder_sync_state("kb", folder_id, [str(doc)])
    before = json.loads(metadata_file.read_text(encoding="utf-8"))["linked_folders"][0]

    manager.update_folder_sync_state("kb", folder_id, [])

    after = json.loads(metadata_file.read_text(encoding="utf-8"))["linked_folders"][0]
    assert after["last_sync"]
    assert after["last_sync"] >= before["last_sync"]
    assert after["synced_files"] == before["synced_files"]


def test_update_folder_sync_state_mtime_failure_is_visible_not_fake_synced(
    tmp_path: Path,
    caplog,
    monkeypatch,
) -> None:
    """A per-file mtime recording failure must surface, not vanish.

    ``update_folder_sync_state()`` used to swallow stat/fromtimestamp errors
    with ``except Exception: pass``. The sync then reported success while the
    file's state was never recorded, hiding the degradation. The failure must
    be logged and the file must stay un-recorded so the next scan re-syncs it.
    """
    manager, metadata_file, folder_id, doc = _manager_with_linked_folder(tmp_path)

    flavour = type(Path())  # PosixPath on POSIX, WindowsPath on Windows

    class _VanishedMidStatPath(flavour):
        """exists() wins the race, stat() loses it (source vanished mid-sync)."""

        def exists(self, **kwargs):
            return True

        def stat(self, **kwargs):
            raise OSError("source vanished before stat")

    monkeypatch.setattr(manager_module, "Path", _VanishedMidStatPath)

    with caplog.at_level(logging.WARNING, logger="deeptutor.knowledge.manager"):
        manager.update_folder_sync_state("kb", folder_id, [str(doc)])

    # Visibility: the swallowed failure is now a logged warning naming the file.
    warnings_for_doc = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and str(doc) in record.getMessage()
    ]
    assert warnings_for_doc, f"expected a logged warning mentioning {doc}, got none"

    # Fallback semantics: the file is not marked synced, so the next scan
    # re-detects it instead of trusting a fabricated "already synced" state.
    on_disk = json.loads(metadata_file.read_text(encoding="utf-8"))
    folder = on_disk["linked_folders"][0]
    assert folder["last_sync"]
    assert str(doc) not in folder["synced_files"]
    assert folder["file_count"] == 0
    assert manager.detect_folder_changes("kb", folder_id)["new_files"] == [str(doc)]


def _synced_workbook(tmp_path: Path):
    """A KB whose raw copies came from a linked ``01-Basic-Arithmetic-Workbook``."""
    manager = KnowledgeBaseManager(base_dir=str(tmp_path / "kbs"))
    kb_dir = manager.base_dir / "math"
    (kb_dir / "raw").mkdir(parents=True)
    manager.register_knowledge_base("math")

    source = tmp_path / "documents" / "math"
    workbook = source / "01-Basic-Arithmetic-Workbook"
    workbook.mkdir(parents=True)
    sources = []
    for name in ("one.md", "two.md", "three.md"):
        path = workbook / name
        path.write_text(name, encoding="utf-8")
        sources.append(path)
        dest = kb_dir / "raw" / "01-Basic-Arithmetic-Workbook" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(name, encoding="utf-8")
    (kb_dir / "raw" / "other.md").write_text("other", encoding="utf-8")

    folder = manager.link_folder("math", str(source))
    collected = [
        str(path) for path in FileTypeRouter.collect_supported_files(source, recursive=True)
    ]
    manager.update_folder_sync_state("math", folder["id"], collected)
    metadata_file = kb_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["file_hashes"] = {
        "01-Basic-Arithmetic-Workbook/one.md": "one",
        "01-Basic-Arithmetic-Workbook/two.md": "two",
        "01-Basic-Arithmetic-Workbook/three.md": "three",
        "other.md": "other",
    }
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    return manager, kb_dir, folder["id"], source, sources


def _synced_relative_paths(manager: KnowledgeBaseManager) -> set[str]:
    folder = manager.get_linked_folders("math")[0]
    root = Path(folder["path"]).resolve()
    return {
        Path(source).resolve().relative_to(root).as_posix()
        for source in folder.get("synced_files", {})
    }


def test_deleting_a_synced_raw_file_is_new_on_the_next_scan(tmp_path: Path) -> None:
    manager, kb_dir, folder_id, source, sources = _synced_workbook(tmp_path)
    original = sources[0].read_bytes()
    raw_copy = kb_dir / "raw" / "01-Basic-Arithmetic-Workbook" / "one.md"

    removal = remove_raw_document(kb_dir, raw_copy)

    assert removal.was_indexed is True
    assert removal.rel_path == "01-Basic-Arithmetic-Workbook/one.md"
    assert not raw_copy.exists()
    assert sources[0].read_bytes() == original
    assert (kb_dir / "raw" / "01-Basic-Arithmetic-Workbook" / "two.md").is_file()
    folder = manager.get_linked_folders("math")[0]
    assert _synced_relative_paths(manager) == {
        "01-Basic-Arithmetic-Workbook/two.md",
        "01-Basic-Arithmetic-Workbook/three.md",
    }
    assert folder["file_count"] == 2
    changes = manager.detect_folder_changes("math", folder_id)
    assert changes["new_files"] == [str(sources[0])]
    assert changes["modified_files"] == []
    assert source.is_dir()


def test_deleting_a_raw_folder_forgets_sync_state_and_keeps_the_source(
    tmp_path: Path,
) -> None:
    manager, kb_dir, folder_id, source, sources = _synced_workbook(tmp_path)
    originals = {path: path.read_bytes() for path in sources}
    external = tmp_path / "secret.md"
    external.write_text("do not delete", encoding="utf-8")
    link = kb_dir / "raw" / "01-Basic-Arithmetic-Workbook" / "link.md"
    link.symlink_to(external)

    removal = remove_raw_tree(kb_dir, kb_dir / "raw" / "01-Basic-Arithmetic-Workbook")

    assert removal.rel_path == "01-Basic-Arithmetic-Workbook"
    assert set(removal.removed_files) == {
        "01-Basic-Arithmetic-Workbook/one.md",
        "01-Basic-Arithmetic-Workbook/two.md",
        "01-Basic-Arithmetic-Workbook/three.md",
    }
    assert removal.was_indexed is True
    assert not (kb_dir / "raw" / "01-Basic-Arithmetic-Workbook").exists()
    assert (kb_dir / "raw" / "other.md").read_text(encoding="utf-8") == "other"
    assert external.read_text(encoding="utf-8") == "do not delete"
    for path, payload in originals.items():
        assert path.read_bytes() == payload
    folder = manager.get_linked_folders("math")[0]
    assert folder["synced_files"] == {}
    assert folder["file_count"] == 0
    changes = manager.detect_folder_changes("math", folder_id)
    assert changes["new_files"] == sorted(str(path) for path in sources)
    assert changes["modified_files"] == []
    metadata = json.loads((kb_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["file_hashes"] == {"other.md": "other"}


def test_deleting_an_empty_folder_forgets_sources_still_marked_synced(
    tmp_path: Path,
) -> None:
    """The leftover folder in the Files panel still owns the sync ledger."""
    manager, kb_dir, folder_id, _source, sources = _synced_workbook(tmp_path)
    workbook = kb_dir / "raw" / "01-Basic-Arithmetic-Workbook"
    for path in workbook.glob("*.md"):
        path.unlink()
    assert manager.get_linked_folders("math")[0]["file_count"] == 3

    remove_raw_tree(kb_dir, workbook)

    assert not workbook.exists()
    folder = manager.get_linked_folders("math")[0]
    assert folder["synced_files"] == {}
    assert folder["file_count"] == 0
    assert manager.detect_folder_changes("math", folder_id)["new_files"] == sorted(
        str(path) for path in sources
    )
    for path in sources:
        assert path.is_file()


def test_missing_raw_copy_is_new_even_when_the_ledger_still_tracks_it(tmp_path: Path) -> None:
    """Stale synced_files entries from a delete that did not update the ledger."""
    manager, kb_dir, folder_id, _source, sources = _synced_workbook(tmp_path)
    for path in (kb_dir / "raw" / "01-Basic-Arithmetic-Workbook").glob("*.md"):
        path.unlink()
    originals = {path: path.read_bytes() for path in sources}

    changes = manager.detect_folder_changes("math", folder_id)

    assert changes["new_files"] == sorted(str(path) for path in sources)
    assert changes["modified_files"] == []
    assert changes["has_changes"] is True
    for path, payload in originals.items():
        assert path.read_bytes() == payload
    assert manager.get_linked_folders("math")[0]["file_count"] == 3


def test_unlink_removes_only_the_source_registration(tmp_path: Path) -> None:
    manager, _metadata_file, folder_id, _doc = _manager_with_linked_folder(tmp_path)
    kb_dir = manager.base_dir / "kb"
    raw_file = kb_dir / "raw" / "note.md"
    index_marker = kb_dir / "version-1" / "index.marker"
    raw_file.parent.mkdir(parents=True)
    index_marker.parent.mkdir(parents=True)
    raw_file.write_text("imported", encoding="utf-8")
    index_marker.write_text("index", encoding="utf-8")

    assert manager.unlink_folder("kb", folder_id) is True

    assert manager.get_linked_folders("kb") == []
    assert raw_file.read_text(encoding="utf-8") == "imported"
    assert index_marker.read_text(encoding="utf-8") == "index"

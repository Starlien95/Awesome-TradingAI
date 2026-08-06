from __future__ import annotations

from pathlib import Path

from quant_bench.artifacts import LocalArtifactStore, verify_run


def test_checksums_detect_corruption(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "run")
    store.write_text("metrics.txt", "ok\n")
    store.write_checksums()
    assert verify_run(store.root) == []
    (store.root / "metrics.txt").write_text("corrupt\n", encoding="utf-8")
    assert verify_run(store.root) == ["checksum mismatch: metrics.txt"]


def test_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "run")
    try:
        store.write_text("../outside.txt", "bad")
    except ValueError as exc:
        assert "inside" in str(exc)
    else:
        raise AssertionError("path traversal was accepted")

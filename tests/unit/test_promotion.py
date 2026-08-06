from __future__ import annotations

from pathlib import Path

import yaml

from quant_bench.runtime.promotion import promote_artifact


def test_promote_artifact_copies_bytes_updates_config_and_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "trained" / "model.pkl"
    source.parent.mkdir()
    source.write_bytes(b"opaque-model-bytes")
    config = tmp_path / "config" / "runtime.yaml"
    config.parent.mkdir()
    config.write_text("model:\n  path: old.pkl\n", encoding="utf-8")

    workspace = tmp_path / "workspace"
    result = promote_artifact(
        source,
        workspace,
        model_id="example_model",
        frequency="1h",
        config_path=config,
    )

    target = Path(result["target"])
    assert target.read_bytes() == source.read_bytes()
    assert Path(result["manifest"]).is_file()
    updated = yaml.safe_load(config.read_text(encoding="utf-8"))
    resolved = (config.parent / updated["model"]["path"]).resolve()
    assert resolved == target


def test_promote_dry_run_does_not_write(tmp_path: Path) -> None:
    source = tmp_path / "model.pth"
    source.write_bytes(b"weights")
    result = promote_artifact(
        source,
        tmp_path / "workspace",
        model_id="model",
        frequency="4h",
        dry_run=True,
    )
    assert not Path(result["target"]).exists()
    assert not Path(result["manifest"]).exists()

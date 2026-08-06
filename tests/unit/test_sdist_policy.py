from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path


def _sdist_checker():
    path = Path(__file__).resolve().parents[2] / "tools" / "check_sdist.py"
    spec = importlib.util.spec_from_file_location("quant_bench_sdist_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_file(archive: tarfile.TarFile, name: str, value: bytes = b"fixture") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def test_sdist_policy_rejects_private_unsafe_and_model_members(tmp_path: Path) -> None:
    checker = _sdist_checker()
    sdist = tmp_path / "unsafe.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        _add_file(archive, "quant_bench-0.1.0/../escape.txt")
        _add_file(archive, "quant_bench-0.1.0/local_data/private.csv")
        _add_file(archive, "quant_bench-0.1.0/model_weights/model.pt")
        _add_file(archive, "quant_bench-0.1.0/.env")
    errors = checker.audit_sdist(sdist)
    joined = "\n".join(errors)
    assert "unsafe source distribution path" in joined
    assert "private or generated directory" in joined
    assert "model or runtime artifact" in joined
    assert "environment file" in joined


def test_sdist_policy_rejects_special_files_and_invalid_metadata(tmp_path: Path) -> None:
    checker = _sdist_checker()
    sdist = tmp_path / "metadata.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        _add_file(archive, "quant_bench-0.1.0/PKG-INFO", b"\xff")
        device = tarfile.TarInfo("quant_bench-0.1.0/device")
        device.type = tarfile.CHRTYPE
        archive.addfile(device)

    errors = checker.audit_sdist(sdist)
    joined = "\n".join(errors)
    assert "special file" in joined
    assert "PKG-INFO is unreadable" in joined

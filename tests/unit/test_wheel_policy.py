from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


def _wheel_checker():
    path = Path(__file__).resolve().parents[2] / "tools" / "check_wheel.py"
    spec = importlib.util.spec_from_file_location("quant_bench_wheel_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wheel_policy_rejects_private_unsafe_and_model_members(tmp_path: Path) -> None:
    checker = _wheel_checker()
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")
        archive.writestr("quant_bench/local_data/private.csv", "private")
        archive.writestr("quant_bench/model_weights/model.pt", "weight")
        archive.writestr("quant_bench/.env", "secret")
    errors = checker.audit_wheel(wheel)
    joined = "\n".join(errors)
    assert "unsafe wheel member path" in joined
    assert "private or generated directory" in joined
    assert "model or runtime artifact" in joined
    assert "environment file" in joined


def test_wheel_policy_reports_invalid_utf8_metadata(tmp_path: Path) -> None:
    checker = _wheel_checker()
    wheel = tmp_path / "metadata.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("quant_bench-0.1.0.dist-info/METADATA", b"\xff")

    errors = checker.audit_wheel(wheel)
    assert any("METADATA is unreadable" in error for error in errors)

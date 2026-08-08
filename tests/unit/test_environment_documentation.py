from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def _dependency_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0]
        .lower()
        .replace("_", "-")
        for requirement in requirements
    }


def test_environment_document_covers_every_declared_extra() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"]["optional-dependencies"])
    environment = (ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")

    missing = sorted(extra for extra in extras if f"`{extra}`" not in environment)
    assert missing == []


def test_all_extra_covers_every_user_capability() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    user_groups = set(extras) - {"all", "dev", "release"}
    required = set().union(*(_dependency_names(extras[group]) for group in user_groups))
    provided = _dependency_names(extras["all"])

    assert sorted(required - provided) == []


def test_news_extra_declares_its_torch_runtime() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    news = _dependency_names(pyproject["project"]["optional-dependencies"]["news"])

    assert "torch" in news


def test_environment_document_is_linked_from_reader_entrypoints() -> None:
    for relative in ("README.md", "README.zh-CN.md", "docs/index.md", "mkdocs.yml"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ENVIRONMENT.md" in text, relative


def test_offline_quickstart_uses_core_install() -> None:
    tutorial = (ROOT / "docs" / "tutorials" / "quickstart.md").read_text(
        encoding="utf-8"
    )
    assert 'python -m pip install -e ".[dev]"' not in tutorial
    assert "python -m pip install -e ." in tutorial

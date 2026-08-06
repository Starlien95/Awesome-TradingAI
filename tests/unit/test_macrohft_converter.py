from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _converter():
    path = REPOSITORY_ROOT / "tools" / "convert_macrohft_external.py"
    spec = importlib.util.spec_from_file_location("quant_bench_macrohft_converter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(tmp_path: Path, confirmation: str) -> argparse.Namespace:
    return argparse.Namespace(
        upstream_root=tmp_path / "upstream",
        hyperagent=tmp_path / "hyperagent.pkl",
        subagents_dir=tmp_path / "subagents",
        output_dir=tmp_path / "output",
        name="macrohft_test",
        module="fake_net",
        subagent_class="Subagent",
        hyperagent_class="Hyperagent",
        state_dim=4,
        trend_dim=4,
        action_dim=2,
        subagent_hidden_dim=3,
        hyperagent_hidden_dim=3,
        confirm=confirmation,
    )


def test_converter_requires_explicit_trust_confirmation(tmp_path: Path) -> None:
    converter = _converter()
    with pytest.raises(PermissionError, match="TRUSTED_MACROHFT_SOURCE"):
        converter.convert(_arguments(tmp_path, "not-confirmed"))


def test_converter_exports_a_torchscript_bundle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    converter = _converter()
    arguments = _arguments(tmp_path, converter.CONFIRMATION)
    arguments.upstream_root.mkdir()
    arguments.subagents_dir.mkdir()
    (arguments.upstream_root / "fake_net.py").write_text(
        """\
import torch


class Subagent(torch.nn.Module):
    def __init__(self, state_dim, trend_dim, action_dim, hidden_dim):
        super().__init__()
        self.projection = torch.nn.Linear(state_dim + trend_dim, action_dim)

    def forward(self, single, trend, previous_action):
        del previous_action
        return self.projection(torch.cat([single, trend], dim=1))


class Hyperagent(torch.nn.Module):
    def __init__(self, state_dim, trend_dim, action_dim, hidden_dim):
        super().__init__()
        self.projection = torch.nn.Linear(state_dim + trend_dim + 2, 6)

    def forward(self, single, trend, context, previous_action):
        del previous_action
        values = self.projection(torch.cat([single, trend, context], dim=1))
        return torch.softmax(values, dim=1)
""",
        encoding="utf-8",
    )

    import sys

    sys.path.insert(0, str(arguments.upstream_root))
    try:
        fake = __import__("fake_net")
        hyperagent = fake.Hyperagent(4, 4, 2, 3)
        torch.save(hyperagent.state_dict(), arguments.hyperagent)
        for name in converter.SUBAGENT_NAMES:
            subagent = fake.Subagent(4, 4, 2, 3)
            torch.save(subagent.state_dict(), arguments.subagents_dir / f"{name}.pkl")

        result = converter.convert(arguments)
    finally:
        sys.path.remove(str(arguments.upstream_root))
        sys.modules.pop("fake_net", None)

    assert Path(result["manifest_path"]).is_file()
    assert (arguments.output_dir / "macrohft_test_hyperagent.pt").is_file()
    assert sorted(path.name for path in (arguments.output_dir / "macrohft_test_subagents").glob("*.pt")) == [
        "slope_1.pt",
        "slope_2.pt",
        "slope_3.pt",
        "vol_1.pt",
        "vol_2.pt",
        "vol_3.pt",
    ]

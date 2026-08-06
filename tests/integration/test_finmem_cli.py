from __future__ import annotations

import json
from pathlib import Path

from quant_bench.cli.app import main


def _dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                f"2024-01-{day:02d}": {"prices": 100.0 + day, "news": []}
                for day in range(1, 9)
            }
        ),
        encoding="utf-8",
    )


def test_finmem_cli_init_map_and_paper_step(tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _dataset(data_dir / "btc.json")
    workspace = tmp_path / "workspace"
    assert (
        main(
            [
                "finmem",
                "init",
                "--workspace",
                str(workspace),
                "--data-dir",
                str(data_dir),
                "--symbol",
                "BTC",
            ]
        )
        == 0
    )
    assert (workspace / "methods" / "finmem" / "configs" / "investorbench.json").is_file()

    action = tmp_path / "action.json"
    action.write_text('{"symbol":"BTC","position":1}', encoding="utf-8")
    assert main(["finmem", "map-action", "--input", str(action)]) == 0
    assert '"signal": "buy"' in capsys.readouterr().out

    state = tmp_path / "paper.json"
    assert (
        main(
            [
                "finmem",
                "paper-step",
                "--state",
                str(state),
                "--price",
                "50000",
                "--target-position",
                "-1",
            ]
        )
        == 0
    )
    assert json.loads(state.read_text(encoding="utf-8"))["position_state"] == "short"


def test_finmem_cycle_rejects_networkless_account_access(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    code = main(
        [
            "finmem",
            "cycle",
            "--workspace",
            str(tmp_path / "workspace"),
            "--config",
            str(config),
            "--mode",
            "demo",
            "--query-account",
        ]
    )
    assert code == 1
    assert "require --allow-network" in capsys.readouterr().err

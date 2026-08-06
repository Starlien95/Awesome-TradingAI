"""Generate a compact Markdown run table without optional plotting dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


def write_workspace_summary(workspace: Path, output: Path | None = None) -> Path:
    workspace_path = workspace.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((workspace_path / "runs").glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        metrics_path = manifest_path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        rows.append(
            {
                "run_id": manifest.get("run_id", ""),
                "recipe": manifest.get("recipe_id", ""),
                "protocol": manifest.get("protocol_id", ""),
                "dataset": manifest.get("dataset_sha256", ""),
                "total_return": metrics.get("total_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
            }
        )

    compatibility_groups = {(row["protocol"], row["dataset"]) for row in rows}
    comparable = len(compatibility_groups) <= 1
    target = output.expanduser().resolve() if output else workspace_path / "reports" / "benchmark_summary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark summary",
        "",
        f"Completed runs: {len(rows)}",
        "",
        f"Directly comparable: {'yes' if comparable else 'no'}",
        "",
    ]
    if not comparable:
        lines.extend(
            [
                "> Runs use more than one protocol or dataset hash. Compare them only after reviewing those differences.",
                "",
            ]
        )
    lines.extend(
        [
            "| run_id | recipe | protocol | total_return | sharpe | max_drawdown |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['recipe']} | {row['protocol']} | "
            f"{_format_metric(row['total_return'])} | {_format_metric(row['sharpe'])} | "
            f"{_format_metric(row['max_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "This report describes offline benchmark runs. It is not evidence of paper, demo, or live trading performance.",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")
    return target

"""Convert trusted, locally obtained MacroHFT checkpoints to TorchScript.

This migration tool imports an upstream source tree supplied by the user.  The
upstream source and checkpoints are never copied into quant-bench.  Run it only
after reviewing the upstream terms and the pickle files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUBAGENT_NAMES = ("slope_1", "slope_2", "slope_3", "vol_1", "vol_2", "vol_3")
CONFIRMATION = "TRUSTED_MACROHFT_SOURCE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_state_dict(torch: Any, path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"checkpoint must contain a state dict: {path}")
    state = loaded.get("state_dict", loaded)
    if not isinstance(state, dict):
        raise TypeError(f"state_dict must be a mapping: {path}")
    return state


def _trace_and_save(torch: Any, module: Any, inputs: tuple[Any, ...], target: Path) -> None:
    module.eval()
    traced = torch.jit.trace(module, inputs, strict=True)
    traced.save(str(target))


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirm != CONFIRMATION:
        raise PermissionError(f"pass --confirm {CONFIRMATION} after reviewing the source and checkpoints")

    upstream_root = args.upstream_root.expanduser().resolve()
    hyperagent_checkpoint = args.hyperagent.expanduser().resolve()
    subagent_checkpoints = args.subagents_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not upstream_root.is_dir():
        raise FileNotFoundError(f"upstream source directory not found: {upstream_root}")
    if not hyperagent_checkpoint.is_file():
        raise FileNotFoundError(f"hyperagent checkpoint not found: {hyperagent_checkpoint}")
    output_root.mkdir(parents=True, exist_ok=True)
    output_subagents = output_root / f"{args.name}_subagents"
    output_subagents.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(upstream_root))
    upstream_module = importlib.import_module(args.module)
    subagent_type = getattr(upstream_module, args.subagent_class)
    hyperagent_type = getattr(upstream_module, args.hyperagent_class)

    import torch

    single = torch.zeros((1, args.state_dim), dtype=torch.float32)
    trend = torch.zeros((1, args.trend_dim), dtype=torch.float32)
    context = torch.zeros((1, 2), dtype=torch.float32)
    previous_action = torch.zeros((1,), dtype=torch.long)

    hyperagent = hyperagent_type(
        args.state_dim,
        args.trend_dim,
        args.action_dim,
        args.hyperagent_hidden_dim,
    )
    hyperagent.load_state_dict(_load_state_dict(torch, hyperagent_checkpoint), strict=True)
    hyperagent_output = output_root / f"{args.name}_hyperagent.pt"
    _trace_and_save(torch, hyperagent, (single, trend, context, previous_action), hyperagent_output)

    sources: list[dict[str, str]] = [
        {"role": "hyperagent", "path": str(hyperagent_checkpoint), "sha256": _sha256(hyperagent_checkpoint)}
    ]
    outputs: list[str] = [str(hyperagent_output)]
    for name in SUBAGENT_NAMES:
        source = subagent_checkpoints / f"{name}.pkl"
        if not source.is_file():
            raise FileNotFoundError(f"subagent checkpoint not found: {source}")
        subagent = subagent_type(
            args.state_dim,
            args.trend_dim,
            args.action_dim,
            args.subagent_hidden_dim,
        )
        subagent.load_state_dict(_load_state_dict(torch, source), strict=True)
        target = output_subagents / f"{name}.pt"
        _trace_and_save(torch, subagent, (single, trend, previous_action), target)
        sources.append({"role": name, "path": str(source), "sha256": _sha256(source)})
        outputs.append(str(target))

    manifest = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "macrohft-torchscript-bundle-v1",
        "upstream_source": str(upstream_root),
        "upstream_commit": _source_commit(upstream_root),
        "upstream_module": args.module,
        "license_review_required": True,
        "source_checkpoints": sources,
        "outputs": outputs,
    }
    manifest_path = output_root / f"{args.name}_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--hyperagent", type=Path, required=True)
    parser.add_argument("--subagents-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="macrohft_v1_5m")
    parser.add_argument("--module", default="model.net")
    parser.add_argument("--subagent-class", default="subagent")
    parser.add_argument("--hyperagent-class", default="hyperagent")
    parser.add_argument("--state-dim", type=int, default=52)
    parser.add_argument("--trend-dim", type=int, default=52)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--subagent-hidden-dim", type=int, default=64)
    parser.add_argument("--hyperagent-hidden-dim", type=int, default=32)
    parser.add_argument("--confirm", required=True)
    return parser


def main() -> None:
    result = convert(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

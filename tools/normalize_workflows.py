"""Normalize every legacy workflow using public constructor defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_bench.config.legacy_workflows import audit_workflows, normalize_workflow_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Workflow root, for example src/quant_bench/resources/qlib_workflows",
    )
    parser.add_argument("--write", action="store_true", help="Write normalized files in place")
    args = parser.parse_args()

    files = sorted(args.root.glob("**/*.yaml"))
    changed = 0
    for path in files:
        file_changed, _ = normalize_workflow_file(path, write=args.write)
        changed += int(file_changed)
    audit = audit_workflows(args.root, require_normalized=args.write)
    print(json.dumps({"files": len(files), "changed": changed, "audit": audit.as_dict()}, indent=2))
    return 0 if (audit.passed or not args.write) else 1


if __name__ == "__main__":
    raise SystemExit(main())

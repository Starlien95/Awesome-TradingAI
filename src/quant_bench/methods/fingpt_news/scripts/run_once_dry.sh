#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/quant-bench}"
quant-bench runtime fingpt-dry-run --workspace "$workspace"

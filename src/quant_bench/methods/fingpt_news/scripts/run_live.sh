#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_live.sh CONFIG [WORKSPACE] DEMO_ORDERS|LIVE_ORDERS" >&2
  exit 2
fi

config="$1"
workspace="${2:-${XDG_DATA_HOME:-$HOME/.local/share}/quant-bench}"
confirmation="${3:-}"

if [[ "$confirmation" != "DEMO_ORDERS" && "$confirmation" != "LIVE_ORDERS" ]]; then
  echo "refusing to start: pass DEMO_ORDERS or LIVE_ORDERS after reviewing CONFIG" >&2
  exit 2
fi

python -m quant_bench.methods.fingpt_news.run_fingpt_news \
  --config "$config" \
  --workspace "$workspace" \
  --allow-network \
  --execute-orders \
  --confirm "$confirmation"

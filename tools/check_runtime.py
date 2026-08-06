from __future__ import annotations

import argparse
import importlib

import requests

GROUPS = {
    "core": ["numpy", "pandas", "yaml", "requests"],
    "traditional_ml": ["sklearn", "joblib", "lightgbm", "xgboost", "torch", "qlib"],
    "reinforcement_learning": ["torch", "scipy", "sklearn"],
    "fingpt_news": ["torch", "transformers", "peft", "bs4"],
    "dashboard": ["streamlit", "plotly", "yaml", "pandas"],
    "okx": ["okx"],
}


def check_imports(groups: list[str]) -> bool:
    ok = True
    for group in groups:
        print(f"[{group}]")
        for module in GROUPS[group]:
            try:
                importlib.import_module(module)
                print(f"  OK   {module}")
            except Exception as exc:
                ok = False
                print(f"  MISS {module}: {type(exc).__name__}: {str(exc)[:120]}")
    return ok


def check_okx_public() -> bool:
    url = "https://www.okx.com/api/v5/market/ticker"
    try:
        response = requests.get(url, params={"instId": "BTC-USDT"}, timeout=10)
        response.raise_for_status()
        payload = response.json()
        success = payload.get("code") == "0" and bool(payload.get("data"))
        print(f"[okx_public] {'OK' if success else 'FAIL'} {str(payload)[:200]}")
        return success
    except Exception as exc:
        print(f"[okx_public] FAIL {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check runtime dependencies and public API connectivity.")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS.keys()), choices=list(GROUPS.keys()))
    parser.add_argument("--check-okx-public", action="store_true")
    args = parser.parse_args()
    ok = check_imports(args.groups)
    if args.check_okx_public:
        ok = check_okx_public() and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

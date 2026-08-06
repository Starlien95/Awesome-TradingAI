"""Compatibility wrapper for packaged prediction consistency checks."""

from quant_bench.integrations.qlib.legacy.verify_prediction_consistency import verify_consistency

if __name__ == "__main__":
    verify_consistency()

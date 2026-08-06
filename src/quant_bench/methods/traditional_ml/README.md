# Traditional ML runtime adapters

The optional runtime includes adapters for MLP, TCN, XGBoost, GATS, LSTM, TRA, DoubleEnsemble, TabNet, and LightGBM artifacts. The process catalog groups them by bar frequency:

```bash
quant-bench runtime list
quant-bench runtime init traditional_1h --output-dir ./qb-workspace/local-config/traditional_1h
```

Target selection calls the shared `threshold_topk_dropout` implementation. The default execution policy is `passive_topk_dropout` with `risk_degree: 0.95`; retained targets do not receive a repeated buy solely to restore equal weights.

Model weights are never packaged. Promote a reviewed local artifact with `quant-bench artifacts promote`, then run `runtime check`. See `docs/how-to/runtime.md` for demo and live gates.

Test an exported config and local artifact without contacting OKX:

```bash
python -m quant_bench.methods.traditional_ml.smoke_test \
  --model mlp \
  --config ./qb-workspace/local-config/traditional_1h/config_mlp.yaml \
  --model-path ./qb-workspace/models/runtime/mlp.pkl
```

The smoke command builds deterministic synthetic OHLCV bars, loads the selected adapter, runs `predict`, and passes the predictions through the shared signal-selection path. It supports every migrated traditional adapter through `--model`.

TabNet and TCN configurations can apply rolling median/MAD normalization when a separately fitted Qlib processor is unavailable. This is a runtime approximation and must be disclosed when comparing it with the original Qlib evaluation path.

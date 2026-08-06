# MacroHFT v1 external integration

The official MacroHFT repository did not include a software license at audited commit `31e5ef41f93b2aea6e63e6ae2267675c997a8e54`. Quant-bench therefore does not redistribute its network implementation or checkpoints. The runtime adapter accepts a user-created TorchScript bundle containing six subagents (`slope_1..3`, `vol_1..3`) and one hyperagent. Its runtime score is `Q[buy] - Q[hold]`.

Available processes:

```text
macrohft_5m
macrohft_15m
macrohft_1h
macrohft_4h
```

Export a safe config with:

```bash
quant-bench runtime init macrohft_5m --output-dir ./qb-workspace/local-config/macrohft_5m
```

The template is single-symbol ETH, uses `risk_degree: 0.95`, and requires both the hyperagent artifact and its subagent directory. No source or weights from the official repository are included in the package.

If you have separately obtained permission to use the upstream source and have reviewed every checkpoint, export it locally:

```bash
python tools/convert_macrohft_external.py \
  --upstream-root /path/to/MacroHFT \
  --hyperagent /path/to/hyperagent.pkl \
  --subagents-dir /path/to/subagents \
  --output-dir ./qb-workspace/models/runtime/5m \
  --name macrohft_v1_5m \
  --confirm TRUSTED_MACROHFT_SOURCE
```

The converter imports the user-supplied source only for the conversion process. It writes traced `.pt` files and a provenance manifest; it never copies upstream Python source into quant-bench. Pickle input remains a trusted-local migration boundary.

Run an offline artifact smoke test before configuring demo trading:

```bash
python -m quant_bench.methods.reinforcement_learning.macrohft_v1.smoke_test \
  --timeframe 5m \
  --config ./qb-workspace/local-config/macrohft_5m/config_macrohft_v1.yaml \
  --model-path ./qb-workspace/models/runtime/5m/macrohft_v1_5m_hyperagent.pt \
  --subagents-dir ./qb-workspace/models/runtime/5m/macrohft_v1_5m_subagents
```

This command uses deterministic synthetic OHLCV bars and does not create an OKX client.

# Configuration reference

`ExperimentConfig` is the authoritative schema. YAML is an input format; runtime code receives a validated Pydantic object.

## Top-level groups

| Group | Purpose |
| --- | --- |
| `experiment` | Name, seed, workspace |
| `data` | Dataset id, source, path, terms, checksum |
| `universe` | Versioned symbol universe |
| `frequency` | Canonical id, pandas rule, annualization |
| `features` | Feature id, plugin, version, parameters |
| `label` | Task, formula, horizon, scale |
| `model` | Catalog id, plugin, backend, class target, parameters |
| `protocol` | Tier, split, strategy, costs, capital, seeds |
| `tracker` | Local, Qlib, MLflow, or disabled tracking |
| `runtime` | Offline research safety boundary |

## Overrides

```bash
quant-bench config render crypto_smoke_v1 \
  --set experiment.seed=7 \
  --set protocol.strategy.top_k=2
```

Overrides must target an existing key. Values are parsed as YAML scalars or collections, validated, and included in the resolved config hash.

## Protocol validation

- `smoke` requires at least one seed.
- `standard` requires at least five unique seeds.
- `publication` requires at least twenty unique seeds.
- Standard and publication protocols require `purge_bars >= label.horizon_bars`.
- Train, validation, and test ranges cannot overlap.
- Research mode rejects network access.

## Model config generation

```bash
quant-bench config init \
  --model qlib_transformer_ts \
  --feature-set crypto_alpha52_v1 \
  --output transformer.yaml
```

The generator starts from constructor defaults and applies structural feature dimensions. It does not add tuned benchmark parameters.

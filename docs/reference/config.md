# Configuration reference

Canonical experiments use a Pydantic-validated YAML document.

| Section | Purpose |
| --- | --- |
| `experiment` | name, seed, workspace |
| `data` | dataset id, source, path, license field, optional checksum |
| `universe` | versioned symbol set |
| `frequency` | id, pandas rule, annualization |
| `features` | feature set id, plugin, version, parameters |
| `label` | id, task, horizon, formula, scale |
| `model` | catalog id, plugin, backend, class information, parameters |
| `protocol` | tier, train/valid/test split, strategy, costs, cash, seeds |
| `tracker` | local or optional tracking backend |
| `runtime` | research mode and explicit network permission |

Render a complete example:

```bash
quant-bench config render crypto_smoke_v1
```

Overrides use dotted paths:

```bash
quant-bench config validate crypto_smoke_v1 \
  --set model.parameters.ridge=0.001 \
  --set protocol.strategy.top_k=2
```

Unknown paths and invalid values are rejected. Every run stores the complete resolved YAML and a canonical JSON hash. Lists are replaced as complete values; nested dictionaries are updated only through an existing dotted field.

Runtime configs use a separate schema because they contain scheduler, account-mode, model-artifact, and execution settings. Export them with `runtime init`; do not add credentials to the file.

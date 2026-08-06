# Add a model

Models can live in this repository or in a separate Python distribution. New integrations should implement the public plugin contract and advertise their capabilities before execution.

## Scaffold a plugin

```bash
quant-bench plugins scaffold my_model --target ./plugins
```

The generated class needs these members:

```python
class MyModelPlugin:
    plugin_id = "my_model"
    contract_version = "1"

    def capabilities(self): ...
    def fit(self, dataset, context=None): ...
    def predict(self, model, dataset): ...
    def save(self, model, target): ...
    def load(self, source): ...
```

Use only public objects from `quant_bench.contracts` in the boundary. Qlib, PyTorch, MLflow, and exchange clients stay inside the plugin implementation.

## Register the entry point

In the plugin distribution's `pyproject.toml`:

```toml
[project.entry-points."quant_bench.models"]
my_model = "my_package.plugin:MyModelPlugin"
```

Install it and confirm discovery:

```bash
python -m pip install -e ./my-package
quant-bench plugins list
```

## Required tests

At minimum, test:

1. capability schema validation;
2. fit and predict on the packaged fixture;
3. canonical prediction columns and index uniqueness;
4. save and load parity;
5. deterministic behavior when the capability claims determinism;
6. a missing optional dependency error with an installation hint;
7. rejection of incompatible feature shapes and dataset kinds.

Include a model card covering the algorithm source, license, dependencies, expected features, resource requirements, artifact format, intended use, and limitations. Do not commit pretrained weights without verified redistribution permission.

## Qlib catalog entries

Qlib wrappers use `src/quant_bench/resources/model_catalog.yaml`. Add aliases, module path, class name, default kwargs, optional extra, dataset kinds, and status. Run:

```bash
quant-bench models show <model_id>
python -m pytest -q tests/unit/test_registry.py
```

Workflow templates are optional. Adding a frequency should reuse a model entry and change only the data, feature, protocol, or runtime configuration.

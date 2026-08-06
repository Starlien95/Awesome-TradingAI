# Direct dependency license inventory

Audit date: 2026-07-18

These packages are declared by `pyproject.toml`. They are installed separately and are not vendored into the quant-bench wheel. The table records the license metadata published by PyPI and the upstream projects on the audit date. Exact resolved versions and transitive packages are captured in the release SBOM and JSON license report.

| Package | Declared profile | License family recorded by upstream metadata |
| --- | --- | --- |
| NumPy | core | BSD-3-Clause with permissive bundled components |
| pandas | core | BSD-3-Clause |
| platformdirs | core | MIT |
| Pydantic | core | MIT |
| PyYAML | core | MIT |
| pyqlib 0.9.7 | qlib, data-qlib | MIT |
| LightGBM | lgbm | MIT |
| XGBoost | xgboost | Apache-2.0 |
| CatBoost | catboost | Apache-2.0 |
| PyTorch | torch | BSD, Apache-2.0, MIT and other permissive bundled components |
| Optuna | tuning | MIT |
| MLflow | tracking | Apache-2.0 |
| CCXT | data-ccxt, data-qlib | MIT |
| Loguru | data-qlib | MIT |
| tqdm | data-qlib | MPL-2.0 and MIT |
| Plotly | dashboard | MIT |
| Streamlit | dashboard | Apache-2.0 |
| Matplotlib | visualization | PSF-based Matplotlib license |
| seaborn | visualization | BSD-3-Clause |
| Material for MkDocs | docs | MIT |
| CycloneDX Python SBOM | release | Apache-2.0 |
| pip-licenses | release | MIT |
| tomli | dev, release on Python 3.10 | MIT |
| Beautiful Soup | runtime | MIT |
| joblib | runtime | BSD-3-Clause |
| Requests | runtime | Apache-2.0 |
| SciPy | runtime | BSD-3-Clause with separately documented binary components |
| python-okx | runtime-okx | MIT |
| Accelerate | news | Apache-2.0 |
| PEFT | news | Apache-2.0 |
| safetensors | news | Apache-2.0 |
| SentencePiece | news | Apache-2.0 |
| Transformers | news | Apache-2.0 |
| build | dev | MIT |
| mypy | dev | MIT |
| pandas-stubs | dev | BSD-3-Clause |
| pytest | dev | MIT |
| pytest-cov | dev | MIT |
| Ruff | dev | MIT |
| types-PyYAML | dev | Apache-2.0 |
| setuptools | build system | MIT |
| wheel | build system | MIT |

The current direct set contains no dependency that requires relicensing quant-bench source. Binary wheels can carry additional notices for compiled libraries, so the generated release report must be reviewed for the exact platform artifact.

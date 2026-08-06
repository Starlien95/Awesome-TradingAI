# Model support matrix

The built-in catalog is verified against Qlib `v0.9.7`. It contains 32 upstream Qlib model wrappers, one local TFT wrapper, and one dependency-light baseline.

## Meaning of support

Every catalog entry provides:

1. A stable `model_id` and legacy aliases.
2. A verified Python module and class target.
3. Constructor defaults without private tuned values.
4. Feature-shape and dataset-kind capabilities.
5. Optional dependency metadata.
6. Typed config generation.
7. Generic Qlib import and instantiation through `QlibModelPlugin`.

Training still requires the declared data contract. A model marked `requires_metadata` is supported by the registry and adapter while execution remains blocked until the required metadata is supplied.

## Catalog

| Model id | Qlib class | Dataset kinds | Extra | Status |
| --- | --- | --- | --- | --- |
| `qlib_lightgbm` | `gbdt.LGBModel` | `DatasetH` | `qlib,lgbm` | stable |
| `qlib_highfreq_lightgbm` | `highfreq_gdbt_model.HFLGBModel` | `DatasetH` | `qlib,lgbm` | requires high-frequency targets |
| `qlib_xgboost` | `xgboost.XGBModel` | `DatasetH` | `qlib,xgboost` | stable |
| `qlib_catboost` | `catboost_model.CatBoostModel` | `DatasetH` | `qlib,catboost` | stable |
| `qlib_double_ensemble` | `double_ensemble.DEnsembleModel` | `DatasetH` | `qlib,lgbm` | stable |
| `qlib_linear` | `linear.LinearModel` | `DatasetH` | `qlib` | stable |
| `qlib_mlp` | `pytorch_nn.DNNModelPytorch` | `DatasetH` | `qlib,torch` | stable |
| `qlib_adarnn` | `pytorch_adarnn.ADARNN` | `DatasetH`, `TSDatasetH` | `qlib,torch` | experimental |
| `qlib_add` | `pytorch_add.ADD` | `DatasetH`, `TSDatasetH` | `qlib,torch` | experimental |
| `qlib_alstm` | `pytorch_alstm.ALSTM` | `DatasetH` | `qlib,torch` | stable |
| `qlib_alstm_ts` | `pytorch_alstm_ts.ALSTM` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_gats` | `pytorch_gats.GATs` | `DatasetH` | `qlib,torch` | stable |
| `qlib_gats_ts` | `pytorch_gats_ts.GATs` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_general_ptnn` | `pytorch_general_nn.GeneralPTNN` | `DatasetH`, `TSDatasetH` | `qlib,torch` | stable |
| `qlib_gru` | `pytorch_gru.GRU` | `DatasetH` | `qlib,torch` | stable |
| `qlib_gru_ts` | `pytorch_gru_ts.GRU` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_hist` | `pytorch_hist.HIST` | `DatasetH` | `qlib,torch` | requires concept metadata |
| `qlib_igmtf` | `pytorch_igmtf.IGMTF` | `DatasetH` | `qlib,torch` | experimental |
| `qlib_krnn` | `pytorch_krnn.KRNN` | `DatasetH`, `TSDatasetH` | `qlib,torch` | stable |
| `qlib_localformer` | `pytorch_localformer.LocalformerModel` | `DatasetH` | `qlib,torch` | stable |
| `qlib_localformer_ts` | `pytorch_localformer_ts.LocalformerModel` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_lstm` | `pytorch_lstm.LSTM` | `DatasetH` | `qlib,torch` | stable |
| `qlib_lstm_ts` | `pytorch_lstm_ts.LSTM` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_sandwich` | `pytorch_sandwich.Sandwich` | `DatasetH`, `TSDatasetH` | `qlib,torch` | experimental |
| `qlib_sfm` | `pytorch_sfm.SFM` | `DatasetH`, `TSDatasetH` | `qlib,torch` | stable |
| `qlib_tabnet` | `pytorch_tabnet.TabnetModel` | `DatasetH` | `qlib,torch` | stable |
| `qlib_tcn` | `pytorch_tcn.TCN` | `DatasetH` | `qlib,torch` | stable |
| `qlib_tcn_ts` | `pytorch_tcn_ts.TCN` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_tcts` | `pytorch_tcts.TCTS` | `DatasetH`, `TSDatasetH` | `qlib,torch` | experimental |
| `qlib_tra` | `pytorch_tra.TRAModel` | `MTSDatasetH` | `qlib,torch` | stable |
| `qlib_transformer` | `pytorch_transformer.TransformerModel` | `DatasetH` | `qlib,torch` | stable |
| `qlib_transformer_ts` | `pytorch_transformer_ts.TransformerModel` | `TSDatasetH` | `qlib,torch` | stable |
| `qlib_tft` | `quant_bench...tft.TFTModel` | `TSDatasetH` | `qlib,torch` | experimental, provenance gate |
| `numpy_linear_v1` | `NumpyLinearPlugin` | canonical DataFrame | base | stable offline baseline |

## Validation commands

```bash
quant-bench models list --verbose
quant-bench models show qlib_tra
quant-bench config init --model qlib_tra --feature-set crypto_alpha52_v1 --output tra.yaml
```

The default CI validates catalog uniqueness, legacy alias coverage, config generation for every entry, and import behavior for installed extras. Scheduled or release CI should install each dependency group and instantiate its matching classes.

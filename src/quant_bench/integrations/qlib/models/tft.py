# Qlib integration portions are adapted from Microsoft Qlib's TFT example.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see LICENSES/QLIB-MIT.txt.
# Modifications copyright (c) 2026 quant-bench contributors.

"""Local Temporal Fusion Transformer integration for Qlib.

This module is migrated from the pre-library implementation in
``qlib_model_trade/workflow/158/tft/tft.py`` and retains the upstream Qlib MIT
notice.  The PyTorch network and crypto workflow changes are quant-bench
modifications.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path
from torch.utils.data import DataLoader


class GLU(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_dimension, output_dimension)
        self.gate = nn.Linear(input_dimension, output_dimension)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.dropout(value)
        return self.projection(value) * torch.sigmoid(self.gate(value))


class AddAndNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(value + skip)


class GRN(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int | None = None,
        dropout: float = 0.0,
        context_dimension: int | None = None,
    ) -> None:
        super().__init__()
        output_dimension = output_dimension or hidden_dimension
        self.first = nn.Linear(input_dimension, hidden_dimension)
        self.activation = nn.ELU()
        self.second = nn.Linear(hidden_dimension, hidden_dimension)
        self.glu = GLU(hidden_dimension, output_dimension, dropout)
        self.add_norm = AddAndNorm(output_dimension)
        self.context = (
            nn.Linear(context_dimension, hidden_dimension, bias=False) if context_dimension else None
        )
        self.skip = (
            nn.Linear(input_dimension, output_dimension) if input_dimension != output_dimension else None
        )

    def forward(self, value: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        skip = self.skip(value) if self.skip else value
        hidden = self.first(value)
        if self.context is not None and context is not None:
            hidden = hidden + self.context(context)
        hidden = self.second(self.activation(hidden))
        return self.add_norm(self.glu(hidden), skip)


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        variable_count: int,
        hidden_dimension: int,
        dropout: float = 0.0,
        context_dimension: int | None = None,
    ) -> None:
        super().__init__()
        self.variable_embeddings = nn.ModuleList(
            [nn.Linear(1, hidden_dimension) for _ in range(variable_count)]
        )
        self.variable_grns = nn.ModuleList(
            [GRN(hidden_dimension, hidden_dimension, dropout=dropout) for _ in range(variable_count)]
        )
        self.selection_grn = GRN(
            variable_count * hidden_dimension,
            hidden_dimension,
            output_dimension=variable_count,
            dropout=dropout,
            context_dimension=context_dimension,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, value: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        chunks = value.split(1, dim=-1)
        embedded = [layer(chunk) for layer, chunk in zip(self.variable_embeddings, chunks, strict=True)]
        processed = [layer(item) for layer, item in zip(self.variable_grns, embedded, strict=True)]
        stacked = torch.stack(processed, dim=-2)
        flattened = torch.cat(embedded, dim=-1)
        weights = self.softmax(self.selection_grn(flattened, context)).unsqueeze(-1)
        return (weights * stacked).sum(dim=-2)


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, head_count: int, model_dimension: int, dropout: float = 0.0) -> None:
        super().__init__()
        if model_dimension % head_count:
            raise ValueError("model_dimension must be divisible by head_count")
        self.head_count = head_count
        self.key_dimension = model_dimension // head_count
        self.query_layers = nn.ModuleList(
            [nn.Linear(model_dimension, self.key_dimension, bias=False) for _ in range(head_count)]
        )
        self.key_layers = nn.ModuleList(
            [nn.Linear(model_dimension, self.key_dimension, bias=False) for _ in range(head_count)]
        )
        self.value_layers = nn.ModuleList(
            [nn.Linear(model_dimension, self.key_dimension, bias=False) for _ in range(head_count)]
        )
        self.output = nn.Linear(self.key_dimension, model_dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        heads: list[torch.Tensor] = []
        for index in range(self.head_count):
            queries = self.query_layers[index](query)
            keys = self.key_layers[index](key)
            values = self.value_layers[index](value)
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.key_dimension)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attention = self.dropout(torch.softmax(scores, dim=-1))
            heads.append(self.dropout(torch.matmul(attention, values)))
        combined = torch.stack(heads, dim=0).mean(dim=0) if self.head_count > 1 else heads[0]
        return self.output(combined)


class StaticCovariateEncoder(nn.Module):
    def __init__(self, model_dimension: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.static_embedding = nn.Parameter(torch.randn(1, model_dimension))
        self.selection_context = GRN(model_dimension, model_dimension, dropout=dropout)
        self.enrichment_context = GRN(model_dimension, model_dimension, dropout=dropout)
        self.hidden_context = GRN(model_dimension, model_dimension, dropout=dropout)
        self.cell_context = GRN(model_dimension, model_dimension, dropout=dropout)

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.static_embedding.expand(batch_size, -1)
        return (
            self.selection_context(embedding),
            self.enrichment_context(embedding),
            self.hidden_context(embedding),
            self.cell_context(embedding),
        )


class TFTNetwork(nn.Module):
    def __init__(
        self,
        feature_count: int,
        model_dimension: int,
        head_count: int,
        layer_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.static_encoder = StaticCovariateEncoder(model_dimension, dropout)
        self.variable_selection = VariableSelectionNetwork(
            feature_count,
            model_dimension,
            dropout,
            context_dimension=model_dimension,
        )
        self.lstm = nn.LSTM(
            model_dimension,
            model_dimension,
            layer_count,
            batch_first=True,
            dropout=dropout if layer_count > 1 else 0.0,
        )
        self.post_lstm_glu = GLU(model_dimension, model_dimension, dropout)
        self.post_lstm_norm = AddAndNorm(model_dimension)
        self.enrichment = GRN(
            model_dimension,
            model_dimension,
            dropout=dropout,
            context_dimension=model_dimension,
        )
        self.attention = InterpretableMultiHeadAttention(head_count, model_dimension, dropout)
        self.post_attention_glu = GLU(model_dimension, model_dimension, dropout)
        self.post_attention_norm = AddAndNorm(model_dimension)
        self.positionwise = GRN(model_dimension, model_dimension, dropout=dropout)
        self.final_glu = GLU(model_dimension, model_dimension, dropout)
        self.final_norm = AddAndNorm(model_dimension)
        self.output = nn.Linear(model_dimension, 1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = source.size()
        selection, enrichment, hidden, cell = self.static_encoder(batch_size)
        selected = self.variable_selection(source, selection.unsqueeze(1).expand(-1, sequence_length, -1))
        hidden_state = hidden.unsqueeze(0).expand(self.layer_count, -1, -1).contiguous()
        cell_state = cell.unsqueeze(0).expand(self.layer_count, -1, -1).contiguous()
        recurrent, _ = self.lstm(selected, (hidden_state, cell_state))
        temporal = self.post_lstm_norm(self.post_lstm_glu(recurrent), selected)
        enriched = self.enrichment(temporal, enrichment.unsqueeze(1).expand(-1, sequence_length, -1))
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, device=source.device))
        attended = self.attention(enriched, enriched, enriched, mask=causal_mask)
        decoded = self.post_attention_norm(self.post_attention_glu(attended), enriched)
        positioned = self.positionwise(decoded)
        final = self.final_norm(self.final_glu(positioned), temporal)
        return self.output(final[:, -1, :]).squeeze(-1)


class TFTModel(Model):
    """Qlib Model wrapper for the local TFT network."""

    def __init__(
        self,
        d_feat: int = 52,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_epochs: int = 100,
        lr: float = 0.001,
        early_stop: int = 10,
        batch_size: int = 128,
        metric: str = "loss",
        loss: str = "mse",
        optimizer: str = "adam",
        reg: float = 1e-4,
        n_jobs: int = 0,
        GPU: int = 0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.n_epochs = n_epochs
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.n_jobs = n_jobs
        self.optimizer = optimizer.lower()
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TFTModel")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.model = TFTNetwork(d_feat, d_model, n_heads, num_layers, dropout)
        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)
        elif self.optimizer == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=reg)
        else:
            raise ValueError(f"unsupported optimizer: {optimizer}")
        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self) -> bool:
        return self.device.type == "cuda"

    @staticmethod
    def _mse(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.mean((prediction.float() - label.float()) ** 2)

    def _loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(label)
        if self.loss != "mse":
            raise ValueError(f"unsupported loss: {self.loss}")
        return self._mse(prediction[mask], label[mask])

    def _metric(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.metric not in {"", "loss"}:
            raise ValueError(f"unsupported metric: {self.metric}")
        return -self._loss(prediction, label)

    def _train_epoch(self, loader: DataLoader[Any]) -> None:
        self.model.train()
        for data in loader:
            features = data[:, :, :-1].to(self.device)
            labels = data[:, -1, -1].to(self.device)
            predictions = self.model(features.float())
            loss = self._loss(predictions, labels)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def _test_epoch(self, loader: DataLoader[Any]) -> tuple[float, float]:
        self.model.eval()
        scores: list[float] = []
        losses: list[float] = []
        for data in loader:
            features = data[:, :, :-1].to(self.device)
            labels = data[:, -1, -1].to(self.device)
            with torch.no_grad():
                predictions = self.model(features.float())
                losses.append(float(self._loss(predictions, labels).item()))
                scores.append(float(self._metric(predictions, labels).item()))
        return float(np.mean(losses)), float(np.mean(scores))

    def fit(
        self,
        dataset: DatasetH,
        evals_result: dict[str, list[float]] | None = None,
        save_path: str | None = None,
    ) -> None:
        results = evals_result if evals_result is not None else {}
        train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if train.empty or valid.empty:
            raise ValueError("training and validation datasets must be non-empty")
        train.config(fillna_type="ffill+bfill")
        valid.config(fillna_type="ffill+bfill")
        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=False,
        )
        target = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        results["train"] = []
        results["valid"] = []
        self.fitted = True
        for epoch in range(self.n_epochs):
            self._train_epoch(train_loader)
            _, train_score = self._test_epoch(train_loader)
            _, valid_score = self._test_epoch(valid_loader)
            results["train"].append(train_score)
            results["valid"].append(valid_score)
            self.logger.info("epoch=%d train=%.6f valid=%.6f", epoch, train_score, valid_score)
            if valid_score > best_score:
                best_score = valid_score
                stop_steps = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break
        if best_state is None:
            raise RuntimeError("TFT training did not produce a finite validation score")
        self.model.load_state_dict(best_state)
        torch.save(best_state, target)
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH) -> pd.Series:
        if not self.fitted:
            raise ValueError("model is not fitted")
        test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        test.config(fillna_type="ffill+bfill")
        loader = DataLoader(test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        predictions: list[np.ndarray] = []
        for data in loader:
            features = data[:, :, :-1].to(self.device)
            with torch.no_grad():
                predictions.append(self.model(features.float()).detach().cpu().numpy())
        if not predictions:
            return pd.Series(dtype=float, index=test.get_index())
        return pd.Series(np.concatenate(predictions), index=test.get_index())

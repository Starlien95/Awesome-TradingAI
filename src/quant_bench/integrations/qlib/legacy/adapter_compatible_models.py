import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path
from torch.utils.data import DataLoader, Dataset


class MLPStrategy(nn.Module):
    """MLP architecture matching strategy_adapters.mlp_adapter."""

    def __init__(self, input_dim=52, hidden_dim=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StockDataset(Dataset):
    def __init__(self, df, d_feat: int, len_seq: int):
        self.d_feat = d_feat
        self.len_seq = len_seq

        if isinstance(df.columns, pd.MultiIndex):
            self.df_feature = df.xs("feature", level=0, axis=1)
            self.df_label = df.xs("label", level=0, axis=1)
        else:
            feature_cols = [col for col in df.columns if "feature" in str(col).lower()]
            label_cols = [col for col in df.columns if "label" in str(col).lower()]
            if feature_cols:
                self.df_feature = df[feature_cols]
                self.df_label = df[label_cols] if label_cols else df.iloc[:, d_feat:]
            else:
                self.df_feature = df.iloc[:, :d_feat]
                self.df_label = df.iloc[:, d_feat:] if df.shape[1] > d_feat else df.iloc[:, -1:]

        self.feature_values = self.df_feature.values
        self.label_values = np.asarray(self.df_label).reshape(len(self.df_label), -1)[:, 0]

    def __len__(self):
        return max(0, len(self.feature_values) - self.len_seq + 1)

    def __getitem__(self, idx):
        feature = self.feature_values[idx : idx + self.len_seq]
        label = self.label_values[idx + self.len_seq - 1]
        return feature, label


def get_stock_loader(df, batch_size: int, d_feat: int, len_seq: int, shuffle: bool = True):
    return DataLoader(
        StockDataset(df, d_feat=d_feat, len_seq=len_seq),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


class AdaRNNModel(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, dropout, len_seq, trans_loss):
        super().__init__()
        self.hidden_size = hidden_size
        self.len_seq = len_seq
        self.trans_loss = trans_loss
        self.features = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x_list):
        pred_list = []
        gru_out_list = []

        for x in x_list:
            out, _ = self.features(x)
            gru_out_list.append(out)
            pred_list.append(self.fc_out(out[:, -1, :]).squeeze(-1))

        if len(x_list) == 1:
            return pred_list, torch.tensor(0.0, device=x_list[0].device)

        loss_transfer = torch.tensor(0.0, device=x_list[0].device)
        feat_flat = [out.reshape(out.size(0), -1) for out in gru_out_list]
        for i in range(len(feat_flat)):
            for j in range(i + 1, len(feat_flat)):
                loss_transfer = loss_transfer + self._compute_mmd(feat_flat[i], feat_flat[j])
        return pred_list, loss_transfer

    @staticmethod
    def _compute_mmd(x, y):
        return torch.norm(x.mean(dim=0) - y.mean(dim=0), p=2)


class AdaRNN(Model):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=100,
        lr=0.001,
        metric="",
        batch_size=1024,
        early_stop=10,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        len_seq=None,
        step_len=None,
        trans_loss=0.005,
        weight_decay=0.0,
        n_splits=5,
        **kwargs,
    ):
        self.logger = get_module_logger("AdaRNN_Custom")
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = float(lr)
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.len_seq = int(len_seq if len_seq is not None else step_len if step_len is not None else 24)
        self.step_len = self.len_seq
        self.trans_loss = trans_loss
        self.weight_decay = weight_decay
        self.n_splits = n_splits

        self.logger.info(
            "AdaRNN parameters: "
            f"d_feat={d_feat}, len_seq={self.len_seq}, trans_loss={trans_loss}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, lr={self.lr}"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.adarnn_model = AdaRNNModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            len_seq=self.len_seq,
            trans_loss=self.trans_loss,
        ).to(self.device)

        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(
                self.adarnn_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "gd":
            self.train_optimizer = optim.SGD(
                self.adarnn_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported")

        self.fitted = False

    def loss_fn(self, pred, label):
        mask = torch.isfinite(label)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        pred_clean = pred[mask]
        label_clean = label[mask]

        if self.loss == "mse":
            return torch.mean((pred_clean - label_clean) ** 2)
        if self.loss in ("cross_entropy", "bce"):
            num_pos = label_clean.sum()
            num_neg = len(label_clean) - num_pos
            pos_weight = None
            if num_pos > 0 and num_neg > 0:
                pos_weight = (num_neg / num_pos * 2.0).clone().detach()
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred_clean, label_clean)
        raise ValueError(f"unknown loss `{self.loss}`")

    def metric_fn(self, pred, label):
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred, label)
        raise ValueError(f"unknown metric `{self.metric}`")

    def train_epoch(self, train_loader_list):
        if not train_loader_list:
            raise ValueError("No valid training periods generated")

        self.adarnn_model.train()
        min_len = min(len(loader) for loader in train_loader_list)
        iter_list = [iter(loader) for loader in train_loader_list]

        for _ in range(min_len):
            data_list = [next(iterator) for iterator in iter_list]
            feature_list = [data[0].float().to(self.device) for data in data_list]
            label_list = [data[1].float().to(self.device) for data in data_list]
            out, loss_transfer = self.adarnn_model(feature_list)

            loss_pred = sum(self.loss_fn(out[i], label_list[i]) for i in range(len(label_list)))
            loss_pred = loss_pred / len(label_list)
            loss = loss_pred + self.trans_loss * loss_transfer

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.adarnn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.adarnn_model.eval()
        scores = []
        losses = []

        for feature, label in data_loader:
            feature = feature.float().to(self.device)
            label = label.float().to(self.device)
            pred, _ = self.adarnn_model([feature])
            pred = pred[0]
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())
            scores.append(self.metric_fn(pred, label).item())

        return np.mean(losses), np.mean(scores)

    def _build_period_loaders(self, df_train):
        dates = df_train.index.get_level_values("datetime").unique().sort_values()
        n_splits = max(1, min(self.n_splits, len(dates)))
        period_len = max(1, len(dates) // n_splits)
        loaders = []

        for idx in range(n_splits):
            start_idx = idx * period_len
            end_idx = (idx + 1) * period_len if idx < n_splits - 1 else len(dates)
            period_dates = dates[start_idx:end_idx]
            mask = df_train.index.get_level_values("datetime").isin(period_dates)
            df_period = df_train[mask]
            if df_period.empty or len(df_period) < self.len_seq:
                self.logger.warning(f"Period {idx} is empty or shorter than len_seq, skipping.")
                continue
            loaders.append(get_stock_loader(df_period, self.batch_size, self.d_feat, self.len_seq, shuffle=True))
        return loaders

    def fit(self, dataset, evals_result=dict(), save_path=None):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key="learn",
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        train_loaders = self._build_period_loaders(df_train)
        self.logger.info(f"Split training data into {len(train_loaders)} periods.")
        valid_loader = get_stock_loader(df_valid, self.batch_size, self.d_feat, self.len_seq, shuffle=False)
        train_loader_full = get_stock_loader(df_train, self.batch_size, self.d_feat, self.len_seq, shuffle=False)

        save_path = get_or_create_path(save_path or "qlib_models/adarnn/adarnn_best.pth")
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.adarnn_model.state_dict())
        evals_result["train"] = []
        evals_result["valid"] = []
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.train_epoch(train_loaders)
            train_loss, train_score = self.test_epoch(train_loader_full)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                best_epoch = step
                stop_steps = 0
                best_param = copy.deepcopy(self.adarnn_model.state_dict())
                torch.save(best_param, save_path)
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.adarnn_model.load_state_dict(best_param)
        if self.device != torch.device("cpu"):
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet")

        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key="infer")
        test_loader = get_stock_loader(df_test, self.batch_size, self.d_feat, self.len_seq, shuffle=False)
        self.adarnn_model.eval()
        preds = []

        with torch.no_grad():
            for feature, _ in test_loader:
                feature = feature.float().to(self.device)
                pred, _ = self.adarnn_model([feature])
                preds.append(pred[0].cpu().numpy())

        if not preds:
            return pd.Series()
        return pd.Series(np.concatenate(preds), index=df_test.index[self.len_seq - 1 :])


class CustomADARNN(AdaRNN):
    pass

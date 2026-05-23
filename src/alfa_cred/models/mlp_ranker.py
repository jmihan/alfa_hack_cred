"""MLP с ListNet loss для diversification ансамбля.

В каждом запросе ровно один позитив, поэтому ListNet редуцируется к
cross-entropy между softmax(scores) и one-hot(positive_idx) по группе.
Это эквивалентно «максимизируй вероятность поставить настоящий позитив
выше всех остальных предложений в группе».
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


class MLPRanker(nn.Module):
    """Простой MLP: вход → 512 → 256 → 128 → 1 с GeLU + dropout."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128), dropout: float = 0.25):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(dims[i + 1]),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)


def _listnet_loss(scores: torch.Tensor, labels: torch.Tensor, groups: np.ndarray) -> torch.Tensor:
    """ListNet-like loss: cross-entropy с one-hot позитивом внутри группы.

    Если в группе нет позитива — она пропускается.
    """
    losses = []
    offset = 0
    for size in groups:
        size = int(size)
        s = scores[offset:offset + size]
        y = labels[offset:offset + size]
        pos_idx = torch.argmax(y)
        if y[pos_idx].item() == 0:
            offset += size
            continue
        log_p = F.log_softmax(s, dim=0)
        losses.append(-log_p[pos_idx])
        offset += size
    if not losses:
        return torch.zeros((), device=scores.device, requires_grad=True)
    return torch.stack(losses).mean()


@dataclass
class MLPRankerTrainer:
    """Обёртка с fit / predict, как у LgbmRanker."""

    feature_columns: list[str] = field(default_factory=list)
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 30
    early_stopping_rounds: int = 5
    requests_per_batch: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    model: MLPRanker | None = None
    feature_means: np.ndarray | None = None
    feature_stds: np.ndarray | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None

    def _standardize(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        arr = X[self.feature_columns].to_numpy(dtype=np.float32)
        # Заменяем NaN/inf на 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.feature_means = arr.mean(axis=0)
            self.feature_stds = arr.std(axis=0) + 1e-6
        arr = (arr - self.feature_means) / self.feature_stds
        # Клипп для устойчивости
        arr = np.clip(arr, -10.0, 10.0)
        return arr

    def _batch_iter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        shuffle: bool,
    ):
        """Батчинг по запросам: каждый батч — это RPS подряд идущих групп."""
        n_groups = len(groups)
        starts = np.cumsum(np.r_[0, groups])
        idx = np.arange(n_groups)
        if shuffle:
            rng = np.random.default_rng(self.seed + (1 if shuffle else 0))
            rng.shuffle(idx)
        for i in range(0, n_groups, self.requests_per_batch):
            batch_groups = idx[i:i + self.requests_per_batch]
            batch_rows = np.concatenate([np.arange(starts[g], starts[g + 1]) for g in batch_groups])
            batch_sizes = groups[batch_groups]
            yield (
                torch.from_numpy(X[batch_rows]).to(self.device),
                torch.from_numpy(y[batch_rows].astype(np.float32)).to(self.device),
                batch_sizes,
            )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Sequence[int],
        groups_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: Sequence[int],
        groups_val: np.ndarray,
    ) -> "MLPRankerTrainer":
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        X_tr = self._standardize(X_train, fit=True)
        y_tr = np.asarray(y_train, dtype=np.float32)
        X_va = self._standardize(X_val, fit=False)
        y_va = np.asarray(y_val, dtype=np.float32)

        self.model = MLPRanker(
            input_dim=X_tr.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        best_val_loss = math.inf
        best_state = None
        best_epoch = 0
        patience_left = self.early_stopping_rounds

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            running_loss, n_batches = 0.0, 0
            for x, y, sizes in self._batch_iter(X_tr, y_tr, groups_train, shuffle=True):
                optimizer.zero_grad()
                preds = self.model(x)
                loss = _listnet_loss(preds, y, sizes)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            train_loss = running_loss / max(n_batches, 1)

            self.model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for x, y, sizes in self._batch_iter(X_va, y_va, groups_val, shuffle=False):
                    preds = self.model(x)
                    loss = _listnet_loss(preds, y, sizes)
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        n_val += 1
            val_loss = val_loss / max(n_val, 1)
            scheduler.step(val_loss)

            LOG.info("MLP epoch %d: train_loss=%.4f, val_loss=%.4f, lr=%.6f",
                     epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"])
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.early_stopping_rounds
            else:
                patience_left -= 1
                if patience_left <= 0:
                    LOG.info("Early stopping at epoch %d (best epoch %d, val_loss=%.4f)",
                             epoch, best_epoch, best_val_loss)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_epoch = best_epoch
        self.best_val_loss = best_val_loss
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена.")
        self.model.eval()
        arr = self._standardize(X, fit=False)
        preds = []
        batch = 16384
        with torch.no_grad():
            for i in range(0, len(arr), batch):
                x = torch.from_numpy(arr[i:i + batch]).to(self.device)
                preds.append(self.model(x).cpu().numpy())
        return np.concatenate(preds)

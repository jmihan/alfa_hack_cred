"""Детерминированный pointwise tabular-MLP для подзадачи B.

NN-компонент диверсити, ортогональный GBDT-бленду (`b_blend`): архитектура
(эмбеддинги категориальных + стандартизованные числовые → MLP → P(is_deal))
даёт ранги со Spearman ≈ 0.84 к `b_blend`, поэтому их смесь сильнее каждого по
отдельности. Внутри заявки ровно один позитив → порядок по P(is_deal)
согласуется с ранжированием.

Детерминизм: фиксированные сиды (numpy/torch) + `cudnn.deterministic`. Метрика
NDCG@5 (зависит только от порядка) устойчива между прогонами и железом.

API: `fit_mlp`/`save_mlp`/`load_mlp`/`predict_mlp` (раздельные train/inference) +
`build_b_pointwise` (all-in-one). `torch` импортируется ДО numpy/pandas — иначе
на Windows конфликт OpenMP/MKL ломает загрузку c10.dll.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alfa_cred.config import TARGET  # noqa: E402

MLP_SEEDS = (42, 137, 314)
HIDDEN = (256, 128)
EPOCHS = 18
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.15


class _TabMLP(nn.Module):
    """Эмбеддинги категориальных + числовые → MLP → логит P(is_deal)."""

    def __init__(self, n_numeric: int, cardinalities: list[int]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c + 2, min(50, c // 2 + 1)) for c in cardinalities])
        in_dim = n_numeric + sum(min(50, c // 2 + 1) for c in cardinalities)
        layers: list[nn.Module] = []
        dim = in_dim
        for hidden in HIDDEN:
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(DROPOUT)]
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.net(torch.cat([x_num] + embedded, dim=1)).squeeze(1)


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fit_preprocessor(fit_df, numeric_cols, cat_cols):
    x_num = fit_df[numeric_cols].astype("float32").to_numpy()
    mean = np.nanmean(x_num, axis=0)
    std = np.nanstd(x_num, axis=0); std[std == 0] = 1.0
    cat_maps, cardinalities = {}, []
    for col in cat_cols:
        uniq = pd.Index(fit_df[col].astype("string").fillna("__NA__").unique())
        cat_maps[col] = {v: i + 1 for i, v in enumerate(uniq)}
        cardinalities.append(len(uniq))
    return mean, std, cat_maps, cardinalities


def _transform(df, numeric_cols, cat_cols, mean, std, cat_maps):
    x_num = np.nan_to_num((df[numeric_cols].astype("float32").to_numpy() - mean) / std, nan=0.0).astype("float32")
    if cat_cols:
        cols = [df[c].astype("string").fillna("__NA__").map(cat_maps[c]).fillna(0).astype("int64").to_numpy() for c in cat_cols]
        x_cat = np.stack(cols, axis=1)
    else:
        x_cat = np.zeros((len(df), 0), dtype="int64")
    return x_num, x_cat


def _train_one(x_num, x_cat, y, cardinalities, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = _TabMLP(x_num.shape[1], cardinalities).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    xn = torch.tensor(x_num, device=device); xc = torch.tensor(x_cat, device=device)
    yt = torch.tensor(y.astype("float32"), device=device)
    gen = torch.Generator().manual_seed(seed); n = len(y)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(xn[idx], xc[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def _enable_determinism(device: str) -> None:
    """Делает обучение MLP воспроизводимым run-to-run.

    На CPU один поток (`set_num_threads(1)`) убирает зависимость порядка
    редукций от числа потоков. `use_deterministic_algorithms(warn_only=True)`
    выбирает детерминированные реализации операций (с откатом+предупреждением,
    если для какой-то операции её нет).
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cpu":
        torch.set_num_threads(1)


def fit_mlp(train_b, feature_cols, cat_cols, seeds=MLP_SEEDS, device: str | None = None) -> dict:
    """Обучает MLP по сидам. Возвращает артефакт (веса + препроцессор + мета)."""
    dev = _resolve_device(device)
    _enable_determinism(dev)
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    mean, std, cat_maps, cardinalities = _fit_preprocessor(train_b, numeric_cols, cat_cols)
    x_num, x_cat = _transform(train_b, numeric_cols, cat_cols, mean, std, cat_maps)
    y = train_b[TARGET].astype(int).to_numpy()
    state_dicts = [_train_one(x_num, x_cat, y, cardinalities, s, dev) for s in seeds]
    return {
        "state_dicts": state_dicts, "cardinalities": cardinalities,
        "numeric_cols": numeric_cols, "cat_cols": list(cat_cols),
        "mean": mean, "std": std, "cat_maps": cat_maps,
    }


def predict_mlp(artifact: dict, predict_df, device: str | None = None) -> np.ndarray:
    """Усреднённые по сидам P(is_deal) для строк `predict_df` из обученного артефакта."""
    dev = _resolve_device(device)
    v_num, v_cat = _transform(predict_df, artifact["numeric_cols"], artifact["cat_cols"],
                              artifact["mean"], artifact["std"], artifact["cat_maps"])
    vn = torch.tensor(v_num, device=dev); vc = torch.tensor(v_cat, device=dev)
    probas = []
    for sd in artifact["state_dicts"]:
        model = _TabMLP(len(artifact["numeric_cols"]), artifact["cardinalities"]).to(dev)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            probas.append(torch.sigmoid(model(vn, vc)).cpu().numpy())
    return np.mean(probas, axis=0)


def save_mlp(artifact: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_dir / "mlp_pointwise.pt")


def load_mlp(in_dir: Path) -> dict:
    return torch.load(Path(in_dir) / "mlp_pointwise.pt", map_location="cpu", weights_only=False)


def build_b_pointwise(train_b, feature_cols, cat_cols, predict_df, seeds=MLP_SEEDS, device: str | None = None) -> np.ndarray:
    """Обучить и сразу предсказать (all-in-one): усреднённые P(is_deal) для `predict_df`."""
    artifact = fit_mlp(train_b, feature_cols, cat_cols, seeds=seeds, device=device)
    return predict_mlp(artifact, predict_df, device=device)

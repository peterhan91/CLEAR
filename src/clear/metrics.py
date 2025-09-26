"""Metric helpers for multilabel classification experiments."""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_curve
from tqdm.auto import tqdm

from .evaluation import compute_cis, compute_mean

DEFAULT_MEAN_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

__all__ = [
    "DEFAULT_MEAN_LABELS",
    "find_optimal_thresholds",
    "binarize_predictions",
    "compute_f1_scores",
    "compute_mcc_scores",
    "bootstrap_metric",
]


def _select_label_column(
    data: np.ndarray,
    label_index: int,
    label_idx_map: Optional[Mapping[str, int]],
    label_name: str,
) -> np.ndarray:
    if label_idx_map is None:
        return data[:, label_index]
    return data[:, label_idx_map[label_name]]


def find_optimal_thresholds(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    label_idx_map: Optional[Mapping[str, int]] = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] = matthews_corrcoef,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    num_thresholds: int = 512,
) -> Dict[str, float]:
    """Compute per-label thresholds that maximize ``metric_func``."""

    thresholds = np.linspace(min_threshold, max_threshold, num=num_thresholds)
    best_thresholds: Dict[str, float] = {}

    for idx, label in enumerate(labels):
        y_true_label = _select_label_column(y_true, idx, label_idx_map, label)
        y_pred_label = y_pred[:, idx]

        if len(np.unique(y_true_label)) < 2:
            best_thresholds[label] = 0.5
            continue

        scores = []
        for threshold in thresholds:
            y_pred_binary = (y_pred_label >= threshold).astype(int)
            score = metric_func(y_true_label, y_pred_binary)
            if np.isnan(score):
                score = -1.0
            scores.append(score)

        best_idx = int(np.argmax(scores))
        best_thresholds[label] = float(thresholds[best_idx])

    return best_thresholds


def binarize_predictions(
    y_pred: np.ndarray,
    labels: Sequence[str],
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """Apply label-specific thresholds to probability predictions."""

    binarized = np.zeros_like(y_pred, dtype=int)
    for idx, label in enumerate(labels):
        threshold = thresholds.get(label, 0.5)
        binarized[:, idx] = (y_pred[:, idx] >= threshold).astype(int)
    return binarized


def _compute_scores(
    y_pred_binary: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    label_idx_map: Optional[Mapping[str, int]],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    scores = {}
    for idx, label in enumerate(labels):
        y_true_label = _select_label_column(y_true, idx, label_idx_map, label)
        y_pred_label = y_pred_binary[:, idx]
        score = metric_func(y_true_label, y_pred_label)
        if np.isnan(score):
            score = 0.0
        scores[label] = [float(score)]

    df = pd.DataFrame(scores)
    mean_labels = [label for label in DEFAULT_MEAN_LABELS if label in df.columns]
    if mean_labels:
        df["Mean"] = compute_mean(df, mean_labels, is_df=True)
    return df


def compute_f1_scores(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    thresholds: Mapping[str, float],
    label_idx_map: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    binarized = binarize_predictions(y_pred, labels, thresholds)
    return _compute_scores(
        binarized,
        y_true,
        labels,
        label_idx_map=label_idx_map,
        metric_func=lambda yt, yp: f1_score(yt, yp, zero_division=0),
    )


def compute_mcc_scores(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    thresholds: Mapping[str, float],
    label_idx_map: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    binarized = binarize_predictions(y_pred, labels, thresholds)
    return _compute_scores(
        binarized,
        y_true,
        labels,
        label_idx_map=label_idx_map,
        metric_func=matthews_corrcoef,
    )


def bootstrap_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray], pd.DataFrame],
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    n_samples: int = 1000,
    random_state: Optional[int] = None,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap arbitrary metrics computed from predictions and targets."""

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y_true))

    frames = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="metric-bootstrap")

    for _ in iterator:
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        frames.append(metric_fn(y_pred[sample_idx], y_true[sample_idx]))

    samples = pd.concat(frames, ignore_index=True)
    return samples, compute_cis(samples)


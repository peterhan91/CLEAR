"""Utility helpers for evaluating multilabel classification models."""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm.auto import tqdm

__all__ = [
    "compute_mean",
    "accuracy",
    "sigmoid",
    "plot_roc",
    "choose_operating_point",
    "plot_pr",
    "evaluate",
    "compute_cis",
    "bootstrap",
]


def compute_mean(
    stats: Mapping[str, Sequence[float]] | pd.DataFrame,
    labels: Optional[Sequence[str]] = None,
    *,
    is_df: bool = True,
) -> float:
    """Compute the mean value across a subset of label metrics.

    Parameters
    ----------
    stats:
        Either a Pandas ``DataFrame`` produced by :func:`evaluate` or a mapping from
        label name to a sequence containing metric values (typically a single-value
        list as produced by the bootstrap helpers).
    labels:
        Optional sequence specifying which label names to include in the mean.
        When omitted, all keys except ``"Mean"`` are used.
    is_df:
        Whether *stats* is a ``DataFrame`` (default) or a mapping. The old API
        exposed this argument, so it is retained for backwards compatibility.
    """

    if is_df:
        if not isinstance(stats, pd.DataFrame):  # defensive for callers
            raise TypeError("Expected a pandas DataFrame when is_df=True")
        if labels is None:
            labels = [c for c in stats.columns if c != "Mean"]
        row = stats.iloc[0]
        values = [float(row[label]) for label in labels if label in row]
    else:
        if not isinstance(stats, Mapping):
            raise TypeError("Expected a mapping when is_df=False")
        if labels is None:
            labels = [k for k in stats.keys() if k != "Mean"]
        values = [float(stats[label][0]) for label in labels if label in stats]

    if not values:
        raise ValueError("No values available to compute a mean")
    return float(np.mean(values))


def accuracy(output, target, topk: Sequence[int] = (1,)) -> List[float]:
    """Compute top-k accuracy for the provided logits and targets."""

    if output.ndim != 2:
        raise ValueError("Expected `output` to be a 2D tensor of logits")
    if target.ndim != 1:
        raise ValueError("Expected `target` to be a 1D tensor of class indices")

    max_k = max(topk)
    _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid implementation for numpy arrays."""

    return 1.0 / (1.0 + np.exp(-x))


def plot_roc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    name: str,
    *,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute ROC curve values and optionally render a plot."""

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    if plot:
        import matplotlib.pyplot as plt  # local import to avoid eager dependency

        plt.figure(dpi=100)
        plt.title(name)
        plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

    return fpr, tpr, thresholds, roc_auc


def choose_operating_point(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
    """Return sensitivity and specificity at the optimal Youden's J index."""

    best_idx = np.argmax(tpr - fpr)
    return float(tpr[best_idx]), float(1 - fpr[best_idx])


def plot_pr(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    name: str,
    *,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute precision-recall values and optionally render a plot."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    if plot:
        import matplotlib.pyplot as plt  # local import to avoid eager dependency

        baseline = float(np.mean(y_true))
        plt.figure(dpi=100)
        plt.title(name)
        plt.plot(recall, precision, "b", label=f"AUC = {pr_auc:.2f}")
        plt.legend(loc="lower right")
        plt.plot([0, 1], [baseline, baseline], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()

    return precision, recall, thresholds, pr_auc


def evaluate(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    label_idx_map: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    """Compute AUROC for each label and return a dataframe of scores."""

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("Predictions and targets must have the same number of samples")

    dataframes = []
    for idx, label in enumerate(labels):
        column = idx if label_idx_map is None else label_idx_map[label]
        fpr, tpr, thresholds, roc_auc = plot_roc(
            y_pred[:, idx],
            y_true[:, column],
            f"{label} ROC Curve",
            plot=False,
        )
        _ = thresholds  # kept for parity with original API
        sens, spec = choose_operating_point(fpr, tpr, thresholds)
        dataframes.append(
            pd.DataFrame(
                {
                    f"{label}_auc": [roc_auc],
                    f"{label}_sens": [sens],
                    f"{label}_spec": [spec],
                }
            )
        )

    return pd.concat(dataframes, axis=1)


def compute_cis(
    data: pd.DataFrame,
    *,
    confidence_level: float = 0.05,
) -> pd.DataFrame:
    """Return mean and confidence intervals for each column in ``data``."""

    lower_quantile = confidence_level / 2
    upper_quantile = 1 - lower_quantile

    columns = {}
    for column in data.columns:
        series = data[column].sort_values()
        lower = float(series.quantile(lower_quantile, interpolation="higher"))
        upper = float(series.quantile(upper_quantile, interpolation="higher"))
        mean = float(series.mean())
        columns[column] = [mean, lower, upper]

    result = pd.DataFrame(columns)
    result.index = ["mean", "lower", "upper"]
    return result


def bootstrap(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    *,
    n_samples: int = 1000,
    label_idx_map: Optional[Mapping[str, int]] = None,
    random_state: Optional[int] = None,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap ROC AUC estimates and return samples + confidence intervals."""

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("Predictions and targets must share the first dimension")

    rng = np.random.default_rng(random_state)
    indices = np.arange(y_true.shape[0])

    boot_samples: List[pd.DataFrame] = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="bootstrap")

    for _ in iterator:
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sample_pred = y_pred[sample_idx]
        sample_true = y_true[sample_idx]
        boot_samples.append(
            evaluate(sample_pred, sample_true, labels, label_idx_map=label_idx_map)
        )

    samples_df = pd.concat(boot_samples, ignore_index=True)
    return samples_df, compute_cis(samples_df)

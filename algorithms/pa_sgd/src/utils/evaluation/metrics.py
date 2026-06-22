"""
Evaluation metrics — platform-agnostic.

Ba nhóm metric:
  1. RegressionEvaluator  – MAE, RMSE, R² cho regression targets
  2. ClassificationEvaluator – Accuracy, Macro F1, Weighted F1, Confusion Matrix
  3. RankingEvaluator     – Spearman, NDCG@K, Precision@K
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from scipy.stats import spearmanr


def _infer_target_names(history: list) -> list[str]:
    """Lấy target names từ history item đầu tiên (y_true_log keys)."""
    if history and hasattr(history[0], "y_true_log"):
        return list(history[0].y_true_log.keys())
    return []


@dataclass
class RegressionMetricRow:
    target: str
    n_samples: int
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "n_samples": self.n_samples,
            "MAE": round(self.mae, 4),
            "RMSE": round(self.rmse, 4),
            "R2": round(self.r2, 4),
        }


class RegressionEvaluator:
    """Computes MAE, RMSE, R² for each regression target."""

    @staticmethod
    def evaluate(history: list, target_names: list[str] | None = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        history : output of QuadRegressionHead.run_online() or UncertaintyWeightedTrainer.run_online()
        target_names : list[str] | None
            If None, inferred from history[0].y_true_log keys.
        """
        if target_names is None:
            target_names = _infer_target_names(history)

        rows = []
        for name in target_names:
            y_true = np.array([r.y_true_log[name] for r in history])
            y_pred = np.array([r.y_pred_log[name] for r in history])

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            rows.append(RegressionMetricRow(
                target=f"y_{name}",
                n_samples=len(history),
                mae=mae, rmse=rmse, r2=r2,
            ).to_dict())

        df = pd.DataFrame(rows)
        avg_row = {
            "target": "AVERAGE",
            "n_samples": len(history),
            "MAE":  round(df["MAE"].mean(),  4),
            "RMSE": round(df["RMSE"].mean(), 4),
            "R2":   round(df["R2"].mean(),   4),
        }
        return pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    @staticmethod
    def per_target_arrays(
        history: list,
        target_names: list[str] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Returns {target: (y_true_raw, y_pred_raw)}."""
        if target_names is None:
            target_names = _infer_target_names(history)
        return {
            name: (
                np.array([r.y_true_raw[name] for r in history]),
                np.array([r.y_pred_raw[name] for r in history]),
            )
            for name in target_names
        }


@dataclass
class ClassificationReport:
    n_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    n_classes: int = 4
    label_names: dict[int, str] | None = None

    def to_dict(self) -> dict:
        return {
            "n_samples":    self.n_samples,
            "Accuracy":     round(self.accuracy,    4),
            "Macro_F1":     round(self.macro_f1,    4),
            "Weighted_F1":  round(self.weighted_f1, 4),
        }


class ClassificationEvaluator:
    """Accuracy, Macro F1, Weighted F1, Confusion Matrix."""

    DEFAULT_LABEL_NAMES = {0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"}

    @staticmethod
    def evaluate(
        history: list,
        n_classes: int = 4,
        label_names: dict[int, str] | None = None,
    ) -> ClassificationReport:
        y_true = np.array([r.y_true for r in history])
        y_pred = np.array([r.y_pred for r in history])
        labels = list(range(n_classes))

        return ClassificationReport(
            n_samples=len(history),
            accuracy=accuracy_score(y_true, y_pred),
            macro_f1=f1_score(y_true, y_pred, average="macro",    zero_division=0, labels=labels),
            weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
            n_classes=n_classes,
            label_names=label_names,
        )

    @staticmethod
    def confusion_matrix_df(
        report: ClassificationReport,
        label_names: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        names_map = label_names or report.label_names or ClassificationEvaluator.DEFAULT_LABEL_NAMES
        labels = [names_map.get(i, str(i)) for i in range(report.n_classes)]
        return pd.DataFrame(
            report.confusion_matrix,
            index=[f"True {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels],
        )

    @staticmethod
    def per_class_f1(
        history: list,
        n_classes: int = 4,
        label_names: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        y_true = np.array([r.y_true for r in history])
        y_pred = np.array([r.y_pred for r in history])
        labels = list(range(n_classes))
        f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

        names_map = label_names or ClassificationEvaluator.DEFAULT_LABEL_NAMES
        names = [names_map.get(i, str(i)) for i in range(n_classes)]
        return pd.DataFrame({"label": names, "F1": np.round(f1s, 4)})


class RankingEvaluator:
    """Spearman correlation, NDCG@K, Precision@K."""

    @staticmethod
    def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        rho, p = spearmanr(y_true, y_pred)
        return float(rho), float(p)

    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        k = min(k, len(y_true))
        ideal_order = np.argsort(y_true)[::-1][:k]
        pred_order  = np.argsort(y_pred)[::-1][:k]

        def dcg(gains: np.ndarray) -> float:
            positions = np.arange(1, len(gains) + 1)
            return float(np.sum(gains / np.log2(positions + 1)))

        idcg = dcg(y_true[ideal_order])
        if idcg == 0:
            return 0.0
        return dcg(y_true[pred_order]) / idcg

    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        k = min(k, len(y_true))
        true_top = set(np.argsort(y_true)[::-1][:k])
        pred_top = set(np.argsort(y_pred)[::-1][:k])
        return len(true_top & pred_top) / k

    @staticmethod
    def evaluate_all(
        y_true_score: np.ndarray,
        y_pred_score: np.ndarray,
        k: int = 10,
    ) -> dict[str, float]:
        rho, p = RankingEvaluator.spearman(y_true_score, y_pred_score)
        ndcg   = RankingEvaluator.ndcg_at_k(y_true_score, y_pred_score, k=k)
        prec   = RankingEvaluator.precision_at_k(y_true_score, y_pred_score, k=k)
        return {
            "Spearman":       round(rho, 4),
            "Spearman_p":     round(p,   6),
            f"NDCG@{k}":      round(ndcg, 4),
            f"Precision@{k}": round(prec, 4),
        }

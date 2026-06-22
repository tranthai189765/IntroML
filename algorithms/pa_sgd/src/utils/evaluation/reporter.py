"""
EvaluationReporter — platform-agnostic.

Tổng hợp tất cả metrics và xuất bảng kết quả cuối cùng.
Nhận score_fn để tính popularity score theo từng platform.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .metrics import (
    RegressionEvaluator,
    ClassificationEvaluator,
    RankingEvaluator,
)


class EvaluationReporter:
    """
    Nhận history từ regression và classification heads, tổng hợp đánh giá.

    Parameters
    ----------
    reg_history : list
        Output của QuadRegressionHead.run_online() hoặc UncertaintyWeightedTrainer.run_online().
    cls_history : list
        Output của PAClassificationHead.run_online() (nếu tách riêng).
        Nếu dùng UncertaintyWeightedTrainer, truyền cùng history.
    score_fn : Callable[[dict[str, np.ndarray]], np.ndarray] | None
        Hàm tính popularity score từ dict {target_name → raw_array}.
        Ví dụ Twitter: lambda d: compute_popularity_score(d["likes"], d["comments"], ...)
        Ví dụ Reddit:  lambda d: compute_reddit_score(d["score"], d["num_comments"], ...)
        Nếu None, bỏ qua ranking metrics.
    target_names : list[str] | None
        Tên regression targets. Nếu None, infer từ history.
    n_classes : int
        Số lớp phân loại. Mặc định 4.
    label_names : dict[int, str] | None
        Tên nhãn. Mặc định Low/Medium/Popular/Viral.
    k : int
        K cho NDCG@K, Precision@K.
    """

    def __init__(
        self,
        reg_history: list,
        cls_history: list,
        score_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
        target_names: list[str] | None = None,
        n_classes: int = 4,
        label_names: dict[int, str] | None = None,
        k: int = 10,
    ):
        self.reg_history = reg_history
        self.cls_history = cls_history
        self.score_fn = score_fn
        self.target_names = target_names
        self.n_classes = n_classes
        self.label_names = label_names
        self.k = k

        self._reg_table: pd.DataFrame | None = None
        self._cls_report = None
        self._ranking_metrics: dict | None = None

    def compute(self) -> "EvaluationReporter":
        self._reg_table = RegressionEvaluator.evaluate(
            self.reg_history, target_names=self.target_names
        )
        self._cls_report = ClassificationEvaluator.evaluate(
            self.cls_history, n_classes=self.n_classes, label_names=self.label_names
        )

        if self.score_fn is not None:
            target_arrays = RegressionEvaluator.per_target_arrays(
                self.reg_history, target_names=self.target_names
            )
            pred_dict = {k: v[1] for k, v in target_arrays.items()}
            true_dict = {k: v[0] for k, v in target_arrays.items()}

            pred_score = self.score_fn(pred_dict)
            true_score = self.score_fn(true_dict)
            self._ranking_metrics = RankingEvaluator.evaluate_all(true_score, pred_score, k=self.k)
            self._pred_score = pred_score
            self._true_score = true_score

        return self

    def print_all(self) -> None:
        if self._reg_table is None:
            self.compute()

        print("\n" + "="*60)
        print("REGRESSION EVALUATION (log space)")
        print("="*60)
        print(self._reg_table.to_string(index=False))

        print("\n" + "="*60)
        print("CLASSIFICATION EVALUATION")
        print("="*60)
        for k, v in self._cls_report.to_dict().items():
            print(f"  {k:20s}: {v}")

        print("\n  Per-class F1:")
        f1_df = ClassificationEvaluator.per_class_f1(
            self.cls_history, n_classes=self.n_classes, label_names=self.label_names
        )
        print(f1_df.to_string(index=False))

        print("\n  Confusion Matrix:")
        cm_df = ClassificationEvaluator.confusion_matrix_df(
            self._cls_report, label_names=self.label_names
        )
        print(cm_df.to_string())

        if self._ranking_metrics is not None:
            print("\n" + "="*60)
            print(f"RANKING EVALUATION (k={self.k})")
            print("="*60)
            for k, v in self._ranking_metrics.items():
                print(f"  {k:20s}: {v}")

    def all_tables(self) -> dict[str, pd.DataFrame]:
        if self._reg_table is None:
            self.compute()

        cls_df = pd.DataFrame([self._cls_report.to_dict()])
        f1_df  = ClassificationEvaluator.per_class_f1(
            self.cls_history, n_classes=self.n_classes, label_names=self.label_names
        )
        cm_df  = ClassificationEvaluator.confusion_matrix_df(
            self._cls_report, label_names=self.label_names
        )

        result = {
            "regression":       self._reg_table,
            "classification":   cls_df,
            "per_class_f1":     f1_df,
            "confusion_matrix": cm_df,
        }
        if self._ranking_metrics is not None:
            result["ranking"] = pd.DataFrame([self._ranking_metrics])

        return result

    def save(self, output_dir: str = "outputs") -> None:
        import os
        if self._reg_table is None:
            self.compute()

        os.makedirs(output_dir, exist_ok=True)
        tables = self.all_tables()

        tables["regression"].to_csv(f"{output_dir}/regression_metrics.csv",       index=False)
        tables["classification"].to_csv(f"{output_dir}/classification_metrics.csv", index=False)
        tables["per_class_f1"].to_csv(f"{output_dir}/per_class_f1.csv",           index=False)
        tables["confusion_matrix"].to_csv(f"{output_dir}/confusion_matrix.csv")
        if "ranking" in tables:
            tables["ranking"].to_csv(f"{output_dir}/ranking_metrics.csv", index=False)

        print(f"[Saved] Evaluation tables → {output_dir}/")

    def ranking_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if "predicted_popularity_score" not in df.columns:
            raise ValueError("df must contain 'predicted_popularity_score'.")
        ranked = df.sort_values("predicted_popularity_score", ascending=False).reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = "rank"
        return ranked

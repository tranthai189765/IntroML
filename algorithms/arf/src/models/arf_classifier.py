"""
src/models/arf_classifier.py
Adaptive Random Forest Classifier cho hai nhiệm vụ phân loại:
  - TopicClassifier  : Dự đoán topic nào sẽ là "popular" (Table 1)
  - PostClassifier   : Dự đoán bài viết nào sẽ là "popular" trong topic (Table 2)

Thuật toán ARF (Gomes et al., 2017):
  - Mở rộng của Hoeffding Tree (VFDT) sang dạng ensemble.
  - Mỗi cây được huấn luyện với online bagging (Poisson resampling, λ=6).
  - Tích hợp ADWIN để phát hiện concept drift và thay thế cây lỗi thời.
  - Phù hợp với data stream thay đổi theo thời gian (trending ≠ stable).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from river import metrics
from river.forest import ARFClassifier
from river.drift import ADWIN
from config import (
    ARF_N_MODELS, ARF_LAMBDA, ARF_GRACE_PERIOD,
    ARF_MAX_DEPTH, ARF_SEED,
)

def _build_arf_classifier(n_models: int = ARF_N_MODELS) -> ARFClassifier:
    """
    Khởi tạo ARF Classifier với ADWIN drift detector.
    river 0.19+: ARF nằm trong river.forest.ARFClassifier
    """
    return ARFClassifier(
        n_models=n_models,
        lambda_value=ARF_LAMBDA,
        grace_period=ARF_GRACE_PERIOD,
        max_depth=ARF_MAX_DEPTH,
        seed=ARF_SEED,
        drift_detector=ADWIN(),
        warning_detector=ADWIN(),
    )


class TopicClassifier:
    """
    Bài toán phân loại Table 1:
    Dự đoán post này thuộc topic 'popular' hay 'not popular' tại 24h.

    Label: is_popular_topic ∈ {0, 1}
      1 = topic của bài viết nằm trong Top-K topics phổ biến nhất tại 24h
      0 = ngược lại
    """

    def __init__(self):
        self.model   = _build_arf_classifier()
        self.acc     = metrics.Accuracy()
        self.f1      = metrics.F1(pos_val=1)
        self.kappa   = metrics.CohenKappa()
        self.n_seen  = 0

    def learn_one(self, x: dict, y: int) -> "TopicClassifier":
        self.model.learn_one(x, y)
        return self

    def predict_one(self, x: dict) -> int:
        return self.model.predict_one(x) or 0

    def predict_proba_one(self, x: dict) -> dict:
        return self.model.predict_proba_one(x)

    def update_metrics(self, y_pred: int, y_true: int) -> None:
        self.acc.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)
        self.kappa.update(y_true, y_pred)
        self.n_seen += 1

    def get_metrics(self) -> dict:
        return {
            "accuracy": self.acc.get(),
            "f1":       self.f1.get(),
            "kappa":    self.kappa.get(),
            "n_seen":   self.n_seen,
        }


class PostClassifier:
    """
    Bài toán phân loại Table 2:
    Dự đoán bài viết có nằm trong top 10% của topic tại 24h hay không.

    Label: is_popular_post ∈ {0, 1}
      1 = post nằm trong top TOP_FRAC_POSTS của topic tại 24h
      0 = ngược lại

    Lưu ý: Đây là imbalanced classification (chỉ ~10% = 1).
    → F1 score quan trọng hơn Accuracy.
    """

    def __init__(self):
        self.model   = _build_arf_classifier()
        self.acc     = metrics.Accuracy()
        self.f1      = metrics.F1(pos_val=1)
        self.kappa   = metrics.CohenKappa()
        self.recall  = metrics.Recall(pos_val=1)
        self.precision = metrics.Precision(pos_val=1)
        self.n_seen  = 0

    def learn_one(self, x: dict, y: int) -> "PostClassifier":
        self.model.learn_one(x, y)
        return self

    def predict_one(self, x: dict) -> int:
        return self.model.predict_one(x) or 0

    def predict_proba_one(self, x: dict) -> dict:
        return self.model.predict_proba_one(x)

    def update_metrics(self, y_pred: int, y_true: int) -> None:
        self.acc.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)
        self.kappa.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        self.precision.update(y_true, y_pred)
        self.n_seen += 1

    def get_metrics(self) -> dict:
        return {
            "accuracy":  self.acc.get(),
            "f1":        self.f1.get(),
            "kappa":     self.kappa.get(),
            "recall":    self.recall.get(),
            "precision": self.precision.get(),
            "n_seen":    self.n_seen,
        }

"""
src/models/arf_multiclass.py
Adaptive Random Forest cho bộ dữ liệu X/Twitter thực:

  ViralityClassifier   – phân loại 4 lớp (0=Low, 1=Medium, 2=Popular, 3=Viral)
  MultiOutputRegressor – dự đoán 4 chỉ số engagement tại 6h (likes, views, comments, reposts)

Cả hai dùng ARF của thư viện river với ADWIN drift detector.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from river import metrics
from river.forest import ARFClassifier, ARFRegressor
from river.drift import ADWIN
from config import (
    ARF_N_MODELS, ARF_LAMBDA, ARF_GRACE_PERIOD,
    ARF_MAX_DEPTH, ARF_SEED,
)


def _clf() -> ARFClassifier:
    return ARFClassifier(
        n_models=ARF_N_MODELS,
        lambda_value=ARF_LAMBDA,
        grace_period=ARF_GRACE_PERIOD,
        max_depth=ARF_MAX_DEPTH,
        seed=ARF_SEED,
        drift_detector=ADWIN(),
        warning_detector=ADWIN(),
    )


def _reg() -> ARFRegressor:
    return ARFRegressor(
        n_models=ARF_N_MODELS,
        lambda_value=ARF_LAMBDA,
        grace_period=ARF_GRACE_PERIOD,
        max_depth=ARF_MAX_DEPTH,
        seed=ARF_SEED,
        drift_detector=ADWIN(),
        warning_detector=ADWIN(),
    )


class ViralityClassifier:
    """
    Phân loại virality bài đăng X/Twitter thành 4 lớp:
      0 = Low/Flop  (bottom 50%)
      1 = Medium    (50–80%)
      2 = Popular   (80–95%)
      3 = Viral     (top 5%)

    Đầu vào : features tại snapshot 0.5h
    Nhãn    : label_6h (= 'label' cột cuối trong CSV)
    """

    LABEL_NAMES = {0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"}

    def __init__(self):
        self.model     = _clf()
        self.acc       = metrics.Accuracy()
        self.macro_f1  = metrics.MacroF1()
        self.weighted_f1 = metrics.WeightedF1()
        self.kappa     = metrics.CohenKappa()
        self.n_seen    = 0

    def learn_one(self, x: dict, y: int) -> "ViralityClassifier":
        self.model.learn_one(x, y)
        return self

    def predict_one(self, x: dict) -> int:
        return self.model.predict_one(x) or 0

    def predict_proba_one(self, x: dict) -> dict:
        return self.model.predict_proba_one(x)

    def update_metrics(self, y_pred: int, y_true: int) -> None:
        self.acc.update(y_true, y_pred)
        self.macro_f1.update(y_true, y_pred)
        self.weighted_f1.update(y_true, y_pred)
        self.kappa.update(y_true, y_pred)
        self.n_seen += 1

    def get_metrics(self) -> dict:
        return {
            "accuracy":    self.acc.get(),
            "macro_f1":    self.macro_f1.get(),
            "weighted_f1": self.weighted_f1.get(),
            "kappa":       self.kappa.get(),
            "n_seen":      self.n_seen,
        }


# Generic next-step target names (không gắn với snapshot cụ thể)
REGRESSION_TARGETS = ["likes_next", "views_next", "comments_next", "reposts_next"]


class MultiOutputRegressor:
    """
    Dự đoán 4 chỉ số engagement tại 6h từ snapshot 0.5h.
    Mỗi target có một ARFRegressor riêng biệt.

    Targets: likes_6h, views_6h, comments_6h, reposts_6h
    """

    def __init__(self):
        self.models = {t: _reg() for t in REGRESSION_TARGETS}
        self.mae    = {t: metrics.MAE()  for t in REGRESSION_TARGETS}
        self.rmse   = {t: metrics.RMSE() for t in REGRESSION_TARGETS}
        self.r2     = {t: metrics.R2()   for t in REGRESSION_TARGETS}
        self.n_seen = 0

    def learn_one(self, x: dict, y: dict) -> "MultiOutputRegressor":
        for t in REGRESSION_TARGETS:
            self.models[t].learn_one(x, y[t])
        return self

    def predict_one(self, x: dict) -> dict:
        return {t: (self.models[t].predict_one(x) or 0.0) for t in REGRESSION_TARGETS}

    def update_metrics(self, y_pred: dict, y_true: dict) -> None:
        for t in REGRESSION_TARGETS:
            self.mae[t].update(y_true[t], y_pred[t])
            self.rmse[t].update(y_true[t], y_pred[t])
            self.r2[t].update(y_true[t], y_pred[t])
        self.n_seen += 1

    def get_metrics(self) -> dict:
        result = {
            t: {
                "mae":  self.mae[t].get(),
                "rmse": self.rmse[t].get(),
                "r2":   self.r2[t].get(),
            }
            for t in REGRESSION_TARGETS
        }
        result["n_seen"] = self.n_seen
        return result

"""
src/models/arf_regressor.py
Adaptive Random Forest Regressor cho nhiệm vụ dự đoán engagement.

Nhiệm vụ Regression:
  Dự đoán 'engagement_target' tại 24h từ features tại 1h.
  engagement_target = W_LIKES*likes_24h + W_SHARES*shares_24h + W_COMMENTS*comments_24h

  Ngoài ra, huấn luyện 3 regressor riêng biệt cho:
    - likes_24h
    - shares_24h
    - comments_24h
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from river import metrics
from river.forest import ARFRegressor
from river.drift import ADWIN
from config import ARF_N_MODELS, ARF_LAMBDA, ARF_GRACE_PERIOD, ARF_MAX_DEPTH, ARF_SEED


def _build_arf_regressor() -> ARFRegressor:
    return ARFRegressor(
        n_models=ARF_N_MODELS,
        lambda_value=ARF_LAMBDA,
        grace_period=ARF_GRACE_PERIOD,
        max_depth=ARF_MAX_DEPTH,
        seed=ARF_SEED,
        drift_detector=ADWIN(),
        warning_detector=ADWIN(),
    )


class EngagementRegressor:
    """
    Bộ dự đoán weighted engagement tổng hợp tại snapshot 24h.

    Target: engagement_target = W_LIKES*likes + W_SHARES*shares + W_COMMENTS*comments
    Sử dụng 1 ARFRegressor duy nhất để giảm thời gian tính toán.

    Phương thức:
      - learn_one(x, y_dict)  : y_dict = {"target": float, "likes": float, ...}
      - predict_one(x)        : tra ve {"target": float}
      - update_metrics(...)   : cap nhat MAE, RMSE, R2
    """

    def __init__(self):
        self.reg     = _build_arf_regressor()
        self.mae     = metrics.MAE()
        self.rmse    = metrics.RMSE()
        self.r2      = metrics.R2()
        self.n_seen  = 0

    def learn_one(self, x: dict, y_dict: dict) -> "EngagementRegressor":
        self.reg.learn_one(x, y_dict["target"])
        return self

    def predict_one(self, x: dict) -> dict:
        val = self.reg.predict_one(x) or 0.0
        return {"target": val}

    def update_metrics(self, y_pred: dict, y_true: dict) -> None:
        self.mae.update(y_true["target"],  y_pred["target"])
        self.rmse.update(y_true["target"], y_pred["target"])
        self.r2.update(y_true["target"],   y_pred["target"])
        self.n_seen += 1

    def get_metrics(self) -> dict:
        return {
            "mae":    self.mae.get(),
            "rmse":   self.rmse.get(),
            "r2":     self.r2.get(),
            "n_seen": self.n_seen,
        }

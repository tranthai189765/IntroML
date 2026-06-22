"""
PA Regression Head — platform-agnostic.

QuadRegressionHead chạy N PA Regressor song song, mỗi cái học 1 target.
Target names mặc định là Twitter convention nhưng hoàn toàn configurable.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import SGDRegressor


# Default target names (Twitter). Platform-specific code override qua target_names=.
TARGET_NAMES = ["likes", "comments", "reposts", "views"]

_PA_LOSS_MAP = {
    "epsilon_insensitive":         ("epsilon_insensitive",         "pa1"),
    "squared_epsilon_insensitive": ("squared_epsilon_insensitive", "pa2"),
}

_LOG_CLIP = 17.0   # expm1(17) ≈ 2.4e7 — above max real (views ~15M), chặn outlier nổ


@dataclass
class RegressionStepResult:
    """Kết quả dự đoán của 1 sample tại 1 bước online."""
    index: int
    y_pred_log: dict[str, float]
    y_true_log: dict[str, float]
    y_pred_raw: dict[str, float]
    y_true_raw: dict[str, float]
    abs_errors: dict[str, float]


class SinglePARegressor:
    """Wrapper mỏng quanh SGDRegressor configured as PA-I/PA-II."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1,
                 loss: str = "epsilon_insensitive"):
        sgd_loss, lr_mode = _PA_LOSS_MAP.get(loss, ("epsilon_insensitive", "pa1"))
        self.model = SGDRegressor(
            loss=sgd_loss, penalty=None, learning_rate=lr_mode,
            eta0=C, epsilon=epsilon, max_iter=1, tol=None,
            random_state=42, warm_start=False,
        )
        self._initialized = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.partial_fit(X, y)
        self._initialized = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_single(self, x: np.ndarray) -> float:
        return float(self.model.predict(x.reshape(1, -1))[0])


class QuadRegressionHead:
    """
    N PA Regressor chạy song song trên cùng input vector z.

    Parameters
    ----------
    target_names : list[str] | None
        Tên các regression targets. Mặc định = TARGET_NAMES (Twitter).
        Reddit ví dụ: ["score", "num_comments", "upvote_ratio"].

    Usage
    -----
    head = QuadRegressionHead(C=1.0, epsilon=0.1)
    results = head.run_online(Z, Y_reg)
    preds   = head.predict(Z_new)
    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        loss: str = "epsilon_insensitive",
        target_names: list[str] | None = None,
    ):
        self.target_names = target_names if target_names is not None else TARGET_NAMES
        self.models: dict[str, SinglePARegressor] = {
            name: SinglePARegressor(C=C, epsilon=epsilon, loss=loss)
            for name in self.target_names
        }
        self.history: list[RegressionStepResult] = []

    # ------------------------------------------------------------------
    # Online training loop
    # ------------------------------------------------------------------

    def run_online(
        self, Z: np.ndarray, Y_reg: np.ndarray
    ) -> list[RegressionStepResult]:
        """
        Predict-then-update online loop.

        Parameters
        ----------
        Z     : (n, feature_dim)
        Y_reg : (n, len(target_names))  log1p-transformed targets
        """
        assert Y_reg.shape[1] == len(self.target_names), (
            f"Y_reg has {Y_reg.shape[1]} columns but target_names has {len(self.target_names)}"
        )
        n = len(Z)
        self.history = []

        # Warm-start với sample đầu tiên
        for i, name in enumerate(self.target_names):
            self.models[name].partial_fit(Z[:1], Y_reg[:1, i])

        for idx in range(1, n):
            x = Z[idx: idx + 1]
            y_true_log = {name: float(Y_reg[idx, i]) for i, name in enumerate(self.target_names)}
            y_pred_log = {name: float(self.models[name].predict(x)[0]) for name in self.target_names}

            y_pred_raw = {k: float(np.expm1(max(v, 0))) for k, v in y_pred_log.items()}
            y_true_raw = {k: float(np.expm1(v)) for k, v in y_true_log.items()}
            abs_errors = {k: abs(y_pred_log[k] - y_true_log[k]) for k in self.target_names}

            self.history.append(RegressionStepResult(
                index=idx,
                y_pred_log=y_pred_log, y_true_log=y_true_log,
                y_pred_raw=y_pred_raw, y_true_raw=y_true_raw,
                abs_errors=abs_errors,
            ))

            for i, name in enumerate(self.target_names):
                self.models[name].partial_fit(x, [float(Y_reg[idx, i])])

        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        """Batch predict. Returns dict target → raw-scale array."""
        return {
            name: np.expm1(np.clip(self.models[name].predict(Z), 0, _LOG_CLIP))
            for name in self.target_names
        }

    def predict_single(self, z: np.ndarray) -> dict[str, float]:
        z = z.reshape(1, -1)
        return {
            name: float(np.expm1(min(max(self.models[name].predict_single(z.flatten()), 0), _LOG_CLIP)))
            for name in self.target_names
        }

    def predict_log(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        return {name: self.models[name].predict(Z) for name in self.target_names}

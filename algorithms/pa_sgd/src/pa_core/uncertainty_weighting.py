"""
Uncertainty Weighting theo Kendall et al. (CVPR 2018) — platform-agnostic.

L_total = (1/2σ₁²) * L_reg + (1/σ₂²) * L_cls + log(σ₁) + log(σ₂)

σ₁ (regression), σ₂ (classification) được học online qua EMA của loss.
Task nào nhiễu hơn (σ lớn) → weight nhỏ hơn → ít ảnh hưởng lên tổng loss.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .regression_head import TARGET_NAMES, RegressionStepResult, _LOG_CLIP
from .classification_head import ClassificationStepResult


class _ManualPAReg:
    """PA Regressor thủ công — dùng trong UncertaintyWeightedTrainer.

    Hỗ trợ AVERAGED weights (trung bình w qua các bước online) -> giảm phụ thuộc
    thứ tự stream, ổn định hơn nhiều. Gọi finalize() sau khi train xong.
    """

    def __init__(self, n_features: int, epsilon: float = 0.1):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.epsilon = epsilon
        self.w_sum = np.zeros(n_features, dtype=np.float64)   # tích lũy cho averaging
        self.b_sum = 0.0
        self.n_upd = 0

    def predict_one(self, x: np.ndarray) -> float:
        return float(x @ self.w + self.b)

    def update(self, x: np.ndarray, y: float, C_eff: float) -> float:
        """Cập nhật với C_eff = C / (2σ₁²). Trả về epsilon-insensitive loss."""
        y_hat = self.predict_one(x)
        residual = y - y_hat
        loss = max(0.0, abs(residual) - self.epsilon)

        if loss > 1e-12:
            norm_sq = float(x @ x) + 1.0
            tau = min(C_eff, loss / norm_sq)
            sign = 1.0 if residual > 0 else -1.0
            self.w += tau * sign * x
            self.b += tau * sign

        self.w_sum += self.w           # average qua MỌI bước (kể cả bước không đổi)
        self.b_sum += self.b
        self.n_upd += 1
        return loss

    def finalize(self) -> None:
        """Thay w bằng trung bình tích lũy (averaged PA) cho inference."""
        if self.n_upd > 0:
            self.w = self.w_sum / self.n_upd
            self.b = self.b_sum / self.n_upd


class _ManualPACls:
    """PA Classifier thủ công — dùng trong UncertaintyWeightedTrainer."""

    def __init__(self, n_features: int, n_classes: int = 4):
        self.W = np.zeros((n_classes, n_features), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)
        self.n_classes = n_classes
        self.W_sum = np.zeros((n_classes, n_features), dtype=np.float64)   # averaging
        self.b_sum = np.zeros(n_classes, dtype=np.float64)
        self.n_upd = 0

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b

    def finalize(self) -> None:
        """Thay W bằng trung bình tích lũy (averaged PA) cho inference."""
        if self.n_upd > 0:
            self.W = self.W_sum / self.n_upd
            self.b = self.b_sum / self.n_upd

    def predict_one(self, x: np.ndarray) -> int:
        return int(np.argmax(self.scores(x)))

    def update(self, x: np.ndarray, y_true: int, C_eff: float) -> float:
        """
        Cập nhật với C_eff = C / σ₂².
        Trả về mean hinge loss = sum(l_c) / (n_classes - 1).

        Mỗi tau scale bởi 1/(K-1) để lực update tỉ lệ với mean loss,
        đồng nhất với định nghĩa L_cls trong uncertainty weighting.
        """
        s = self.scores(x)
        norm_sq = float(x @ x)
        K = self.n_classes
        total_loss = 0.0

        for c in range(K):
            if c == y_true:
                continue
            margin = 1.0 - s[y_true] + s[c]
            loss_c = max(0.0, margin)
            total_loss += loss_c
            if loss_c > 1e-12:
                denom = 2.0 * norm_sq if norm_sq > 1e-12 else 1e-12
                tau = min(C_eff / (K - 1), loss_c / denom)
                self.W[y_true] += tau * x
                self.b[y_true] += tau
                self.W[c]      -= tau * x
                self.b[c]      -= tau

        self.W_sum += self.W           # average qua MỌI bước
        self.b_sum += self.b
        self.n_upd += 1
        return total_loss / (K - 1)


# ──────────────────────────────────────────────────────────────────────────
# Online SGD heads — CÙNG objective với PA (epsilon-insensitive reg + multiclass
# hinge cls), chỉ khác LUẬT CẬP NHẬT: thay bước closed-form τ của PA bằng bước
# (sub)gradient với learning-rate schedule η_t = η₀/√t + L2 weight decay.
# Cùng feature, cùng target, cùng uncertainty weighting -> so sánh PA vs SGD
# có kiểm soát (chỉ optimizer khác).
#
#   PA  reg:  w += min(C_eff, loss/‖x‖²) · sign(r) · x     (bước tự co giãn)
#   SGD reg:  w += (η₀/√t) · sign(r) · x                    (bước theo lịch LR)
#
# η₀ = giá trị c_eff truyền vào (đã gồm uncertainty weight C/2σ²), nên SGD đang
# đi gradient descent trên ĐÚNG L_total đa nhiệm. Polyak–Ruppert averaging
# (finalize) song song với averaged-PA.
# ──────────────────────────────────────────────────────────────────────────

class _SGDReg:
    """Online SGD regressor (subgradient của epsilon-insensitive loss)."""

    def __init__(self, n_features: int, epsilon: float = 0.1, l2: float = 0.0):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.epsilon = epsilon
        self.l2 = l2
        self.t = 0                                            # bước cho schedule η₀/√t
        self.w_sum = np.zeros(n_features, dtype=np.float64)   # ASGD averaging
        self.b_sum = 0.0
        self.n_upd = 0

    def predict_one(self, x: np.ndarray) -> float:
        return float(x @ self.w + self.b)

    def update(self, x: np.ndarray, y: float, c_eff: float) -> float:
        """c_eff = η₀·weight_reg (base LR đã gồm uncertainty weight). Trả về eps-insensitive loss."""
        residual = y - self.predict_one(x)                   # tính grad tại w hiện tại
        loss = max(0.0, abs(residual) - self.epsilon)

        self.t += 1
        eta = c_eff / np.sqrt(self.t)                        # η_t = η₀/√t

        if self.l2 > 0.0:                                     # subgrad của (λ/2)‖w‖² (không phạt bias)
            self.w *= (1.0 - eta * self.l2)
        if loss > 1e-12:                                     # subgrad của eps-insensitive loss
            sign = 1.0 if residual > 0 else -1.0
            self.w += eta * sign * x
            self.b += eta * sign

        self.w_sum += self.w
        self.b_sum += self.b
        self.n_upd += 1
        return loss

    def finalize(self) -> None:
        if self.n_upd > 0:
            self.w = self.w_sum / self.n_upd
            self.b = self.b_sum / self.n_upd


class _SGDCls:
    """Online SGD classifier (subgradient của multiclass hinge loss)."""

    def __init__(self, n_features: int, n_classes: int = 4, l2: float = 0.0,
                 class_weights: np.ndarray | None = None):
        self.W = np.zeros((n_classes, n_features), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)
        self.n_classes = n_classes
        self.l2 = l2
        # Cost-sensitive: trọng số mỗi lớp (None = đồng đều). Lớp hiếm w lớn -> đẩy
        # margin mạnh hơn -> tăng recall/F1 lớp hiếm mà KHÔNG cần tăng LR toàn cục.
        self.class_weights = None if class_weights is None else np.asarray(class_weights, float)
        self.t = 0
        self.W_sum = np.zeros((n_classes, n_features), dtype=np.float64)
        self.b_sum = np.zeros(n_classes, dtype=np.float64)
        self.n_upd = 0

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b

    def predict_one(self, x: np.ndarray) -> int:
        return int(np.argmax(self.scores(x)))

    def update(self, x: np.ndarray, y_true: int, c_eff: float) -> float:
        """c_eff = η₀·weight_cls. Trả về mean hinge loss = sum(l_c)/(K-1)."""
        s = self.scores(x)                                   # grad tại W hiện tại
        K = self.n_classes
        self.t += 1
        eta = c_eff / np.sqrt(self.t)

        if self.l2 > 0.0:
            self.W *= (1.0 - eta * self.l2)

        total_loss = 0.0
        cw = 1.0 if self.class_weights is None else float(self.class_weights[y_true])
        step = eta * cw / (K - 1)                            # chia (K-1): subgrad của mean-hinge
        for c in range(K):
            if c == y_true:
                continue
            margin = 1.0 - s[y_true] + s[c]
            loss_c = max(0.0, margin)
            total_loss += loss_c
            if loss_c > 1e-12:                               # subgrad hinge cho cặp (y_true, c)
                self.W[y_true] += step * x
                self.b[y_true] += step
                self.W[c]      -= step * x
                self.b[c]      -= step

        self.W_sum += self.W
        self.b_sum += self.b
        self.n_upd += 1
        return total_loss / (K - 1)

    def finalize(self) -> None:
        if self.n_upd > 0:
            self.W = self.W_sum / self.n_upd
            self.b = self.b_sum / self.n_upd


class UncertaintyWeighter:
    """
    Theo dõi và cập nhật σ₁, σ₂ online qua EMA của loss.

    σ² ← (1−α) * σ² + α * loss_t  →  σ = sqrt(σ²)
    """

    def __init__(
        self,
        init_sigma_reg: float = 1.0,
        init_sigma_cls: float = 1.0,
        ema_alpha: float = 0.05,
        min_sigma: float = 0.1,
        max_sigma: float = 10.0,
    ):
        self.sigma_reg = init_sigma_reg
        self.sigma_cls = init_sigma_cls
        self.alpha = ema_alpha
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_reg_history: list[float] = [init_sigma_reg]
        self.sigma_cls_history: list[float] = [init_sigma_cls]

    def update(self, l_reg: float, l_cls: float) -> None:
        new_var_reg = (1 - self.alpha) * self.sigma_reg**2 + self.alpha * max(l_reg, 0.0)
        new_var_cls = (1 - self.alpha) * self.sigma_cls**2 + self.alpha * max(l_cls, 0.0)
        self.sigma_reg = float(np.clip(np.sqrt(max(new_var_reg, 1e-8)), self.min_sigma, self.max_sigma))
        self.sigma_cls = float(np.clip(np.sqrt(max(new_var_cls, 1e-8)), self.min_sigma, self.max_sigma))
        self.sigma_reg_history.append(self.sigma_reg)
        self.sigma_cls_history.append(self.sigma_cls)

    @property
    def weight_reg(self) -> float:
        return 1.0 / (2.0 * self.sigma_reg ** 2)

    @property
    def weight_cls(self) -> float:
        return 1.0 / (self.sigma_cls ** 2)

    def c_eff_reg(self, C: float) -> float:
        return C * self.weight_reg

    def c_eff_cls(self, C: float) -> float:
        return C * self.weight_cls

    def total_loss(self, l_reg: float, l_cls: float) -> float:
        """L_total = (1/2σ₁²)*L_reg + (1/σ₂²)*L_cls + log(σ₁) + log(σ₂)"""
        return (
            self.weight_reg * l_reg
            + self.weight_cls * l_cls
            + np.log(max(self.sigma_reg, 1e-8))
            + np.log(max(self.sigma_cls, 1e-8))
        )


@dataclass
class UWStepResult:
    index: int
    y_pred_log: dict[str, float]
    y_true_log: dict[str, float]
    y_pred_raw: dict[str, float]
    y_true_raw: dict[str, float]
    reg_losses: dict[str, float]
    l_reg: float
    y_pred_cls: int
    y_true_cls: int
    l_cls: float
    sigma_reg: float
    sigma_cls: float
    weight_reg: float
    weight_cls: float
    l_total: float


class UncertaintyWeightedTrainer:
    """
    Trainer kết hợp Uncertainty Weighting vào PA online learning.

    Tại mỗi bước t:
      1. Predict ŷ_reg, ŷ_cls
      2. Tính l_reg (mean epsilon-insensitive), l_cls (mean hinge / (K-1))
      3. L_total = (1/2σ₁²)*l_reg + (1/σ₂²)*l_cls + log σ₁ + log σ₂
      4. Update σ₁, σ₂ via EMA
      5. C_eff_reg = C/(2σ₁²),  C_eff_cls = C/σ₂²
      6. Update models

    Parameters
    ----------
    target_names : list[str] | None
        Tên regression targets. Mặc định = TARGET_NAMES (Twitter).
    """

    def __init__(
        self,
        n_features: int,
        C: float = 1.0,
        epsilon: float = 0.1,
        init_sigma_reg: float = 1.0,
        init_sigma_cls: float = 1.0,
        ema_alpha: float = 0.05,
        n_classes: int = 4,
        target_names: list[str] | None = None,
        algo: str = "pa",
        sgd_l2: float = 0.0,
    ):
        self.C = C
        self.epsilon = epsilon
        self.n_features = n_features
        self.algo = algo
        self.target_names = target_names if target_names is not None else TARGET_NAMES

        # Cùng objective + uncertainty weighting; chỉ khác luật cập nhật của head.
        if algo == "sgd":
            self.reg_models = {
                name: _SGDReg(n_features, epsilon=epsilon, l2=sgd_l2)
                for name in self.target_names
            }
            self.cls_model = _SGDCls(n_features, n_classes=n_classes, l2=sgd_l2)
        elif algo == "pa":
            self.reg_models = {
                name: _ManualPAReg(n_features, epsilon=epsilon)
                for name in self.target_names
            }
            self.cls_model = _ManualPACls(n_features, n_classes=n_classes)
        else:
            raise ValueError(f"Unknown algo {algo!r} (expected 'pa' or 'sgd')")
        self.uw = UncertaintyWeighter(
            init_sigma_reg=init_sigma_reg,
            init_sigma_cls=init_sigma_cls,
            ema_alpha=ema_alpha,
        )
        self.history: list[UWStepResult] = []

    def run_online(
        self,
        Z: np.ndarray,
        Y_reg: np.ndarray,
        y_cls: np.ndarray,
    ) -> list[UWStepResult]:
        """
        Predict-then-update với uncertainty weighting.

        Parameters
        ----------
        Z     : (n, n_features)
        Y_reg : (n, len(target_names))  log1p targets
        y_cls : (n,)  int labels
        """
        assert Y_reg.shape[1] == len(self.target_names)
        n = len(Z)
        self.history = []

        # Warm-start
        x0 = Z[0]
        for i, name in enumerate(self.target_names):
            self.reg_models[name].update(x0, float(Y_reg[0, i]), self.C)
        self.cls_model.update(x0, int(y_cls[0]), self.C)

        for idx in range(1, n):
            x = Z[idx]

            # ── 1. Predict ───────────────────────────────────────────────
            y_pred_log = {name: self.reg_models[name].predict_one(x) for name in self.target_names}
            y_true_log = {name: float(Y_reg[idx, i]) for i, name in enumerate(self.target_names)}
            y_pred_raw = {k: float(np.expm1(np.clip(v, 0, _LOG_CLIP))) for k, v in y_pred_log.items()}
            y_true_raw = {k: float(np.expm1(v)) for k, v in y_true_log.items()}
            y_pred_cls = self.cls_model.predict_one(x)
            y_true_cls = int(y_cls[idx])

            # ── 2. Compute losses ────────────────────────────────────────
            reg_losses = {
                name: max(0.0, abs(y_true_log[name] - y_pred_log[name]) - self.epsilon)
                for name in self.target_names
            }
            l_reg = sum(reg_losses.values()) / len(self.target_names)

            K = self.cls_model.n_classes
            s = self.cls_model.scores(x)
            l_cls = sum(
                max(0.0, 1.0 - s[y_true_cls] + s[c])
                for c in range(K) if c != y_true_cls
            ) / (K - 1)

            # ── 3–6. Total loss → update σ → compute C_eff → update models
            l_total = self.uw.total_loss(l_reg, l_cls)
            self.uw.update(l_reg, l_cls)
            c_eff_reg = self.uw.c_eff_reg(self.C)
            c_eff_cls = self.uw.c_eff_cls(self.C)

            for i, name in enumerate(self.target_names):
                self.reg_models[name].update(x, float(Y_reg[idx, i]), c_eff_reg)
            self.cls_model.update(x, y_true_cls, c_eff_cls)

            self.history.append(UWStepResult(
                index=idx,
                y_pred_log=y_pred_log, y_true_log=y_true_log,
                y_pred_raw=y_pred_raw, y_true_raw=y_true_raw,
                reg_losses=reg_losses, l_reg=l_reg,
                y_pred_cls=y_pred_cls, y_true_cls=y_true_cls,
                l_cls=l_cls, sigma_reg=self.uw.sigma_reg,
                sigma_cls=self.uw.sigma_cls, weight_reg=self.uw.weight_reg,
                weight_cls=self.uw.weight_cls, l_total=l_total,
            ))

        return self.history

    def finalize(self) -> None:
        """Chuyển mọi head sang averaged weights (gọi 1 lần sau khi train xong)."""
        for m in self.reg_models.values():
            m.finalize()
        self.cls_model.finalize()

    def predict(self, Z: np.ndarray) -> dict:
        reg_preds = {}
        for name in self.target_names:
            log_preds = np.array([self.reg_models[name].predict_one(Z[i]) for i in range(len(Z))])
            reg_preds[name] = np.expm1(np.clip(log_preds, 0, _LOG_CLIP))
        cls_preds = np.array([self.cls_model.predict_one(Z[i]) for i in range(len(Z))])
        return {**reg_preds, "label": cls_preds}

    def predict_single(self, x: np.ndarray) -> dict:
        x = x.flatten()
        reg = {
            name: float(np.expm1(np.clip(self.reg_models[name].predict_one(x), 0, _LOG_CLIP)))
            for name in self.target_names
        }
        return {**reg, "label": self.cls_model.predict_one(x)}

    @property
    def sigma_history(self) -> dict[str, list[float]]:
        return {"sigma_reg": self.uw.sigma_reg_history, "sigma_cls": self.uw.sigma_cls_history}

    @property
    def final_weights(self) -> dict[str, float]:
        return {
            "sigma_reg": self.uw.sigma_reg,
            "sigma_cls": self.uw.sigma_cls,
            "weight_reg (1/2σ₁²)": self.uw.weight_reg,
            "weight_cls (1/σ₂²)":  self.uw.weight_cls,
        }
